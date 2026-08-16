"""Core context type and root context implementation.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/context.ts`` and the proxy traps of
``vendor/cordis/src/reflect.ts``).

Python mapping: the root context is not a real proxy. Property reads go
through ``Context.__getattr__``, which first walks the parent chain (the
prototype inheritance of ``extend()``) and then resolves services through
the reflection layer with inject gating. The service-method mixins that the
TypeScript ``ReflectService`` constructor registers as accessors are plain
``Context`` methods here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .events import EventsService
from .fiber import EffectDisposer, Fiber
from .logger import LoggerService
from .reflect import AccessorProperty, ReflectService
from .registry import RegistryService
from .utils import ChainMap, rebind

FILTER_ATTR = '_cordis_filter'
ISOLATE_ATTR = '_cordis_isolate'
INTERCEPT_ATTR = '_cordis_intercept'

_RESERVED_WORDS = ('prototype', 'then')
_MISSING = object()


def _is_special_property(name: str) -> bool:
    return name.startswith('_') or name in _RESERVED_WORDS or name.isdigit()


def _find_inherited(ctx: Context | None, name: str) -> Any:
    while ctx is not None:
        if name in ctx.__dict__:
            return ctx.__dict__[name]
        ctx = ctx.__dict__.get('_parent')
    return _MISSING


class Context:
    """Root and child dependency containers for Cordis plugins.

    Normal property reads go through the service resolver, while
    ``extend()``, ``isolate()``, and ``intercept()`` create scoped child
    contexts without mutating their parent.
    """

    #: Meta key under which ``extend()`` installs a listener filter consulted
    #: on every event dispatch (``Context[symbols.filter]`` in TypeScript).
    filter = FILTER_ATTR

    # Typed surface of the root-installable built-ins. Values live in the
    # root instance dict and resolve through the parent chain.
    root: Context
    base_url: str | None
    fiber: Fiber
    events: EventsService
    logger: LoggerService
    reflect: ReflectService
    registry: RegistryService

    def __init__(self) -> None:
        self.__dict__['_parent'] = None
        self.__dict__[ISOLATE_ATTR] = ChainMap()
        self.__dict__[INTERCEPT_ATTR] = ChainMap()
        self.__dict__['root'] = self
        self.__dict__['base_url'] = None
        self.__dict__['fiber'] = Fiber(self, {}, {}, None)
        self.__dict__['reflect'] = ReflectService(self)
        self.__dict__['registry'] = RegistryService(self)
        self.__dict__['events'] = EventsService(self)
        self.__dict__['logger'] = LoggerService(self)
        # Effects registered during construction are cleared at once: the
        # built-in services never unload.
        self.fiber._disposables.clear()

    def __repr__(self) -> str:
        try:
            return f'Context <{self.fiber.name}>'
        except AttributeError:
            return 'Context <uninitialized>'

    @classmethod
    def is_(cls, value: Any) -> bool:
        """Return true for Cordis contexts."""
        return isinstance(value, Context)

    def extend(self, meta: dict[str, Any] | None = None) -> Context:
        """Create a child context with extra metadata on top of this scope.

        The child inherits every property of this context; entries of
        ``meta`` shadow the inherited ones. The parent is not mutated.
        """
        child = Context.__new__(Context)
        child.__dict__['_parent'] = self
        if meta:
            child.__dict__.update(meta)
        return child

    def isolate(self, name: str, label: Any = None) -> Context:
        """Create a child context with an independent scope for ``name``.

        Passing the same ``label`` to two ``isolate()`` calls joins their
        scopes.
        """
        from .reflect import Label

        shadow = ChainMap(parent=self._cordis_isolate)
        shadow[name] = label if label is not None else Label(name)
        return self.extend({ISOLATE_ATTR: shadow})

    def intercept(self, name: str, config: Any) -> Context:
        """Add service-specific intercept config for plugins started below.

        Entries added closer to the root apply first; the parent context is
        not affected.
        """
        intercept = ChainMap(parent=self._cordis_intercept)
        intercept[name] = config
        return self.extend({INTERCEPT_ATTR: intercept})

    def __getattr__(self, name: str) -> Any:
        # Only misses reach here: own attributes and class members resolve
        # through normal lookup, mirroring the proxy's `Reflect.has` check.
        parent = self.__dict__.get('_parent')
        value = _find_inherited(parent, name)
        if value is not _MISSING:
            return rebind(self, value)
        if _is_special_property(name):
            raise AttributeError(name)
        return self._resolve_property(name)

    def _resolve_property(self, name: str) -> Any:
        error = AttributeError(f'cannot get property "{name}" without inject')
        reflect: ReflectService = self.reflect
        prop = reflect.props.get(name)
        if isinstance(prop, AccessorProperty):
            return prop.get(self, error)
        if self.fiber.runtime is None:
            return reflect.get(name, False)
        return self.events.waterfall(
            'internal/get',
            self,
            name,
            error,
            lambda *_args: self._walk_service(name, error),
        )

    def _walk_service(self, name: str, error: AttributeError) -> Any:
        key = self.root._cordis_isolate.get(name)
        fiber: Fiber = self.fiber
        while True:
            store = fiber.store
            if store is not None:
                impl = store.get(name)
                if impl is not None:
                    return rebind(self, impl.value)
            if name in fiber.inject:
                raise AttributeError(
                    f'cannot get required service "{name}" in inactive context'
                ) from None
            if fiber.runtime is None:
                raise error
            if fiber.parent._cordis_isolate.get(name) is not key:
                raise error
            fiber = fiber.parent.fiber

    def __setattr__(self, name: str, value: Any) -> None:
        if _is_special_property(name):
            object.__setattr__(self, name, value)
            return
        reflect = _find_inherited(self, 'reflect')
        if reflect is _MISSING:
            object.__setattr__(self, name, value)
            return
        prop = reflect.props.get(name)
        if prop is None:
            fiber = _find_inherited(self, 'fiber')
            if fiber is _MISSING or fiber.runtime is None:
                object.__setattr__(self, name, value)
                return
            raise AttributeError(f'cannot set property "{name}" without provide')
        error = RuntimeError(f'cannot set property "{name}" without provide')
        if isinstance(prop, AccessorProperty):
            if prop.set is None or not prop.set(self, value, error):
                raise AttributeError(f'cannot set property "{name}"')
            return
        result = self.events.waterfall(
            'internal/set',
            self,
            name,
            value,
            error,
            lambda *_args: rebind(self, reflect).set(name, value, error),
        )
        if not result:
            raise AttributeError(f'cannot set property "{name}"')

    # Mixed-in service methods. In TypeScript these are accessors registered
    # by the ReflectService constructor (`this.mixin('reflect', ...)` etc.);
    # here they are plain methods forwarding with caller-context rebinding.

    def get(self, name: str, strict: bool = True) -> Any:
        """Read a service from the store without the inject requirement."""
        return self.reflect.get(name, strict)

    def set(self, name: str, value: Any) -> bool:
        """Overwrite a provided service's value (same-fiber providers only)."""
        return self.reflect.set(name, value)

    def provide(
        self,
        name: str,
        value: Any = None,
        check: Callable[[], bool] | None = None,
    ) -> EffectDisposer:
        """Register a service implementation owned by the current fiber."""
        return self.reflect.provide(name, value, check)

    def accessor(self, name: str, *, get: Any, set: Any = None) -> EffectDisposer:
        """Define a computed context property backed by get/set hooks."""
        return self.reflect.accessor(name, get=get, set=set)

    def mixin(self, source: Any, mixins: list[str] | dict[str, str]) -> EffectDisposer:
        """Expose selected members of a service directly on ``ctx``."""
        return self.reflect.mixin(source, mixins)

    @property
    def runtime(self) -> Any:
        """The plugin runtime of the fiber owning this context, if any."""
        return self.fiber.runtime

    def effect(
        self,
        execute: Callable[[], Any],
        label: str = 'anonymous',
    ) -> EffectDisposer:
        """Register a cleanup-aware effect on the current fiber."""
        return self.fiber.effect(execute, label)

    def plugin(self, plugin: Any, config: Any = None) -> Fiber:
        """Load a plugin in the current context; awaitable fiber handle."""
        return self.registry.plugin(plugin, config)

    def inject(self, deps: Any, callback: Callable[..., Any]) -> Fiber:
        """Run a callback once the requested services are available."""
        return self.registry.inject(deps, callback)

    def on(
        self,
        name: str,
        listener: Callable[..., Any],
        options: bool | dict[str, Any] | None = None,
    ) -> Any:
        """Register an event listener owned by the current fiber."""
        return self.events.on(name, listener, options)

    def once(
        self,
        name: str,
        listener: Callable[..., Any],
        options: bool | dict[str, Any] | None = None,
    ) -> Any:
        """Register an event listener that disposes itself after one call."""
        return self.events.once(name, listener, options)

    def parallel(self, *args: Any) -> Any:
        """Dispatch an event, running all listeners concurrently."""
        return self.events.parallel(*args)

    def emit(self, *args: Any) -> None:
        """Dispatch an event synchronously, ignoring listener return values."""
        self.events.emit(*args)

    def serial(self, *args: Any) -> Any:
        """Dispatch an event, awaiting listeners in order until one bails."""
        return self.events.serial(*args)

    def bail(self, *args: Any) -> Any:
        """Dispatch an event, calling listeners in order until one bails."""
        return self.events.bail(*args)

    def waterfall(self, *args: Any) -> Any:
        """Dispatch an event whose last argument is a ``next`` continuation."""
        return self.events.waterfall(*args)
