"""Reflection and service-resolution layer installed as ``ctx.reflect``.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/reflect.ts``). The context proxy traps are mapped onto
``Context.__getattr__``/``Context.__setattr__`` in ``context.py``; accessor
hooks receive ``(ctx, error)`` / ``(ctx, value, error)`` since the traceable
receiver machinery is not ported.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .fiber import FiberState
from .utils import Tracker, rebind

if TYPE_CHECKING:
    from .context import Context
    from .fiber import EffectDisposer, Fiber


class Label:
    """Isolation scope label (JavaScript ``Symbol(name)``)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f'Label({self.name})'


class ServiceProperty:
    """Context property backed by a provided implementation."""

    type = 'service'


class AccessorProperty:
    """Computed context property backed by custom get/set hooks."""

    type = 'accessor'

    def __init__(
        self,
        get: Callable[[Context, BaseException], Any],
        set: Callable[[Context, Any, BaseException], bool] | None = None,
    ) -> None:
        self.get = get
        self.set = set


class Impl:
    """Concrete service implementation record stored in the root store."""

    __slots__ = ('check', 'fiber', 'name', 'value')

    def __init__(
        self,
        name: str,
        fiber: Fiber,
        value: Any = None,
        check: Callable[[], bool] | None = None,
    ) -> None:
        self.name = name
        self.fiber = fiber
        self.value = value
        self.check = check


class ReflectService:
    """Reflection service powering service resolution, accessors, and mixins."""

    _cordis_tracker = Tracker(property='ctx', no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        # Service implementations, keyed by isolation label.
        self.store: dict[Any, Impl] = {}
        # Declared context properties (services and accessors), by name.
        self.props: dict[str, ServiceProperty | AccessorProperty] = {}

    def get(self, name: str, strict: bool = True) -> Any:
        """Read a service from the store without the inject requirement.

        With ``strict`` (the default) only implementations whose providing
        fiber is currently active are returned.
        """
        impl = self._get_impl(name, strict)
        if impl is None:
            return None
        return rebind(self.ctx, impl.value)

    def _get_impl(self, name: str, strict: bool = True) -> Impl | None:
        key = self.ctx._cordis_isolate.get(name)
        if key is None:
            return None
        impl = self.store.get(key)
        if impl is None:
            return None
        if strict and impl.fiber.state is not FiberState.ACTIVE:
            return None
        return impl

    def set(self, name: str, value: Any, error: BaseException | None = None) -> bool:
        """Overwrite a provided service's value (same-fiber providers only)."""
        key = self.ctx._cordis_isolate.get(name)
        impl = self.store.get(key)
        if impl is None:
            raise RuntimeError(f'cannot set property "{name}" without provide')
        if impl.fiber is not self.ctx.fiber:
            raise RuntimeError(f'cannot set property "{name}" in multiple fibers')
        impl.value = value
        return True

    def provide(
        self,
        name: str,
        value: Any = None,
        check: Callable[[], bool] | None = None,
    ) -> EffectDisposer:
        """Register a service implementation owned by the current fiber.

        The service becomes visible to dependents in the same isolation scope
        once the fiber is active; it is unregistered (waking dependents) when
        the returned disposer runs or the fiber unloads. Throws if the name
        is already provided in this scope or declared as an accessor.
        """
        def effect() -> Callable[[], Any]:
            prop = self.props.get(name)
            if prop is not None and prop.type != 'service':
                raise RuntimeError(
                    f'property "{name}" is already declared as {prop.type}'
                )
            self.props[name] = ServiceProperty()

            root_isolate = self.ctx.root._cordis_isolate
            root_isolate.setdefault(name, Label(name))
            key = self.ctx._cordis_isolate.get(name)
            impl = Impl(name, self.ctx.fiber, value, check)
            existing = self.store.get(key)
            if existing is not None:
                raise RuntimeError(
                    f'service "{name}" has been registered at <{existing.fiber.name}>'
                )
            self.store[key] = impl
            fiber = self.ctx.fiber
            assert fiber.store is not None
            fiber.store[name] = impl
            if fiber.state is FiberState.ACTIVE:
                self.notify([name])

            async def dispose() -> None:
                del self.store[key]
                fibers = self.notify([name])
                await asyncio.gather(
                    *(dependent.wait() for dependent in fibers),
                    return_exceptions=True,
                )
                # ensure self access before dependencies cleanup
                assert fiber.store is not None
                del fiber.store[name]

            return dispose

        return self.ctx.fiber.effect(effect, f'ctx.provide({name!r})')

    def notify(
        self,
        names: list[str],
        filter: Callable[[Context, str], bool] | None = None,
    ) -> list[Fiber]:
        """Re-evaluate every fiber that requires one of the given services.

        Returns the fibers whose dependency state was refreshed.
        """
        if filter is None:
            provider_ctx = self.ctx

            def filter(ctx: Context, name: str) -> bool:
                return bool(
                    ctx._cordis_isolate.get(name)
                    == provider_ctx._cordis_isolate.get(name)
                )

        fibers: list[Fiber] = []
        for runtime in list(self.ctx.registry.values()):
            for fiber in list(runtime.fibers):
                has_update = False
                for name in names:
                    if name not in fiber.inject:
                        continue
                    if not filter(fiber.ctx, name):
                        continue
                    has_update = True
                    fiber._check_impl(name)
                if not has_update:
                    continue
                fiber._refresh()
                fibers.append(fiber)
        for name in names:
            scope = self.ctx.extend({})
            scope.__dict__['_cordis_filter'] = (
                lambda target, _name=name: filter(target, _name)
            )
            impl = self._get_impl(name, False)
            self.ctx.events.emit(
                scope, 'internal/service', name, impl.value if impl else None
            )
        return fibers

    def accessor(
        self,
        name: str,
        *,
        get: Callable[[Context, BaseException], Any],
        set: Callable[[Context, Any, BaseException], bool] | None = None,
    ) -> EffectDisposer:
        """Define a computed context property backed by get/set hooks."""
        def effect() -> Callable[[], Any]:
            if name in self.props:
                raise RuntimeError(
                    f'property "{name}" is already declared as {self.props[name].type}'
                )
            self.props[name] = AccessorProperty(get, set)
            return lambda: self.props.pop(name, None)

        return self.ctx.fiber.effect(effect, f'ctx.accessor({name!r})')

    def mixin(self, source: Any, mixins: list[str] | dict[str, str]) -> EffectDisposer:
        """Expose selected members of a service directly on ``ctx``.

        ``mixins`` is a list of forwarded keys or a source-key → ctx-key map.
        Mixins are removed when the current fiber unloads.
        """
        if isinstance(mixins, dict):
            entries = list(mixins.items())
        else:
            entries = [(key, key) for key in mixins]

        def effect() -> Any:
            for key, value in entries:
                def get(ctx: Context, error: BaseException, _key: str = key) -> Any:
                    # TODO enhance error message
                    service = (
                        getattr(ctx, source) if isinstance(source, str) else source
                    )
                    if service is None:
                        return None
                    return getattr(service, _key)

                def set(
                    ctx: Context,
                    new_value: Any,
                    error: BaseException,
                    _key: str = key,
                ) -> bool:
                    service = (
                        getattr(ctx, source) if isinstance(source, str) else source
                    )
                    setattr(service, _key, new_value)
                    return True

                yield self.accessor(value, get=get, set=set)

        return self.ctx.fiber.effect(effect, f'ctx.mixin({source!r})')
