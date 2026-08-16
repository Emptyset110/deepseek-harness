"""Plugin registry, dependency injection, and plugin entrypoint types.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/registry.ts``).

Python plugin shapes: a plain function ``(ctx, config)``, a class
constructed with ``(ctx, config)``, or an object with an ``apply(ctx,
config)`` method. Optional attributes ``name``, ``Config`` (a callable
``config -> validated config``), and ``inject`` (a list of service names or
a name → intercept-config dict) are read with ``getattr``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from .fiber import Fiber
from .utils import DisposableList, Tracker

if TYPE_CHECKING:
    from .context import Context


def resolve_inject(inject: Any, result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert array/object inject declarations into a plain map.

    Each service name maps to its intercept config or ``None``.
    """
    if result is None:
        result = {}
    if not inject:
        return result
    if isinstance(inject, (list, tuple, set)):
        for name in inject:
            result[name] = None
    else:
        for name, config in inject.items():
            result[name] = config if config is not None else None
    return result


class PluginRuntime:
    """Mutable registry record shared by all fibers of one plugin callback."""

    def __init__(
        self,
        name: str | None,
        callback: Callable[..., Any],
        Config: Callable[[Any], Any] | None = None,
    ) -> None:
        self.name = name
        self.callback = callback
        self.fibers: DisposableList[Fiber] = DisposableList()
        self.Config = Config


class _InjectPlugin:
    """Object plugin shape built by ``RegistryService.inject()``."""

    def __init__(self, inject: Any, callback: Callable[..., Any]) -> None:
        self.inject = inject
        self.apply = callback
        self.name = getattr(callback, '__name__', None)


class RegistryService:
    """Plugin registry installed as ``ctx.registry`` and mixed into contexts.

    Normalizes plugin shapes, tracks plugin runtimes (shared per normalized
    callback), starts fibers, and exposes map-like inspection.
    """

    _cordis_tracker = Tracker(property='ctx', no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._counter = 0
        self._internal: dict[Callable[..., Any], PluginRuntime] = {}

    @property
    def counter(self) -> int:
        """Allocate the next fiber uid (increments on every read)."""
        self._counter += 1
        return self._counter

    @property
    def size(self) -> int:
        """Number of registered plugin runtimes."""
        return len(self._internal)

    def resolve(self, plugin: Any) -> Callable[..., Any] | None:
        """Resolve a supported plugin shape to its executable callback."""
        # plugin.apply may throw
        try:
            if isinstance(plugin, type) or callable(plugin):
                return cast(Callable[..., Any], plugin)
            apply = getattr(plugin, 'apply', None)
            if callable(apply):
                return cast(Callable[..., Any], apply)
        except Exception:
            pass
        return None

    def get(self, plugin: Any) -> PluginRuntime | None:
        """Look up the runtime record for a plugin."""
        key = self.resolve(plugin)
        if key is None:
            return None
        return self._internal.get(key)

    def has(self, plugin: Any) -> bool:
        """Check whether a plugin has a registered runtime."""
        key = self.resolve(plugin)
        return key is not None and key in self._internal

    def delete(self, plugin: Any) -> PluginRuntime | None:
        """Dispose every running fiber for a plugin and remove its runtime."""
        key = self.resolve(plugin)
        if key is None:
            return None
        runtime = self._internal.get(key)
        if runtime is None:
            return None
        del self._internal[key]
        for fiber in runtime.fibers:
            fiber.dispose()
        return runtime

    def keys(self) -> Iterator[Callable[..., Any]]:
        """Iterate the registered plugin callbacks."""
        return iter(list(self._internal.keys()))

    def values(self) -> Iterator[PluginRuntime]:
        """Iterate the registered plugin runtimes."""
        return iter(list(self._internal.values()))

    def entries(self) -> Iterator[tuple[Callable[..., Any], PluginRuntime]]:
        """Iterate ``(callback, runtime)`` pairs."""
        return iter(list(self._internal.items()))

    def for_each(
        self,
        callback: Callable[[PluginRuntime, Callable[..., Any]], Any],
    ) -> None:
        """Visit every registered runtime."""
        for key, value in list(self._internal.items()):
            callback(value, key)

    def inject(self, inject: Any, callback: Callable[..., Any]) -> Fiber:
        """Start a callback once the requested dependencies are available.

        The callback is unloaded and re-run whenever a required service
        changes.
        """
        return self.plugin(_InjectPlugin(inject, callback))

    def plugin(self, plugin: Any, config: Any = None) -> Fiber:
        """Start a plugin in the current context and return its fiber.

        Awaiting the fiber settles once loading finished (raising config or
        startup errors); ``fiber.dispose()`` unloads it.
        """
        callback = self.resolve(plugin)
        if callback is None:
            raise RuntimeError(
                'invalid plugin, expect function or object with an "apply"'
                f' method, received {type(plugin).__name__}'
            )
        self.ctx.fiber.assert_active()

        runtime = self._internal.get(callback)
        if runtime is None:
            name = getattr(plugin, 'name', None)
            if name is None:
                name = getattr(plugin, '__name__', None)
            if name == 'apply':
                name = None
            runtime = PluginRuntime(
                name, callback, getattr(plugin, 'Config', None)
            )
            self._internal[callback] = runtime

        return Fiber(
            self.ctx, config, resolve_inject(getattr(plugin, 'inject', None)), runtime
        )
