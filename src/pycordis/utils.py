"""Shared internal helpers used by context, services, and plugin fibers.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/utils.ts``).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterator
from types import MethodType
from typing import Any, Generic, NamedTuple, TypeVar

T = TypeVar('T')


class symbols:
    """Unique markers mirroring cordis' shared well-known symbols.

    JavaScript symbols have no Python attribute-key equivalent, so internals
    store these under reserved ``_cordis_*`` attribute names instead; the
    markers remain for API parity with the TypeScript surface.
    """

    # internal symbols
    shadow = object()
    receiver = object()
    original = object()
    metadata = object()
    init_hooks = object()
    check_proto = object()

    # context symbols
    effect = object()
    filter = object()
    isolate = object()
    intercept = object()

    # service symbols
    init = object()
    check = object()
    config = object()
    invoke = object()
    extend = object()
    tracker = object()
    resolve_config = object()


class DisposableList(Generic[T]):
    """Ordered collection of disposable values with O(1) deletion by value."""

    def __init__(self) -> None:
        self._sn = 0
        self._map: dict[int, T] = {}
        self._index: dict[Any, int] = {}

    @property
    def length(self) -> int:
        return len(self._map)

    def push(self, value: T) -> Callable[[], bool]:
        """Append a value; returns a remover deleting exactly this entry."""
        self._sn += 1
        sn = self._sn
        self._map[sn] = value
        self._index[value] = sn

        def remove() -> bool:
            return self._map.pop(sn, None) is not None

        return remove

    def delete(self, value: T) -> bool:
        sn = self._index.get(value)
        if sn is None:
            return False
        return self._map.pop(sn, None) is not None

    def clear(self) -> list[T]:
        """Empty the list and return its values in reverse insertion order."""
        values = list(self._map.values())
        self._map.clear()
        values.reverse()
        return values

    def __iter__(self) -> Iterator[T]:
        return iter(list(self._map.values()))

    def __repr__(self) -> str:
        return repr(list(self))


class ChainMap:
    """Prototype-chained string map: writes are own, reads walk parents.

    Mirrors the vendored ``Object.create(parentMap)`` isolation/intercept
    maps: each level shadows ancestor entries without mutating them.
    """

    __slots__ = ('own', 'parent')

    def __init__(self, parent: ChainMap | None = None) -> None:
        self.own: dict[str, Any] = {}
        self.parent = parent

    def get(self, key: str, default: Any = None) -> Any:
        node: ChainMap | None = self
        while node is not None:
            if key in node.own:
                return node.own[key]
            node = node.parent
        return default

    def __contains__(self, key: str) -> bool:
        node: ChainMap | None = self
        while node is not None:
            if key in node.own:
                return True
            node = node.parent
        return False

    def __setitem__(self, key: str, value: Any) -> None:
        self.own[key] = value

    def setdefault(self, key: str, default: Any) -> Any:
        if key in self:
            return self.get(key)
        self.own[key] = default
        return default


class Tracker(NamedTuple):
    """Metadata used to rebind services to the caller's active context."""

    associate: str | None = None
    property: str | None = None
    no_shadow: bool = False


class ServiceView:
    """Per-access rebind of a tracked service to the caller's context.

    Python mapping of cordis' traceable proxies: methods and property writes
    see the accessing context as ``self.ctx`` while mutable service state
    (stores, hook lists, counters) stays shared on the target instance.
    """

    _cordis_tracker: None = None

    def __init__(self, target: Any, ctx: Any) -> None:
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_ctx', ctx)

    def __getattr__(self, name: str) -> Any:
        if name == 'ctx':
            return object.__getattribute__(self, '_ctx')
        target = object.__getattribute__(self, '_target')
        value = getattr(target, name)
        if isinstance(value, MethodType) and value.__self__ is target:
            return value.__func__.__get__(self, type(self))
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, '_target'), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = self.__getattr__('__call__')
        return call(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, '_target'))


def rebind(ctx: Any, value: T) -> T:
    """Attach the accessing context to a tracked service.

    Values without tracker metadata (plain data, fibers, contexts) pass
    through unchanged.
    """
    if getattr(value, '_cordis_tracker', None) is None:
        return value
    return ServiceView(value, ctx)  # type: ignore[return-value]


def is_constructor(func: Any) -> bool:
    """Return true when a plugin callback should be constructed with ``new``."""
    return isinstance(func, type)


async def maybe_await(value: Any) -> Any:
    """Await awaitables, pass plain values through (JavaScript ``await``)."""
    if inspect.isawaitable(value):
        return await value
    return value


def schedule_logged(
    awaitable: Any,
    log_error: Callable[[BaseException], Any],
) -> asyncio.Future[Any]:
    """Schedule a fire-and-forget coroutine, logging any failure.

    Python mapping of Node's floating promise plus unhandled-rejection
    diagnostic for listener results discarded by ``emit()``.
    """
    future: asyncio.Future[Any] = asyncio.ensure_future(awaitable)

    def _done(done: asyncio.Future[Any]) -> None:
        if done.cancelled():
            return
        error = done.exception()
        if error is not None:
            log_error(error)

    future.add_done_callback(_done)
    return future
