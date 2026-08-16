"""Event bus, dispatch modes, and event augmentation types.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/events.ts``).

Python mapping: JavaScript's ``this`` binding for listeners is replaced by
prepending the dispatch ``thisArg`` (when given) as the listener's first
parameter, so e.g. an ``internal/update`` listener has the signature
``(fiber, config, no_save, next)``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from .utils import DisposableList, Tracker, maybe_await, schedule_logged

if TYPE_CHECKING:
    from .context import Context


def is_bailed(value: Any) -> bool:
    """Return whether an event result should stop a bail-style dispatch.

    Everything except ``None`` and ``False`` bails — including ``0``, ``''``
    and ``NaN`` (identity comparison, not truthiness).
    """
    return value is not None and value is not False


class Hook:
    """Registered listener record stored by the event service."""

    __slots__ = ('callback', 'ctx', 'global_', 'prepend')

    def __init__(
        self,
        ctx: Context,
        callback: Callable[..., Any],
        prepend: bool = False,
        global_: bool = False,
    ) -> None:
        self.ctx = ctx
        self.callback = callback
        self.prepend = prepend
        self.global_ = global_


def _on_internal_listener(
    ctx: Context,
    name: str,
    listener: Callable[..., Any],
    options: dict[str, Any],
) -> Any:
    """Replace registration of fiber-local ``internal/update`` listeners."""
    if name == 'internal/update' and not options.get('global'):
        hooks = ctx.fiber._hooks.setdefault('internal/update', DisposableList())
        method = 'unshift' if options.get('prepend') else 'push'
        # Mirrors the vendored source: DisposableList has no `unshift`, so a
        # prepended internal/update listener fails here.
        return getattr(hooks, method)(listener)
    return None


def _on_internal_update(
    fiber: Any,
    config: Any,
    no_save: bool,
    next_: Callable[[], Any],
) -> Any:
    """Chain fiber-local ``internal/update`` hooks before the outer next."""
    cbs = list(fiber._hooks.get('internal/update') or [])

    def _next() -> Any:
        if cbs:
            cb = cbs.pop(0)
            return cb(fiber, config, no_save, _next)
        return next_()

    return _next()


class EventsService:
    """Event bus installed as ``ctx.events`` and mixed into every context.

    Supports concurrent, synchronous, serial, bail, and waterfall dispatch
    and automatically disposes listeners with their owning fiber.
    """

    _cordis_tracker = Tracker(property='ctx', no_shadow=True)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._hooks: dict[str, list[Hook]] = {}

        self.on('internal/listener', _on_internal_listener)
        self.on(
            'internal/update',
            _on_internal_update,
            {'global': True, 'prepend': True},
        )

    def dispatch(self, type_: str, args: list[Any]) -> list[Callable[..., Any]]:
        """Resolve listeners for one dispatch and apply context filtering."""
        this_arg: Any = None
        if args and not isinstance(args[0], str):
            this_arg = args.pop(0)
        name: str = args.pop(0)
        if not name.startswith('internal/'):
            self.emit('internal/dispatch', type_, name, args, this_arg)
        filter_fn = (
            getattr(this_arg, '_cordis_filter', None)
            if this_arg is not None
            else None
        )
        callbacks: list[Callable[..., Any]] = []
        for hook in self._hooks.get(name) or []:
            if not (hook.global_ or filter_fn is None or filter_fn(hook.ctx)):
                continue
            if this_arg is not None:
                callbacks.append(partial(hook.callback, this_arg))
            else:
                callbacks.append(hook.callback)
        return callbacks

    async def parallel(self, *raw_args: Any) -> None:
        """Run listeners concurrently and wait for all of them."""
        args = list(raw_args)
        callbacks = self.dispatch('emit', args)

        async def run(callback: Callable[..., Any]) -> None:
            await maybe_await(callback(*args))

        results = await asyncio.gather(
            *(run(callback) for callback in callbacks),
            return_exceptions=True,
        )
        # ExceptionGroup requires Exception members; a BaseException such as
        # CancelledError propagates on its own instead.
        errors = [r for r in results if isinstance(r, BaseException)]
        for error in errors:
            if not isinstance(error, Exception):
                raise error
        exceptions = [e for e in errors if isinstance(e, Exception)]
        if exceptions:
            raise ExceptionGroup('parallel dispatch failed', exceptions)

    def emit(self, *raw_args: Any) -> None:
        """Run listeners synchronously without waiting for returned awaitables."""
        args = list(raw_args)
        for callback in self.dispatch('emit', args):
            result = callback(*args)
            if inspect.isawaitable(result):
                schedule_logged(result, self.ctx.logger.error)

    async def serial(self, *raw_args: Any) -> Any:
        """Run listeners in order, awaiting each, until one returns a bail value."""
        args = list(raw_args)
        for callback in self.dispatch('serial', args):
            result = await maybe_await(callback(*args))
            if is_bailed(result):
                return result
        return None

    def bail(self, *raw_args: Any) -> Any:
        """Run listeners synchronously until one returns a bail value."""
        args = list(raw_args)
        for callback in self.dispatch('bail', args):
            result = callback(*args)
            if is_bailed(result):
                return result
        return None

    def waterfall(self, *raw_args: Any) -> Any:
        """Compose listeners around the final ``next`` callback.

        The last dispatch argument is the innermost ``next``. Listeners run
        outermost-first; a listener that does not call ``next()`` vetoes the
        rest of the chain, including the built-in behavior. Returns the
        outermost listener's return value synchronously (an async listener's
        result stays a coroutine).
        """
        args = list(raw_args)
        callbacks = self.dispatch('waterfall', args)
        inner = args.pop()

        def next_() -> Any:
            callback = callbacks.pop(0) if callbacks else inner
            return callback(*args)

        args.append(next_)
        return next_()

    def register(
        self,
        label: str,
        hooks: list[Hook],
        callback: Callable[..., Any],
        options: dict[str, Any],
    ) -> Any:
        """Store a listener record as an effect on the current fiber."""
        def effect() -> Callable[[], bool]:
            hook = Hook(
                self.ctx,
                callback,
                bool(options.get('prepend')),
                bool(options.get('global')),
            )
            if options.get('prepend'):
                hooks.insert(0, hook)
            else:
                hooks.append(hook)
            return lambda: self.unregister(hooks, callback)

        return self.ctx.fiber.effect(effect, label)

    def unregister(self, hooks: list[Hook], callback: Callable[..., Any]) -> bool:
        """Remove a stored listener record; ``True`` when it was present."""
        for index, hook in enumerate(hooks):
            if hook.callback is callback:
                del hooks[index]
                return True
        return False

    def on(
        self,
        name: str,
        listener: Callable[..., Any],
        options: bool | dict[str, Any] | None = None,
    ) -> Any:
        """Register an event listener owned by the current fiber.

        ``options`` is ``{'prepend': bool, 'global': bool}``; a boolean is
        shorthand for ``prepend``. Returns a disposer removing the listener,
        or the replacement provided by an ``internal/listener`` bail hook.
        """
        if isinstance(options, dict):
            opts = options
        else:
            opts = {'prepend': bool(options)}

        # handle special events
        self.ctx.fiber.assert_active()
        result = self.bail(self.ctx, 'internal/listener', name, listener, opts)
        if result:
            return result

        hooks = self._hooks.setdefault(name, [])
        return self.register(f'ctx.on({name!r})', hooks, listener, opts)

    def once(
        self,
        name: str,
        listener: Callable[..., Any],
        options: bool | dict[str, Any] | None = None,
    ) -> Any:
        """Register an event listener that disposes itself after the first call."""
        def wrapper(*args: Any) -> Any:
            dispose()
            return listener(*args)

        dispose = self.on(name, wrapper, options)
        return dispose
