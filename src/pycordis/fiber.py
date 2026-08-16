"""Plugin fiber lifecycle, effects, and config validation helpers.

Ported from pycordis (MIT, cordiverse), vendored in deepseek-harness
(``vendor/cordis/src/fiber.ts``), including the dsh lifecycle hardening:
effect wrappers are registered before setup runs, setup failure rolls back,
effects are rejected while ``UNLOADING``, child fibers register their
disposer on the parent before ``internal/plugin`` is published, and
``Fiber.update()`` returns the ``internal/update`` waterfall result.
"""

from __future__ import annotations

import asyncio
import inspect
import weakref
from collections.abc import Callable, Coroutine, Generator, Iterable
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

from .utils import DisposableList, is_constructor, schedule_logged

if TYPE_CHECKING:
    from .context import Context
    from .registry import PluginRuntime

INACTIVE = '__INACTIVE__'


class CordisError(Exception):
    """Framework error with a stable machine-readable code."""

    CODES: ClassVar[dict[str, str]] = {
        'INACTIVE_EFFECT': 'cannot create effect on inactive context',
    }

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message if message is not None else self.CODES[code])


class ValidationError(TypeError):
    """Error raised when plugin configuration fails validation."""


def resolve_config(runtime: PluginRuntime, config: Any) -> Any:
    """Validate and normalize config for a plugin runtime before it starts.

    Python mapping: ``runtime.Config`` is a callable ``config -> validated
    config``; raising rejects the config. Runtimes without ``Config`` pass
    the config through unchanged.
    """
    if runtime.Config is None:
        return config
    return runtime.Config(config)


class FiberState(IntEnum):
    """Lifecycle state for one plugin fiber."""

    PENDING = 0
    LOADING = 1
    ACTIVE = 2
    FAILED = 3
    DISPOSED = 4
    UNLOADING = 5


class EffectDisposer:
    """Single-shot, awaitable disposer returned by ``Fiber.effect()``.

    Calling it tears the effect down (once; repeats are no-ops); awaiting it
    settles once the effect's setup finished and yields a disposal function.
    """

    def __init__(
        self,
        call: Callable[[], Any],
        await_impl: Callable[[], Coroutine[Any, Any, Any]],
        label: str,
    ) -> None:
        self._call = call
        self._await_impl = await_impl
        self.label = label

    def __call__(self) -> Any:
        return self._call()

    def __await__(self) -> Generator[Any, None, Any]:
        return self._await_impl().__await__()

    def __repr__(self) -> str:
        return f'EffectDisposer({self.label!r})'


class _Runner:
    """Effect execution record: current epoch plus the collect callback."""

    __slots__ = ('collect', 'epoch', 'execute')

    def __init__(
        self,
        epoch: Any,
        execute: Callable[[], Any],
        collect: Callable[[Callable[[], Any]], None],
    ) -> None:
        self.epoch = epoch
        self.execute = execute
        self.collect = collect


# Public effect disposers remain single-shot, but structural owners and outer
# effects must still be able to join a cleanup that another caller started.
_effect_inertia: weakref.WeakKeyDictionary[Any, Callable[[], Any]] = (
    weakref.WeakKeyDictionary()
)


def _run_disposable(dispose: Callable[[], Any]) -> Any:
    result = dispose()
    inertia = _effect_inertia.get(dispose)
    if inertia is not None:
        inner = inertia()
        if inner is not None:
            return inner
    return result


def _emit_plugin_disposed(context: Context, fiber: Fiber) -> None:
    """Notify plugin teardown without one observer breaking ownership cleanup."""
    args: list[Any] = ['internal/plugin', fiber]
    try:
        callbacks = context.events.dispatch('emit', args)
    except Exception as error:
        context.logger.error(error)
        return
    for callback in callbacks:
        try:
            returned = callback(*args)
            if inspect.isawaitable(returned):
                schedule_logged(returned, context.logger.error)
        except Exception as error:
            context.logger.error(error)


async def _chain_disposable(
    task: asyncio.Future[Any],
    disposable: Callable[[], Any],
) -> None:
    await task
    result = _run_disposable(disposable)
    if inspect.isawaitable(result):
        await result


class Fiber:
    """Runtime instance of one plugin application.

    Tracks dependency state, validated config, lifecycle effects, and
    cleanup for the plugin context returned by ``ctx.plugin()``.
    """

    def __init__(
        self,
        parent: Context,
        config: Any,
        inject: dict[str, Any],
        runtime: PluginRuntime | None,
    ) -> None:
        self.uid: int | None
        self.ctx: Context
        self.context: Context
        self.config: Any = None
        self._config = config
        self.state = FiberState.PENDING
        self.store: dict[str, Any] | None = None
        self.inertia: asyncio.Future[Any] | None = None
        self.parent = parent
        self.inject = inject
        self.runtime = runtime
        self._hooks: dict[str, DisposableList[Any]] = {}
        self._disposables: DisposableList[Callable[[], Any]] = DisposableList()
        self._error: BaseException | None = None
        self._store: dict[str, Any] = {}

        def collect(dispose: Callable[[], Any]) -> None:
            self._disposables.push(dispose)

        if runtime is not None:
            self.uid = parent.registry.counter
            self.ctx = self.context = parent.extend({'fiber': self})

            if self.inject:
                intercept = type(parent._cordis_intercept)(parent._cordis_intercept)
                for name, inject_config in self.inject.items():
                    if inject_config is None:
                        continue
                    intercept[name] = inject_config
                self.ctx.__dict__['_cordis_intercept'] = intercept

            def execute() -> Any:
                callback = runtime.callback
                if is_constructor(callback):
                    instance = callback(self.ctx, self.config)
                    for hook in getattr(instance, 'cordis_init_hooks', None) or []:
                        hook()
                    init = getattr(instance, 'cordis_init', None)
                    if init is not None:
                        return init()
                    return None
                return callback(self.ctx, self.config)

            self._runner = _Runner(INACTIVE, execute, collect)

            def plugin_effect() -> Callable[[], Any]:
                remove = runtime.fibers.push(self)

                async def dispose() -> None:
                    self.uid = None
                    _emit_plugin_disposed(self.context, self)
                    if self.ctx.registry.has(runtime.callback):
                        remove()
                        if not runtime.fibers.length:
                            self.ctx.registry.delete(runtime.callback)
                    self._set_epoch(INACTIVE)
                    # A PENDING fiber can already own effects registered by an
                    # internal/plugin observer. Its epoch is still INACTIVE, so
                    # _set_epoch() has no transition to drive; explicitly unload
                    # that pre-activation work before reporting disposal done.
                    if self.inertia is None:
                        def unload() -> FiberState:
                            self.inertia = asyncio.ensure_future(self._unload())
                            return FiberState.UNLOADING

                        self._update_state(unload)
                    # `self.inertia` never rejects: both `_reload` and `_unload`
                    # swallow their own work errors via `ctx.logger.error`.
                    while self.inertia is not None:
                        await self.inertia

                return dispose

            self._dispose: Callable[[], Any] = parent.fiber.effect(
                plugin_effect, 'ctx.plugin()'
            )

            try:
                # Publish only after the parent owns a fully assigned disposer.
                # A synchronous observer may dispose this fiber or its parent.
                self.context.emit('internal/plugin', self)
            except BaseException:
                # Publication failed synchronously. The disposer removes the
                # child from both the parent and runtime before control escapes.
                result = self.dispose()
                if inspect.isawaitable(result):
                    schedule_logged(result, self._log_error)
                raise

            # Keep the initial notification's historical PENDING view. The
            # loader may also extend `inject` in that notification, so resolve
            # dependencies only after publication. A reentrant parent unload
            # makes the child disposer drain any PENDING effects instead.
            if self.uid is not None and parent.fiber.state is not FiberState.UNLOADING:
                for name in self.inject:
                    self._check_impl(name)
                self._refresh()
        else:
            self.uid = 0
            self.ctx = self.context = parent
            self.state = FiberState.ACTIVE
            self.store = {}
            self._runner = _Runner('', lambda: None, collect)
            self._dispose = self.restart

    @property
    def name(self) -> str:
        """Display name from the nearest named ancestor, else ``'root'``."""
        fiber = self
        while True:
            if fiber.runtime is not None and fiber.runtime.name:
                return fiber.runtime.name
            parent_fiber = fiber.parent.fiber
            if parent_fiber is fiber:
                return 'root'
            fiber = parent_fiber

    def assert_active(self) -> None:
        """Throw ``CordisError('INACTIVE_EFFECT')`` once uid is cleared."""
        if self.uid is not None:
            return
        raise CordisError('INACTIVE_EFFECT')

    def dispose(self) -> Any:
        """Unload the plugin, then settle once cleanup finished."""
        return self._dispose()

    def _log_error(self, error: BaseException) -> None:
        self.ctx.logger.error(error)

    def _execute(self, runner: _Runner) -> asyncio.Future[Any] | None:
        old_epoch = runner.epoch

        def safe_collect(dispose: Any) -> None:
            if callable(dispose):
                runner.collect(dispose)
            elif dispose is not None:
                raise TypeError('Invalid effect')

        effect = runner.execute()
        if callable(effect):
            runner.collect(effect)
            return None
        if effect is None:
            return None
        if inspect.isawaitable(effect):
            async def collect_awaitable() -> None:
                safe_collect(await effect)

            return asyncio.ensure_future(collect_awaitable())
        if inspect.isasyncgen(effect):
            async def consume() -> None:
                # force a checkpoint so the epoch guard sees owner transitions
                await asyncio.sleep(0)
                while True:
                    if runner.epoch != old_epoch:
                        return
                    try:
                        value = await effect.__anext__()
                    except StopAsyncIteration:
                        return
                    safe_collect(value)

            return asyncio.ensure_future(consume())
        if isinstance(effect, Iterable) and not isinstance(effect, (str, bytes)):
            iterator = iter(effect)
            while True:
                try:
                    value = next(iterator)
                except StopIteration as stop:
                    safe_collect(stop.value)
                    return None
                safe_collect(value)
        raise TypeError('Invalid effect')

    def effect(
        self,
        execute: Callable[[], Any],
        label: str = 'anonymous',
    ) -> EffectDisposer:
        """Register a cleanup-aware effect on this fiber.

        ``execute`` runs immediately; the disposers it produces (a function,
        an awaitable of one, a sync generator, or an async generator) run in
        reverse order when the returned disposer is called or the fiber
        unloads, whichever comes first. Calling the disposer twice is a
        no-op. Throws ``CordisError('INACTIVE_EFFECT')`` when the fiber is
        disposed or unloading, and ``TypeError`` for invalid result shapes.
        """
        self.assert_active()
        if self.state is FiberState.UNLOADING:
            raise CordisError('INACTIVE_EFFECT')

        disposables: list[Callable[[], Any]] = []
        disposing = False
        disposal_task: asyncio.Future[Any] | None = None

        def dispose() -> asyncio.Future[Any] | None:
            nonlocal disposing, disposal_task
            if disposing:
                return disposal_task
            disposing = True
            task: asyncio.Future[Any] | None = None
            pending = list(reversed(disposables))
            disposables.clear()
            for disposable in pending:
                if task is not None:
                    task = asyncio.ensure_future(
                        _chain_disposable(task, disposable)
                    )
                else:
                    result = _run_disposable(disposable)
                    if inspect.isawaitable(result):
                        task = asyncio.ensure_future(result)
            disposal_task = task
            return task

        def collect_into_effect(dispose: Callable[[], Any]) -> None:
            disposables.append(dispose)
            self._disposables.delete(dispose)

        runner = _Runner(True, execute, collect_into_effect)

        executing = True
        setup_failed = False
        in_flight: asyncio.Future[Any] | None = None
        task: asyncio.Future[Any] | None = None
        setup_barrier: asyncio.Future[None] | None = None
        def _noop_remove() -> bool:
            return False

        remove_wrapper: Callable[[], bool] = _noop_remove

        def wait_for_setup() -> asyncio.Future[None]:
            nonlocal setup_barrier
            if setup_barrier is None:
                setup_barrier = asyncio.Future()
            return setup_barrier

        def dispose_after(setup: asyncio.Future[Any]) -> asyncio.Future[None]:
            async def run() -> None:
                try:
                    await setup
                except BaseException as reason:
                    inner = dispose()
                    if inspect.isawaitable(inner):
                        await inner
                    raise reason
                inner = dispose()
                if inspect.isawaitable(inner):
                    await inner

            return asyncio.ensure_future(run())

        def finalize_disposal(callback: Callable[[], Any]) -> Any:
            nonlocal in_flight
            try:
                result = callback()
            except BaseException:
                remove_wrapper()
                raise
            if inspect.isawaitable(result):
                pending = asyncio.ensure_future(result)

                def pending_done(done: asyncio.Future[Any]) -> None:
                    nonlocal in_flight
                    remove_wrapper()
                    if in_flight is pending:
                        in_flight = None

                pending.add_done_callback(pending_done)
                in_flight = pending
                return pending
            remove_wrapper()
            return result

        def call() -> Any:
            # A synchronous setup failure can race an owner unload that already
            # captured this wrapper but has not invoked it yet. The failed
            # effect is never returned publicly, so let that internal caller
            # await rollback.
            if not runner.epoch:
                return in_flight if setup_failed else None
            runner.epoch = False

            def callback() -> Any:
                if executing:
                    return dispose_after(wait_for_setup())
                if task is not None:
                    return dispose_after(task)
                return dispose()

            return finalize_disposal(callback)

        def dispose_async() -> Any:
            if not runner.epoch:
                return None
            runner.epoch = False
            return finalize_disposal(dispose)

        async def await_impl() -> Callable[[], Any]:
            if task is not None:
                await task
            return dispose_async

        wrapper = EffectDisposer(call, await_impl, label)
        _effect_inertia[wrapper] = lambda: in_flight

        # Make the effect visible to a reentrant owner unload before execute()
        # runs any plugin code. Async teardown stays owner-visible until it
        # settles, allowing an outer effect to join cleanup another caller began.
        remove_wrapper = self._disposables.push(wrapper)
        try:
            executed = self._execute(runner)
        except BaseException as reason:
            executing = False
            setup_failed = True
            runner.epoch = False
            try:
                cleanup = finalize_disposal(dispose)
            finally:
                if setup_barrier is not None and not setup_barrier.done():
                    setup_barrier.set_exception(reason)
            if inspect.isawaitable(cleanup):
                schedule_logged(cleanup, self._log_error)
            raise
        executing = False
        task = executed
        if setup_barrier is not None:
            if task is None:
                if not setup_barrier.done():
                    setup_barrier.set_result(None)
            else:
                def barrier_done(done: asyncio.Future[Any]) -> None:
                    if setup_barrier is None or setup_barrier.done():
                        return
                    if done.cancelled():
                        setup_barrier.set_exception(asyncio.CancelledError())
                        return
                    error = done.exception()
                    if error is not None:
                        setup_barrier.set_exception(error)
                    else:
                        setup_barrier.set_result(None)

                task.add_done_callback(barrier_done)

        # Mirror the floating `task.catch(...)`: a rejected effect task tears
        # the effect down, and disposal errors are logged, never raised here.
        if task is not None:
            def task_done(done: asyncio.Future[Any]) -> None:
                if done.cancelled():
                    error: BaseException | None = asyncio.CancelledError()
                else:
                    error = done.exception()
                if error is None:
                    return
                try:
                    if not runner.epoch:
                        result = dispose()
                    else:
                        result = finalize_disposal(dispose)
                except BaseException as dispose_error:
                    self._log_error(dispose_error)
                    return
                if inspect.isawaitable(result):
                    schedule_logged(result, self._log_error)

            task.add_done_callback(task_done)

        return wrapper

    def _get_state(self) -> FiberState:
        if self.uid is None:
            return FiberState.DISPOSED
        if self._error is not None:
            return FiberState.FAILED
        if self._runner.epoch != INACTIVE:
            return FiberState.ACTIVE
        return FiberState.PENDING

    def _update_state(self, callback: Callable[[], FiberState | None]) -> None:
        old_state = self.state
        result = callback()
        self.state = result if result is not None else self._get_state()
        if old_state == self.state:
            return
        self.context.emit('internal/status', self, old_state)

        # only notify changes between ACTIVE and NON-ACTIVE states
        if old_state is not FiberState.ACTIVE and self.state is not FiberState.ACTIVE:
            return
        for impl in list(self.context.reflect.store.values()):
            if impl.fiber is not self:
                continue
            self.context.reflect.notify([impl.name])

    def _check_impl(self, name: str) -> None:
        impl = self.ctx.reflect._get_impl(name, True)
        if impl is None:
            self._store.pop(name, None)
            return
        try:
            if impl.check is not None and not impl.check():
                self._store.pop(name, None)
                return
        except Exception as error:
            impl.fiber.ctx.logger.error(error)
            self._store.pop(name, None)
            return
        self._store[name] = impl

    def _refresh(self) -> None:
        epoch = ''
        for name in self.inject:
            impl = self._store.get(name)
            if impl is None:
                epoch = INACTIVE
                break
            epoch += ':' + str(impl.fiber.uid)
        self._set_epoch(epoch)

    def _set_epoch(self, epoch: str) -> None:
        old_epoch = self._runner.epoch
        if epoch == old_epoch:
            return
        self._runner.epoch = epoch
        if self.inertia is not None:
            return

        def transition() -> FiberState:
            if epoch != INACTIVE and old_epoch == INACTIVE:
                self.inertia = asyncio.ensure_future(self._reload())
                return FiberState.LOADING
            self.inertia = asyncio.ensure_future(self._unload())
            return FiberState.UNLOADING

        self._update_state(transition)

    def _resolve_config(self, config: Any) -> Any:
        config = self.context.events.waterfall(
            self, 'internal/config', config, lambda *_args: config
        )
        if self.runtime is not None:
            return resolve_config(self.runtime, config)
        return config

    def _reload(self) -> Any:
        self.store = dict(self._store)
        old_epoch = self._runner.epoch

        async def body() -> None:
            try:
                await asyncio.sleep(0)
                # A disposer queued before this checkpoint may already have
                # invalidated the load. Do not run plugin code for a stale
                # epoch; the state update below drains effects collected while
                # the fiber was PENDING.
                if self._runner.epoch == old_epoch:
                    self.config = self._resolve_config(self._config)
                    executed = self._execute(self._runner)
                    if executed is not None:
                        await executed
                    self._error = None
            except Exception as reason:
                self.ctx.logger.error(reason)
                self._error = reason
                self._runner.epoch = INACTIVE

            def transition() -> FiberState | None:
                if self._runner.epoch == old_epoch:
                    self.inertia = None
                    return None
                self.inertia = asyncio.ensure_future(self._unload())
                return FiberState.UNLOADING

            self._update_state(transition)

        return body()

    def _unload(self) -> Any:
        disposables = self._disposables.clear()

        async def body() -> None:
            async def run_one(dispose: Callable[[], Any]) -> None:
                try:
                    await asyncio.sleep(0)
                    result = _run_disposable(dispose)
                    if inspect.isawaitable(result):
                        await result
                except Exception as reason:
                    self.ctx.logger.error(reason)

            # Top-level effects: reverse start, concurrently awaited.
            await asyncio.gather(*(run_one(dispose) for dispose in disposables))
            self.store = None

            def transition() -> FiberState | None:
                if self._runner.epoch == INACTIVE:
                    self.inertia = None
                    return None
                self.inertia = asyncio.ensure_future(self._reload())
                return FiberState.LOADING

            self._update_state(transition)

        return body()

    async def wait(self) -> Fiber:
        """Wait for current lifecycle work and rethrow startup errors."""
        while self.inertia is not None:
            await self.inertia
        if self._error is not None:
            raise self._error
        return self

    def __await__(self) -> Generator[Any, None, Fiber]:
        return self.wait().__await__()

    async def restart(self) -> None:
        """Dispose and immediately reload this plugin with its current config."""
        self.assert_active()
        self._set_epoch(INACTIVE)
        self._refresh()
        await self.wait()

    def update(self, config: Any, no_save: bool = False) -> Any:
        """Validate and apply new config, then restart the plugin.

        Runs the ``internal/update`` waterfall first, so update hooks can
        veto or replace the restart; returns the waterfall result (the
        default restart returns a coroutine).
        """
        self.assert_active()
        self._config = config
        if self.state is not FiberState.ACTIVE:
            # Config resolution may access injected services, so defer it
            # until the fiber can activate.
            self._error = None
            self._set_epoch(INACTIVE)
            self._refresh()
            return None
        config = self._resolve_config(config)

        def final(*_args: Any) -> Any:
            self.config = config
            self._error = None
            return self.restart()

        return self.context.events.waterfall(
            self, 'internal/update', config, no_save, final
        )
