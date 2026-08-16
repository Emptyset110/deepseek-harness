"""Behavior tests for fiber lifecycle, effects, and the plugin registry.

Ported-semantics tests for the cordis Python port (MIT, cordiverse;
vendored in deepseek-harness).
"""

from __future__ import annotations

import asyncio

import pytest

from pycordis import Context, CordisError, Fiber, FiberState


async def test_effect_disposers_reverse_serial_within_one_effect() -> None:
    ctx = Context()
    order: list[str] = []

    def d1() -> None:
        order.append('d1')

    async def d2() -> None:
        order.append('d2-start')
        await asyncio.sleep(0)
        order.append('d2-end')

    disposer = ctx.effect(lambda: [d1, d2])
    result = disposer()
    assert result is not None
    await result
    # reverse registration order; async disposers are awaited serially
    assert order == ['d2-start', 'd2-end', 'd1']


async def test_effect_double_dispose_is_noop() -> None:
    ctx = Context()
    calls: list[str] = []
    disposer = ctx.effect(lambda: lambda: calls.append('disposed'))
    disposer()
    disposer()
    assert calls == ['disposed']


async def test_effect_invalid_shape_throws() -> None:
    ctx = Context()
    with pytest.raises(TypeError, match='Invalid effect'):
        ctx.effect(lambda: 42)


async def test_effect_generator_collects_each_yield() -> None:
    ctx = Context()
    order: list[str] = []

    def body() -> object:
        yield lambda: order.append('first')
        yield lambda: order.append('second')

    disposer = ctx.effect(body)
    disposer()
    assert order == ['second', 'first']


async def test_effect_async_generator() -> None:
    ctx = Context()
    order: list[str] = []

    async def body() -> object:
        yield lambda: order.append('clean')

    disposer = ctx.effect(body)
    # awaiting the disposer settles once the async generator is consumed
    await disposer
    result = disposer()
    if result is not None:
        await result
    assert order == ['clean']


async def test_effect_on_disposed_fiber_throws() -> None:
    ctx = Context()

    def plugin(c: Context, config: object) -> None:
        return None

    fiber = ctx.plugin(plugin)
    await fiber
    await fiber.dispose()
    assert fiber.state is FiberState.DISPOSED
    with pytest.raises(CordisError) as exc_info:
        fiber.ctx.effect(lambda: None)
    assert exc_info.value.code == 'INACTIVE_EFFECT'
    with pytest.raises(CordisError) as exc_info:
        fiber.ctx.on('event', lambda: None)
    assert exc_info.value.code == 'INACTIVE_EFFECT'


async def test_fiber_unload_reversed_start_concurrent_await() -> None:
    ctx = Context()
    starts: list[str] = []
    finished: list[str] = []
    slow_started = asyncio.Event()
    fast_done = asyncio.Event()

    async def slow() -> None:
        starts.append('slow')
        slow_started.set()
        # circular waits only resolve when both disposers run concurrently;
        # any serial teardown order deadlocks and hits the timeout
        await asyncio.wait_for(fast_done.wait(), 1)
        finished.append('slow')

    async def fast() -> None:
        starts.append('fast')
        await asyncio.wait_for(slow_started.wait(), 1)
        fast_done.set()
        finished.append('fast')

    def plugin(c: Context, config: object) -> None:
        c.effect(lambda: slow)
        c.effect(lambda: fast)

    fiber = ctx.plugin(plugin)
    await fiber
    await fiber.dispose()
    # reverse start order, both awaited concurrently
    assert starts == ['fast', 'slow']
    assert finished == ['fast', 'slow']
    assert fiber.state is FiberState.DISPOSED


async def test_disposer_error_is_logged_not_propagated() -> None:
    ctx = Context()

    def raising() -> None:
        raise RuntimeError('boom')

    def plugin(c: Context, config: object) -> None:
        c.effect(lambda: raising)

    fiber = ctx.plugin(plugin)
    await fiber
    await fiber.dispose()
    assert fiber.state is FiberState.DISPOSED
    assert any(
        any('boom' in str(arg) for arg in message.args)
        for message in ctx.logger.buffer
    )


async def test_child_plugin_cascades_with_parent() -> None:
    ctx = Context()
    order: list[str] = []

    def child(c: Context, config: object) -> object:
        order.append('child-load')
        return lambda: order.append('child-unload')

    children: list[Fiber] = []

    def parent(c: Context, config: object) -> object:
        order.append('parent-load')
        children.append(c.plugin(child))
        return lambda: order.append('parent-unload')

    fiber = ctx.plugin(parent)
    await fiber
    # the child fiber loads on its own task
    await children[0]
    assert order == ['parent-load', 'child-load']
    assert ctx.registry.has(child)

    await fiber.dispose()
    assert sorted(order[2:]) == ['child-unload', 'parent-unload']
    assert fiber.state is FiberState.DISPOSED
    # the child's runtime is removed once its last fiber is gone
    assert not ctx.registry.has(child)


async def test_function_class_and_apply_plugin_forms() -> None:
    ctx = Context()
    log: list[object] = []

    def function_plugin(c: Context, config: object) -> object:
        log.append(('function', config))
        return lambda: log.append('function-gone')

    class ClassPlugin:
        def __init__(self, c: Context, config: object) -> None:
            log.append(('class', config))

        def cordis_init(self) -> object:
            return lambda: log.append('class-gone')

    class ApplyPlugin:
        def apply(self, c: Context, config: object) -> object:
            log.append(('apply', config))
            return lambda: log.append('apply-gone')

    f1 = ctx.plugin(function_plugin, 1)
    f2 = ctx.plugin(ClassPlugin, 2)
    f3 = ctx.plugin(ApplyPlugin(), 3)
    await f1
    await f2
    await f3
    assert log == [('function', 1), ('class', 2), ('apply', 3)]
    await f1.dispose()
    await f2.dispose()
    await f3.dispose()
    assert sorted(log[3:]) == ['apply-gone', 'class-gone', 'function-gone']


async def test_same_plugin_multiple_independent_fibers() -> None:
    ctx = Context()
    instances: list[str] = []

    def plugin(c: Context, config: str) -> None:
        instances.append(config)

    first = ctx.plugin(plugin, 'one')
    second = ctx.plugin(plugin, 'two')
    await first
    await second
    assert instances == ['one', 'two']
    runtime = ctx.registry.get(plugin)
    assert runtime is not None
    assert runtime.fibers.length == 2

    await first.dispose()
    assert first.state is FiberState.DISPOSED
    assert second.state is FiberState.ACTIVE
    assert ctx.registry.has(plugin)

    await second.dispose()
    assert not ctx.registry.has(plugin)


async def test_config_validation_failure() -> None:
    ctx = Context()

    def plugin(c: Context, config: object) -> None:
        return None

    def validate(config: dict[str, int]) -> dict[str, int]:
        if config['x'] < 0:
            raise ValueError('x must be non-negative')
        return config

    plugin.Config = validate  # type: ignore[attr-defined]
    fiber = ctx.plugin(plugin, {'x': -1})
    with pytest.raises(ValueError, match='non-negative'):
        await fiber
    assert fiber.state is FiberState.FAILED


async def test_config_validation_transforms() -> None:
    ctx = Context()
    received: list[object] = []

    def plugin(c: Context, config: object) -> None:
        received.append(config)

    plugin.Config = lambda config: {'x': config['x'] * 2}  # type: ignore[attr-defined]
    fiber = ctx.plugin(plugin, {'x': 3})
    await fiber
    assert received == [{'x': 6}]


async def test_update_restarts_with_new_config() -> None:
    ctx = Context()
    configs: list[object] = []

    def plugin(c: Context, config: object) -> None:
        configs.append(config)

    fiber = ctx.plugin(plugin, {'x': 1})
    await fiber
    result = fiber.update({'x': 5})
    # update() returns the internal/update waterfall result; the default
    # final handler's restart is a coroutine
    await result
    assert configs == [{'x': 1}, {'x': 5}]
    assert fiber.config == {'x': 5}
    assert fiber.state is FiberState.ACTIVE


async def test_update_vetoed_by_fiber_listener() -> None:
    ctx = Context()
    configs: list[object] = []

    def plugin(c: Context, config: object) -> None:
        configs.append(config)

    fiber = ctx.plugin(plugin, {'x': 1})
    await fiber
    # a non-global internal/update listener lands on the fiber's own hooks
    # and can veto the restart by not calling next()
    fiber.ctx.on('internal/update', lambda f, config, no_save, next_: 'vetoed')
    result = fiber.update({'x': 9})
    assert result == 'vetoed'
    assert configs == [{'x': 1}]


async def test_restart() -> None:
    ctx = Context()
    log: list[str] = []

    def plugin(c: Context, config: object) -> object:
        log.append('load')
        return lambda: log.append('unload')

    fiber = ctx.plugin(plugin)
    await fiber
    await fiber.restart()
    assert log == ['load', 'unload', 'load']
    assert fiber.state is FiberState.ACTIVE


async def test_registry_delete_disposes_all_fibers() -> None:
    ctx = Context()

    def plugin(c: Context, config: object) -> None:
        return None

    first = ctx.plugin(plugin)
    second = ctx.plugin(plugin)
    await first
    await second
    removed = ctx.registry.delete(plugin)
    assert removed is not None
    assert not ctx.registry.has(plugin)
    for _ in range(100):
        if first.state is FiberState.DISPOSED and second.state is FiberState.DISPOSED:
            break
        await asyncio.sleep(0.01)
    assert first.state is FiberState.DISPOSED
    assert second.state is FiberState.DISPOSED


async def test_plugin_awaitable_returns_fiber() -> None:
    ctx = Context()

    def plugin(c: Context, config: object) -> None:
        return None

    fiber = ctx.plugin(plugin)
    settled = await fiber
    assert settled is fiber
    assert isinstance(fiber, Fiber)


async def test_internal_config_waterfall() -> None:
    ctx = Context()
    received: list[object] = []
    ctx.on(
        'internal/config',
        lambda fiber, config, next_: {**config, 'injected': True},
    )

    def plugin(c: Context, config: object) -> None:
        received.append(config)

    fiber = ctx.plugin(plugin, {'a': 1})
    await fiber
    assert received == [{'a': 1, 'injected': True}]


async def test_effect_rejected_while_unloading() -> None:
    ctx = Context()
    log: list[str] = []

    def plugin(c: Context, config: object) -> object:
        def on_unload() -> None:
            try:
                c.effect(lambda: None)
                log.append('no-error')
            except CordisError:
                log.append('rejected')

        return on_unload

    fiber = ctx.plugin(plugin)
    await fiber
    await fiber.dispose()
    assert log == ['rejected']


async def test_update_on_pending_fiber_defers() -> None:
    ctx = Context()
    seen: list[object] = []

    def plugin(c: Context, config: object) -> None:
        seen.append(config)

    fiber = ctx.inject(['dep'], plugin)
    assert fiber.update({'v': 1}) is None
    assert fiber.state is FiberState.PENDING
    ctx.provide('dep', True)
    await fiber
    assert seen == [{'v': 1}]
    assert fiber.config == {'v': 1}


async def test_invalid_plugin_throws() -> None:
    ctx = Context()
    with pytest.raises(RuntimeError, match='invalid plugin'):
        ctx.plugin(42)
