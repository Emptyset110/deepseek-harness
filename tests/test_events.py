"""Behavior tests for the five event dispatch modes and listener options.

Ported-semantics tests for the cordis Python port (MIT, cordiverse;
vendored in deepseek-harness). The dispatch ``thisArg`` (JavaScript ``this``)
is prepended as the listener's first parameter when present.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from pycordis import Context


async def test_emit_is_synchronous_fire_and_forget() -> None:
    ctx = Context()
    called: list[tuple[int, int]] = []
    ctx.on('e', lambda a, b: called.append((a, b)))
    assert ctx.emit('e', 1, 2) is None
    assert called == [(1, 2)]


async def test_emit_schedules_async_listeners() -> None:
    ctx = Context()
    done = asyncio.Event()

    async def listener(value: int) -> None:
        done.set()

    ctx.on('e', listener)
    ctx.emit('e', 1)
    await asyncio.wait_for(done.wait(), 1)


async def test_parallel_runs_all_and_raises_exception_group() -> None:
    ctx = Context()
    results: list[str] = []

    async def ok(value: int) -> None:
        await asyncio.sleep(0)
        results.append('ok')

    async def bad(value: int) -> None:
        raise ValueError('boom')

    ctx.on('p', ok)
    ctx.on('p', bad)
    with pytest.raises(ExceptionGroup) as exc_info:
        await ctx.parallel('p', 1)
    assert len(exc_info.value.exceptions) == 1
    # all listeners ran despite the failure (all-settled semantics)
    assert results == ['ok']


async def test_serial_awaits_in_order_until_bail() -> None:
    ctx = Context()
    order: list[str] = []

    async def first(value: int) -> None:
        order.append('first')
        return None

    async def second(value: int) -> str:
        order.append('second')
        return 'stop'

    def third(value: int) -> None:
        order.append('third')

    ctx.on('s', first)
    ctx.on('s', second)
    ctx.on('s', third)
    assert await ctx.serial('s', 1) == 'stop'
    assert order == ['first', 'second']


async def test_bail_value_identity_not_truthiness() -> None:
    ctx = Context()
    extra: list[str] = []
    ctx.on('b', lambda: 0)
    ctx.on('b', lambda: extra.append('unreachable'))
    # 0 bails even though it is falsy
    assert ctx.bail('b') == 0
    assert extra == []

    ctx.on('c', lambda: '')
    assert ctx.bail('c') == ''

    ctx.on('d', lambda: False)
    ctx.on('d', lambda: None)
    assert ctx.bail('d') is None


async def test_waterfall_next_passthrough() -> None:
    ctx = Context()
    order: list[str] = []

    def outer(value: int, next_: object) -> object:
        order.append('outer-in')
        result = next_()
        order.append('outer-out')
        return result + 1

    def final(value: int, next_: object) -> int:
        order.append('final')
        return value * 10

    ctx.on('w', outer)
    assert ctx.waterfall('w', 1, final) == 11
    assert order == ['outer-in', 'final', 'outer-out']


async def test_waterfall_veto() -> None:
    ctx = Context()
    ctx.on('w', lambda value, next_: 'veto')
    final_called: list[bool] = []
    result = ctx.waterfall(
        'w', 1, lambda value, next_: final_called.append(True)
    )
    # not calling next() vetoes the rest of the chain; the listener's return
    # value becomes the overall result
    assert result == 'veto'
    assert final_called == []


async def test_waterfall_returns_coroutine_unawaited() -> None:
    ctx = Context()

    async def async_listener(value: int, next_: object) -> object:
        return next_()

    ctx.on('w', async_listener)
    result = ctx.waterfall('w', 2, lambda value, next_: value + 1)
    # waterfall itself is synchronous: an async listener's result is a
    # coroutine the caller may await
    assert inspect.isawaitable(result)
    assert await result == 3


async def test_dispatch_filter_and_global() -> None:
    ctx = Context()
    hits: list[str] = []
    ctx.on('f', lambda scope, value: hits.append('normal'))
    ctx.on('f', lambda scope, value: hits.append('global'), {'global': True})
    scoped = ctx.extend({Context.filter: lambda target: False})
    ctx.emit(scoped, 'f', 1)
    # non-global listeners are filtered out by the thisArg's context filter
    assert hits == ['global']


async def test_prepend_option() -> None:
    ctx = Context()
    order: list[str] = []
    ctx.on('e', lambda: order.append('second'))
    ctx.on('e', lambda: order.append('first'), True)
    ctx.emit('e')
    assert order == ['first', 'second']


async def test_once() -> None:
    ctx = Context()
    hits: list[int] = []
    ctx.once('o', lambda value: hits.append(value))
    ctx.emit('o', 1)
    ctx.emit('o', 2)
    assert hits == [1]


async def test_listener_disposer() -> None:
    ctx = Context()
    hits: list[int] = []
    dispose = ctx.on('e', lambda value: hits.append(value))
    ctx.emit('e', 1)
    dispose()
    # repeated calls are a no-op
    dispose()
    ctx.emit('e', 2)
    assert hits == [1]


async def test_internal_listener_replaces_registration() -> None:
    ctx = Context()
    replacement = lambda: 'custom'  # noqa: E731

    def interceptor(
        scope: Context, name: str, listener: object, options: dict[str, object]
    ) -> object:
        if name == 'special':
            return replacement
        return None

    ctx.on('internal/listener', interceptor)
    result = ctx.on('special', lambda: None)
    # the bail value from internal/listener replaces normal registration
    assert result is replacement
    # and the listener was never stored
    ctx.emit('special')


async def test_internal_plugin_event() -> None:
    ctx = Context()
    seen: list[object] = []
    ctx.on('internal/plugin', lambda fiber: seen.append(fiber))

    def plugin(c: Context, config: object) -> None:
        return None

    fiber = ctx.plugin(plugin)
    await fiber
    assert seen == [fiber]
    await fiber.dispose()
    # fired again with uid cleared on disposal
    assert seen == [fiber, fiber]
    assert fiber.uid is None
