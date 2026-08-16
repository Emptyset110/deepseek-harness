"""Behavior tests for service provision and the inject lifecycle.

Ported-semantics tests for the cordis Python port (MIT, cordiverse;
vendored in deepseek-harness).
"""

from __future__ import annotations

import pytest

from pycordis import Context, FiberState


async def test_provide_and_get() -> None:
    ctx = Context()
    ctx.provide('foo', 42)
    assert ctx.get('foo') == 42
    assert ctx.get('missing') is None
    assert ctx.get('missing', strict=False) is None
    # root context attribute access resolves provided services
    assert ctx.foo == 42


async def test_duplicate_provide_throws() -> None:
    ctx = Context()
    ctx.provide('foo', 1)
    with pytest.raises(RuntimeError, match='has been registered'):
        ctx.provide('foo', 2)
    # the first registration survives the failed second provide
    assert ctx.get('foo') == 1


async def test_inject_pending_activate_unload() -> None:
    ctx = Context()
    loads: list[int] = []
    unloads: list[int] = []

    def consumer(c: Context, config: object = None) -> object:
        loads.append(c.foo)
        return lambda: unloads.append(c.foo)

    fiber = ctx.inject(['foo'], consumer)
    # missing dependency keeps the fiber PENDING
    assert fiber.state is FiberState.PENDING
    assert loads == []

    dispose_foo = ctx.provide('foo', 1)
    await fiber
    assert fiber.state is FiberState.ACTIVE
    assert loads == [1]

    # revoking the service unloads the dependent; awaiting the returned
    # disposal settles once dependents finished unloading
    await dispose_foo()
    assert unloads == [1]
    assert fiber.state is FiberState.PENDING


async def test_provider_swap_reloads_dependent() -> None:
    ctx = Context()
    loads: list[str] = []
    unloads: list[str] = []

    def provider_a(c: Context, config: object = None) -> None:
        c.provide('foo', 'A')

    def provider_b(c: Context, config: object = None) -> None:
        c.provide('foo', 'B')

    def consumer(c: Context, config: object = None) -> object:
        loads.append(c.foo)
        return lambda: unloads.append(c.foo)

    consumer_fiber = ctx.inject(['foo'], consumer)
    fiber_a = ctx.plugin(provider_a)
    await fiber_a
    await consumer_fiber
    assert loads == ['A']
    assert consumer_fiber.state is FiberState.ACTIVE

    await fiber_a.dispose()
    assert unloads == ['A']
    assert consumer_fiber.state is FiberState.PENDING

    fiber_b = ctx.plugin(provider_b)
    await fiber_b
    await consumer_fiber
    # a new provider uid forces a full unload + reload of the dependent
    assert loads == ['A', 'B']
    assert consumer_fiber.state is FiberState.ACTIVE


async def test_value_change_same_owner_does_not_reload() -> None:
    ctx = Context()
    loads: list[str] = []

    def provider(c: Context, config: object = None) -> None:
        c.provide('foo', 'one')

    def consumer(c: Context, config: object = None) -> None:
        loads.append(c.foo)

    provider_fiber = ctx.plugin(provider)
    consumer_fiber = ctx.inject(['foo'], consumer)
    await provider_fiber
    await consumer_fiber
    assert loads == ['one']

    # same owner fiber, new value: no reload, but the snapshot sees the value
    provider_fiber.ctx.set('foo', 'two')
    await consumer_fiber.wait()
    assert loads == ['one']
    assert consumer_fiber.ctx.foo == 'two'


async def test_attribute_access_requires_inject() -> None:
    ctx = Context()

    def provider(c: Context, config: object = None) -> None:
        c.provide('bar', 'x')

    provider_fiber = ctx.plugin(provider)
    await provider_fiber

    def peek(c: Context, config: object = None) -> None:
        _ = c.bar  # not declared in inject

    failed = ctx.plugin(peek)
    with pytest.raises(AttributeError, match='without inject'):
        await failed
    assert failed.state is FiberState.FAILED

    seen: list[str] = []
    fiber = ctx.inject(['bar'], lambda c, config=None: seen.append(c.bar))
    await fiber
    assert seen == ['x']


async def test_attribute_access_in_inactive_context() -> None:
    ctx = Context()
    fiber = ctx.inject(['missing'], lambda c, config=None: None)
    assert fiber.state is FiberState.PENDING
    with pytest.raises(AttributeError, match='inactive context'):
        _ = fiber.ctx.missing
    with pytest.raises(AttributeError, match='without inject'):
        _ = fiber.ctx.other


async def test_accessor() -> None:
    ctx = Context()
    stored: list[int] = []
    ctx.accessor('computed', get=lambda c, error: 42)
    assert ctx.computed == 42
    ctx.accessor(
        'rw',
        get=lambda c, error: stored,
        set=lambda c, value, error: stored.append(value) is None or True,
    )
    ctx.rw = 5
    assert stored == [5]


async def test_accessor_conflict_with_provide() -> None:
    ctx = Context()
    ctx.accessor('thing', get=lambda c, error: 1)
    with pytest.raises(RuntimeError, match='already declared as accessor'):
        ctx.provide('thing', 2)


async def test_mixin() -> None:
    class Greeter:
        def greet(self) -> str:
            return 'hi'

    ctx = Context()
    ctx.provide('greeter', Greeter())
    ctx.mixin('greeter', ['greet'])
    assert ctx.greet() == 'hi'


async def test_internal_service_event() -> None:
    ctx = Context()
    seen: list[tuple[str, object]] = []
    # internal/service is dispatched with a scope context as thisArg, which
    # is prepended to the listener arguments
    ctx.on('internal/service', lambda scope, name, value: seen.append((name, value)))
    ctx.provide('foo', 1)
    assert ('foo', 1) in seen
