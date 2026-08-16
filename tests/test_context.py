"""Behavior tests for context scoping: root, extend, isolate, intercept.

Ported-semantics tests for the cordis Python port (MIT, cordiverse;
vendored in deepseek-harness).
"""

from __future__ import annotations

from pycordis import Context, FiberState, Service


async def test_root_context() -> None:
    ctx = Context()
    assert ctx.fiber.uid == 0
    assert ctx.fiber.state is FiberState.ACTIVE
    assert ctx.root is ctx
    assert ctx.fiber.name == 'root'
    assert Context.is_(ctx)
    assert not Context.is_(object())


async def test_extend_shadows_without_mutating_parent() -> None:
    ctx = Context()
    child = ctx.extend({'extra': 1})
    assert child.extra == 1
    assert child.root is ctx
    assert child.fiber is ctx.fiber
    # the parent is not mutated; unknown services resolve to None at root
    assert ctx.extra is None


async def test_isolate_scopes_are_independent() -> None:
    ctx = Context()
    ctx.provide('foo', 'root')
    isolated = ctx.isolate('foo')
    # providing the same name in a fresh scope does not clash with root
    isolated.provide('foo', 'isolated')
    assert ctx.get('foo') == 'root'
    assert isolated.get('foo') == 'isolated'


async def test_isolate_same_label_joins_scopes() -> None:
    ctx = Context()
    label = object()
    first = ctx.isolate('foo', label)
    second = ctx.isolate('foo', label)
    first.provide('foo', 'joined')
    assert second.get('foo') == 'joined'
    # a fresh label is a different scope
    third = ctx.isolate('foo')
    assert third.get('foo') is None


async def test_isolated_dependent_follows_scope_provider() -> None:
    ctx = Context()
    ctx.provide('foo', 'root')
    isolated = ctx.isolate('foo')

    seen: list[str] = []

    def consumer(c: Context, config: object = None) -> None:
        seen.append(c.foo)

    fiber = isolated.inject(['foo'], consumer)
    assert fiber.state is FiberState.PENDING
    isolated.provide('foo', 'inner')
    await fiber
    assert seen == ['inner']
    assert fiber.state is FiberState.ACTIVE


async def test_intercept_merges_root_first() -> None:
    class Dummy(Service):
        provide = 'dummy'

    ctx = Context()
    base = ctx.intercept('dummy', {'a': 1, 'b': 1})
    child = base.intercept('dummy', {'b': 2})
    service = Dummy(child)
    assert service.resolve_config() == {'a': 1, 'b': 2}
    # the parent context is not affected (isolated scope avoids the
    # duplicate-provide error from registering the same name twice)
    plain = Dummy(ctx.isolate('dummy'))
    assert plain.resolve_config() == {}


async def test_intercept_logger_name() -> None:
    ctx = Context()
    scoped = ctx.intercept('logger', {'name': 'custom'})
    assert scoped.logger().name == 'custom'
    assert ctx.logger('explicit').name == 'explicit'
    assert ctx.logger().name == 'root'
