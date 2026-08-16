"""Behavior tests for the merged scope module (``pydsh.scope``).

Ported-semantics tests for the Python port of deepseek-harness
``packages/core/scope`` (MIT), covering scope minting and tags, the parent
chain, scope-carrier event filtering (untagged listeners global, identity
plus ancestor-chain admission, base-filter preservation), and the
insertion-ordered entry stores.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pycordis import Context, Service
from pydsh.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    bind_scope_parent,
    create_scope,
    scope_chain_of,
    scope_of,
    scope_parent_of,
    scope_target,
)


async def test_create_scope_tags_contexts_and_nearest_tag_wins() -> None:
    ctx = Context()
    outer_key, inner_key = object(), object()
    outer = create_scope(ctx, outer_key)
    inner = create_scope(outer.ctx, inner_key)

    assert scope_of(ctx) is None
    assert scope_of(outer.ctx) is outer_key
    assert scope_of(outer.ctx.extend({})) is outer_key
    assert scope_of(inner.ctx) is inner_key

    await inner.dispose()
    await outer.dispose()


async def test_scope_dispose_runs_registrations_and_shares_quiescence() -> None:
    ctx = Context()
    scope = create_scope(ctx, object())
    events: list[str] = []
    scope.ctx.effect(lambda: lambda: events.append('disposed'), 'test')

    first = scope.dispose()
    second = scope.dispose()
    # racing calls await the same completion
    assert first is second
    await asyncio.gather(first, second, scope.dispose())
    assert events == ['disposed']
    # teardown is done even when the raw disposer was already claimed
    await scope.dispose()
    assert events == ['disposed']


async def test_scope_raw_dispose_exposes_the_exact_fiber_disposer() -> None:
    ctx = Context()
    scope = create_scope(ctx, object())
    events: list[str] = []
    scope.ctx.effect(lambda: lambda: events.append('disposed'), 'test')
    result = scope.raw_dispose()
    if asyncio.iscoroutine(result):
        await result
    await scope.dispose()
    assert events == ['disposed']


async def test_scope_target_routes_by_key_and_admits_untagged_globally() -> None:
    ctx = Context()
    key_a, key_b = object(), object()
    scope_a = create_scope(ctx, key_a)
    scope_b = create_scope(ctx, key_b)
    heard: list[str] = []
    ctx.on('scope-test/ping', lambda carrier, v: heard.append(f'global:{v}'))
    scope_a.ctx.on('scope-test/ping', lambda carrier, v: heard.append(f'A:{v}'))
    scope_b.ctx.on('scope-test/ping', lambda carrier, v: heard.append(f'B:{v}'))

    ctx.emit(scope_target(ctx, key_a), 'scope-test/ping', 'a')
    ctx.emit(scope_target(ctx, key_b), 'scope-test/ping', 'b')
    # an unscoped subject reaches only untagged listeners
    ctx.emit(scope_target(ctx, None), 'scope-test/ping', 'none')

    assert heard == ['global:a', 'A:a', 'global:b', 'B:b', 'global:none']
    await asyncio.gather(scope_a.dispose(), scope_b.dispose())


async def test_scope_target_global_listeners_skip_filtering() -> None:
    ctx = Context()
    scope = create_scope(ctx, object())
    heard: list[str] = []
    scope.ctx.on(
        'scope-test/ping',
        lambda carrier, v: heard.append(v),
        {'global': True},
    )
    ctx.emit(scope_target(ctx, object()), 'scope-test/ping', 'foreign')
    ctx.emit(scope_target(ctx, None), 'scope-test/ping', 'none')
    assert heard == ['foreign', 'none']
    await scope.dispose()


async def test_scope_target_preserves_plain_subject_cordis_filter() -> None:
    ctx = Context()
    heard: list[str] = []
    ctx.on('scope-test/ping', lambda carrier, v: heard.append(v))

    class Subject:
        def _cordis_filter(self, target: Context) -> bool:
            return False

    ctx.emit(scope_target(Subject(), object()), 'scope-test/ping', 'vetoed')
    assert heard == []


async def test_scope_target_preserves_service_filter() -> None:
    ctx = Context()

    class Probe(Service):
        def filter(self, target: Context) -> bool:
            return False

    probe = Probe(ctx, 'probe')
    heard: list[str] = []
    ctx.on('scope-test/ping', lambda carrier, v: heard.append(v))
    ctx.emit(scope_target(probe, object()), 'scope-test/ping', 'vetoed')
    assert heard == []


async def test_scope_carrier_exposes_key_but_not_subject() -> None:
    key = object()
    subject = {'value': 1}
    carrier = scope_target(subject, key)
    assert carrier.key is key
    assert not hasattr(carrier, 'value')
    unkeyed = scope_target(subject, None)
    assert unkeyed.key is None


async def test_scope_target_admits_ancestor_tags_never_descendants() -> None:
    ctx = Context()
    preset, agent, other = object(), object(), object()
    preset_scope = create_scope(ctx, preset)
    agent_scope = create_scope(ctx, agent, parent=preset)
    other_scope = create_scope(ctx, other)
    heard: list[str] = []

    def record(tag: str) -> Any:
        return lambda carrier, v: heard.append(f'{tag}:{v}')

    ctx.on('scope-test/ping', record('global'))
    preset_scope.ctx.on('scope-test/ping', record('preset'))
    agent_scope.ctx.on('scope-test/ping', record('agent'))
    other_scope.ctx.on('scope-test/ping', record('other'))

    # events flow UP the chain: the enclosing scope observes its descendants
    ctx.emit(scope_target(ctx, agent), 'scope-test/ping', 'down')
    assert heard == ['global:down', 'preset:down', 'agent:down']

    heard.clear()
    # never DOWN: a descendant tag is excluded from an ancestor's dispatch
    ctx.emit(scope_target(ctx, preset), 'scope-test/ping', 'up')
    assert heard == ['global:up', 'preset:up']

    heard.clear()
    # a key outside the chain is excluded entirely
    ctx.emit(scope_target(ctx, other), 'scope-test/ping', 'side')
    assert heard == ['global:side', 'other:side']


async def test_scope_parent_chain_links_walks_and_rejects_cycles() -> None:
    ctx = Context()
    preset, agent = object(), object()
    create_scope(ctx, preset)
    create_scope(ctx, agent, parent=preset)

    assert scope_parent_of(agent) is preset
    assert scope_parent_of(preset) is None
    assert scope_chain_of(agent) == [agent, preset]
    assert scope_chain_of(None) == []
    with pytest.raises(ValueError, match='cycle'):
        bind_scope_parent(preset, agent)
    with pytest.raises(ValueError, match='cycle'):
        bind_scope_parent(preset, preset)


def test_scope_rebind_only_through_the_original_binding() -> None:
    preset_a, preset_b, agent = object(), object(), object()
    binding = bind_scope_parent(agent, preset_a)
    with pytest.raises(ValueError, match='already bound'):
        bind_scope_parent(agent, preset_b)
    binding.rebind(preset_b)
    assert scope_chain_of(agent) == [agent, preset_b]
    # the rebind keeps the cycle check: a parent may not adopt its ancestor
    child = object()
    bind_scope_parent(child, agent)
    with pytest.raises(ValueError, match='cycle'):
        binding.rebind(child)


def test_named_entries_insert_undo_and_drain_detach() -> None:
    entries: NamedEntries[int] = NamedEntries(
        lambda name: ValueError(f'duplicate: {name}')
    )
    undo_a = entries.insert('a', 1)
    entries.insert('b', 2)
    with pytest.raises(ValueError, match='duplicate: a'):
        entries.insert('a', 3)
    assert list(entries.entries()) == [('a', 1), ('b', 2)]
    assert entries.get('a') == 1
    assert entries.has('b')
    assert not entries.is_empty()

    undo_a()
    undo_a()  # idempotent
    assert list(entries.keys()) == ['b']
    assert entries.get('a') is None


def test_named_entries_stale_undo_cannot_touch_new_generation() -> None:
    entries: NamedEntries[int] = NamedEntries(lambda name: ValueError(name))
    undo1 = entries.insert('x', 1)
    undo2 = entries.insert('y', 2)
    undo1()
    undo2()  # drains and detaches the table
    entries.insert('x', 10)
    undo1()  # stale: must not delete the new generation's 'x'
    assert entries.get('x') == 10


def test_anonymous_entries_keep_equal_values_independent() -> None:
    entries: AnonymousEntries[int] = AnonymousEntries()
    undo_first = entries.append(1)
    entries.append(1)
    assert list(entries.values()) == [1, 1]
    undo_first()
    undo_first()  # idempotent
    assert list(entries.values()) == [1]
    assert not entries.is_empty()


class _Layer:
    def __init__(self, scope: object) -> None:
        self.scope = scope
        self.entries: NamedEntries[str] = NamedEntries(lambda name: ValueError(name))

    def is_empty(self) -> bool:
        return self.entries.is_empty()


async def test_scoped_layers_peek_chain_merge_and_reclaim() -> None:
    ctx = Context()
    changes: list[str] = []
    layers: ScopedLayers[_Layer] = ScopedLayers(
        _Layer, lambda: changes.append('change')
    )
    parent_key, child_key = object(), object()
    bind_scope_parent(child_key, parent_key)
    parent_scope = create_scope(ctx, parent_key)
    child_scope = create_scope(ctx, child_key)

    # reads never create scoped layers
    assert layers.peek(parent_key) is None
    assert layers.chain_layers(child_key) == []

    layers.effect(
        ctx,
        lambda layer: layer.entries.insert('g', 'global'),
        'test.global',
    )
    layers.effect(
        parent_scope.ctx,
        lambda layer: layer.entries.insert('p', 'parent'),
        'test.parent',
    )
    dispose_child = layers.effect(
        child_scope.ctx,
        lambda layer: layer.entries.insert('p', 'child'),
        'test.child',
    )
    assert changes == ['change'] * 3

    # peek is chain-blind; chain_layers runs farthest ancestor first
    assert layers.peek(parent_key) is not None
    chain = layers.chain_layers(child_key)
    assert [layer.scope for layer in chain] == [parent_key, child_key]

    # merge: the nearest scope shadows a global/ancestor name
    merged = layers.merge(child_key, lambda layer: layer.entries)
    assert merged == {'g': 'global', 'p': 'child'}
    assert layers.merge(None, lambda layer: layer.entries) == {'g': 'global'}

    # disposal reclaims only a completely empty scoped layer
    dispose_child()
    assert layers.peek(child_key) is None
    assert layers.peek(parent_key) is not None
    assert changes == ['change'] * 4


async def test_scoped_layers_failed_action_reclaims_created_layer() -> None:
    ctx = Context()
    layers: ScopedLayers[_Layer] = ScopedLayers(_Layer, lambda: None)
    key = object()
    scope = create_scope(ctx, key)

    def bad_action(layer: _Layer) -> Any:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        layers.effect(scope.ctx, bad_action, 'test.bad')
    assert layers.peek(key) is None


async def test_scoped_layers_notify_can_be_suppressed() -> None:
    ctx = Context()
    changes: list[str] = []
    layers: ScopedLayers[_Layer] = ScopedLayers(
        _Layer, lambda: changes.append('change')
    )
    dispose = layers.effect(
        ctx,
        lambda layer: layer.entries.insert('g', 'global'),
        'test.quiet',
        notify=False,
    )
    dispose()
    assert changes == []
