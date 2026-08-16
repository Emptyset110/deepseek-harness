"""Behavior tests for the agent registry service (``ctx.agents``).

Covers registration/detachment/query semantics and the paired
``agent/created`` / ``agent/disposed`` lifecycle events. Ported-semantics
tests for the Python port of deepseek-harness ``packages/core/agent`` (MIT).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from pycordis import Context, LoggerLevel
from pydsh.agent import (
    AgentOptions,
    AgentRegistry,
    SessionId,
    scoped_context,
)


class FakeInbox:
    @property
    def next_turn(self) -> list[Any]:
        return []

    @property
    def next_step(self) -> list[Any]:
        return []

    @property
    def has_pending(self) -> bool:
        return False


class FakeSession:
    def __init__(self, session_id: SessionId) -> None:
        self.id = session_id


class FakeAgent:
    """Minimal ``Agent`` protocol implementation for registry tests."""

    def __init__(
        self,
        ctx: Context,
        agent_id: str,
        session_id: str | None = None,
    ) -> None:
        self.id = SessionId(agent_id)
        self.options = AgentOptions()
        session = session_id if session_id is not None else agent_id
        self.session = FakeSession(SessionId(session))
        self.inbox = FakeInbox()
        self.status = 'idle'
        self.ctx = ctx


def make_registry() -> tuple[Context, AgentRegistry]:
    ctx = Context()
    registry = AgentRegistry(ctx)
    return ctx, registry


class WarnRecorder:
    """Exporter capturing warn records (the default level gate drops warns)."""

    levels: ClassVar[dict[str, LoggerLevel]] = {'default': LoggerLevel.WARN}

    def __init__(self) -> None:
        self.messages: list[str] = []

    def export(self, message: Any) -> None:
        if message.type == 'warn':
            self.messages.append(str(message.args[0]))


def record_warns(ctx: Context) -> list[str]:
    recorder = WarnRecorder()
    ctx.logger.exporter(recorder)
    return recorder.messages


async def flush_async_listener_results() -> None:
    # schedule_logged delivers a rejection to the logger one loop tick after
    # the listener coroutine completes; yield a few ticks to observe it
    for _ in range(3):
        await asyncio.sleep(0)


async def test_service_is_installed_as_ctx_agents() -> None:
    ctx, registry = make_registry()
    # property access resolves through the service store (a rebound view)
    assert registry.name == 'agents'
    assert ctx.get('agents').name == 'agents'
    assert ctx.agents.get(SessionId('missing')) is None


async def test_register_emits_created_and_supports_queries() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    created: list[dict[str, Any]] = []
    ctx.on('agent/created', lambda carrier, payload: created.append(payload))

    dispose = registry.register(agent)

    assert created == [{'agent': agent}]
    assert registry.get(SessionId('a1')) is agent
    assert registry.list() == [agent]
    assert registry.roots() == [agent]
    dispose()


async def test_register_rejects_duplicate_id() -> None:
    ctx, registry = make_registry()
    registry.register(FakeAgent(ctx, 'a1'))
    with pytest.raises(ValueError, match='agent "a1" is already registered'):
        registry.register(FakeAgent(ctx, 'a1'))


async def test_register_rejects_session_id_mismatch() -> None:
    ctx, registry = make_registry()
    with pytest.raises(ValueError, match='does not match session id'):
        registry.register(FakeAgent(ctx, 'a1', session_id='other'))


async def test_disposer_unregisters_and_emits_disposed_once() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    disposed: list[dict[str, Any]] = []
    ctx.on('agent/disposed', lambda carrier, payload: disposed.append(payload))

    dispose = registry.register(agent)
    dispose()

    assert disposed == [{'agent': agent}]
    assert registry.get(SessionId('a1')) is None
    assert registry.list() == []
    # single-shot: a repeat call is a no-op
    dispose()
    assert len(disposed) == 1


async def test_register_from_agent_scoped_context_records_ownership() -> None:
    ctx, registry = make_registry()
    owner = FakeAgent(ctx, 'parent')
    registry.register(owner)

    child_ctx = ctx.extend({'agent': owner})
    child = FakeAgent(ctx, 'child')
    child_ctx.agents.register(child)

    assert registry.is_owned_by(SessionId('child'), owner)
    assert not registry.is_owned_by(SessionId('parent'), owner)
    assert registry.roots() == [owner]
    assert registry.list() == [owner, child]


async def test_enter_without_announce_rolls_back_without_events() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    events: list[str] = []
    ctx.on('agent/created', lambda carrier, payload: events.append('created'))
    ctx.on('agent/disposed', lambda carrier, payload: events.append('disposed'))

    detach = registry.enter(agent, None)
    assert registry.get(SessionId('a1')) is agent
    detach()

    # an unannounced insertion was never externally created: no disposal edge
    assert registry.get(SessionId('a1')) is None
    assert events == []
    # idempotent
    detach()


async def test_announce_emits_created_after_enter() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    created: list[dict[str, Any]] = []
    ctx.on('agent/created', lambda carrier, payload: created.append(payload))

    detach = registry.enter(agent, None)
    assert created == []
    registry.announce(agent)
    assert created == [{'agent': agent}]
    detach()
    assert registry.get(SessionId('a1')) is None


async def test_announce_rejects_unknown_and_repeat() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    with pytest.raises(ValueError, match='is not live in this registry'):
        registry.announce(agent)

    registry.enter(agent, None)
    registry.announce(agent)
    with pytest.raises(ValueError, match='was already announced'):
        registry.announce(agent)
    # a same-id different object is not the live entry
    with pytest.raises(ValueError, match='is not live in this registry'):
        registry.announce(FakeAgent(ctx, 'a1'))


async def test_detach_during_announce_waits_for_dispatch_to_unwind() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    detach = registry.enter(agent, None)
    observed_during_dispatch: list[Any] = []
    disposed: list[str] = []
    ctx.on('agent/disposed', lambda carrier, payload: disposed.append('disposed'))

    def on_created(carrier: Any, payload: dict[str, Any]) -> None:
        detach()
        # removal is deferred: the entry is still live inside the dispatch
        observed_during_dispatch.append(registry.get(SessionId('a1')))

    ctx.on('agent/created', on_created)
    registry.announce(agent)

    assert observed_during_dispatch == [agent]
    assert registry.get(SessionId('a1')) is None
    assert disposed == ['disposed']


async def test_created_listener_throw_vetoes_publication_and_rolls_back() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    events: list[str] = []
    ctx.on('agent/created', lambda carrier, payload: events.append('created-early'))

    def veto(carrier: Any, payload: dict[str, Any]) -> None:
        raise RuntimeError('nope')

    ctx.on('agent/created', veto)
    ctx.on('agent/disposed', lambda carrier, payload: events.append('disposed'))

    with pytest.raises(RuntimeError, match='nope'):
        registry.register(agent)

    # the earlier listener already observed creation and must see disposal
    assert events == ['created-early', 'disposed']
    assert registry.get(SessionId('a1')) is None


async def test_disposed_listener_failures_are_contained_and_logged() -> None:
    ctx, registry = make_registry()
    agent = FakeAgent(ctx, 'a1')
    survived: list[str] = []

    def throwing(carrier: Any, payload: dict[str, Any]) -> None:
        raise RuntimeError('sync boom')

    async def rejecting(carrier: Any, payload: dict[str, Any]) -> None:
        raise RuntimeError('async boom')

    ctx.on('agent/disposed', throwing)
    ctx.on('agent/disposed', rejecting)
    ctx.on('agent/disposed', lambda carrier, payload: survived.append('ok'))
    warnings = record_warns(ctx)

    dispose = registry.register(agent)
    dispose()

    # neither failure vetoed teardown or starved the later listener
    assert survived == ['ok']
    assert registry.get(SessionId('a1')) is None
    await flush_async_listener_results()
    assert any('listener threw' in w and 'sync boom' in w for w in warnings)
    assert any('listener rejected' in w and 'async boom' in w for w in warnings)


async def test_lifecycle_events_are_scope_filtered_per_agent() -> None:
    ctx, registry = make_registry()
    agent_a = FakeAgent(ctx, 'a')
    agent_b = FakeAgent(ctx, 'b')
    hits: list[str] = []
    scoped_context(ctx, agent_a).on(
        'agent/created', lambda carrier, payload: hits.append('a-listener')
    )
    scoped_context(ctx, agent_b).on(
        'agent/created', lambda carrier, payload: hits.append('b-listener')
    )

    registry.register(agent_a)
    assert hits == ['a-listener']
    registry.register(agent_b)
    assert hits == ['a-listener', 'b-listener']
