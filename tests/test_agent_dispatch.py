"""Behavior tests for the fused agent-scoped dispatcher (``agent_events``).

Covers the scope-carrier filtering (matching agent and its scope ancestors,
untagged listeners admitted globally, ``{'global': True}`` listeners
unfiltered), per-listener failure containment on emit, and the waterfall
semantics of ``agent/pre-step`` and ``agent/request``. Ported-semantics
tests for the Python port of deepseek-harness ``packages/core/agent`` (MIT).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, ClassVar

import pytest

from pycordis import Context, LoggerLevel
from pydsh.agent import (
    AgentOptions,
    LlmCallConfig,
    PreStepEnter,
    PreStepReject,
    SessionId,
    UserMessage,
    agent_carrier,
    agent_events,
    emit_agent_event,
    scoped_context,
)
from pydsh.scope import bind_scope_parent


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
    """Minimal ``Agent`` protocol implementation whose ctx carries its scope tag."""

    def __init__(self, ctx: Context, agent_id: str) -> None:
        self.id = SessionId(agent_id)
        self.options = AgentOptions()
        self.session = FakeSession(self.id)
        self.inbox = FakeInbox()
        self.status = 'idle'
        self.ctx = scoped_context(ctx, self)


def make_agent(ctx: Context, agent_id: str = 'a') -> FakeAgent:
    return FakeAgent(ctx, agent_id)


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


async def test_emit_fuses_subject_and_filters_by_agent_scope() -> None:
    ctx = Context()
    agent_a = make_agent(ctx, 'a')
    agent_b = make_agent(ctx, 'b')
    hits: list[str] = []

    agent_a.ctx.on('agent/status', lambda carrier, payload: hits.append('scoped-a'))
    agent_b.ctx.on('agent/status', lambda carrier, payload: hits.append('scoped-b'))
    # untagged listeners are admitted for every agent
    ctx.on('agent/status', lambda carrier, payload: hits.append('untagged'))
    # global listeners skip filtering entirely
    ctx.on(
        'agent/status',
        lambda carrier, payload: hits.append('global'),
        {'global': True},
    )

    agent_events(ctx, agent_a).emit('agent/status', {'status': 'running'})
    assert hits == ['scoped-a', 'untagged', 'global']

    hits.clear()
    agent_events(ctx, agent_b).emit('agent/status', {'status': 'idle'})
    assert hits == ['scoped-b', 'untagged', 'global']


async def test_emit_admits_ancestor_scope_listeners_never_descendants() -> None:
    # the carrier admits a listener tagged with the dispatch key or any of
    # its scope ancestors (dsh-scope's bindScopeParent chain)
    ctx = Context()
    preset_agent = make_agent(ctx, 'preset')
    child_agent = make_agent(ctx, 'child')
    bind_scope_parent(child_agent, preset_agent)
    hits: list[str] = []

    preset_agent.ctx.on('agent/status', lambda carrier, payload: hits.append('preset'))
    child_agent.ctx.on('agent/status', lambda carrier, payload: hits.append('child'))
    ctx.on('agent/status', lambda carrier, payload: hits.append('untagged'))

    # events flow UP the chain: the enclosing scope observes descendants
    agent_events(ctx, child_agent).emit('agent/status', {'status': 'running'})
    assert hits == ['preset', 'child', 'untagged']

    hits.clear()
    # never DOWN: the descendant tag is excluded from the ancestor's dispatch
    agent_events(ctx, preset_agent).emit('agent/status', {'status': 'idle'})
    assert hits == ['preset', 'untagged']


async def test_emit_injects_agent_subject_into_payload() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    seen: list[dict[str, Any]] = []
    carriers: list[Any] = []

    def listener(carrier: Any, payload: dict[str, Any]) -> None:
        carriers.append(carrier)
        seen.append(payload)

    ctx.on('agent/status', listener)

    carrier = agent_carrier(agent)
    dispatch = agent_events(ctx, agent, carrier)
    dispatch.emit('agent/status', {'status': 'running'})

    assert seen == [{'status': 'running', 'agent': agent}]
    assert carriers == [carrier]
    assert carrier.key is agent


async def test_payload_agent_key_cannot_override_injected_subject() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    other = make_agent(ctx, 'other')
    seen: list[dict[str, Any]] = []
    ctx.on('agent/status', lambda carrier, payload: seen.append(payload))

    agent_events(ctx, agent).emit('agent/status', {'status': 'idle', 'agent': other})
    assert seen[0]['agent'] is agent


async def test_emit_contains_sync_and_async_listener_failures() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    survived: list[str] = []

    def throwing(carrier: Any, payload: dict[str, Any]) -> None:
        raise RuntimeError('sync boom')

    async def rejecting(carrier: Any, payload: dict[str, Any]) -> None:
        raise RuntimeError('async boom')

    ctx.on('agent/error', throwing)
    ctx.on('agent/error', rejecting)
    ctx.on('agent/error', lambda carrier, payload: survived.append('ok'))
    warnings = record_warns(ctx)

    agent_events(ctx, agent).emit('agent/error', {'turn': 1, 'step': 0, 'error': 'x'})

    assert survived == ['ok']
    await flush_async_listener_results()
    assert any('listener threw' in w and 'sync boom' in w for w in warnings)
    assert any('listener rejected' in w and 'async boom' in w for w in warnings)


async def test_carrier_preserves_base_filter() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    # a base subject whose own cordis filter rejects every context
    agent._cordis_filter = lambda target: False  # type: ignore[attr-defined]
    hits: list[str] = []
    ctx.on('agent/status', lambda carrier, payload: hits.append('hit'))
    ctx.on(
        'agent/status',
        lambda carrier, payload: hits.append('global'),
        {'global': True},
    )

    agent_events(ctx, agent).emit('agent/status', {'status': 'running'})
    assert hits == ['global']


async def test_emit_agent_event_one_shot_helper() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    seen: list[dict[str, Any]] = []
    agent.ctx.on('agent/session-start', lambda carrier, payload: seen.append(payload))

    emit_agent_event(ctx, agent, 'agent/session-start', {'source': 'startup'})
    assert seen == [{'source': 'startup', 'agent': agent}]


async def test_serial_turn_stopping_runs_in_order_until_bail() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    order: list[str] = []

    async def first(carrier: Any, payload: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        order.append('first')

    def second(carrier: Any, payload: dict[str, Any]) -> str:
        order.append('second')
        return 'stop'

    def unreachable(carrier: Any, payload: dict[str, Any]) -> None:
        order.append('unreachable')

    agent.ctx.on('agent/turn-stopping', first)
    agent.ctx.on('agent/turn-stopping', second)
    ctx.on('agent/turn-stopping', unreachable)

    result = await agent_events(ctx, agent).serial('agent/turn-stopping', {'turn': 3})
    assert result == 'stop'
    assert order == ['first', 'second']


async def test_serial_is_scope_filtered() -> None:
    ctx = Context()
    agent_a = make_agent(ctx, 'a')
    agent_b = make_agent(ctx, 'b')
    order: list[str] = []
    agent_a.ctx.on('agent/turn-stopping', lambda carrier, payload: order.append('a'))
    agent_b.ctx.on('agent/turn-stopping', lambda carrier, payload: order.append('b'))

    await agent_events(ctx, agent_a).serial('agent/turn-stopping', {'turn': 1})
    assert order == ['a']


async def test_pre_step_waterfall_passthrough_and_replacement() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    messages = [UserMessage(id='m1', content='hello')]
    default = PreStepEnter(messages=list(messages))

    # a listener that calls next() delegates, then may wrap the decision
    def delegating(carrier: Any, payload: dict[str, Any], next_: Any) -> Any:
        assert payload['agent'] is agent
        assert payload['messages'] == messages
        assert payload['turn'] == 1 and payload['step'] == 0
        decision = next_()
        assert decision is default
        return decision

    ctx.on('agent/pre-step', delegating)
    result = agent_events(ctx, agent).waterfall(
        'agent/pre-step',
        {'messages': messages, 'turn': 1, 'step': 0, 'signal': None},
        lambda *_args: default,
    )
    assert result is default


async def test_pre_step_waterfall_short_circuits_without_next() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    inner_called: list[bool] = []

    def rejecting(carrier: Any, payload: dict[str, Any], next_: Any) -> PreStepReject:
        return PreStepReject()

    def inner(carrier: Any, payload: dict[str, Any], next_: Any) -> Any:
        inner_called.append(True)
        return next_()

    ctx.on('agent/pre-step', rejecting)
    ctx.on('agent/pre-step', inner)
    result = agent_events(ctx, agent).waterfall(
        'agent/pre-step',
        {'messages': [], 'turn': 1, 'step': 0, 'signal': None},
        lambda *_args: PreStepEnter(messages=[]),
    )
    # returning without next() vetoes the rest of the chain, default included
    assert result == PreStepReject()
    assert inner_called == []


async def test_request_waterfall_replaces_frozen_call_config() -> None:
    ctx = Context()
    agent = make_agent(ctx)
    seed = LlmCallConfig(provider='seed', model='seed', temperature=0.2)
    replacement = LlmCallConfig(provider='alpha', model='a1')

    def switching(carrier: Any, payload: dict[str, Any], next_: Any) -> LlmCallConfig:
        assert next_() is seed
        return replacement

    ctx.on('agent/request', switching)
    result = agent_events(ctx, agent).waterfall(
        'agent/request',
        {'turn': 1, 'step': 0, 'signal': None},
        lambda *_args: seed,
    )
    assert result is replacement


async def test_llm_call_config_is_frozen_and_defensively_copied() -> None:
    stop = ['</end>']
    config = LlmCallConfig(provider='p', model='m', stop=stop)
    stop.append('mutated')
    assert config.stop == ('</end>',)
    assert isinstance(config.stop, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model = 'other'  # type: ignore[misc]


async def test_waterfall_is_scope_filtered() -> None:
    ctx = Context()
    agent_a = make_agent(ctx, 'a')
    agent_b = make_agent(ctx, 'b')
    order: list[str] = []
    agent_b.ctx.on('agent/pre-step', lambda carrier, payload, next_: order.append('b'))

    decision = PreStepEnter(messages=[])
    result = agent_events(ctx, agent_a).waterfall(
        'agent/pre-step',
        {'messages': [], 'turn': 1, 'step': 0, 'signal': None},
        lambda *_args: decision,
    )
    assert result is decision
    assert order == []
