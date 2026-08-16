"""Behavior tests for the approval service (``ctx.approval``).

Covers the turn-enclosed audit pair, the fail-closed waterfall semantics
(missing/throwing/rogue answerers → ``'unavailable'``), the deterministic
``'never'`` policy, policy fold and switch event shapes, abort racing, and
agent-scoped dispatch. Ported-semantics tests for the Python port of
deepseek-harness ``packages/interaction/user-approval`` (MIT).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pycordis import Context
from pydsh.agent import scoped_context
from pydsh.approval import (
    APPROVAL_POLICIES,
    ApprovalRequest,
    ApprovalService,
    effective_approval_policy,
    set_approval_policy,
)
from pydsh.approval.types import ApprovalRequestId
from pydsh.session import Session, SessionEvent, SessionStore
from pydsh.subprocess.types import AbortController


class FakeAgent:
    """Minimal ``Agent`` stand-in: identity, live session, scope-tagged ctx."""

    def __init__(self, ctx: Context, session: Session) -> None:
        self.id = session.id
        self.session = session
        self.ctx = scoped_context(ctx, self)


def make_agent(ctx: Context, store: SessionStore) -> FakeAgent:
    return FakeAgent(ctx, store.create())


def open_turn(session: Session, turn: int = 1) -> None:
    session.append('turn/start', {'turn': turn})


def close_turn(session: Session, turn: int = 1) -> None:
    session.append('turn/end', {'turn': turn, 'reason': {'kind': 'completed'}})


def events_of(session: Session, type_: str) -> list[SessionEvent]:
    return [event for event in session.events if event['type'] == type_]


async def test_request_without_answerer_fails_closed_with_audit_pair() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))

    assert outcome == 'unavailable'
    asked = events_of(agent.session, 'approval/asked')
    decided = events_of(agent.session, 'approval/decided')
    assert len(asked) == 1 and len(decided) == 1
    assert asked[0]['data'] == {'id': decided[0]['data']['id'], 'toolName': 'bash'}
    assert decided[0]['data']['outcome'] == 'unavailable'
    # the pair is turn-enclosed: asked/decided sit between turn/start and the end
    types = [event['type'] for event in agent.session.events]
    assert types == ['turn/start', 'approval/asked', 'approval/decided']


async def test_request_outside_open_turn_raises_before_appending() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)

    with pytest.raises(RuntimeError, match='outside an open turn'):
        await service.request(ApprovalRequest(agent=agent, tool_name='bash'))

    open_turn(agent.session)
    close_turn(agent.session)
    with pytest.raises(RuntimeError, match='outside an open turn'):
        await service.request(ApprovalRequest(agent=agent, tool_name='bash'))

    assert events_of(agent.session, 'approval/asked') == []


async def test_audit_event_carries_call_id_and_reason_when_present() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    await service.request(
        ApprovalRequest(
            agent=agent,
            tool_name='edit',
            call_id='call-1',
            reason='hook denied by policy',
        )
    )

    asked = events_of(agent.session, 'approval/asked')[0]['data']
    assert asked['toolName'] == 'edit'
    assert asked['callId'] == 'call-1'
    assert asked['reason'] == 'hook denied by policy'


async def test_answerer_grants_and_waterfall_passthrough() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    def granting(carrier: Any, req: ApprovalRequest, next_: Any) -> str:
        assert req.tool_name == 'bash'
        return 'allowed-once'

    ctx.on('approval/request', granting)
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'allowed-once'

    # a delegating answerer wraps the chain's inner result
    def delegating(carrier: Any, req: ApprovalRequest, next_: Any) -> Any:
        return next_()

    ctx.on('approval/request', delegating, {'prepend': True})
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'allowed-once'


async def test_waterfall_delegation_reaches_fail_closed_default() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    async def delegating(carrier: Any, req: ApprovalRequest, next_: Any) -> Any:
        return await next_()

    ctx.on('approval/request', delegating)
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'unavailable'


async def test_throwing_answerers_fail_the_question_closed() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    def sync_throw(carrier: Any, req: ApprovalRequest, next_: Any) -> Any:
        raise RuntimeError('sync boom')

    ctx.on('approval/request', sync_throw)
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'unavailable'


async def test_async_rejecting_answerer_fails_closed() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    async def async_throw(carrier: Any, req: ApprovalRequest, next_: Any) -> Any:
        raise RuntimeError('async boom')

    ctx.on('approval/request', async_throw)
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'unavailable'


async def test_rogue_answerer_return_is_normalized_to_unavailable() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    ctx.on('approval/request', lambda carrier, req, next_: 'maybe')
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'unavailable'
    decided = events_of(agent.session, 'approval/decided')
    assert decided[0]['data']['outcome'] == 'unavailable'


async def test_effective_approval_policy_folds_last_switch() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    session = store.create()

    assert effective_approval_policy(session.events) is None
    set_approval_policy(session, 'never')
    assert effective_approval_policy(session.events) == 'never'
    set_approval_policy(session, 'ask')
    assert effective_approval_policy(session.events) == 'ask'

    policy_events = events_of(session, 'approval/policy')
    assert [event['data'] for event in policy_events] == [
        {'policy': 'never'},
        {'policy': 'ask'},
    ]


async def test_set_approval_policy_validates_before_appending() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    session = store.create()

    with pytest.raises(TypeError, match='one of "ask" or "never"'):
        set_approval_policy(session, 'sometimes')  # type: ignore[arg-type]
    assert session.events == ()
    for policy in APPROVAL_POLICIES:
        set_approval_policy(session, policy)
    assert [e['data']['policy'] for e in events_of(session, 'approval/policy')] == [
        'ask',
        'never',
    ]


async def test_never_policy_rejects_deterministically_before_dispatch() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)
    calls: list[str] = []

    # even a prepended granting listener must not see the ask
    ctx.on(
        'approval/request',
        lambda carrier, req, next_: calls.append('called') or 'allowed-once',
        {'prepend': True},
    )
    set_approval_policy(agent.session, 'never')
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'rejected'
    assert calls == []
    decided = events_of(agent.session, 'approval/decided')
    assert decided[0]['data']['outcome'] == 'rejected'


async def test_session_override_wins_over_configured_default() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx, {'policy': 'never'})
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    assert service.override_of(agent.session) is None
    # configured default applies without a session override
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'rejected'

    # the session override takes over
    set_approval_policy(agent.session, 'ask')
    assert service.override_of(agent.session) == 'ask'
    outcome = await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    assert outcome == 'unavailable'


async def test_config_validation_fails_loud() -> None:
    with pytest.raises(TypeError, match='one of "ask" or "never"'):
        ApprovalService(Context(), {'policy': 'sometimes'})  # type: ignore[typeddict-item]


async def test_pre_aborted_signal_cancels_without_dispatch() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)
    calls: list[str] = []
    ctx.on('approval/request', lambda carrier, req, next_: calls.append('called'))

    controller = AbortController()
    controller.abort()
    outcome = await service.request(
        ApprovalRequest(agent=agent, tool_name='bash', signal=controller.signal)
    )
    assert outcome == 'cancelled'
    assert calls == []
    decided = events_of(agent.session, 'approval/decided')
    assert decided[0]['data']['outcome'] == 'cancelled'


async def test_abort_during_pending_answer_cancels_and_discards_late_answer() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)
    release = asyncio.Event()

    async def slow_answerer(carrier: Any, req: ApprovalRequest, next_: Any) -> str:
        await release.wait()
        return 'allowed-once'

    ctx.on('approval/request', slow_answerer)
    controller = AbortController()
    task = asyncio.ensure_future(
        service.request(
            ApprovalRequest(agent=agent, tool_name='bash', signal=controller.signal)
        )
    )
    await asyncio.sleep(0)  # let the answerer start waiting
    controller.abort()
    assert await task == 'cancelled'

    # the late answer is discarded; the logged outcome stays 'cancelled'
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    decided = events_of(agent.session, 'approval/decided')
    assert [event['data']['outcome'] for event in decided] == ['cancelled']


async def test_answer_completes_normally_alongside_live_signal() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)
    ctx.on('approval/request', lambda carrier, req, next_: 'rejected')

    controller = AbortController()
    outcome = await service.request(
        ApprovalRequest(agent=agent, tool_name='bash', signal=controller.signal)
    )
    assert outcome == 'rejected'


async def test_dispatch_is_scoped_to_the_requesting_agent() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent_a = make_agent(ctx, store)
    agent_b = make_agent(ctx, store)
    open_turn(agent_a.session)
    hits: list[str] = []

    agent_b.ctx.on('approval/request', lambda carrier, req, next_: hits.append('b'))
    ctx.on('approval/request', lambda carrier, req, next_: hits.append('untagged'))

    outcome = await service.request(ApprovalRequest(agent=agent_a, tool_name='bash'))
    assert outcome == 'unavailable'
    # the untagged listener is admitted globally; agent b's scoped one is not
    assert hits == ['untagged']


async def test_request_id_is_fresh_per_request() -> None:
    ctx = Context()
    store = SessionStore(ctx)
    service = ApprovalService(ctx)
    agent = make_agent(ctx, store)
    open_turn(agent.session)

    await service.request(ApprovalRequest(agent=agent, tool_name='bash'))
    await service.request(ApprovalRequest(agent=agent, tool_name='bash'))

    ids = [
        event['data']['id'] for event in events_of(agent.session, 'approval/asked')
    ]
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert all(isinstance(id_, str) for id_ in ids)
    assert ApprovalRequestId(ids[0]) == ids[0]
