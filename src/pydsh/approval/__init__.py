"""Approval capability seam: policy-gated, fail-closed permission questions.

Ported from deepseek-harness packages/interaction/user-approval (MIT).
``ApprovalService`` registers itself as ``ctx.approval``. Each
:meth:`ApprovalService.request` appends an ``approval/asked`` +
``approval/decided`` audit pair inside the requesting session's open turn,
applies the session's approval policy BEFORE any answerer sees the ask, and
dispatches the ``approval/request`` waterfall through a scope carrier keyed
by the requesting agent (``pydsh.scope.scope_target``), so agent-scoped
answerers receive only their own agent's questions.

Session event vocabulary (the TS ``SessionEventMap`` augmentation; durable
field names stay camelCase exactly as in TS):

- ``approval/asked`` — log-only audit; ``id`` pairs it with the
  ``approval/decided`` that always follows. See :class:`ApprovalAskedData`.
- ``approval/decided`` — exactly one per ask: a decision, a cancellation,
  or the fail-closed ``'unavailable'``. See :class:`ApprovalDecidedData`.
- ``approval/policy`` — the LAST such event is the session's policy
  override (:func:`effective_approval_policy`). See
  :class:`ApprovalPolicyData`.

Python mappings and deferrals:

- TS ``AbortSignal`` → the :class:`ApprovalSignal` protocol (DOM-shaped:
  ``aborted`` plus ``add_listener``/``remove_listener``, with immediate
  fire when registered on an already-aborted signal — satisfied
  structurally by ``pydsh.subprocess``'s ``AbortSignal``). The loop layer
  owns the real cancellation primitive; deduplicate at integration.
- ``pydsh.agent``'s ``Agent.session`` is the real
  :class:`pydsh.session.Session` (the stand-in was deduplicated when the
  agent-loop layer landed).
- NOT ported: the ``ctx.inject(['systemPrompt'], ...)`` runtime-context
  sentences and ``setPolicy``'s model-facing switch notice — they need the
  SystemPrompt section wiring and dsh-llm's ``createUserMessage``, neither
  of which exists in pydsh yet. Session initialization switches policy
  through :func:`set_approval_policy`, which needs neither.
- NOT ported: the ``invariant.ts`` companion — its precommit staging hooks
  cordis' ``internal/dispatch`` over bus-dispatched ``session/event``, but
  pydsh's session observers run through a contained, non-bus channel.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict, cast
from uuid import uuid4

from pycordis import Context, Service
from pycordis.utils import maybe_await
from pydsh.agent import Agent
from pydsh.scope import scope_target
from pydsh.session import CallId, Session, SessionEvent

from .types import ApprovalOutcome, ApprovalRequestId

__all__ = [
    'APPROVAL_POLICIES',
    'OUTCOMES',
    'ApprovalAskedData',
    'ApprovalDecidedData',
    'ApprovalOutcome',
    'ApprovalPolicy',
    'ApprovalPolicyData',
    'ApprovalRequest',
    'ApprovalRequestId',
    'ApprovalService',
    'ApprovalSignal',
    'Config',
    'effective_approval_policy',
    'set_approval_policy',
]

#: Every :data:`ApprovalOutcome`, for runtime normalization of answerer returns.
OUTCOMES: tuple[ApprovalOutcome, ...] = (
    'allowed-once',
    'rejected',
    'cancelled',
    'unavailable',
)

ApprovalPolicy = Literal['ask', 'never']
"""A session's approval policy — what happens to an ask BEFORE any
interactive answerer sees it:

- ``'ask'`` (the default) — delegate to the composed answerers; with none
  composed the chain falls through to the fail-closed ``'unavailable'``.
- ``'never'`` — never prompt anyone: every ask resolves ``'rejected'``
  deterministically. The strict headless stance (CI, unattended runs) and
  the policy whose outcome is knowable without asking.
"""

#: Every :data:`ApprovalPolicy`, for option advertisement and runtime
#: validation of untrusted policy strings.
APPROVAL_POLICIES: tuple[ApprovalPolicy, ...] = ('ask', 'never')


class ApprovalAskedData(TypedDict):
    """``approval/asked`` payload: an approval question was put to the
    answerer chain — log-only audit (no ``surfaceOp``)."""

    #: Pairs with the ``approval/decided`` that always follows.
    id: ApprovalRequestId
    #: The tool the question is about.
    toolName: str
    #: The exact tool call, when the asker had one.
    callId: NotRequired[CallId]
    #: The asker's human-readable explanation of why it is asking.
    reason: NotRequired[str]


class ApprovalDecidedData(TypedDict):
    """``approval/decided`` payload: the outcome of a prior ``approval/asked``
    (same ``id``) — log-only audit, appended when the outcome is known."""

    id: ApprovalRequestId
    outcome: ApprovalOutcome


class ApprovalPolicyData(TypedDict):
    """``approval/policy`` payload: the session's approval policy was
    switched — log-only, durable, replayable, never in the model transcript.
    The LAST such event is the session's override."""

    policy: ApprovalPolicy
    #: Marks an override seeded into a child at delegation; an absent source
    #: is a runtime switch.
    source: NotRequired[Literal['delegation']]


class ApprovalSignal(Protocol):
    """DOM-shaped abort-signal stand-in for :class:`ApprovalRequest`.

    Aborting withdraws the question: the request settles ``'cancelled'``
    immediately and a late answer from a still-pending answerer is
    discarded. ``add_listener`` on an already-aborted signal fires
    immediately, closing the check-then-register race.
    """

    #: True once the signal has fired.
    aborted: bool

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Register an abort listener; fires immediately if already aborted."""
        ...

    def remove_listener(self, listener: Callable[[], None]) -> None:
        """Remove a previously registered listener; unknown listeners no-op."""
        ...


@dataclass(frozen=True)
class ApprovalRequest:
    """Readonly same-process permission question.

    ``call_id`` links to an already presented tool call, so arguments are
    not duplicated here. Field names are snake_case at the Python API; the
    durable audit event keeps the TS camelCase keys.
    """

    #: The agent on whose behalf the question is asked. Routes the question
    #: (a UI answerer only answers for agents it owns) and receives the
    #: audit events on its session log.
    agent: Agent
    #: The tool the question is about (presentation and audit).
    tool_name: str
    #: The exact tool call being decided, when the asker has one.
    call_id: CallId | None = None
    #: The asker's human-readable explanation of WHY it is asking.
    reason: str | None = None
    #: Withdraws the question when aborted.
    signal: ApprovalSignal | None = None


class Config(TypedDict, total=False):
    """Plugin config. All optional — an omitted ``policy`` defaults to
    ``'ask'`` (the TS schemastery default)."""

    #: The deployment's default :data:`ApprovalPolicy` for sessions without
    #: an ``approval/policy`` override.
    policy: ApprovalPolicy


def effective_approval_policy(
    events: Sequence[SessionEvent],
) -> ApprovalPolicy | None:
    """The session's approval-policy override: the last ``approval/policy``
    event in the log, or ``None`` when the session never switched (callers
    apply the plugin's configured default).

    The pure fold — resume needs no catch-up machinery because replaying the
    log IS the state.

    :param events: session events in log order (other event types are
        skipped).
    :return: the policy of the last switch event, or ``None`` without one.
    """
    for event in reversed(events):
        if event['type'] == 'approval/policy':
            return cast(ApprovalPolicyData, event['data'])['policy']
    return None


def set_approval_policy(session: Session, policy: ApprovalPolicy) -> None:
    """Append the sole durable representation of a session policy override.

    Invalid values raise before the log changes; consumers fold the new
    value on each read.

    :param session: the session the override belongs to.
    :param policy: the policy in effect until the next switch.
    :raises TypeError: when ``policy`` is not a known approval policy.
    """
    if policy not in APPROVAL_POLICIES:
        raise TypeError('approval policy must be one of "ask" or "never"')
    session.append('approval/policy', ApprovalPolicyData(policy=policy))


def _has_open_turn(events: Sequence[SessionEvent]) -> bool:
    """Whether the log currently sits inside an open turn (a ``turn/start``
    not yet closed by a ``turn/end``) — the :meth:`ApprovalService.request`
    precondition.

    The audit pair must be turn-enclosed: the turn is the durable log's
    commit/replay boundary, so a bare event appended between turns is
    indistinguishable from a crash tail and silently dropped on reload.
    """
    for event in reversed(events):
        type_ = event['type']
        if type_ == 'turn/start':
            return True
        if type_ == 'turn/end':
            return False
    return False


def _session_of(agent: Agent) -> Session:
    return agent.session


class ApprovalService(Service):
    """Approval service that applies session policy before answerers and
    logs every ask/outcome pair to the requesting session.

    Registered as ``ctx.approval``. Missing answerers fail closed; grants
    apply only to the requested action.
    """

    def __init__(self, ctx: Context, config: Config | None = None) -> None:
        super().__init__(ctx, 'approval')
        configured = (config or {}).get('policy')
        if configured is not None and configured not in APPROVAL_POLICIES:
            raise TypeError('approval policy must be one of "ask" or "never"')
        self.config: Config = config if config is not None else {}

    async def request(self, req: ApprovalRequest) -> ApprovalOutcome:
        """Ask the composed answerers to decide one readonly same-process
        request.

        The service borrows the request, agent, session, and live signal
        directly. The request requires an open turn because the audit pair
        must be enclosed by the durable log's commit/replay boundary; an
        idle ask raises before appending anything. The answerer phase always
        produces an outcome: an aborted signal yields ``'cancelled'``, a
        missing or throwing answerer yields ``'unavailable'`` (fail closed),
        and a rogue non-vocabulary return value is normalized to
        ``'unavailable'``.

        :param req: the pending decision (agent, tool identity, reason,
            signal).
        :return: the closed outcome; ``'allowed-once'`` is the only grant.
        :raises RuntimeError: when no turn is open.
        """
        session = _session_of(req.agent)
        if not _has_open_turn(session.events):
            raise RuntimeError(
                'approval.request() outside an open turn: the approval/asked + '
                'approval/decided audit pair must be turn-enclosed (a bare event '
                'between turns is crash-tail garbage on reload). Ask from inside '
                'the turn that needs the decision.'
            )
        request_id = ApprovalRequestId(str(uuid4()))
        asked = ApprovalAskedData(id=request_id, toolName=req.tool_name)
        if req.call_id is not None:
            asked['callId'] = req.call_id
        if req.reason is not None:
            asked['reason'] = req.reason
        session.append('approval/asked', asked)
        outcome = await self._decide(req, session)
        session.append(
            'approval/decided',
            ApprovalDecidedData(id=request_id, outcome=outcome),
        )
        return outcome

    def override_of(self, session: Session) -> ApprovalPolicy | None:
        """Read the session override without applying the configured default.

        :param session: session whose log supplies the override.
        :return: the last logged policy, or ``None`` without one.
        """
        return effective_approval_policy(session.events)

    def _effective_policy(self, session: Session) -> ApprovalPolicy:
        """The session's effective policy: its own ``approval/policy`` fold,
        else the configured default (``'ask'`` when unconfigured)."""
        override = self.override_of(session)
        if override is not None:
            return override
        configured = self.config.get('policy')
        return configured if configured is not None else 'ask'

    async def _decide(self, req: ApprovalRequest, session: Session) -> ApprovalOutcome:
        """Dispatch the waterfall, contained and raced against the request
        signal.

        :param req: the borrowed public request.
        :param session: the request agent's session used for policy lookup.
        :return: the normalized closed outcome.
        """
        signal = req.signal
        if signal is not None and signal.aborted:
            return 'cancelled'
        # The 'never' policy is decided HERE, before any dispatch: only the
        # service's own request path can keep the documented promise that
        # 'never' rejects deterministically regardless of listener
        # registration order (a prepended listener would sit ahead of any
        # listener-shaped gate).
        if self._effective_policy(session) == 'never':
            return 'rejected'

        async def ask() -> ApprovalOutcome:
            # The waterfall is entered inside this coroutine, so a listener
            # that throws SYNCHRONOUSLY lands in the same containment as an
            # async rejection (TS: Promise.resolve().then(...)).
            try:
                outcome: ApprovalOutcome = await maybe_await(
                    self.ctx.waterfall(
                        scope_target(self, req.agent),
                        'approval/request',
                        req,
                        # cordis calls the innermost next with the event
                        # args; the final handler ignores them.
                        lambda *_args: 'unavailable',
                    )
                )
            except Exception:
                # A throwing answerer must fail the QUESTION closed, not the
                # caller's tool call open — the seam contains its callbacks.
                return 'unavailable'
            # Normalize a rogue (non-vocabulary) answerer return to the
            # fail-closed outcome instead of leaking it into callers'
            # closed-union switches.
            return outcome if outcome in OUTCOMES else 'unavailable'

        if signal is None:
            return await ask()

        result: asyncio.Future[ApprovalOutcome] = (
            asyncio.get_running_loop().create_future()
        )

        def settle(outcome: ApprovalOutcome) -> None:
            # After an abort won the race this is a settled-future no-op:
            # the late answer is discarded by construction.
            if not result.done():
                result.set_result(outcome)

        def on_abort() -> None:
            signal.remove_listener(on_abort)
            settle('cancelled')

        def on_answer(task: asyncio.Task[ApprovalOutcome]) -> None:
            signal.remove_listener(on_abort)
            if task.cancelled():
                settle('unavailable')
                return
            error = task.exception()
            settle('unavailable' if error is not None else task.result())

        signal.add_listener(on_abort)
        answer = asyncio.ensure_future(ask())
        answer.add_done_callback(on_answer)
        return await result
