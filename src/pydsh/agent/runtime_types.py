"""Public agent runtime types and live-runtime event declarations.

Ported from deepseek-harness ``packages/core/agent/src/runtime-types.ts``
(MIT). Durable transcript facts and turn/step boundaries remain session
events; this module carries only the live-runtime surface: the ``Agent``
protocol, its lifecycle status, and the payload shapes of the agent-subject
events dispatched through ``pydsh.agent.dispatch``.

Python mappings:

- TypeScript ``interface`` → ``typing.Protocol`` (``Agent``) or frozen
  dataclasses (value types).
- Discriminated-union decisions (``PreStepDecision``, ``RequestErrorAction``)
  become frozen dataclasses with a ``kind`` discriminant.
- Declaration-merged ``Events`` cannot exist in Python; event names, dispatch
  modes, and payload contracts are declared here as constants, the
  ``AgentSubjectEvent`` literal, and one ``TypedDict`` per event.
- Integration-time deduplication (done with the agent-loop layer):
  ``SessionId``, ``Session``, ``Message``, and ``AgentCancelCause`` are
  re-exported from :mod:`pydsh.session`, ``LlmFailure`` /
  ``ResolvedRetryPolicy`` from :mod:`pydsh.llm`, and ``AbortSignal`` from
  :mod:`pydsh.subprocess.types` (the DOM-primitive port the loop's
  cancellation signal extends). ``UserMessage`` below remains a lightweight
  attribute-accessible value for registry/tool producers that do not build
  full session messages; the loop layer works in canonical ``Message``
  dicts. ``LlmCallConfig`` likewise remains the frozen registry-facing
  value type — the loop's ``agent/request`` waterfall carries the
  session-layer camelCase config dict (``EpochHeader['config']``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypedDict,
)

from pydsh.llm import LlmFailure as LlmFailure
from pydsh.llm import ResolvedRetryPolicy as ResolvedRetryPolicy
from pydsh.session import AgentCancelCause as AgentCancelCause
from pydsh.session import Message as Message
from pydsh.session import Session as Session
from pydsh.session import SessionId as SessionId
from pydsh.subprocess.types import AbortSignal as AbortSignal

if TYPE_CHECKING:
    from pycordis import Context

    from .inbox import Inbox


@dataclass(frozen=True)
class UserMessage:
    """Lightweight identified user-context value for registry/tool producers.

    The canonical model/durable message shape is :class:`pydsh.session.Message`
    (a plain JSON dict); this frozen stand-in predates that deduplication and
    stays for producers that only carry an id plus opaque content.
    """

    id: str
    content: Any
    source: Any = None


#: Session-layer call configuration (camelCase ``EpochHeader['config']``
#: shape) threaded through the ``agent/request`` waterfall: a waterfall
#: listener may only REPLACE the config wholesale, never mutate it in place.
LoopCallConfig = dict[str, Any]


@dataclass(frozen=True)
class LlmCallConfig:
    """Frozen call configuration threaded through the ``agent/request`` waterfall.

    Instances are immutable and defensively copied (``stop`` is stored as a
    tuple): a waterfall listener may only REPLACE the config wholesale, never
    mutate it in place. The waterfall cannot rewrite messages at all — the
    model-visible ⟺ logged invariant requires anything that reaches a model
    request to be reconstructable from the session log, so model-visible
    content must flow through logged channels, not this waterfall.
    """

    provider: str
    model: str
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: tuple[str, ...] = field(default=())

    def __init__(
        self,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Iterable[str] = (),
    ) -> None:
        object.__setattr__(self, 'provider', provider)
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'reasoning_effort', reasoning_effort)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'stop', tuple(stop))


# ---------------------------------------------------------------------------
# Agent options, status, and decisions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentOptions:
    """Agent creation options (TS merge-extensible ``AgentOptions``).

    Persona belongs to system-prompt sections.
    """

    provider: str | None = None
    model: str | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class CancelOptions:
    """Options for ``Agent.cancel``.

    ``keep_inbox`` preserves queued and steering inbox items instead of
    discarding them: the active turn is still aborted, but un-started and
    pending work survives for a later turn and no canceled inbox splice is
    logged.
    """

    keep_inbox: bool = False


#: An agent's lifecycle state, emitted on every transition as
#: ``agent/status``: ``idle`` means no driver is active; ``running`` begins
#: when waking input starts cancellable pre-step processing and lasts while
#: the driver drains, closes, or checkpoints turns. Disposal removes the agent
#: from its registry; it is not a third observable status.
AgentStatus = Literal['idle', 'running']

#: One of the two ordered pending-message lists owned by an agent.
InboxTarget = Literal['next-turn', 'next-step']


@dataclass(frozen=True)
class PreStepReject:
    """The loop does not enter the proposed step."""

    kind: Literal['reject'] = 'reject'


@dataclass(frozen=True)
class PreStepEnter:
    """The loop enters the proposed step with these messages."""

    messages: list[Message]
    kind: Literal['enter'] = 'enter'


#: Whether and with which messages the loop enters a proposed step.
PreStepDecision = PreStepReject | PreStepEnter


@dataclass(frozen=True)
class RequestErrorRetry:
    """Action of an ``agent/request-error`` listener owning model-request recovery."""

    kind: Literal['retry'] = 'retry'


#: Action returned by a listener that owns model-request recovery; the default
#: ``None`` leaves the failure terminal.
RequestErrorAction = RequestErrorRetry | None

#: Why a session lifecycle began; seeded creates are ``startup``, persisted
#: loads are ``resume``.
SessionStartSource = Literal['startup', 'resume', 'clear', 'compact']


class Agent(Protocol):
    """Public live-agent handle (TS ``Agent`` interface)."""

    @property
    def id(self) -> SessionId:
        """The single identity shared with ``session``."""
        ...

    @property
    def options(self) -> AgentOptions:
        """The provider route and model this agent's requests use."""
        ...

    @property
    def session(self) -> Session:
        """The live session this agent drives.

        Its log is the durable source of truth.
        """
        ...

    @property
    def inbox(self) -> Inbox:
        """The agent-owned projection of durable pending work."""
        ...

    @property
    def status(self) -> AgentStatus:
        """The current lifecycle state.

        Mirrored on every ``agent/status`` transition.
        """
        ...

    @property
    def ctx(self) -> Context:
        """Agent-scoped context.

        Its contributions are agent-local and unwind on disposal.
        """
        ...

    def cancel(
        self,
        cause: AgentCancelCause,
        options: CancelOptions | None = None,
    ) -> None:
        """Clear queued and steering work — unless ``keep_inbox``.

        Also aborts the active turn or between-turn task. The first cause
        wins for that activity. With no active activity, cancellation is a
        no-op and does not arm later work.
        """
        ...

    def when_idle(self) -> Awaitable[None]:
        """Resolve after the current whole-agent activity reaches quiescence.

        Follows replacement work started before the observed driver retires,
        but does not identify the settlement of any particular message.
        """
        ...

    def run_maintenance(
        self,
        task: Callable[[AbortSignal], Awaitable[Any]],
    ) -> Awaitable[Any]:
        """Run one non-turn maintenance task from the true idle phase.

        The task starts synchronously after claiming that phase; later waking
        input remains in the inbox until the task settles, while public
        status stays ``idle``. Raises synchronously when turn-driving or
        another maintenance task already owns the agent.
        """
        ...

    def send(self, message: Message, target: InboxTarget, wakeup: bool) -> None:
        """Route identified input to an inbox boundary.

        ``wakeup`` controls whether delivery may wake the driver.
        """
        ...

    def followup(self, message: Message) -> None:
        """Queue an ordinary follow-up turn and wake the driver."""
        ...

    def steer(self, message: Message) -> None:
        """Submit steering for the nearest step (an idle driver starts a turn)."""
        ...

    def inject(self, message: Message) -> None:
        """Queue model-facing context for the next pre-step.

        Does not wake the driver.
        """
        ...


# ---------------------------------------------------------------------------
# Agent-subject event declarations.
#
# Every event below is an AGENT-SUBJECT event: the payload carries the
# ``agent`` subject and dispatch goes through a scope carrier built by
# ``pydsh.agent.dispatch.agent_carrier``, so agent-scoped listeners (registered
# on a context tagged with that agent's scope key or one of its scope
# ancestors) receive their own agent's events — plus every descendant
# scope's — untagged listeners receive every agent's events, and
# ``{'global': True}`` listeners skip filtering entirely.
# ---------------------------------------------------------------------------

AGENT_CREATED = 'agent/created'
AGENT_DISPOSED = 'agent/disposed'
AGENT_STATUS = 'agent/status'
AGENT_INBOX_INSERTED = 'agent/inbox/inserted'
AGENT_INBOX_CLAIMED = 'agent/inbox/claimed'
AGENT_INBOX_DISCARDED = 'agent/inbox/discarded'
AGENT_SESSION_START = 'agent/session-start'
AGENT_PRE_STEP = 'agent/pre-step'
AGENT_REQUEST = 'agent/request'
AGENT_REQUEST_ERROR = 'agent/request-error'
AGENT_TURN_STOPPING = 'agent/turn-stopping'
AGENT_ERROR = 'agent/error'

#: The event names whose subject is an agent (TS ``AgentSubjectEvent``).
AgentSubjectEvent = Literal[
    'agent/created',
    'agent/disposed',
    'agent/status',
    'agent/inbox/inserted',
    'agent/inbox/claimed',
    'agent/inbox/discarded',
    'agent/session-start',
    'agent/pre-step',
    'agent/request',
    'agent/request-error',
    'agent/turn-stopping',
    'agent/error',
]


class AgentCreatedPayload(TypedDict):
    """Mode: emit. A fully configured agent and live session were published.

    A synchronous listener failure vetoes publication and rolls back; a
    returned-awaitable rejection is logged. Detach requested during dispatch
    waits until every creation listener has observed the stable entry.
    """

    agent: Agent


class AgentDisposedPayload(TypedDict):
    """Mode: emit. An agent left the registry.

    Listener failures are contained and logged.
    """

    agent: Agent


class AgentStatusPayload(TypedDict):
    """Mode: emit. Agent status changed (``idle`` ⇄ ``running``).

    ``status`` is the destination just entered.
    """

    agent: Agent
    status: AgentStatus


class AgentInboxInsertedPayload(TypedDict):
    """Mode: emit. One message entered the live inbox."""

    agent: Agent
    message: Message


class AgentInboxClaimedPayload(TypedDict):
    """Mode: emit. One message left the inbox inside its open turn.

    If the proposed step is rejected, the claimed message ends here: it is
    neither discarded nor re-emitted, and the turn closes without a step.
    """

    agent: Agent
    message: Message
    turn: int


class AgentInboxDiscardedPayload(TypedDict):
    """Mode: emit. One message was discarded from the live inbox."""

    agent: Agent
    message: Message


class AgentSessionStartPayload(TypedDict):
    """Mode: emit. The session lifecycle began, once before the first turn.

    This is a notification, not a veto; disposal requested by a lifecycle
    owner is rechecked before the driver starts.
    """

    agent: Agent
    source: SessionStartSource


class AgentPreStepPayload(TypedDict):
    """Mode: waterfall. Reject a proposed step or replace the messages that enter it.

    Calling ``next()`` preserves the current messages; returning a decision
    without calling ``next()`` short-circuits the chain. The waterfall
    returns a ``PreStepDecision``.
    """

    agent: Agent
    messages: list[Message]
    turn: int
    step: int
    signal: AbortSignal


class AgentRequestPayload(TypedDict):
    """Mode: waterfall. Replace the frozen call configuration for one model request.

    ``await next()`` yields the ``LlmCallConfig`` the machine would use; a
    listener returns a replacement to switch. Model-visible content must use
    logged channels; this waterfall cannot mutate messages (the
    model-visible ⟺ logged invariant — see ``LlmCallConfig``).
    """

    agent: Agent
    turn: int
    step: int
    signal: AbortSignal


class AgentRequestErrorPayload(TypedDict):
    """Mode: waterfall. Handle one failed model-request attempt.

    Runs before the loop retries or closes the step. A listener returns
    ``RequestErrorRetry()`` without calling ``next()`` when it owns recovery,
    or calls ``next()`` to delegate. The default ``None`` leaves the failure
    terminal.
    """

    agent: Agent
    turn: int
    step: int
    provider: str
    failure: LlmFailure
    retry_policy: ResolvedRetryPolicy | None
    signal: AbortSignal


class AgentTurnStoppingPayload(TypedDict):
    """Mode: serial. The turn is about to close: the model owes no response.

    Awaited before the boundary commits — a listener that objects steers
    (``agent.steer(...)``) and the machine re-reads its inbox: fresh steering
    runs another step, none closes the turn. Data decides, so listener order
    cannot change the outcome.
    """

    agent: Agent
    turn: int
    signal: AbortSignal


class AgentErrorPayload(TypedDict):
    """Mode: emit. A step or turn errored.

    Reported here even when the error has no in-turn position for a durable
    record.
    """

    agent: Agent
    turn: int
    step: int
    error: Any
