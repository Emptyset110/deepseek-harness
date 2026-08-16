"""Session data model types.

Ported from deepseek-harness packages/core/session (MIT), ``src/types.ts``.

Python mappings:

- Opaque cross-boundary ids use :class:`typing.NewType`; calling
  ``SessionId(raw)`` brands a plain string, mirroring the TS cast function.
- Durable JSON field names stay camelCase exactly as in the event log and
  storage headers (``createdAt``, ``surfaceOp``, ``sourceEventSeqs``, ...);
  Python-level API parameters stay snake_case.
- LLM-owned shapes (messages, chunks, usage, tool schemas, call config) are
  minimal local stand-ins covering only the fields the session model reads;
  the future ``pydsh.llm`` port owns the full definitions (integration-time
  dedup is expected, per the porting plan).
- The TS discriminated ``SessionEvent`` union and the merge-extensible
  ``SessionEventMap`` have no Python equivalent: ``SessionEvent`` is one
  TypedDict with ``data: Any``, and the type/data pairing is enforced by the
  ``Session.append`` overloads plus runtime validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NewType, NotRequired, TypeAlias, TypedDict

from .json import JsonValue

SessionId = NewType('SessionId', str)
"""Identifies one session in the store (and its persistence artifacts)."""

SESSION_FORMAT_VERSION: Final = 0
"""The on-disk session format version, stamped into every new SessionHeader.

Pinned at ``0`` while the harness is unreleased: no compatibility is
implied, incompatible logs are rejected, and no migration is provided. Only
structural changes (header shape, event envelope, core event semantics, the
surface mechanism) bump it; ordinary event vocabulary growth is covered by
the per-event ``ignorable`` marker instead.
"""

CallId: TypeAlias = str
"""Minimal local stand-in for dsh-llm's branded tool-call id."""


# --- Minimal LLM-owned stand-ins (owned by the future pydsh.llm port) ---


class MessageSource(TypedDict, total=False):
    """Where a message came from; ``kind`` discriminates the variant."""

    kind: str
    provider: str
    model: str
    callId: CallId


ContentBlock: TypeAlias = dict[str, JsonValue]
"""One message content block (text, tool-call, tool-result, ...)."""


class Message(TypedDict, total=False):
    """One LLM message; the session model reads id/role/source/content."""

    id: str
    role: str
    source: MessageSource
    content: list[ContentBlock]


UserMessage: TypeAlias = Message
AssistantMessage: TypeAlias = Message
ToolResultMessage: TypeAlias = Message

StreamChunk: TypeAlias = dict[str, JsonValue]
TokenUsage: TypeAlias = dict[str, JsonValue]
ToolSchema: TypeAlias = dict[str, JsonValue]


class LlmFailure(TypedDict, total=False):
    """Structured LLM failure facts carried by an ``error`` turn end."""

    message: str
    code: str


class LlmCallConfig(TypedDict, total=False):
    """Call configuration: provider route, model, and sampling scalars."""

    provider: str
    model: str
    reasoningEffort: str
    maxTokens: int


class LlmCallConfigAdapterDefaults(TypedDict, total=False):
    """Fields materialized from the adapter rather than proposed by a caller."""

    reasoningEffort: bool
    maxTokens: bool


# --- Session header and creation options ---


class SessionHeader(TypedDict):
    """Immutable validated storage metadata, kept outside the event log."""

    #: On-disk format version, stamped from SESSION_FORMAT_VERSION.
    version: int
    #: The session's id (mirrors the Session's id).
    id: SessionId
    #: Non-negative safe-integer Unix epoch milliseconds of creation.
    createdAt: int
    #: Absolute working directory the session was created in (if any).
    cwd: NotRequired[str]
    #: The session this one was forked from (seed lineage), if any.
    parentSession: NotRequired[SessionId]
    #: How many leading events were inherited through a seed.
    seedLength: NotRequired[int]
    #: Coarse product classification for a subagent child session.
    origin: NotRequired[Literal['subagent']]
    #: Delegation depth; absent (zero) for a top-level session.
    delegationDepth: NotRequired[int]
    #: Id of the agent preset this session's agent was composed from.
    agentPreset: NotRequired[str]


class CreateSessionMeta(TypedDict, total=False):
    """Caller-supplied storage fields folded into a SessionHeader."""

    cwd: str
    parentSession: SessionId
    createdAt: int
    seedLength: int
    origin: Literal['subagent']
    delegationDepth: int
    agentPreset: str


# --- Session event vocabulary (the core SessionEventMap) ---


class TurnStartData(TypedDict):
    """Opens turn ``turn`` before the loop claims queued input."""

    turn: int


class AgentCancelCause(TypedDict, total=False):
    """Why an active agent driver was cancelled (``hook`` adds ``reason``)."""

    kind: Literal['user', 'parent', 'hook', 'disposed', 'legacy']
    reason: str


class TurnEndReason(TypedDict, total=False):
    """Why a turn ended; merge-extensible in TS, an open record here.

    Core ``kind`` values: ``completed``, ``aborted`` (with ``reason``),
    ``blocked``, ``error`` (with ``error``), ``max-tokens``,
    ``interrupted``.
    """

    kind: str
    reason: AgentCancelCause
    error: LlmFailure


class TurnEndData(TypedDict):
    """Closes turn ``turn`` with the reason that ended it."""

    turn: int
    reason: TurnEndReason


class StepStartData(TypedDict):
    """Opens one step: one model call plus the tool executions it requested."""

    turn: int
    step: int


class StepEndData(TypedDict):
    """Closes step ``step`` of turn ``turn``."""

    turn: int
    step: int


class AssistantChunkData(TypedDict):
    """Raw stream chunk — token-level replay fidelity."""

    turn: int
    step: int
    chunk: StreamChunk


class AssistantMessageData(TypedDict):
    """Assembled assistant message for one step (derived history uses this)."""

    turn: int
    step: int
    message: AssistantMessage
    #: Token accounting, present only when the adapter reported it.
    usage: NotRequired[TokenUsage]


class ToolCallData(TypedDict):
    """The model requested one tool invocation with raw argument JSON."""

    turn: int
    step: int
    callId: CallId
    name: str
    arguments: str


class ToolResultError(TypedDict):
    """Internal failure identity of a completed tool call."""

    name: str
    code: str


class ToolResultData(TypedDict):
    """A completed tool call's model-facing result."""

    turn: int
    step: int
    message: ToolResultMessage
    error: NotRequired[ToolResultError]
    #: Tool-private presentation payload; opaque to the core, JSON-required.
    meta: NotRequired[JsonValue]


class TodoItem(TypedDict):
    """One entry in an agent's todo list (whole-list snapshot unit)."""

    content: str
    status: Literal['pending', 'in_progress', 'completed']


class TodoWriteData(TypedDict):
    """Whole-list snapshot; latest write wins on replay. Log-only UI state."""

    todos: list[TodoItem]


class EpochHeader(TypedDict):
    """Logged request state: call config, system prompt, and tools."""

    config: LlmCallConfig
    adapterDefaults: NotRequired[LlmCallConfigAdapterDefaults]
    #: Rendered system prompt text; absent for a system-less request.
    system: NotRequired[str]
    #: Assembled tool schemas; absent for a tool-less request.
    tools: NotRequired[list[ToolSchema]]


RequestHeaderReason: TypeAlias = Literal['initial', 'resume', 'change']
"""Why a ``request/header`` snapshot was appended."""


class RequestHeaderData(TypedDict):
    """Full header for the next request, appended inside its step."""

    header: EpochHeader
    reason: RequestHeaderReason


class RequestContext(TypedDict):
    """Registration-bound metadata for one resolved model route."""

    provider: str
    model: str
    #: Maximum combined request/response context in tokens, when advertised.
    contextWindow: NotRequired[int]


SessionEventType: TypeAlias = str
"""An appendable event-type key; core keys are the append() overload set."""

SurfaceEventType: TypeAlias = Literal[
    'user/message', 'assistant/message', 'tool/result'
]
"""Event types whose events produce LLM messages and join the surface."""


class SurfaceReplace(TypedDict):
    """Replace surface nodes ``start``..``end`` (inclusive seqs) with this node."""

    op: Literal['replace']
    start: int
    end: int


SurfaceOp: TypeAlias = Literal['append'] | SurfaceReplace
"""How a session event entered the ordered surface."""


class SurfaceIntent(TypedDict, total=False):
    """Surface placement and cited source-event seqs for ``Session.append``.

    Required on message-producing events and forbidden on log-only events.
    """

    surfaceOp: SurfaceOp
    #: Complete set of known source-event seqs; an empty list is allowed only
    #: on assistant/message (a known empty provider stream).
    sourceEventSeqs: list[int]


class SessionEvent(TypedDict):
    """One immutable entry in the session log.

    ``surfaceOp``/``sourceEventSeqs`` exist only on surface-eligible events
    (enforced at runtime by the surface fold, and at type-check time by the
    ``Session.append`` overloads). ``ignorable`` marks an event a reader may
    skip when it does not recognize ``type``; absent means required-on-read.
    """

    type: str
    #: Monotonic sequence number within the session.
    seq: int
    #: Unix epoch milliseconds.
    time: int
    data: Any
    ignorable: NotRequired[Literal[True]]
    surfaceOp: NotRequired[SurfaceOp]
    sourceEventSeqs: NotRequired[list[int]]


# --- Store creation options ---


@dataclass(frozen=True)
class CreateSessionOptions:
    """Options for creating a Session via the store.

    ``seed`` replays/forks an existing event log; ``meta`` carries the
    caller-supplied storage fields the store folds into a SessionHeader.
    """

    #: Initial replay or fork history supplied at construction.
    seed: Sequence[SessionEvent] | None = None
    #: Storage metadata read once before publication.
    meta: CreateSessionMeta | None = None


@dataclass(frozen=True)
class RestoredSessionOptions:
    """Fresh storage values transferred to ``SessionStore.prepare``.

    Persistence ownership-transfer path: callers retain no mutable aliases;
    the values are validated and frozen in place (frozen by convention — see
    the store module docstring).
    """

    #: Fresh detached storage events to validate and adopt in place.
    seed: Sequence[SessionEvent]
    #: Fresh detached storage metadata to validate and adopt in place.
    meta: SessionHeader
    #: Selects the persistence ownership-transfer path.
    seed_source: Literal['persistence'] = 'persistence'


PrepareSessionOptions: TypeAlias = CreateSessionOptions | RestoredSessionOptions
"""Inputs accepted while constructing an unpublished Session."""
