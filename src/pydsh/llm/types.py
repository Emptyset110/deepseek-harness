"""Canonical provider-neutral message and streaming vocabulary.

Ported from deepseek-harness ``packages/llm/llm/src/types.ts``,
``message.ts``, ``content.ts``, ``brand.ts``, and ``call-config.ts`` (MIT).

Python mapping: TS interfaces become ``TypedDict``s with the exact same
(camelCase) keys, so values stay JSON-serializable for the session log.
``Branded<B>`` ids become ``NewType`` aliases. ``GenerateOptions`` is a
``dict`` subclass so the process-local agent-loop identity marker can hold
weak references to request objects (TS uses a ``WeakSet``).

Read-only convention (in place of TS ``Object.freeze`` / ``deepFreeze``):
harness data dicts are treated as immutable after creation. Nothing in this
package mutates a ``GenerateOptions`` or a ``Message``; when the runtime
materializes adapter defaults it builds a new dict instead. The
``llm/stream`` waterfall payload obeys the "model-visible ⟺ logged"
invariant: listeners may read the request or short-circuit with their own
chunk stream, but must never rewrite message content, because the request
must remain reconstructable from the session log.
"""

from __future__ import annotations

import weakref
from collections.abc import AsyncIterator
from typing import Any, Literal, NewType, TypedDict

# --- branded ids (brand.ts) ---

#: Stable identity carried by one message across inbox, log, and request boundaries.
MessageId = NewType("MessageId", str)
#: Correlates a model-issued tool call with its result.
CallId = NewType("CallId", str)
#: Provider-issued request identifier retained for diagnostics.
ProviderRequestId = NewType("ProviderRequestId", str)
#: Adapter-owned identifier for one model's selectable reasoning effort.
ReasoningEffortId = NewType("ReasoningEffortId", str)


# --- failures ---


class LlmFailure(TypedDict, total=False):
    """Serializable provider or transport failure facts; policy decides retryability."""

    #: Human-readable provider or transport failure.
    message: str  # required
    #: Stable provider-neutral machine-routing code.
    code: str  # required
    #: HTTP status returned by the provider, when available.
    status: int
    #: Provider-requested delay in milliseconds, when valid and available.
    providerRetryAfterMs: float
    #: Opaque provider-issued request identifier for diagnostics.
    requestId: ProviderRequestId


# --- content blocks ---


class TextBlock(TypedDict):
    """Plain text visible to the end user."""

    type: Literal["text"]
    text: str


class ReasoningBlock(TypedDict):
    """Reasoning / thinking content, distinct from visible text."""

    type: Literal["reasoning"]
    text: str


class ImageBlock(TypedDict):
    """A durable raster image reference, valid in user or assistant content."""

    type: Literal["image"]
    #: Immutable bytes and display metadata owned by the attachment service.
    attachment: Any


class ToolCallBlock(TypedDict):
    """A tool invocation requested by the model."""

    type: Literal["tool-call"]
    #: Provider-issued call id; correlates with the matching tool result.
    id: CallId
    name: str
    #: Raw JSON string as produced by the model.
    arguments: str


class ToolResultBlock(TypedDict, total=False):
    """The result of a tool invocation, sent back to the model."""

    type: Literal["tool-result"]  # required
    toolCallId: CallId  # required
    content: list[ContentBlock]  # required
    isError: bool


#: Any known content block. Plugins may add valid unknown cases: switch on
#: ``type`` and fall through unknowns.
ContentBlock = TextBlock | ReasoningBlock | ImageBlock | ToolCallBlock | ToolResultBlock


def content_has_image(content: list[ContentBlock]) -> bool:
    """True when typed model content contains an image block, walking nested
    tool-result content (the one recursive image walk shared by every image
    policy)."""
    return any(
        block["type"] == "image"
        or (
            block["type"] == "tool-result"
            and content_has_image(block.get("content", []))
        )
        for block in content
    )


# --- finish reasons ---


class StopFinish(TypedDict):
    kind: Literal["stop"]


class ToolCallsFinish(TypedDict):
    kind: Literal["tool-calls"]


class MaxTokensFinish(TypedDict):
    kind: Literal["max-tokens"]


class AbortedFinish(TypedDict):
    kind: Literal["aborted"]
    failure: LlmFailure


class ErrorFinish(TypedDict):
    kind: Literal["error"]
    failure: LlmFailure


#: Why a model response stopped. Merge-extensible in TS; switch on ``kind``
#: and fall through unknowns.
FinishReason = (
    StopFinish | ToolCallsFinish | MaxTokensFinish | AbortedFinish | ErrorFinish
)


# --- token accounting ---


class TokenUsage(TypedDict, total=False):
    """Token accounting for one model call (cache fields are optional).

    Counts are DISJOINT: ``inputTokens`` is uncached input only; cached input
    is reported separately as ``cacheReadTokens``/``cacheWriteTokens``.
    Adapters whose providers fold cache hits into a total prompt count
    (DeepSeek's ``prompt_tokens``) subtract them out.
    """

    inputTokens: int  # required
    outputTokens: int  # required
    cacheReadTokens: int
    cacheWriteTokens: int
    reasoningTokens: int


# --- stream chunks ---


class BlockStartChunk(TypedDict):
    type: Literal["block-start"]
    index: int
    blockType: str


class TextDeltaChunk(TypedDict):
    type: Literal["text-delta"]
    index: int
    text: str


class ReasoningDeltaChunk(TypedDict):
    type: Literal["reasoning-delta"]
    index: int
    text: str


class ToolCallDeltaChunk(TypedDict, total=False):
    type: Literal["tool-call-delta"]  # required
    index: int  # required
    id: CallId  # required
    argumentsDelta: str  # required
    name: str


class BlockEndChunk(TypedDict):
    type: Literal["block-end"]
    index: int
    block: ContentBlock


class UsageChunk(TypedDict):
    type: Literal["usage"]
    usage: TokenUsage


class FinishChunk(TypedDict, total=False):
    type: Literal["finish"]  # required
    reason: FinishReason  # required
    #: Adapter-private lossless-JSON state for replaying a successful response.
    replayState: Any


#: Raw streaming protocol emitted by adapters. Block indexes correlate
#: interleaved deltas; ``block-end`` carries the assembled block. Adapters
#: emit usage before the terminal finish and nothing afterward. An adapter
#: implementation may raise, but ``LlmRuntime.stream()`` normalizes that
#: failure to a terminal ``error`` or ``aborted`` finish chunk.
StreamChunk = (
    BlockStartChunk
    | TextDeltaChunk
    | ReasoningDeltaChunk
    | ToolCallDeltaChunk
    | BlockEndChunk
    | UsageChunk
    | FinishChunk
)


# --- tools and provider/model metadata ---


class ToolSchema(TypedDict):
    """JSON-schema description of a tool, as sent to the model."""

    name: str
    description: str
    #: JSON Schema object for the arguments.
    parameters: dict[str, Any]


class LlmProviderInfo(TypedDict):
    """Display metadata for one registered provider route."""

    #: Provider route key used by ``GenerateOptions['provider']``.
    id: str
    #: Human-readable provider name for selectors and diagnostics.
    name: str


#: Provider model modality vocabulary (merge-extensible in TS).
ModelModality = Literal["text", "image"]


class LlmModelInfo(TypedDict, total=False):
    """One adapter-discovered model; catalog membership is advisory, not validation."""

    provider: str  # required
    id: str  # required
    name: str  # required
    description: str
    #: Accepted request modalities; absent means unknown, an explicit list is a claim.
    inputModalities: list[ModelModality]


class LlmModelContext(TypedDict):
    """Provider-owned context capacity for one exact provider/model route."""

    #: Maximum combined request and response context in tokens.
    contextWindow: int


class LlmReasoningEffortInfo(TypedDict, total=False):
    """Display metadata for one adapter-owned reasoning effort."""

    id: ReasoningEffortId  # required
    name: str  # required
    description: str


class LlmModelReasoningInfo(TypedDict, total=False):
    """Selectable reasoning efforts for one exact provider/model route."""

    #: Supported efforts in adapter-preferred display order.
    efforts: list[LlmReasoningEffortInfo]  # required
    #: Adapter-configured default materialized into requests when callers omit one.
    defaultEffort: ReasoningEffortId


class LlmResolvedModelInfo(LlmModelInfo, total=False):
    """Exact-route model metadata resolved by its owning adapter."""

    context: LlmModelContext
    #: Adapter-configured per-request output cap materialized when callers omit one.
    defaultMaxTokens: int
    reasoning: LlmModelReasoningInfo


# --- messages (message.ts) ---


class ModelMessageSource(TypedDict, total=False):
    """Required source of an assistant message produced by a routed model."""

    kind: Literal["model"]  # required
    #: Provider route that produced the message.
    provider: str  # required
    #: Provider model id that produced the message.
    model: str  # required
    #: Lossless-JSON adapter state needed to replay the provider response.
    replayState: Any


class UserMessageSource(TypedDict):
    kind: Literal["user"]


class PluginMessageSource(TypedDict, total=False):
    kind: Literal["plugin"]  # required
    plugin: str  # required
    #: Producer-declared context form; absent means opaque content.
    form: str


class ToolMessageSource(TypedDict):
    kind: Literal["tool"]
    callId: CallId


#: Where a message came from. Merge-extensible in TS; switch on ``kind`` and
#: fall through unknowns.
MessageSource = (
    ModelMessageSource | UserMessageSource | PluginMessageSource | ToolMessageSource
)


class Message(TypedDict):
    """One immutable message shared by delivery, durable history, and model requests."""

    #: Stable identity preserved across every representation boundary.
    id: MessageId
    #: Provider-neutral conversation role.
    role: Literal["system", "user", "assistant"]
    #: Exact model-facing blocks.
    content: list[ContentBlock]
    #: Required source fields supplied by the producer.
    source: MessageSource


# --- the request envelope ---


class GenerateOptions(dict[str, Any]):
    """A single model request, fully assembled.

    A plain ``dict`` subclass so values stay JSON-shaped while remaining
    weak-referenceable for :func:`mark_agent_loop_request`. Identity-hashed
    (like the TS object identity the ``WeakSet`` marker relies on).
    Recognized keys (TS ``GenerateOptions`` fields, unchanged):

    - ``provider: str`` — registered provider route selecting the adapter.
    - ``model: str``
    - ``reasoningEffort: ReasoningEffortId`` — adapter-owned effort for this model.
    - ``messages: list[Message]`` — ordered conversation messages.
    - ``system: str`` — system prompt text (mapped to the provider's system slot).
    - ``tools: list[ToolSchema]``
    - ``temperature: float``
    - ``maxTokens: int``
    - ``stop: list[str]`` — generation halts on any of these strings.
    - ``signal: asyncio.Event`` — caller cancellation; a set event means aborted.
    - ``sessionId: str`` — session identity for request routing.
    - ``purpose: 'compaction' | 'session-title'`` — auxiliary-call classification.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        # Identity hash: dict.__hash__ is None, but the agent-loop marker
        # tracks request objects by identity like the TS WeakSet does.
        return id(self)


#: Process-local identities of request objects assembled by the agent loop.
_AGENT_LOOP_REQUESTS: weakref.WeakSet[GenerateOptions] = weakref.WeakSet()


def mark_agent_loop_request(request: GenerateOptions) -> GenerateOptions:
    """Mark one exact request object as assembled by the agent loop."""
    _AGENT_LOOP_REQUESTS.add(request)
    return request


def is_agent_loop_request(request: GenerateOptions) -> bool:
    """Test whether the exact request object was assembled by the agent loop."""
    return request in _AGENT_LOOP_REQUESTS


def is_token_delta(chunk: StreamChunk) -> bool:
    """Whether a stream chunk carries visible model output (the first-token
    boundary). Empty deltas (heartbeats, empty tool-call frames) do not count."""
    chunk_type = chunk["type"]
    if chunk_type in ("text-delta", "reasoning-delta"):
        return chunk["text"] != ""  # type: ignore[typeddict-item]
    if chunk_type == "tool-call-delta":
        return chunk["argumentsDelta"] != "" or "name" in chunk  # type: ignore[typeddict-item]
    return False


ChunkStream = AsyncIterator[StreamChunk]
