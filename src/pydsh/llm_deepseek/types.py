"""DeepSeek chat-completions wire format (OpenAI-compatible). Types only.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/types.ts`` (MIT).

Python mapping: the TS discriminated ``WireMessage`` union flattens into one
``total=False`` dict (JSON bodies are plain dicts anyway); requiredness is
enforced by the serializer, not the type.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class WireToolFunction(TypedDict):
    name: str
    description: str
    #: JSON Schema object.
    parameters: dict[str, Any]


class WireTool(TypedDict):
    """One entry of the request ``tools`` array."""

    type: Literal["function"]
    function: WireToolFunction


class WireToolCallFunction(TypedDict, total=False):
    #: Present on the first delta of each call only.
    name: str
    #: Argument JSON fragment (concatenate across deltas).
    arguments: str


class WireToolCall(TypedDict):
    """A completed tool call replayed on an assistant history message."""

    id: str
    type: Literal["function"]
    function: WireToolCallFunction


class WireMessage(TypedDict, total=False):
    """One entry of the request ``messages`` array, discriminated on ``role``.

    - system/user: ``content`` is a single string.
    - tool: the result of one tool call, keyed by ``tool_call_id``.
    - assistant: ``content`` is ``''`` (never ``None``) on tool-call-only
      turns; ``reasoning_content`` (CoT passback) is REQUIRED on assistant
      turns that carried tool calls and omitted otherwise.
    """

    role: str  # required
    content: str | None
    tool_call_id: str
    reasoning_content: str
    tool_calls: list[WireToolCall]


class WireThinking(TypedDict):
    type: Literal["enabled", "disabled"]


class WireStreamOptions(TypedDict):
    include_usage: Literal[True]


class WireRequest(TypedDict, total=False):
    """Request body for ``POST {baseURL}/chat/completions``."""

    model: str  # required
    messages: list[WireMessage]  # required
    stream: Literal[True]  # required
    stream_options: WireStreamOptions  # required
    #: Thinking-mode toggle (top level, NOT inside extra_body on the wire).
    thinking: WireThinking
    #: Thinking effort (official levels; low/medium map to high server-side).
    reasoning_effort: Literal["high", "max"]
    tools: list[WireTool]
    temperature: float
    max_tokens: int
    #: Stop sequences (OpenAI ``stop``); mapped from ``GenerateOptions['stop']``.
    stop: list[str]


class WireToolCallDelta(TypedDict, total=False):
    """A streamed fragment of one tool call; fragments sharing an ``index``
    concatenate into one call."""

    #: Disambiguates parallel tool calls; stable across a call's deltas.
    index: int  # required
    #: Present on the first delta of each call only.
    id: str
    type: Literal["function"]
    function: WireToolCallFunction


class WireDelta(TypedDict, total=False):
    """The incremental content of one streamed choice; any subset of fields
    may be present per chunk."""

    role: str
    #: Visible text. Null/empty on reasoning/tool-call chunks.
    content: str | None
    #: Thinking-mode CoT. The FIRST chunk carries an empty string (must not
    #: open a reasoning block); absent entirely in non-thinking mode.
    reasoning_content: str | None
    tool_calls: list[WireToolCallDelta]


class WireChoice(TypedDict, total=False):
    """One streamed choice; ``finish_reason`` is non-null only on its terminal chunk."""

    delta: WireDelta
    finish_reason: str | None


class WirePromptTokensDetails(TypedDict, total=False):
    #: OpenAI-compat spelling of the cache hit count.
    cached_tokens: int


class WireCompletionTokensDetails(TypedDict, total=False):
    reasoning_tokens: int


class WireUsage(TypedDict, total=False):
    """Wire token accounting. ``prompt_tokens`` INCLUDES cache hits (it equals
    ``prompt_cache_hit_tokens + prompt_cache_miss_tokens``); ``map_usage``
    subtracts them to keep the harness convention of disjoint counts."""

    prompt_tokens: int  # required
    completion_tokens: int  # required
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    prompt_tokens_details: WirePromptTokensDetails
    completion_tokens_details: WireCompletionTokensDetails


class WireChunk(TypedDict, total=False):
    """One parsed SSE ``data:`` payload (a chat.completion.chunk)."""

    choices: list[WireChoice]
    #: Arrives attached to the finish chunk and/or as a trailing usage-only chunk.
    usage: WireUsage | None


class WireErrorBody(TypedDict, total=False):
    message: str
    type: str
    code: str


class WireError(TypedDict, total=False):
    """Non-2xx error body."""

    error: WireErrorBody


__all__ = [name for name in dir() if name.startswith("Wire")]
