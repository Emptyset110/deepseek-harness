"""Serialize harness messages into DeepSeek chat completions.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/serialize.ts``
(MIT). User text is joined; assistant text becomes ``content``, tool calls
become ``tool_calls``, and tool results become separate tool messages.
Assistant reasoning is replayed as ``reasoning_content`` only on tool-call
turns, as required by thinking-mode passback. Core image blocks are rejected
explicitly because this wire route is text-only; unknown (plugin-added)
block types retain the adapter's documented extension fallback: they are
skipped by the text flattening paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from pydsh.llm import (
    GenerateOptions,
    LlmError,
    Message,
    content_has_image,
)

from .types import WireMessage, WireRequest, WireTool, WireToolCall


@dataclass(frozen=True)
class RequestDefaults:
    """Adapter-level request defaults (from plugin config)."""

    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["off", "high", "max"] | None = None


@dataclass(frozen=True)
class _ResolvedThinking:
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["high", "max"] | None = None


def _reasoning_effort(effort: Any) -> Literal["off", "high", "max"]:
    """Validate the adapter-owned effort before resolving its wire fields."""
    if effort in ("off", "high", "max"):
        return cast(Literal["off", "high", "max"], effort)
    raise LlmError(
        f'DeepSeek does not support reasoning effort "{effort}"',
        "UNSUPPORTED_REASONING_EFFORT",
    )


def _resolve_thinking(
    options: GenerateOptions, defaults: RequestDefaults
) -> _ResolvedThinking:
    """Resolve one legal thinking/effort pair without exposing ``off`` as a
    wire effort."""
    if options.get("purpose") == "session-title":
        return _ResolvedThinking(thinking="disabled")
    effort = (
        defaults.reasoning_effort
        if options.get("reasoningEffort") is None
        else _reasoning_effort(options["reasoningEffort"])
    )
    if defaults.thinking == "disabled" and effort is not None and effort != "off":
        raise LlmError(
            f'DeepSeek deployment does not support reasoning effort "{effort}"',
            "UNSUPPORTED_REASONING_EFFORT",
        )
    if effort == "off":
        return _ResolvedThinking(thinking="disabled")
    if effort in ("high", "max"):
        return _ResolvedThinking(thinking="enabled", reasoning_effort=effort)
    if defaults.thinking is None:
        return _ResolvedThinking()
    return _ResolvedThinking(thinking=defaults.thinking)


def _flatten_text(blocks: list[Any]) -> str:
    """Join the text blocks of a message (used for user/tool-result content)."""
    return "".join(block["text"] for block in blocks if block.get("type") == "text")


def _assert_text_only(blocks: list[Any]) -> None:
    """Reject core image content before any text-flattening path can silently
    erase it."""
    if content_has_image(blocks):
        raise LlmError(
            "The DeepSeek chat-completions adapter does not support image content.",
            "UNSUPPORTED_CONTENT",
        )


def _serialize_assistant(message: Message) -> WireMessage:
    """Serialize one assistant message (text + reasoning + tool calls)."""
    blocks: list[Any] = message["content"]
    text = _flatten_text(blocks)
    reasoning = "".join(
        block["text"] for block in blocks if block.get("type") == "reasoning"
    )
    tool_calls: list[WireToolCall] = [
        {
            "id": block["id"],
            "type": "function",
            "function": {"name": block["name"], "arguments": block["arguments"]},
        }
        for block in blocks
        if block.get("type") == "tool-call"
    ]
    wire: WireMessage = {
        "role": "assistant",
        # Text-less turns send "" — NEVER null. Pure tool-call turns replay
        # message.content verbatim (which is ""); reasoning-only turns exist
        # (the model can answer entirely in the reasoning channel), and the
        # live API rejects null-content/no-tool_calls assistant messages with
        # a 400 — a null here would brick every later turn of that session.
        "content": text,
    }
    # Official passback rule (guides/thinking_mode.mdx): reasoning_content
    # must return on tool-call turns; it is ignored on plain turns, so we
    # drop it there to save tokens.
    if tool_calls and reasoning:
        wire["reasoning_content"] = reasoning
    if tool_calls:
        wire["tool_calls"] = tool_calls
    return wire


def serialize_messages(messages: list[Message]) -> list[WireMessage]:
    """Serialize the conversation. ``tool-result`` blocks become standalone
    ``role: 'tool'`` messages; the harness puts each tool result in its own
    user-role message, so a mixed user message contributes its text first and
    its tool results as separate wire messages after. Order preserved."""
    wire: list[WireMessage] = []
    for message in messages:
        _assert_text_only(message["content"])
        if message["role"] == "system":
            wire.append(
                {"role": "system", "content": _flatten_text(message["content"])}
            )
            continue
        if message["role"] == "assistant":
            wire.append(_serialize_assistant(message))
            continue
        # user role: tool results ride in user messages in the harness
        # vocabulary, but DeepSeek wants them as role:'tool' messages.
        blocks: list[Any] = message["content"]
        tool_results = [block for block in blocks if block.get("type") == "tool-result"]
        text = _flatten_text(blocks)
        if len(text) > 0 or len(tool_results) == 0:
            wire.append({"role": "user", "content": text})
        for result in tool_results:
            wire.append(
                {
                    "role": "tool",
                    "tool_call_id": result["toolCallId"],
                    # Empty tool output still needs SOME content on the wire.
                    "content": _flatten_text(result.get("content", []))
                    or "(no output)",
                }
            )
    return wire


def serialize_request(
    options: GenerateOptions,
    defaults: RequestDefaults | None = None,
) -> WireRequest:
    """Build the full wire request. Always streaming (``stream: true``, usage
    reporting on); optional fields are omitted rather than sent as null, so
    provider defaults apply."""
    if defaults is None:
        defaults = RequestDefaults()
    messages: list[WireMessage] = []
    if options.get("system") is not None:
        messages.append({"role": "system", "content": options["system"]})
    messages.extend(serialize_messages(options["messages"]))

    tools: list[WireTool] | None = (
        [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in options["tools"]
        ]
        if options.get("tools") is not None
        else None
    )
    # A short title budget must produce visible text; conversation and
    # compaction calls continue to inherit the adapter's thinking defaults.
    resolved_thinking = _resolve_thinking(options, defaults)

    request: WireRequest = {
        "model": options["model"],
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if resolved_thinking.thinking is not None:
        request["thinking"] = {"type": resolved_thinking.thinking}
    if resolved_thinking.reasoning_effort is not None:
        request["reasoning_effort"] = resolved_thinking.reasoning_effort
    if tools:
        request["tools"] = tools
    if options.get("temperature") is not None:
        request["temperature"] = options["temperature"]
    if options.get("maxTokens") is not None:
        request["max_tokens"] = options["maxTokens"]
    if options.get("stop") is not None:
        request["stop"] = options["stop"]
    return request
