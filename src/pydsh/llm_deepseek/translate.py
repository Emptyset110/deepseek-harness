"""Translate DeepSeek wire chunks into the harness ``StreamChunk`` protocol.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/translate.ts``
(MIT). One stateful harness block per content, reasoning, or tool-call
index. An empty initial reasoning delta does not open a block. Finish reason
and the latest usage are deferred until ``[DONE]``, covering both
finish-attached and trailing usage-only shapes while ensuring no chunk
follows ``finish``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Literal

from pydsh.llm import (
    EMPTY_RESPONSE_CODE,
    CallId,
    ContentBlock,
    FinishReason,
    LlmError,
    StreamChunk,
    TokenUsage,
    ToolCallDeltaChunk,
)

from .sse import DONE
from .types import WireChunk, WireUsage


@dataclass
class _OpenBlock:
    """One open block under assembly."""

    index: int
    kind: Literal["text", "reasoning", "tool-call"]
    text: str = ""
    call_id: str | None = None
    name: str | None = None


def map_finish_reason(reason: str) -> FinishReason:
    """Map the wire ``finish_reason`` vocabulary to the harness FinishReason.

    Unrecognized values (``content_filter``, …) become an ``error`` finish
    with the uppercased value as ``code``.
    """
    if reason == "stop":
        return {"kind": "stop"}
    if reason == "tool_calls":
        return {"kind": "tool-calls"}
    if reason == "length":
        return {"kind": "max-tokens"}
    # content_filter, insufficient_system_resource, future additions.
    return {
        "kind": "error",
        "failure": {"message": f"model stopped: {reason}", "code": reason.upper()},
    }


def map_usage(usage: WireUsage) -> TokenUsage:
    """Map wire usage fields to disjoint harness counts.

    DeepSeek's ``prompt_tokens`` INCLUDES cache hits, so cache reads are
    subtracted out of ``inputTokens``; cache/reasoning fields are present
    only when the wire reported them.
    """
    details_cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cache_read = (
        details_cached
        if details_cached is not None
        else usage.get("prompt_cache_hit_tokens")
    )
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    result: TokenUsage = {
        "inputTokens": usage["prompt_tokens"] - (cache_read or 0),
        "outputTokens": usage["completion_tokens"],
    }
    if cache_read is not None:
        result["cacheReadTokens"] = cache_read
    if reasoning is not None:
        result["reasoningTokens"] = reasoning
    return result


def _close_block(block: _OpenBlock) -> ContentBlock:
    """Assemble the final ContentBlock for one open block."""
    if block.kind == "text":
        return {"type": "text", "text": block.text}
    if block.kind == "reasoning":
        return {"type": "reasoning", "text": block.text}
    return {
        "type": "tool-call",
        "id": CallId(block.call_id or ""),
        "name": block.name or "",
        "arguments": block.text,
    }


async def translate(payloads: AsyncIterable[str]) -> AsyncGenerator[StreamChunk]:
    """Consume SSE data payloads (ending with ``[DONE]``) and yield StreamChunks.

    Malformed JSON payloads abort the stream with ``MALFORMED_RESPONSE``.
    Deltas are yielded as they arrive; ``block-end``s, ``usage``, and
    ``finish`` are all deferred to the ``[DONE]`` sentinel. A ``stop`` (or
    absent) finish with no opened blocks is a degenerate provider completion
    and maps to an ``EMPTY_RESPONSE`` error finish instead of a successful
    empty message.
    """
    next_index = 0
    text_block: _OpenBlock | None = None
    reasoning_block: _OpenBlock | None = None
    tool_blocks: dict[int, _OpenBlock] = {}
    order: list[_OpenBlock] = []
    pending_finish: FinishReason | None = None
    pending_usage: TokenUsage | None = None

    def open_block(kind: Literal["text", "reasoning", "tool-call"]) -> _OpenBlock:
        nonlocal next_index
        block = _OpenBlock(index=next_index, kind=kind)
        next_index += 1
        order.append(block)
        return block

    async for payload in payloads:
        if payload == DONE:
            for finished in order:
                yield {
                    "type": "block-end",
                    "index": finished.index,
                    "block": _close_block(finished),
                }
            if pending_usage is not None:
                yield {"type": "usage", "usage": pending_usage}
            reason: FinishReason = (
                pending_finish if pending_finish is not None else {"kind": "stop"}
            )
            if reason["kind"] == "stop" and len(order) == 0:
                reason = {
                    "kind": "error",
                    "failure": {
                        "message": "model returned a completed response with"
                        " no content",
                        "code": EMPTY_RESPONSE_CODE,
                    },
                }
            yield {"type": "finish", "reason": reason}
            return

        try:
            chunk: WireChunk = json.loads(payload)
        except json.JSONDecodeError:
            raise LlmError(
                f"malformed SSE payload: {payload[:120]}", "MALFORMED_RESPONSE"
            ) from None

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}

            # Reasoning first: thinking mode interleaves it before text. The
            # empty-string first chunk must not open a block.
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and len(reasoning) > 0:
                if reasoning_block is None:
                    reasoning_block = open_block("reasoning")
                    yield {
                        "type": "block-start",
                        "index": reasoning_block.index,
                        "blockType": "reasoning",
                    }
                reasoning_block.text += reasoning
                yield {
                    "type": "reasoning-delta",
                    "index": reasoning_block.index,
                    "text": reasoning,
                }

            content = delta.get("content")
            if isinstance(content, str) and len(content) > 0:
                if text_block is None:
                    text_block = open_block("text")
                    yield {
                        "type": "block-start",
                        "index": text_block.index,
                        "blockType": "text",
                    }
                text_block.text += content
                yield {"type": "text-delta", "index": text_block.index, "text": content}

            for call in delta.get("tool_calls") or []:
                tool_block = tool_blocks.get(call["index"])
                if tool_block is None:
                    tool_block = open_block("tool-call")
                    tool_blocks[call["index"]] = tool_block
                    yield {
                        "type": "block-start",
                        "index": tool_block.index,
                        "blockType": "tool-call",
                    }
                if call.get("id") is not None:
                    tool_block.call_id = call["id"]
                function = call.get("function") or {}
                if function.get("name") is not None:
                    tool_block.name = function["name"]
                fragment = function.get("arguments") or ""
                tool_block.text += fragment
                tool_delta: ToolCallDeltaChunk = {
                    "type": "tool-call-delta",
                    "index": tool_block.index,
                    "id": CallId(tool_block.call_id or ""),
                    "argumentsDelta": fragment,
                }
                if tool_block.name is not None:
                    tool_delta["name"] = tool_block.name
                yield tool_delta

            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                pending_finish = map_finish_reason(finish_reason)

        # Usage may arrive attached to the finish chunk or as a trailing
        # usage-only chunk — keep the latest.
        usage = chunk.get("usage")
        if usage is not None:
            pending_usage = map_usage(usage)

    # parse_sse guarantees the [DONE] sentinel (or throws); reaching here
    # means the payload source violated that contract.
    raise LlmError("SSE payload stream ended without [DONE]", "STREAM_CLOSED")
