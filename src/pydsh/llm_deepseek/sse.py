"""Decode an SSE byte stream into event ``data`` payloads.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/sse.ts`` (MIT).
TS delegates framing to ``eventsource-parser``; this module implements the
same framing directly: chunk reassembly, UTF-8/CRLF/BOM handling, comment
and non-data field skipping, and multi-``data:`` joining.

The DeepSeek protocol on top: the literal ``[DONE]`` is yielded so the
caller owns final flushing, and EOF before it raises ``LlmError``
(``STREAM_CLOSED``). Framing is spec-strict: an event dispatches only on its
blank-line terminator, so an unterminated tail at EOF is truncation, not a
flushable payload.
"""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import AsyncGenerator, AsyncIterable, Callable

from pydsh.llm import LlmError

#: The terminal payload DeepSeek (and OpenAI) send after the last chunk.
DONE = "[DONE]"


def _pop_line(buffer: str, final: bool) -> tuple[str | None, str]:
    """Split one complete line off ``buffer``.

    Line terminators are ``\\n``, ``\\r\\n``, and lone ``\\r``. A trailing
    ``\\r`` with more data possibly coming is held back to disambiguate
    ``\\r\\n`` from a lone ``\\r``.
    """
    for index, char in enumerate(buffer):
        if char == "\n":
            return buffer[:index], buffer[index + 1 :]
        if char == "\r":
            if index + 1 == len(buffer) and not final:
                return None, buffer
            if buffer[index + 1 : index + 2] == "\n":
                return buffer[:index], buffer[index + 2 :]
            return buffer[:index], buffer[index + 1 :]
    if final and buffer:
        return buffer, ""
    return None, buffer


async def parse_sse(
    stream: AsyncIterable[bytes],
    on_comment: Callable[[str], None] | None = None,
    idle_timeout: float | None = None,
) -> AsyncGenerator[str]:
    """Parse an SSE byte stream into data payloads.

    Yields ``[DONE]`` as the final value and returns; raises
    ``LlmError('STREAM_CLOSED')`` when the stream ends without it (a truncated
    response cannot be trusted). Raises ``asyncio.TimeoutError`` when one read
    exceeds ``idle_timeout`` seconds (the adapter maps it to ``TIMEOUT``).

    :param stream: raw SSE bytes; reads may split anywhere, including
        mid-UTF-8 sequence.
    :param on_comment: optional transport-activity callback; comments never
        enter the yielded payload stream.
    """
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    buffer = ""
    data_lines: list[str] = []
    iterator = stream.__aiter__()

    async def read() -> bytes:
        if idle_timeout is None:
            return await iterator.__anext__()
        return await asyncio.wait_for(iterator.__anext__(), idle_timeout)

    exhausted = False
    while True:
        if not exhausted:
            try:
                chunk = await read()
            except StopAsyncIteration:
                exhausted = True
                buffer += decoder.decode(b"", final=True)
            else:
                buffer += decoder.decode(bytes(chunk))
        while True:
            line, buffer = _pop_line(buffer, final=exhausted)
            if line is None:
                break
            if line == "":
                # Blank line: dispatch the pending event, if any.
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    yield payload
                    if payload == DONE:
                        return
                continue
            if line.startswith(":"):
                if on_comment is not None:
                    on_comment(line[1:].removeprefix(" "))
                continue
            field, _, value = line.partition(":")
            if field == "data":
                data_lines.append(value.removeprefix(" "))
        if exhausted:
            # An unterminated tail (a non-empty buffer or pending data lines
            # that never saw their blank line) is truncation, not a payload.
            raise LlmError("SSE stream ended without [DONE]", "STREAM_CLOSED")
