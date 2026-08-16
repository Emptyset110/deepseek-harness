"""LLM-side helpers the loop needs, pending deduplication into ``pydsh.llm``.

Ported from deepseek-harness (MIT): ``packages/llm/llm/src/assembler.ts``
(:class:`BlockAssembler`), ``message.ts`` (the ``create*Message``
constructors), ``never.ts`` (:func:`assert_never`), ``error.ts``
(:func:`error_chain`), and ``call-config.ts`` (:func:`deep_freeze`).

These canonically live in the llm package; they are kept private to the loop
layer because the current port boundary does not allow modifying
``pydsh.llm``. Python mappings:

- ``deepFreeze(structuredClone(value))`` becomes :func:`detached` — a deep
  copy; immutability is by convention (see ``pydsh.llm.types``).
- Message identity minting uses :func:`uuid.uuid4` (TS ``crypto.randomUUID``).
- The block assembler is the single canonical chunk-to-message assembly
  algorithm: the loop feeds it while logging raw chunks for replay fidelity.
"""

from __future__ import annotations

import copy
from typing import Any, NoReturn, cast
from uuid import uuid4

from pydsh.llm import (
    CallId,
    ContentBlock,
    FinishReason,
    MessageId,
    StreamChunk,
    TokenUsage,
)
from pydsh.session import Message

__all__ = [
    'BlockAssembler',
    'assert_never',
    'create_assistant_message',
    'create_tool_result_message',
    'create_user_message',
    'detached',
    'error_chain',
]


def detached(value: Any) -> Any:
    """Deep-copy one JSON value (TS ``deepFreeze(structuredClone(value))``).

    The copy is treated as read-only by convention — the harness never
    mutates harness data dicts after creation.
    """
    return copy.deepcopy(value)


def error_chain(error: Any) -> str:
    """Flatten an exception and its cause chain into one message.

    Simplified port of TS ``errorChain``: follows ``__cause__`` /
    ``__context__``, flagging true cycles; AggregateError bracketing has no
    Python counterpart (``ExceptionGroup`` members are bracketed instead).
    """
    path: set[int] = set()

    def render(current: Any) -> str:
        if id(current) in path:
            return '<circular cause>'
        path.add(id(current))
        try:
            if not isinstance(current, BaseException):
                return str(current)
            message = str(current) or type(current).__name__
            members = (
                f' [{"; ".join(render(member) for member in current.exceptions)}]'
                if isinstance(current, BaseExceptionGroup)
                else ''
            )
            cause = current.__cause__ or current.__context__
            suffix = f' ← {render(cause)}' if cause is not None else ''
            return f'{message}{members}{suffix}'
        finally:
            path.discard(id(current))

    return render(error)


def assert_never(value: NoReturn, label: str) -> NoReturn:
    """Closed-union exhaustiveness guard (TS ``assertNever``)."""
    raise AssertionError(f'unexpected {label}: {value!r}')


def create_user_message(
    content: list[ContentBlock], source: dict[str, Any]
) -> Message:
    """Create one identified user-role message with a fresh stable identity."""
    # The session Message TypedDict is a JSON stand-in; llm content blocks
    # and merge-extensible sources satisfy it structurally at runtime.
    return Message(
        id=MessageId(str(uuid4())),
        role='user',
        content=cast(Any, content),
        source=cast(Any, source),
    )


def create_assistant_message(
    content: list[ContentBlock],
    *,
    provider: str,
    model: str,
    replay_state: Any = None,
    has_replay_state: bool = False,
) -> Message:
    """Create one identified model-produced assistant message.

    :param has_replay_state: distinguishes an absent replay state from a
        JSON ``None`` state (TS optional-field presence).
    """
    source: dict[str, Any] = {'kind': 'model', 'provider': provider, 'model': model}
    if has_replay_state:
        source['replayState'] = replay_state
    return Message(
        id=MessageId(str(uuid4())),
        role='assistant',
        content=cast(Any, content),
        source=cast(Any, source),
    )


def create_tool_result_message(
    call_id: CallId, content: list[ContentBlock], is_error: bool
) -> Message:
    """Create one identified tool-result message."""
    return create_user_message(
        [
            {
                'type': 'tool-result',
                'toolCallId': call_id,
                'content': content,
                'isError': is_error,
            }
        ],
        {'kind': 'tool', 'callId': call_id},
    )


class _PartialBlock:
    """One in-flight block, closed authoritatively by its ``block-end``."""

    __slots__ = (
        'block',
        'block_type',
        'text',
        'tool_call_arguments',
        'tool_call_id',
        'tool_call_name',
    )

    def __init__(self, block_type: str) -> None:
        self.block_type = block_type
        self.text = ''
        self.tool_call_arguments = ''
        self.tool_call_id: CallId | None = None
        self.tool_call_name: str | None = None
        self.block: ContentBlock | None = None


class BlockAssembler:
    """Incrementally assembles raw stream chunks into content blocks.

    Tolerant of delta-only protocols (no block-start/end); deltas arriving
    for an index already closed by ``block-end`` are ignored (malformed
    stream) so a misbehaving adapter cannot grow memory or corrupt a
    completed block.
    """

    def __init__(self) -> None:
        self._partials: dict[int, _PartialBlock] = {}
        self._order: list[int] = []
        self._usage: TokenUsage | None = None
        self._finish: FinishReason | None = None
        self._replay_state: Any = None
        self._has_replay_state = False

    def push(self, chunk: StreamChunk) -> None:
        """Feed one chunk into the assembly state, in stream order."""
        # TypedDict unions do not narrow on the discriminant under mypy;
        # the loop's own chunk log keeps the exact shape, so read loosely.
        data: dict[str, Any] = chunk  # type: ignore[assignment]
        chunk_type = data['type']
        if chunk_type == 'block-start':
            if data['index'] not in self._partials:
                self._order.append(data['index'])
                self._partials[data['index']] = _PartialBlock(data['blockType'])
            return
        if chunk_type in ('text-delta', 'reasoning-delta'):
            partial = self._ensure(
                data['index'], 'text' if chunk_type == 'text-delta' else 'reasoning'
            )
            if partial.block is not None:
                return  # closed by block-end; ignore stragglers
            partial.text += data['text']
            return
        if chunk_type == 'tool-call-delta':
            partial = self._ensure(data['index'], 'tool-call')
            if partial.block is not None:
                return
            partial.tool_call_id = data['id']
            name = data.get('name')
            if name:
                partial.tool_call_name = name
            partial.tool_call_arguments += data['argumentsDelta']
            return
        if chunk_type == 'block-end':
            block: ContentBlock = data['block']
            partial = self._ensure(data['index'], block['type'])
            # First close wins; ignoring re-close stragglers keeps streamed
            # output and the final assembled block in agreement.
            if partial.block is not None:
                return
            partial.block = block
            return
        if chunk_type == 'usage':
            self._usage = data['usage']
            return
        if chunk_type == 'finish':
            self._finish = data['reason']
            self._has_replay_state = 'replayState' in data
            self._replay_state = data.get('replayState')
            return
        assert_never(chunk, 'BlockAssembler.push')  # type: ignore[arg-type]

    def _ensure(self, index: int, block_type: str) -> _PartialBlock:
        partial = self._partials.get(index)
        if partial is None:
            partial = _PartialBlock(block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    def _assemble(self, partial: _PartialBlock, index: int) -> ContentBlock:
        if partial.block is not None:
            return partial.block
        if partial.block_type == 'text':
            return {'type': 'text', 'text': partial.text}
        if partial.block_type == 'reasoning':
            return {'type': 'reasoning', 'text': partial.text}
        if partial.block_type == 'tool-call':
            return {
                'type': 'tool-call',
                'id': (
                    partial.tool_call_id
                    if partial.tool_call_id is not None
                    else CallId(f'call-{index}')
                ),
                'name': partial.tool_call_name or '',
                'arguments': partial.tool_call_arguments,
            }
        raise ValueError(
            f'cannot assemble incomplete block of type "{partial.block_type}"'
        )

    def blocks(self) -> list[ContentBlock]:
        """Assemble all blocks seen so far, in stream order.

        Max-token truncation drops tool calls that cannot be executed
        safely; an open block assembles from its accumulated deltas (an
        unknown block type never closed by ``block-end`` raises).
        """
        assembled = [
            self._assemble(self._partials[index], index) for index in self._order
        ]
        if self.finish['kind'] == 'max-tokens':
            return [block for block in assembled if block['type'] != 'tool-call']
        return assembled

    @property
    def usage(self) -> TokenUsage | None:
        """Usage from the ``usage`` chunk; ``None`` until one arrives."""
        return self._usage

    @property
    def finish(self) -> FinishReason:
        """The finish reason; ``{'kind': 'stop'}`` when the stream ended without one."""
        if self._finish is not None:
            return self._finish
        return {'kind': 'stop'}

    @property
    def replay_state(self) -> tuple[bool, Any]:
        """``(present, state)`` from the terminal finish chunk."""
        return self._has_replay_state, self._replay_state
