"""Surface layer on top of the session event log.

Ported from deepseek-harness packages/core/session (MIT), ``src/surface.ts``.

The surface is an ordered view of the events that produce LLM messages; the
append-only log remains the source of truth. The module holds the pure
whole-log fold (:func:`fold_surface`), the incremental live view
(:class:`SurfaceManager`), the per-node message projection
(:func:`derive_event_message`), and the runtime guards over surface metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, TypedDict, TypeGuard, cast

from .json import _deep_equal_json
from .types import Message, SessionEvent, SurfaceOp, SurfaceReplace

__all__ = [
    'SurfaceFoldReplacement',
    'SurfaceFoldResult',
    'SurfaceManager',
    'derive_event_message',
    'fold_surface',
    'is_append_surface_event',
    'is_replacement_surface_event',
    'is_surface_eligible_type',
    'is_surface_event',
]

_MAX_SAFE_INTEGER: Final = 2**53 - 1

#: Runtime counterpart of the message-producing event union.
SURFACE_EVENT_TYPES: Final = frozenset(
    {'user/message', 'assistant/message', 'tool/result'}
)


def is_surface_eligible_type(type: str) -> bool:
    """Whether an event type can join the model-visible surface.

    :param type: event type to test.
    :returns: true for one of the three message-producing event types.
    """
    return type in SURFACE_EVENT_TYPES


def is_surface_event(event: SessionEvent) -> bool:
    """Narrow an event to a surface event carrying its required marker.

    :param event: event to test.
    :returns: true when both the type and marker identify a surface event.
    """
    return is_surface_eligible_type(event['type']) and 'surfaceOp' in event


def is_append_surface_event(event: SessionEvent) -> bool:
    """Narrow an event to an append-origin surface event.

    Append-origin events are the durable source material of a human
    transcript; replacement copies stay model-only.

    :param event: event to test.
    :returns: true when the event appended to the surface tail.
    """
    return is_surface_event(event) and event['surfaceOp'] == 'append'


def is_replacement_surface_event(event: SessionEvent) -> bool:
    """Narrow an event to a surface replacement.

    :param event: event to test.
    :returns: true when the event replaced a surface range.
    """
    return is_surface_event(event) and event['surfaceOp'] != 'append'


def derive_event_message(event: SessionEvent) -> Message | None:
    """Project a single event into the LLM message it derives to.

    THE per-node projection rule: ``Session.derive_messages`` folds it over
    the live surface, and pure projections fold it over a log prefix's
    surface. A non-surface event (chunk, boundary, log-only record) and an
    empty-content ``assistant/message`` (which exists only to host usage)
    project to no message. Message content is passed through verbatim —
    framing of injected context is producer-owned, never re-added here.

    :param event: the event to project.
    :returns: the derived message, or None when the event produces none.
    """
    type_ = event['type']
    if type_ == 'user/message':
        return cast(Message, event['data'])
    if type_ == 'assistant/message':
        message: Message = event['data']['message']
        # An empty-content assistant/message exists only to host a max-tokens
        # step's usage and must not enter the provider transcript.
        if len(message['content']) == 0:
            return None
        return message
    if type_ == 'tool/result':
        return cast(Message, event['data']['message'])
    # Merge-extensible union in TS: unknown types project to no message.
    return None


class SurfaceFoldReplacement(TypedDict):
    """One replacement operation observed while folding a session surface."""

    #: Seq of the event that replaced the prior surface range.
    seq: int
    #: Declared inclusive start seq of the replaced surface range.
    start: int
    #: Declared inclusive end seq of the replaced surface range.
    end: int
    #: Actual surface entries removed by the operation, in surface order.
    shadowedSeqs: list[int]


class SurfaceFoldResult(TypedDict):
    """Complete result of replaying the surface operations in a session log."""

    #: Current surface event sequences in model-visible order.
    nodes: list[int]
    #: Replacement operations in event order.
    replacements: list[SurfaceFoldReplacement]


class _FoldState:
    """Mutable state shared by complete and incremental folds."""

    __slots__ = ('nodes', 'replace_generation')

    def __init__(self) -> None:
        self.nodes: list[int] = []
        self.replace_generation = 0


@dataclass(frozen=True)
class _AppendPlan:
    """A validated tail-append transition not yet committed."""

    seq: int


@dataclass(frozen=True)
class _ReplacePlan:
    """A validated positional-replacement transition not yet committed."""

    seq: int
    start: int
    end: int
    start_idx: int
    end_idx: int
    shadowed_seqs: list[int]


_SurfacePlan = _AppendPlan | _ReplacePlan


def _is_event_seq(value: Any) -> bool:
    """Whether a runtime value is a non-negative safe event sequence."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_SAFE_INTEGER
    )


def _is_replace_op(value: Any) -> TypeGuard[SurfaceReplace]:
    """Whether a runtime value is the exact positional-replacement shape."""
    return (
        isinstance(value, dict)
        and set(value.keys()) == {'op', 'start', 'end'}
        and value['op'] == 'replace'
        and _is_event_seq(value['start'])
        and _is_event_seq(value['end'])
    )


def _surface_op_of(event: SessionEvent) -> SurfaceOp | None:
    """Validate event-local surface eligibility and return its operation."""
    type_ = event['type']
    if not is_surface_eligible_type(type_):
        if 'surfaceOp' in event:
            raise ValueError(
                f'session event "{type_}" is not surface-eligible'
                ' and cannot carry surfaceOp'
            )
        if 'sourceEventSeqs' in event:
            raise ValueError(
                f'session event "{type_}" is not surface-eligible'
                ' and cannot carry sourceEventSeqs'
            )
        return None
    if 'surfaceOp' not in event:
        raise ValueError(
            f'session event "{type_}" is surface-eligible'
            ' and requires a surfaceOp marker'
        )
    op: Any = event['surfaceOp']
    if op == 'append':
        return 'append'
    if not isinstance(op, dict):
        raise ValueError(f'session event "{type_}" carries an invalid surfaceOp')
    if not _is_replace_op(op):
        raise ValueError(
            f'session event "{type_}" carries an invalid replace surfaceOp'
        )
    return op


def _assert_provenance(event: SessionEvent, shadowed_seqs: Sequence[int]) -> None:
    """Validate cited source-event seqs against prior log entries and the range."""
    sources: set[int] = set()
    if 'sourceEventSeqs' in event:
        raw: Any = event['sourceEventSeqs']
        if not isinstance(raw, list):
            raise ValueError(
                f'sourceEventSeqs on event at seq {event["seq"]}'
                ' must be an array when present'
            )
        if len(raw) == 0 and event['type'] != 'assistant/message':
            raise ValueError(
                'sourceEventSeqs must not be empty except on assistant/message'
            )
        non_earlier_source: int | None = None
        for source in raw:
            if not _is_event_seq(source):
                raise ValueError(
                    f'session event "{event["type"]}" sourceEventSeqs must'
                    ' densely contain non-negative safe integers'
                )
            sources.add(source)
            if non_earlier_source is None and source >= event['seq']:
                non_earlier_source = source
        if len(sources) != len(raw):
            raise ValueError('sourceEventSeqs must not contain duplicates')
        if non_earlier_source is not None:
            raise ValueError(
                'sourceEventSeqs must reference earlier events:'
                f' {non_earlier_source} >= current seq {event["seq"]}'
            )
    missing = [seq for seq in shadowed_seqs if seq not in sources]
    if missing:
        raise ValueError(
            'surface replace: sourceEventSeqs must include every shadowed'
            f' surface node; missing {", ".join(str(seq) for seq in missing)}'
        )


def _replacement_range(
    state: _FoldState, op: SurfaceReplace
) -> tuple[int, int, list[int]]:
    """Locate one replacement range without mutating the current fold state."""
    try:
        start_idx = state.nodes.index(op['start'])
    except ValueError:
        raise ValueError(
            f'surface replace: start seq {op["start"]} not found in surface'
        ) from None
    try:
        end_idx = state.nodes.index(op['end'])
    except ValueError:
        raise ValueError(
            f'surface replace: end seq {op["end"]} not found in surface'
        ) from None
    if start_idx > end_idx:
        raise ValueError(
            f'surface replace: start seq {op["start"]} (index {start_idx})'
            f' is after end seq {op["end"]} (index {end_idx})'
        )
    return start_idx, end_idx, state.nodes[start_idx : end_idx + 1]


def _message_with_cleared_content(message: Message) -> dict[str, Any]:
    """Copy a message with its first block's content nulled for comparison."""
    cleared = dict(message)
    content = message.get('content')
    block: Any = content[0] if isinstance(content, list) and content else None
    replaced = dict(block) if isinstance(block, dict) else {}
    replaced['content'] = None
    cleared['content'] = [replaced]
    return cleared


def _assert_tool_result_rewrite(
    event: SessionEvent,
    shadowed_seqs: Sequence[int],
    events: Sequence[SessionEvent],
    base_seq: int,
) -> None:
    """Restrict a tool-result replacement to one current result's content."""
    if event['type'] != 'tool/result':
        return
    if len(shadowed_seqs) != 1:
        raise ValueError(
            'tool/result surface replacement must rewrite exactly one current node'
        )
    for original_seq in shadowed_seqs:
        original = events[original_seq - base_seq]
        if original['type'] != 'tool/result':
            raise ValueError(
                'tool/result surface replacement must target a current tool/result'
            )
        original_rest = dict(original['data'])
        replacement_rest = dict(event['data'])
        original_rest['message'] = _message_with_cleared_content(
            original['data']['message']
        )
        replacement_rest['message'] = _message_with_cleared_content(
            event['data']['message']
        )
        if not _deep_equal_json(original_rest, replacement_rest):
            raise ValueError(
                'tool/result surface replacement may change only content'
            )


def _plan_surface_event(
    state: _FoldState,
    event: SessionEvent,
    expected_seq: int,
    events: Sequence[SessionEvent],
    base_seq: int,
) -> _SurfacePlan | None:
    """Validate one event at its replay boundary and prepare its transition."""
    if event['seq'] != expected_seq:
        raise ValueError(
            f'session event seq {event["seq"]} is not contiguous;'
            f' expected {expected_seq}'
        )
    surface_op = _surface_op_of(event)
    if surface_op is None:
        return None
    if surface_op == 'append':
        _assert_provenance(event, [])
        return _AppendPlan(seq=event['seq'])
    start_idx, end_idx, shadowed_seqs = _replacement_range(state, surface_op)
    _assert_provenance(event, shadowed_seqs)
    _assert_tool_result_rewrite(event, shadowed_seqs, events, base_seq)
    return _ReplacePlan(
        seq=event['seq'],
        start=surface_op['start'],
        end=surface_op['end'],
        start_idx=start_idx,
        end_idx=end_idx,
        shadowed_seqs=shadowed_seqs,
    )


def _apply_surface_plan(
    state: _FoldState, plan: _SurfacePlan | None
) -> SurfaceFoldReplacement | None:
    """Commit one previously validated surface transition."""
    if isinstance(plan, _AppendPlan):
        state.nodes.append(plan.seq)
        return None
    if plan is None:
        return None
    state.nodes[plan.start_idx : plan.end_idx + 1] = [plan.seq]
    state.replace_generation += 1
    return SurfaceFoldReplacement(
        seq=plan.seq,
        start=plan.start,
        end=plan.end,
        shadowedSeqs=plan.shadowed_seqs,
    )


def fold_surface(events: Sequence[SessionEvent]) -> SurfaceFoldResult:
    """Replay a complete session log through the canonical surface fold.

    :param events: session events in contiguous seq order.
    :returns: detached current sequences and replacement history.
    :raises ValueError: when an event violates surface metadata, source-event
        references, range, or tool-result rewrite rules.
    """
    state = _FoldState()
    replacements: list[SurfaceFoldReplacement] = []
    for index, event in enumerate(events):
        plan = _plan_surface_event(state, event, index, events, 0)
        replacement = _apply_surface_plan(state, plan)
        if replacement is not None:
            replacements.append(replacement)
    return SurfaceFoldResult(nodes=list(state.nodes), replacements=replacements)


class SurfaceManager:
    """Incremental ordered surface view and append-boundary validator.

    Shares the session's live log list, so reads fold only the events
    appended since the previous access. ``validate_next`` plans one
    candidate without committing it, so a rejected append leaves the
    surface untouched.
    """

    def __init__(self, log: list[SessionEvent], base_seq: int = 0) -> None:
        """Bind the manager to a contiguous log (or loaded event window).

        :param log: contiguous complete log or loaded event window; the live
            list object is shared, not copied.
        :param base_seq: absolute sequence of the window's first event.
        """
        self._log = log
        self._base_seq = base_seq
        self._state = _FoldState()
        self._last_processed_seq = base_seq - 1
        self._pending_plan: tuple[SessionEvent, int, _SurfacePlan | None] | None = (
            None
        )

    def validate_next(self, event: SessionEvent) -> None:
        """Validate the next candidate without mutating the committed surface.

        :param event: candidate event that has not entered the log yet.
        :raises ValueError: when the candidate violates the surface contract.
        """
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        expected_seq = self._base_seq + len(self._log)
        self._pending_plan = (
            event,
            expected_seq,
            _plan_surface_event(
                self._state, event, expected_seq, self._log, self._base_seq
            ),
        )

    @property
    def replace_generation(self) -> int:
        """Monotonic count of committed positional replacements."""
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return self._state.replace_generation

    @property
    def nodes(self) -> list[int]:
        """Surface event sequences in model-visible order (a fresh copy)."""
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return list(self._state.nodes)

    def _process_delta(self) -> None:
        """Fold events appended since the previous access."""
        tail_seq = self._base_seq + len(self._log) - 1
        for seq in range(self._last_processed_seq + 1, tail_seq + 1):
            index = seq - self._base_seq
            event = self._log[index]
            pending = self._pending_plan
            if pending is not None and pending[0] is event and pending[1] == seq:
                _apply_surface_plan(self._state, pending[2])
            else:
                plan = _plan_surface_event(
                    self._state, event, seq, self._log, self._base_seq
                )
                _apply_surface_plan(self._state, plan)
            if pending is not None and pending[1] <= seq:
                self._pending_plan = None
            self._last_processed_seq = seq
