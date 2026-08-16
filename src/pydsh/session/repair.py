"""Crash-recovery repair for an interrupted session log.

Ported from deepseek-harness packages/core/session (MIT), ``src/repair.ts``.

The repair preserves a fully written final turn and supplies the missing
tool, step, and turn boundaries needed to resume with a provider-valid
transcript. It is a pure function over the durable log; persistence backends
apply it on reload when a crash orphaned a turn.
"""

from __future__ import annotations

from typing import Final, cast

from .types import SessionEvent, ToolResultMessage

__all__ = [
    'TOOL_NOT_STARTED',
    'TOOL_OUTCOME_UNKNOWN',
    'interrupted_turn_closers',
]

#: Recovery code for an assistant tool request that never reached a recorded
#: call start.
TOOL_NOT_STARTED: Final = 'TOOL_NOT_STARTED'

#: Recovery code for a recorded tool call whose completed outcome was not
#: durably recorded.
TOOL_OUTCOME_UNKNOWN: Final = 'TOOL_OUTCOME_UNKNOWN'

_OUTCOME_UNKNOWN_TEXT: Final = (
    'The tool call was interrupted after it was recorded, but no result was'
    ' durably recorded. Its outcome is unknown. Decide whether to retry from'
    ' the tool semantics: retry only if the operation is read-only or'
    ' idempotent; if it may have side effects, first verify external state or'
    ' ask the user. Do not retry blindly.'
)

_NOT_STARTED_TEXT: Final = (
    'The tool call was interrupted before the Harness recorded it as started.'
    ' Retry it if it is still needed.'
)


def interrupted_turn_closers(events: list[SessionEvent]) -> list[SessionEvent]:
    """Return deterministic synthetic events that close an open tail turn.

    Unmatched calls receive error results first, followed by an open
    ``step/end`` and an interrupted ``turn/end``; sequences continue the log
    and timestamps reuse the last real event. A balanced or empty log
    returns no events.

    :param events: the loaded durable log to scan (a valid committed prefix,
        possibly with a crash tail).
    :returns: the synthetic closer events to append after ``events``, in
        order; empty when the log is already balanced.
    """
    open_turn: int | None = None
    open_step: int | None = None
    # Reset at each turn boundary so earlier calls cannot leak into tail
    # repair. Assistant blocks register calls; later tool/call events add
    # their seqs to call_seqs. Dict order preserves transcript order.
    pending_steps: dict[str, int] = {}
    call_seqs: dict[str, int] = {}
    for event in events:
        type_ = event['type']
        if type_ == 'turn/start':
            open_turn = event['data']['turn']
            open_step = None
            pending_steps.clear()
            call_seqs.clear()
        elif type_ == 'turn/end':
            open_turn = None
            open_step = None
            pending_steps.clear()
            call_seqs.clear()
        elif type_ == 'step/start':
            open_step = event['data']['step']
        elif type_ == 'step/end':
            pending_steps.clear()
            call_seqs.clear()
            open_step = None
        elif type_ == 'assistant/message':
            # The assistant message carries the tool-call blocks; each is
            # pending until a tool/result event with the same callId is
            # logged.
            for block in event['data']['message'].get('content') or []:
                if block.get('type') == 'tool-call':
                    pending_steps[block['id']] = event['data']['step']
        elif type_ == 'tool/call':
            # Cite the tool/call seq from the synthetic result.
            call_id = event['data']['callId']
            if call_id in pending_steps:
                call_seqs[call_id] = event['seq']
        elif type_ == 'tool/result':
            call_id = event['data']['message']['source']['callId']
            pending_steps.pop(call_id, None)
            call_seqs.pop(call_id, None)
        # Other event types do not move the turn/step boundary cursor.

    # Balanced log (no crash mid-turn): nothing to close. An open turn
    # implies events is non-empty (its turn/start was logged).
    last = events[-1] if events else None
    if open_turn is None or last is None:
        return []

    # The last real event supplies the seq base and the timestamp for the
    # synthetic closers (reusing the last timestamp keeps them deterministic
    # and never invents a "future" time).
    seq = last['seq'] + 1
    time = last['time']
    closers: list[SessionEvent] = []

    # Close calls before their step: providers reject dangling assistant
    # calls.
    for call_id, step in pending_steps.items():
        started = call_id in call_seqs
        message: ToolResultMessage = {
            'id': f'interrupted-tool-result-{call_id}-{seq}',
            'role': 'user',
            'source': {'kind': 'tool', 'callId': call_id},
            'content': [
                {
                    'type': 'tool-result',
                    'toolCallId': call_id,
                    'isError': True,
                    'content': [
                        {
                            'type': 'text',
                            'text': _OUTCOME_UNKNOWN_TEXT
                            if started
                            else _NOT_STARTED_TEXT,
                        }
                    ],
                }
            ],
        }
        closer: dict[str, object] = {
            'type': 'tool/result',
            'seq': seq,
            'time': time,
            'data': {
                'turn': open_turn,
                'step': step,
                'message': message,
                'error': {
                    'name': 'ToolOutcomeUnknownError'
                    if started
                    else 'ToolNotStartedError',
                    'code': TOOL_OUTCOME_UNKNOWN if started else TOOL_NOT_STARTED,
                },
            },
            'surfaceOp': 'append',
        }
        if started:
            closer['sourceEventSeqs'] = [call_seqs[call_id]]
        closers.append(cast(SessionEvent, closer))
        seq += 1

    # Close an open step next — a turn/end while a step is open is an
    # invariant violation, so the step's boundary must be synthesized before
    # the turn's.
    if open_step is not None:
        closers.append(
            SessionEvent(
                type='step/end',
                seq=seq,
                time=time,
                data={'turn': open_turn, 'step': open_step},
            )
        )
        seq += 1
    closers.append(
        SessionEvent(
            type='turn/end',
            seq=seq,
            time=time,
            data={'turn': open_turn, 'reason': {'kind': 'interrupted'}},
        )
    )
    return closers
