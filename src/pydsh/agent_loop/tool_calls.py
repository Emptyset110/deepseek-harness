"""Schedules one assistant step's tool calls.

Ported from deepseek-harness ``packages/core/agent-loop/src/tool-calls.ts``
(MIT). Exclusive calls form barriers; parallel calls use a bounded rolling
pool and are reclassified before start. Dispatch may overlap, while policy,
results, and result context remain model-ordered. Abort or an internal
scheduler failure stops replenishment and drains started calls.

Abort records synthetic error results for skipped calls so replay stays
valid. A terminal scheduler failure preserves already-recorded ``tool/call``
events without fabricating results.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pycordis import Context
from pydsh.agent import Message
from pydsh.llm import CallId, ToolCallBlock
from pydsh.session import Session
from pydsh.subprocess.types import AbortSignal
from pydsh.tools import (
    TOOL_ABORTED_BEFORE_DISPATCH,
    ScheduledDispatch,
    ScheduledFinalResult,
    ScheduledPostResult,
    ToolExecutionInput,
    ToolExecutionResult,
    ToolExecutionSuccess,
)
from pydsh.tools.runtime import _UNSET

from .llm_helpers import assert_never, create_tool_result_message

if TYPE_CHECKING:
    from pydsh.tools import ToolExecution


@dataclass
class _PlannedCall:
    """One tool call after argument parsing, ready to schedule."""

    block: ToolCallBlock
    exec: ToolExecutionInput


@dataclass
class _Slot:
    """Settled dispatch awaiting model-order finalization."""

    exec: ToolExecution
    result: ToolExecutionResult
    needs_post: bool


@dataclass
class _GroupOutcome:
    """One scheduler group outcome, including a drained cancellation."""

    consumed: int
    aborted: bool
    #: Whether any committed result carried ``concludes_turn``.
    concluded: bool


def _parse_arguments(raw: str) -> Any:
    """Parse model arguments: invalid JSON stays text, empty input is ``{}``."""
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return raw


async def execute_tool_calls(
    ctx: Context,
    turn: int,
    step: int,
    tool_calls: list[ToolCallBlock],
    signal: AbortSignal,
    accept_context: Callable[[Message], None],
) -> bool:
    """Schedule one assistant step's tool calls by their live concurrency mode.

    Ordinary completion and abort commit started-call results in order.
    Abort drains them, records synthetic results for unstarted calls, and
    returns with the signal still aborted after accepting started-call
    context through the caller-supplied acceptor (the machine stages it in
    its next-step inbox for the step boundary). An internal scheduler failure
    stops new dispatches, drains already-started dispatches, and raises the
    first failure without fabricating tool results.

    The committed step's AgentLoop driver boundary supplies the initiating
    agent that becomes each explicit ``ToolExecutionInput.agent``.

    :param ctx: loop context owning the tool registry and the initiator.
    :param turn: current turn number.
    :param step: current step number.
    :param tool_calls: assistant calls in model order.
    :param signal: abort signal shared by the step.
    :param accept_context: accepts committed result context for the next
        step boundary.
    :returns: whether any committed result concluded the turn.
    """
    agent = ctx.agents.require_initiator()
    session = agent.session

    # Inputs are distinct because tools/execute wrappers may replace
    # ``exec.signal``.
    planned = [
        _PlannedCall(
            block=block,
            exec=ToolExecutionInput(
                call_id=CallId(block['id']),
                name=block['name'],
                arguments=_parse_arguments(block['arguments']),
                signal=signal,
                agent=agent,
            ),
        )
        for block in tool_calls
    ]

    next_index = 0
    concluded = False
    while next_index < len(planned):
        # Commit before classifying again so registry changes affect
        # unstarted calls.
        first = planned[next_index]
        mode = ctx.tools.execution_mode(first.exec)
        group = planned[next_index:] if mode == 'parallel' else [first]
        outcome = await _run_group(ctx, turn, step, group, mode, signal, accept_context)
        next_index += outcome.consumed
        concluded = concluded or outcome.concluded
        if outcome.aborted:
            for call in planned[next_index:]:
                _append_skipped_tool_call(session, turn, step, call.block)
            return concluded
    return concluded


async def _run_group(
    ctx: Context,
    turn: int,
    step: int,
    group: list[_PlannedCall],
    mode: str,
    signal: AbortSignal,
    accept_context: Callable[[Message], None],
) -> _GroupOutcome:
    """Run one exclusive barrier or parallel pool.

    Later calls are reclassified before start; an exclusive reclassification
    waits for the current pool to drain and remains for the caller's next
    barrier. Results and contexts commit in model order. Abort stops starts,
    drains and commits started calls, accepts their contexts into the owning
    batch, records results for skipped calls, and returns an aborted outcome.
    Scheduler failure drains dispatches without committing synthetic recovery
    results.
    """
    session = ctx.agents.require_initiator().session
    max_parallel = ctx.agentLoop.config.max_parallel_tool_calls
    scheduler = ctx.tools.scheduler
    slots: list[_Slot | None] = [None] * len(group)
    # Started slots retain their ``tool/call`` seq so the result can cite it.
    call_seqs = [-1] * len(group)
    next_to_start = 0
    committed = 0
    started = 0
    aborted = signal.aborted
    concluded = False
    scheduler_failure: BaseException | None = None

    def throw_scheduler_failure() -> None:
        if scheduler_failure is not None:
            raise scheduler_failure

    async def commit_ready() -> None:
        # ``committed`` advances only across contiguous model-order slots.
        nonlocal committed, concluded
        while committed < len(group):
            slot = slots[committed]
            if slot is None:
                break
            call = group[committed]
            if slot.needs_post:
                result = await scheduler.finalize(slot.exec, slot.result)
            else:
                result = scheduler.finish(slot.exec, slot.result)
            _append_tool_result(
                session, turn, step, call.block, result, call_seqs[committed]
            )
            for context in result.additional_contexts or ():
                # The tools layer annotates contexts with the pydsh.agent
                # stand-in; at runtime they are canonical message dicts.
                accept_context(context)
            concluded = concluded or (
                isinstance(result, ToolExecutionSuccess) and result.concludes_turn
            )
            committed += 1

    in_flight: dict[int, asyncio.Task[int]] = {}

    async def start_call(index: int) -> None:
        nonlocal started, scheduler_failure
        call = group[index]
        call_seqs[index] = _append_tool_call(session, turn, step, call.block)
        started += 1
        prepared = await scheduler.prepare(call.exec)
        throw_scheduler_failure()
        if isinstance(prepared, ScheduledDispatch):

            async def run_dispatch() -> int:
                nonlocal scheduler_failure
                try:
                    outcome = await scheduler.dispatch(prepared.exec)
                except BaseException as error:
                    if scheduler_failure is None:
                        scheduler_failure = error
                    return index
                slots[index] = _Slot(
                    exec=prepared.exec,
                    result=outcome.result,
                    needs_post=outcome.kind == 'post-result',
                )
                return index

            in_flight[index] = asyncio.ensure_future(run_dispatch())
        elif isinstance(prepared, ScheduledPostResult):
            slots[index] = _Slot(
                exec=prepared.exec, result=prepared.result, needs_post=True
            )
        elif isinstance(prepared, ScheduledFinalResult):
            slots[index] = _Slot(
                exec=prepared.exec, result=prepared.result, needs_post=False
            )
        else:
            assert_never(prepared, 'tool-call scheduler prepare result')

    async def fill_pool() -> None:
        nonlocal next_to_start, aborted
        while (
            not aborted
            and next_to_start < len(group)
            and len(in_flight) < max_parallel
        ):
            # Re-read later modes after ordered commits so registry changes
            # can create a barrier.
            next_call = group[next_to_start]
            if (
                next_to_start > 0
                and mode == 'parallel'
                and ctx.tools.execution_mode(next_call.exec) != 'parallel'
            ):
                break
            await start_call(next_to_start)
            next_to_start += 1
            throw_scheduler_failure()
            await commit_ready()
            throw_scheduler_failure()
            # Abort may arrive while pre-execute awaits.
            if signal.aborted:
                aborted = True

    # Ordered pre-execute may await; only dispatch/body overlaps. A scheduler
    # failure stops new dispatches and reaches the turn boundary after every
    # already-started dispatch settles.
    try:
        await fill_pool()
        while len(in_flight) > 0:
            done, _pending = await asyncio.wait(
                in_flight.values(), return_when=asyncio.FIRST_COMPLETED
            )
            settled_index = next(iter(done)).result()
            del in_flight[settled_index]
            throw_scheduler_failure()
            await commit_ready()
            throw_scheduler_failure()
            # Abort may arrive while a tool or ordered commit awaits.
            if signal.aborted:
                aborted = True
            await fill_pool()
    except BaseException as error:
        if scheduler_failure is None:
            scheduler_failure = error
        await asyncio.gather(*in_flight.values(), return_exceptions=True)
        raise scheduler_failure from error

    if aborted:
        # Started calls and accepted context settle first; every remaining
        # model call then receives an ordered synthetic result before the
        # turn aborts.
        for call in group[started:]:
            _append_skipped_tool_call(session, turn, step, call.block)
        return _GroupOutcome(consumed=len(group), aborted=True, concluded=concluded)
    if committed != started:
        raise RuntimeError('tool-call scheduler: uncommitted settled calls')
    return _GroupOutcome(consumed=started, aborted=False, concluded=concluded)


def _append_skipped_tool_call(
    session: Session, turn: int, step: int, block: ToolCallBlock
) -> None:
    """Append the durable call/result pair for a call skipped after cancellation."""
    call_seq = _append_tool_call(session, turn, step, block)
    _append_tool_result_dict(
        session,
        turn,
        step,
        block,
        content=[{'type': 'text', 'text': 'Error: tool call aborted before dispatch'}],
        is_error=True,
        error={'name': 'AbortError', 'code': TOOL_ABORTED_BEFORE_DISPATCH},
        meta=_UNSET,
        call_seq=call_seq,
    )


def _append_tool_call(
    session: Session, turn: int, step: int, block: ToolCallBlock
) -> int:
    """Append a started call and return the event seq its result must cite."""
    event = session.append(
        'tool/call',
        {
            'turn': turn,
            'step': step,
            'callId': block['id'],
            'name': block['name'],
            'arguments': block['arguments'],
        },
    )
    return event['seq']


def _append_tool_result(
    session: Session,
    turn: int,
    step: int,
    block: ToolCallBlock,
    result: ToolExecutionResult,
    call_seq: int,
) -> None:
    """Append a model-ordered result linked to its call event."""
    error = getattr(result, 'error', None)
    info = error.info if error is not None else None
    _append_tool_result_dict(
        session,
        turn,
        step,
        block,
        content=result.content,
        is_error=result.is_error,
        error=(
            {'name': info.name, 'code': info.code} if info is not None else None
        ),
        meta=result.meta,
        call_seq=call_seq,
    )


def _append_tool_result_dict(
    session: Session,
    turn: int,
    step: int,
    block: ToolCallBlock,
    *,
    content: Any,
    is_error: bool,
    error: dict[str, str] | None,
    meta: Any,
    call_seq: int,
) -> None:
    message = create_tool_result_message(CallId(block['id']), content, is_error)
    data: dict[str, Any] = {'turn': turn, 'step': step, 'message': message}
    if error is not None:
        data['error'] = error
    # The tool's private presentation payload (e.g. a result-time diff),
    # persisted so a UI bridge reproduces the card on replay.
    if meta is not _UNSET:
        data['meta'] = meta
    session.append(
        'tool/result', data, {'surfaceOp': 'append', 'sourceEventSeqs': [call_seq]}
    )
