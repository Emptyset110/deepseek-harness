"""Default Agent driver over queued turns and step-boundary input.

Ported from deepseek-harness ``packages/core/agent-loop/src/agent.ts``
(MIT). Every request is derived from the session log.

Cancellation mapping: TS ``AbortController``/``AbortSignal`` become
:class:`LoopAbortController`/:class:`LoopAbortSignal`, which extend the
``pydsh.subprocess.types`` DOM-primitive port with the ``asyncio.Event``-shaped
reads (``is_set``/``wait``) the LLM request wire expects. ``asyncio``
futures stand in for ``Promise.withResolvers``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, cast

from pycordis import Context
from pycordis.utils import maybe_await
from pydsh.agent import (
    AgentCancelCause,
    AgentEventDispatch,
    AgentOptions,
    CancelOptions,
    Inbox,
    InboxNotifications,
    InboxTarget,
    Message,
    PreStepEnter,
    PreStepReject,
    agent_events,
    assemble_context_for,
)
from pydsh.llm import (
    GenerateOptions,
    LlmError,
    LlmFailure,
    ResolvedRetryPolicy,
    ToolCallBlock,
    mark_agent_loop_request,
)
from pydsh.scope import create_scope
from pydsh.session import (
    EpochHeader,
    Session,
    SessionId,
    canonical_header,
    header_equals,
)
from pydsh.subprocess.types import AbortController, AbortSignal
from pydsh.system_prompt import (
    PromptAssembly,
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from pydsh.system_prompt import (
    ToolSchema as PromptToolSchema,
)

from .llm_helpers import (
    BlockAssembler,
    create_assistant_message,
    detached,
    error_chain,
)
from .runtime_context import RuntimeContextProjection
from .tool_calls import execute_tool_calls


class LoopAbortSignal(AbortSignal):
    """The turn's cancellation signal, adding ``asyncio.Event``-shaped reads."""

    def is_set(self) -> bool:
        """Whether the signal is aborted (``asyncio.Event`` convention)."""
        return self.aborted

    async def wait(self) -> bool:
        """Wait until the signal aborts (``asyncio.Event`` convention)."""
        if self.aborted:
            return True
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def on_abort() -> None:
            if not future.done():
                future.set_result(True)

        self.add_listener(on_abort)
        try:
            return await future
        finally:
            self.remove_listener(on_abort)


class LoopAbortController(AbortController):
    """Owner side of a :class:`LoopAbortSignal`."""

    def __init__(self) -> None:
        self.signal: LoopAbortSignal = LoopAbortSignal()


@dataclass
class _IdlePhase:
    kind: Literal['idle']
    last_turn: int


@dataclass
class _MaintenancePhase:
    abort: LoopAbortController
    last_turn: int
    wake_requested: bool
    kind: Literal['maintenance'] = 'maintenance'


@dataclass
class _RunningPhase:
    abort: LoopAbortController
    turn: int
    step: int
    wake_requested: bool
    kind: Literal['running'] = 'running'


_Phase = _IdlePhase | _MaintenancePhase | _RunningPhase

#: Extract of TurnEndReason: only completed/max-tokens end a step.
_StepEndReason = dict[str, Any]


@dataclass
class _PreparedEnter:
    messages: list[Message]
    assembly: PromptAssembly
    kind: Literal['enter'] = 'enter'


_PreparedStep = PreStepReject | _PreparedEnter


def _request_proposal(header: EpochHeader) -> dict[str, Any]:
    """Remove adapter-derived values before plugins propose the next config."""
    proposal = dict(header['config'])
    defaults = header.get('adapterDefaults')
    if defaults is None:
        return proposal
    if defaults.get('reasoningEffort') is True:
        proposal.pop('reasoningEffort', None)
    if defaults.get('maxTokens') is True:
        proposal.pop('maxTokens', None)
    return proposal


def _new_future() -> asyncio.Future[None]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    return loop.create_future()


class ReactLoopAgent:
    """Drives one session through turn and step boundaries."""

    def __init__(
        self,
        loop_ctx: Context,
        id: SessionId,
        options: AgentOptions,
        session: Session,
    ) -> None:
        self._loop_ctx = loop_ctx
        self.id = id
        self.options = options
        self.session = session
        self._dispatch: AgentEventDispatch = agent_events(loop_ctx, self)
        self._notifications = _InboxNotifications(self._dispatch)
        self.inbox = Inbox(session, self._notifications)
        last_turn = next(
            (
                event['data']['turn']
                for event in reversed(session.events)
                if event['type'] == 'turn/start'
            ),
            0,
        )
        self._phase: _Phase = _IdlePhase(kind='idle', last_turn=last_turn)
        self._activity_done: asyncio.Future[None] = _new_future()
        self._activity_done.set_result(None)
        #: Strong references to driver tasks until they settle.
        self._background_tasks: set[asyncio.Future[None]] = set()
        #: The agent-scoped registration boundary; the lifecycle owner unwinds
        #: it after the driver exits.
        self.scope = create_scope(loop_ctx, self)
        self.ctx = self.scope.ctx.extend({'agent': self})
        #: Whether this loop instance has appended its initial/resume anchor.
        self._request_header_logged = False
        self._runtime_context = RuntimeContextProjection(self.ctx, session)

    @property
    def status(self) -> Literal['idle', 'running']:
        return 'idle' if self._phase.kind in ('idle', 'maintenance') else 'running'

    def _set_phase(self, next_phase: _Phase) -> None:
        """Commit a phase and publish its externally visible status transition."""
        previous_status = self.status
        self._phase = next_phase
        status = self.status
        if status != previous_status:
            self._dispatch.emit('agent/status', {'status': status})

    def send(self, message: Message, target: InboxTarget, wakeup: bool) -> None:
        """Route input to an inbox boundary; ``wakeup`` may wake the driver."""
        # Waking input cannot join an aborted activity, so it starts the next
        # turn. Captured before the insertion so a reentrant cancel from a
        # splice observer cannot reclassify it.
        waking_after_abort = (
            wakeup and self._phase.kind != 'idle' and self._phase.abort.signal.aborted
        )
        resolved_target: InboxTarget = 'next-turn' if waking_after_abort else target
        self.inbox.append(resolved_target, message)
        if wakeup:
            self._wake_driver(waking_after_abort)

    def followup(self, message: Message) -> None:
        """Queue an ordinary follow-up turn and wake the driver."""
        self.send(message, 'next-turn', True)

    def steer(self, message: Message) -> None:
        """Submit steering for the nearest step (an idle driver starts a turn)."""
        self.send(message, 'next-step', True)

    def inject(self, message: Message) -> None:
        """Queue model-facing context for the next pre-step without waking."""
        self.send(message, 'next-step', False)

    def cancel(
        self, cause: AgentCancelCause, options: CancelOptions | None = None
    ) -> None:
        """Clear queued and steering work — unless ``keep_inbox`` — and abort."""
        phase = self._phase
        if options is None or not options.keep_inbox:
            self.inbox.clear()
            if isinstance(phase, (_MaintenancePhase, _RunningPhase)):
                phase.wake_requested = False
        if isinstance(phase, (_MaintenancePhase, _RunningPhase)):
            phase.abort.abort(cause)

    def run_maintenance(
        self, job: Callable[[AbortSignal], Any]
    ) -> asyncio.Task[Any]:
        """Run one non-turn maintenance task from the true idle phase."""
        if self._phase.kind != 'idle':
            raise RuntimeError(f'agent "{self.id}" already has active work')
        done: asyncio.Future[None] = _new_future()
        maintenance = _MaintenancePhase(
            abort=LoopAbortController(),
            last_turn=self._phase.last_turn,
            wake_requested=False,
        )
        self._set_phase(maintenance)
        self._activity_done = done

        async def run() -> Any:
            try:
                return await maybe_await(job(maintenance.abort.signal))
            finally:
                self._set_phase(
                    _IdlePhase(kind='idle', last_turn=maintenance.last_turn)
                )
                if maintenance.wake_requested and self.inbox.has_pending:
                    self._wake_driver()
                if not done.done():
                    done.set_result(None)

        return asyncio.ensure_future(run())

    def _wake_driver(self, wake_after_abort: bool = False) -> None:
        """Start one driver, or latch its wake behind maintenance/abort."""
        if self._phase.kind != 'idle':
            # Maintenance and aborted drivers cannot deliver the wake: latch
            # it for replay at convergence. Live drivers claim queued work
            # themselves; disposal never latches, so teardown waits on no
            # model turn.
            phase = self._phase
            reason = phase.abort.signal.reason
            if (
                not (isinstance(reason, dict) and reason.get('kind') == 'disposed')
                and (phase.kind == 'maintenance' or wake_after_abort)
            ):
                phase.wake_requested = True
            return
        driver: asyncio.Future[None] = _new_future()
        self._activity_done = driver
        self._set_phase(
            _RunningPhase(
                abort=LoopAbortController(),
                turn=self._phase.last_turn,
                step=0,
                wake_requested=False,
            )
        )

        async def drive() -> None:
            try:
                await self._loop_ctx.agents.with_initiator(self, self.kick)
            except BaseException as error:
                if not driver.done():
                    driver.set_exception(error)
                    # A driver failure is contained at this boundary; nobody
                    # observes the future, so retrieve it immediately.
                    driver.exception()
                return
            if not driver.done():
                driver.set_result(None)

        task = asyncio.ensure_future(drive())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def when_idle(self) -> None:
        """Resolve after the current whole-agent activity reaches quiescence."""
        while True:
            activity = self._activity_done
            await activity
            if activity is self._activity_done:
                return

    def _throw_error(self, error: BaseException) -> NoReturn:
        """Report one failure at its live boundary, then preserve it for containment."""
        phase = self._phase
        turn = phase.turn if isinstance(phase, _RunningPhase) else phase.last_turn
        step = phase.step if isinstance(phase, _RunningPhase) else 0
        self._dispatch.emit('agent/error', {'turn': turn, 'step': step, 'error': error})
        raise error

    async def kick(self) -> None:
        """The driver body: run turns until the inbox stays drained."""
        try:
            while await self.turn():
                pass
        except Exception:
            # Reported failures and cancellation are contained at the driver
            # boundary.
            pass
        finally:
            phase = self._phase
            if isinstance(phase, _RunningPhase):
                self._set_phase(_IdlePhase(kind='idle', last_turn=phase.turn))
                if phase.wake_requested and self.inbox.has_pending:
                    self._wake_driver()

    async def _pre_step(
        self, target: InboxTarget, position: dict[str, int]
    ) -> _PreparedStep:
        if self._phase.kind != 'running':
            raise RuntimeError(f'agent "{self.id}": pre-step outside running phase')
        signal = self._phase.abort.signal
        claimed = self.inbox.claim(target, position['turn'])
        assembly = await self._loop_ctx.systemPrompt.assemble(
            assemble_context_for(self, signal)
        )
        signal.throw_if_aborted()
        sections = render_context_sections(assembly)
        context = self._runtime_context.project(
            join_context_sections(sections), sections
        )

        async def default(*_args: Any) -> PreStepEnter:
            return PreStepEnter(
                messages=[*claimed, context] if context is not None else list(claimed)
            )

        decision: PreStepEnter | PreStepReject = await maybe_await(
            self._dispatch.waterfall(
                'agent/pre-step',
                {
                    'messages': claimed,
                    'turn': position['turn'],
                    'step': position['step'],
                    'signal': signal,
                },
                default,
            )
        )
        signal.throw_if_aborted()
        if isinstance(decision, PreStepReject):
            return decision
        return _PreparedEnter(messages=decision.messages, assembly=assembly)

    async def turn(self) -> bool:
        """Open one turn before claiming its first proposed step."""
        if self._phase.kind != 'running':
            self._throw_error(
                RuntimeError(f'agent "{self.id}": turn without driver reservation')
            )
        phase = self._phase
        signal = phase.abort.signal
        signal.throw_if_aborted()
        turn = phase.turn + 1
        try:
            self.session.append('turn/start', {'turn': turn})
        except Exception as error:
            self._throw_error(error)
        phase.turn = turn
        turn_ends: dict[str, Any] | None = None
        target: InboxTarget = 'next-turn'
        try:
            while True:
                signal.throw_if_aborted()
                step = phase.step + 1
                decision = await self._pre_step(target, {'turn': turn, 'step': step})
                if decision.kind == 'reject':
                    turn_ends = {'kind': 'blocked'}
                    return False
                if turn_ends and len(decision.messages) == 0:
                    break
                # A removed waking message or an enter decision rewritten to
                # empty still owns the initial turn boundary, but it spends no
                # model call.
                if phase.step == 0 and len(decision.messages) == 0:
                    turn_ends = {'kind': 'completed'}
                    return False
                signal.throw_if_aborted()
                self.session.append('step/start', {'turn': turn, 'step': step})
                phase.step = step
                try:
                    for message in decision.messages:
                        self.session.append(
                            'user/message', message, {'surfaceOp': 'append'}
                        )
                    # max-tokens is sticky: once any step hits the ceiling,
                    # later steps that complete normally must not downgrade
                    # the turn outcome.
                    step_end = await self._step(decision.assembly)
                    if turn_ends is None or turn_ends.get('kind') != 'max-tokens':
                        turn_ends = step_end
                finally:
                    self.session.append('step/end', {'turn': turn, 'step': step})
                signal.throw_if_aborted()
                if turn_ends and len(self.inbox.next_step) == 0:
                    await self._dispatch.serial(
                        'agent/turn-stopping', {'turn': turn, 'signal': signal}
                    )
                    signal.throw_if_aborted()
                if turn_ends and len(self.inbox.next_step) == 0:
                    break
                target = 'next-step'
        except Exception as error:
            if signal.aborted:
                turn_ends = {'kind': 'aborted', 'reason': signal.reason}
                raise
            # Every failure is structured: an ``LlmError`` keeps its facts,
            # anything else flattens to ``error_chain`` text under UNKNOWN.
            failure: LlmFailure = (
                cast(LlmFailure, dict(error.failure))
                if isinstance(error, LlmError)
                else {'message': error_chain(error), 'code': 'UNKNOWN'}
            )
            turn_ends = {'kind': 'error', 'error': failure}
            self._throw_error(error)
        finally:
            try:
                # Every exit above assigns a turn ending.
                self.session.append(
                    'turn/end', {'turn': turn, 'reason': cast(Any, turn_ends)}
                )
            except Exception as error:
                self._throw_error(error)
        if not self.inbox.has_pending:
            return False
        phase.abort = LoopAbortController()
        # A fresh controller makes a latch set on the old one stale: the live
        # driver claims the queue itself.
        phase.wake_requested = False
        phase.step = 0
        return True

    async def _step(self, assembly: PromptAssembly) -> _StepEndReason | None:
        """Run one step: model call plus the tool executions it requested."""
        if self._phase.kind != 'running':
            raise RuntimeError(f'agent "{self.id}": step outside running phase')
        phase = self._phase
        turn = phase.turn
        step = phase.step
        signal = phase.abort.signal
        signal.throw_if_aborted()
        system = render_prompt(assembly)

        while True:
            request = await self._build_request(
                turn,
                step,
                assembly.tools,
                system,
                self.session.derive_messages(),
                signal,
            )
            assembler = BlockAssembler()
            chunk_seqs: list[int] = []
            stream = self._loop_ctx.llm.stream(request)
            signal.throw_if_aborted()
            async for chunk in stream:
                signal.throw_if_aborted()
                event = self.session.append(
                    'assistant/chunk', {'turn': turn, 'step': step, 'chunk': chunk}
                )
                chunk_seqs.append(event['seq'])
                assembler.push(chunk)
            signal.throw_if_aborted()
            finish = assembler.finish
            if finish['kind'] in ('error', 'aborted'):
                failure = finish['failure']  # type: ignore[typeddict-item,unused-ignore]

                async def no_recovery(*_args: Any) -> None:
                    return None

                action = await maybe_await(
                    self._dispatch.waterfall(
                        'agent/request-error',
                        {
                            'turn': turn,
                            'step': step,
                            'provider': request['provider'],
                            'failure': failure,
                            'retry_policy': self._retry_policy(request['provider']),
                            'signal': signal,
                        },
                        no_recovery,
                    )
                )
                signal.throw_if_aborted()
                if not (action is not None and action.kind == 'retry'):
                    raise LlmError(
                        failure['message'],
                        failure['code'],
                        status=failure.get('status'),
                        provider_retry_after_ms=failure.get('providerRetryAfterMs'),
                        request_id=failure.get('requestId'),
                    )
                continue

            has_replay_state, replay_state = assembler.replay_state
            message = create_assistant_message(
                assembler.blocks(),
                provider=request['provider'],
                model=request['model'],
                replay_state=replay_state,
                has_replay_state=has_replay_state,
            )
            data: dict[str, Any] = {'turn': turn, 'step': step, 'message': message}
            if assembler.usage is not None:
                data['usage'] = assembler.usage
            self.session.append(
                'assistant/message',
                data,
                {'surfaceOp': 'append', 'sourceEventSeqs': chunk_seqs},
            )
            if finish['kind'] == 'max-tokens':
                return {'kind': 'max-tokens'}

            tool_call_blocks = [
                block for block in message['content'] if block['type'] == 'tool-call'
            ]
            if len(tool_call_blocks) == 0:
                return {'kind': 'completed'}
            def accept_context(context: Message) -> None:
                self.inbox.splice('next-step', len(self.inbox.next_step), 0, [context])

            concluded = await execute_tool_calls(
                self._loop_ctx,
                turn,
                step,
                cast(list[ToolCallBlock], tool_call_blocks),
                signal,
                accept_context,
            )
            return {'kind': 'completed'} if concluded else None

    def _retry_policy(self, provider: str) -> ResolvedRetryPolicy | None:
        """The provider route's captured retry policy, if an adapter owns it."""
        try:
            policy: ResolvedRetryPolicy = self._loop_ctx.llm.provider_retry_policy(
                provider
            )
            return policy
        except LlmError:
            return None

    async def _build_request(
        self,
        turn: int,
        step: int,
        tools: list[PromptToolSchema],
        system: str,
        boundary_messages: list[Message],
        signal: LoopAbortSignal,
    ) -> GenerateOptions:
        """Compose one frozen request after the ``agent/request`` waterfall.

        TS resolves the exact-model adapter defaults through
        ``llm.prepareCall``; the Python llm runtime has no prepared-call
        object, so this port uses ``resolve_call_config`` and computes the
        adapter-default markers by diffing the proposed and resolved configs
        (``NO_ADAPTER`` still falls back to the proposed config).
        """
        session = self.session

        # A loop instance starts from its declared route, restoring only an
        # explicit effort owned by that exact model. Later steps re-resolve
        # marked defaults.
        persisted_header = session.request_header()
        persisted_config = persisted_header['config'] if persisted_header else None
        route = {
            'provider': self.options.provider or '',
            'model': self.options.model or '',
        }
        reasoning_effort = (
            persisted_config.get('reasoningEffort')
            if persisted_config is not None
            and persisted_config.get('provider') == route['provider']
            and persisted_config.get('model') == route['model']
            and persisted_header is not None
            and persisted_header.get('adapterDefaults', {}).get('reasoningEffort')
            is not True
            else None
        )
        if self._request_header_logged and persisted_header is not None:
            seed_source: dict[str, Any] = _request_proposal(persisted_header)
        else:
            seed_source = dict(route)
            if reasoning_effort is not None:
                seed_source['reasoningEffort'] = reasoning_effort
            if self.options.max_tokens is not None:
                seed_source['maxTokens'] = self.options.max_tokens
        seed_config: dict[str, Any] = detached(seed_source)

        async def seed(*_args: Any) -> dict[str, Any]:
            return seed_config

        proposed_config: dict[str, Any] = await maybe_await(
            self._dispatch.waterfall(
                'agent/request', {'turn': turn, 'step': step, 'signal': signal}, seed
            )
        )
        signal.throw_if_aborted()
        if not proposed_config.get('provider') or not proposed_config.get('model'):
            raise RuntimeError(
                f'agent "{self.id}" has no provider/model: set'
                ' AgentOptions.provider and AgentOptions.model or supply both'
                ' via the agent/request waterfall'
            )
        adapter_defaults: dict[str, bool] = {}
        try:
            config: dict[str, Any] = self._loop_ctx.llm.resolve_call_config(
                proposed_config
            )
            for key in ('reasoningEffort', 'maxTokens'):
                if (
                    proposed_config.get(key) is None
                    and config.get(key) is not None
                ):
                    adapter_defaults[key] = True
        except LlmError as error:
            # Middleware may serve an unregistered route; terminal dispatch
            # still requires an adapter.
            if error.code != 'NO_ADAPTER':
                raise
            config = proposed_config
        signal.throw_if_aborted()

        header_input: dict[str, Any] = {'config': config}
        if adapter_defaults:
            header_input['adapterDefaults'] = adapter_defaults
        if system:
            header_input['system'] = system
        if len(tools) > 0:
            header_input['tools'] = [
                {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters,
                }
                for tool in tools
            ]
        header = canonical_header(header_input)  # type: ignore[arg-type]
        baseline = session.request_header()
        if not self._request_header_logged:
            session.append(
                'request/header',
                {
                    'header': header,
                    'reason': 'initial' if baseline is None else 'resume',
                },
            )
            self._request_header_logged = True
        elif baseline is None or not header_equals(baseline, header):
            session.append('request/header', {'header': header, 'reason': 'change'})

        context_window = self._context_window(config['provider'], config['model'])
        request_context: dict[str, Any] = {
            'provider': config['provider'],
            'model': config['model'],
        }
        if context_window is not None:
            request_context['contextWindow'] = context_window
        previous_context = session.request_context()
        if (
            previous_context is None
            or previous_context['provider'] != request_context['provider']
            or previous_context['model'] != request_context['model']
            or previous_context.get('contextWindow') != context_window
        ):
            session.append('request/context', request_context)  # type: ignore[arg-type,unused-ignore]
        signal.throw_if_aborted()

        request = GenerateOptions({
            **header['config'],
            'messages': boundary_messages,
            **({'system': header['system']} if 'system' in header else {}),
            **({'tools': header['tools']} if 'tools' in header else {}),
            'sessionId': session.id,
            'signal': signal,
        })
        return mark_agent_loop_request(request)

    def _context_window(self, provider: str, model: str) -> int | None:
        """The exact route's advertised context window, when resolvable."""
        try:
            info = self._loop_ctx.llm.resolve_model_info(provider, model)
        except LlmError:
            return None
        context = info.get('context')
        return context['contextWindow'] if context is not None else None


class _InboxNotifications(InboxNotifications):
    """Publishes inbox mutations through the fused dispatcher."""

    def __init__(self, dispatch: AgentEventDispatch) -> None:
        self._dispatch = dispatch

    def inserted(self, message: Message) -> None:
        self._dispatch.emit('agent/inbox/inserted', {'message': message})

    def discarded(self, message: Message) -> None:
        self._dispatch.emit('agent/inbox/discarded', {'message': message})

    def claimed(self, message: Message, turn: int) -> None:
        self._dispatch.emit('agent/inbox/claimed', {'message': message, 'turn': turn})
