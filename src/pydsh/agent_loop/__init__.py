"""Concrete agent-loop plugin: creates scoped ReactLoopAgents, publishes them
through the agent/session registries, and owns their ordered teardown.

Ported from deepseek-harness ``packages/core/agent-loop/src/index.ts``
(MIT).

Simplifications from the TypeScript original:

- The ``settingsNamespace('agent-loop')`` settings seam is not ported (no
  settings capability exists on the Python side). ``maxParallelToolCalls``
  is a plugin ``Config`` field validated at load (fail loud) and frozen for
  the service lifetime; the scheduler still re-reads it at every tool group,
  so a later settings wiring only needs to swap the value's source.
- ``LlmRuntime.prepareCall`` does not exist in ``pydsh.llm``; the request
  builder uses ``resolve_call_config`` and diffs proposed/resolved configs
  for the adapter-default markers (see ``agent.py``).
- Config keys follow the local Python convention (snake_case field names in
  the raw mapping match the dataclass fields).

Asyncio mapping: ``Promise.withResolvers`` becomes ``asyncio`` futures,
``AbortSignal.any`` becomes listener fusion over the
``pydsh.subprocess.types`` DOM-primitive port, and the fiber teardown effect
returns the ownership's dispose task.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, TypeVar
from uuid import uuid4

from pycordis import Context, Fiber, FiberState, Service, ValidationError
from pycordis.utils import maybe_await
from pydsh.agent import (
    Agent,
    AgentHandle,
    AgentOptions,
    AgentSetup,
    AgentSetupCommit,
    CreateAgentOptions,
    ResumeAgentOptions,
    SessionStartSource,
    emit_agent_event,
)
from pydsh.session import (
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionId,
    SessionPreparation,
)
from pydsh.subprocess.types import AbortController, AbortSignal

from .agent import LoopAbortController, LoopAbortSignal, ReactLoopAgent
from .constants import DEFAULT_MAX_PARALLEL_TOOL_CALLS
from .llm_helpers import error_chain

__all__ = [
    'CONFIGURED_AGENT_IDENTITIES_KEY',
    'DEFAULT_MAX_PARALLEL_TOOL_CALLS',
    'AgentLoop',
    'Config',
    'ConfiguredAgent',
    'ConfiguredAgentIdentities',
    'LauncherAgentIdentity',
    'SessionPersistence',
]

_T = TypeVar('_T')

#: Fiber states that cannot own or serve a new lifecycle.
_INACTIVE_STATES = frozenset({
    FiberState.UNLOADING,
    FiberState.DISPOSED,
    FiberState.FAILED,
})

#: Context key a launcher sets before any loader entry mounts
#: (``ctx.provide(CONFIGURED_AGENT_IDENTITIES_KEY, identities)``) to fix
#: configured agents' session identities without a config key.
CONFIGURED_AGENT_IDENTITIES_KEY = 'configuredAgentIdentities'


class SessionPersistence(Protocol):
    """The optional ``sessionPersistence`` service face the resume path needs.

    A persistence backend plugin provides this service; the loop only reads
    it, so the structural contract is enough.
    """

    def prepare(self, id: SessionId, signal: AbortSignal) -> Any:
        """Load one persisted session as an awaitable SessionPreparation."""
        ...

    def list(self) -> Any:
        """Awaitable list of persisted session headers."""
        ...


@dataclass(frozen=True)
class LauncherAgentIdentity:
    """One launcher-selected session identity for a configured agent.

    ``resume`` distinguishes rehydrating existing persisted history from
    creating the session fresh under that exact id.
    """

    #: Exact session id to create fresh or resume.
    id: SessionId
    #: Resume existing persisted history instead of creating fresh.
    resume: bool


#: Launcher-selected identities keyed by the configured agent's ``id``.
ConfiguredAgentIdentities = dict[str, LauncherAgentIdentity]


@dataclass(frozen=True)
class ConfiguredAgent:
    """One declarative agent entry created or resumed at plugin startup."""

    #: Stable config label used in logs and as the fresh combined-id prefix.
    id: str
    provider: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    #: Optional stable identity; remounts resume its materialized history,
    #: while first use creates it fresh.
    session_id: SessionId | None = None
    #: Optional workspace for a fresh session.
    cwd: str | None = None
    #: Persisted session to resume instead of creating a fresh session.
    resume_session_id: SessionId | None = None


@dataclass(frozen=True)
class Config:
    """Agent-loop plugin configuration."""

    #: Maximum parallel-safe calls in flight per agent step. ``1`` is serial.
    max_parallel_tool_calls: int = DEFAULT_MAX_PARALLEL_TOOL_CALLS
    #: Agents created or resumed at plugin startup.
    agents: tuple[ConfiguredAgent, ...] = ()


_AGENT_ENTRY_KEYS = frozenset({
    'id',
    'provider',
    'model',
    'max_tokens',
    'session_id',
    'cwd',
    'resume_session_id',
})


def _resolve_max_parallel_tool_calls(value: Any) -> int:
    """Resolve the deployment-wide scheduler cap at the config boundary."""
    cap = DEFAULT_MAX_PARALLEL_TOOL_CALLS if value is None else value
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValidationError('maxParallelToolCalls must be a positive integer')
    return cap


def _resolve_agent_entry(index: int, raw: Any) -> ConfiguredAgent:
    """Validate one declarative agent entry, failing loud at load."""
    if not isinstance(raw, dict):
        raise ValidationError(f'agents[{index}] must be a mapping')
    unknown = sorted(set(raw) - _AGENT_ENTRY_KEYS)
    if unknown:
        raise ValidationError(f'agents[{index}]: unknown keys: {", ".join(unknown)}')
    agent_id = raw.get('id')
    if not isinstance(agent_id, str) or not agent_id:
        raise ValidationError(f'agents[{index}].id must be a non-empty string')
    session_id = raw.get('session_id')
    resume_session_id = raw.get('resume_session_id')
    if session_id and resume_session_id:
        raise ValidationError(
            f'agent "{agent_id}": session_id and resume_session_id are'
            ' mutually exclusive'
        )
    max_tokens = raw.get('max_tokens')
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
    ):
        raise ValidationError(f'agent "{agent_id}": max_tokens must be an integer')
    return ConfiguredAgent(
        id=agent_id,
        provider=raw.get('provider'),
        model=raw.get('model'),
        max_tokens=max_tokens,
        session_id=SessionId(session_id) if session_id else None,
        cwd=raw.get('cwd'),
        resume_session_id=SessionId(resume_session_id) if resume_session_id else None,
    )


def _validate_configured_agents(agents: Sequence[ConfiguredAgent]) -> None:
    """Reject duplicate exact identities before any configured agent starts."""
    exact: dict[SessionId, str] = {}
    for agent in agents:
        identity = agent.resume_session_id or agent.session_id
        if identity is None:
            continue
        first = exact.get(identity)
        if first is not None:
            raise ValidationError(
                f'agents "{first}" and "{agent.id}" use duplicate exact session'
                f' identity "{identity}"'
            )
        exact[identity] = agent.id


def _resolve_config(config: Config | dict[str, Any] | None) -> Config:
    """Validate a raw plugin config mapping into a ``Config``.

    Installed as ``AgentLoop.Config`` so the Cordis loader applies it; also
    accepts an already-built ``Config`` for direct construction.
    """
    if config is None:
        return Config()
    if isinstance(config, Config):
        _validate_configured_agents(config.agents)
        return config
    if not isinstance(config, dict):
        raise ValidationError(f'invalid agentLoop config: {type(config).__name__}')
    unknown = sorted(set(config) - {'max_parallel_tool_calls', 'agents'})
    if unknown:
        raise ValidationError(f'unknown agentLoop config keys: {", ".join(unknown)}')
    agents = tuple(
        _resolve_agent_entry(index, entry)
        for index, entry in enumerate(config.get('agents') or [])
    )
    _validate_configured_agents(agents)
    return Config(
        max_parallel_tool_calls=_resolve_max_parallel_tool_calls(
            config.get('max_parallel_tool_calls')
        ),
        agents=agents,
    )


def _apply_launcher_identities(
    agents: tuple[ConfiguredAgent, ...],
    identities: ConfiguredAgentIdentities | None,
) -> tuple[ConfiguredAgent, ...]:
    """Apply launcher-owned identities over the configured agents."""
    if identities is None:
        return agents
    applied: list[ConfiguredAgent] = []
    for agent in agents:
        identity = identities.get(agent.id)
        if identity is None:
            applied.append(agent)
            continue
        if identity.resume:
            applied.append(
                ConfiguredAgent(
                    id=agent.id,
                    provider=agent.provider,
                    model=agent.model,
                    max_tokens=agent.max_tokens,
                    cwd=agent.cwd,
                    resume_session_id=identity.id,
                )
            )
        else:
            applied.append(
                ConfiguredAgent(
                    id=agent.id,
                    provider=agent.provider,
                    model=agent.model,
                    max_tokens=agent.max_tokens,
                    cwd=agent.cwd,
                    session_id=identity.id,
                )
            )
    return tuple(applied)


def _abort_reason(signal: AbortSignal, id: SessionId) -> BaseException:
    """The reason a fused setup abort raises (always an exception)."""
    reason = signal.reason
    if isinstance(reason, BaseException):
        return reason
    return RuntimeError(f'agent "{id}" creation aborted')


def _silence(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned task's exception so the loop never warns."""
    if task.cancelled():
        return
    task.exception()


async def _await_value(value: Any) -> Any:
    """Normalize a plain value or awaitable into an awaitable."""
    return await maybe_await(value)


async def race_abort(
    operation: Any, signal: AbortSignal, id: SessionId
) -> Any:
    """Await ``operation``, or raise the signal's reason as soon as it aborts.

    The operation keeps running when the abort wins (TS ``Promise.race``
    semantics); its late settlement is silenced here, and callers that need
    the late value use :func:`race_abort_call`.
    """
    if signal.aborted:
        raise _abort_reason(signal, id)
    task = operation if isinstance(operation, asyncio.Future) else None
    if task is None:
        task = asyncio.ensure_future(_await_value(operation))
    aborted: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def on_abort() -> None:
        if not aborted.done():
            aborted.set_result(None)

    signal.add_listener(on_abort)
    try:
        await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
        if aborted.done() and not task.done():
            task.add_done_callback(_silence)
            raise _abort_reason(signal, id)
        return await task
    finally:
        signal.remove_listener(on_abort)


async def race_abort_call(
    operation: Callable[[], Any],
    signal: AbortSignal,
    id: SessionId,
    release_abandoned: Callable[[Any], None] | None = None,
) -> Any:
    """Start an abortable operation and release a value arriving after cancellation."""
    if signal.aborted:
        raise _abort_reason(signal, id)
    pending = asyncio.ensure_future(_await_value(operation()))
    try:
        return await race_abort(pending, signal, id)
    except BaseException:
        if signal.aborted and release_abandoned is not None:

            def release(task: asyncio.Future[Any]) -> None:
                if task.cancelled() or task.exception() is not None:
                    return
                release_abandoned(task.result())

            pending.add_done_callback(release)
        raise


class FactoryOwnership:
    """Factory-level ownership: live agent teardowns plus config startup work."""

    def __init__(self, fiber: Fiber) -> None:
        self._fiber = fiber
        self._accepting = True
        self._teardown = AbortController()
        self._inactive: asyncio.Future[None] | None = None
        self._live_agents: set[Callable[[], Any]] = set()
        self._startup_tasks: set[asyncio.Future[Any]] = set()

    @property
    def signal(self) -> AbortSignal:
        """Abort on factory teardown (reason: `agent loop is not active`)."""
        return self._teardown.signal

    def is_active(self) -> bool:
        return self._accepting and self._fiber.state not in _INACTIVE_STATES

    def track(self, dispose: Callable[[], Any]) -> Callable[[], None]:
        """Track one live agent's shared teardown until it has run."""
        self._live_agents.add(dispose)

        def untrack() -> None:
            self._live_agents.discard(dispose)

        return untrack

    def track_startup(self, job: asyncio.Future[Any]) -> None:
        """Join config startup work that begins before an agent exists."""
        self._startup_tasks.add(job)
        job.add_done_callback(self._startup_tasks.discard)

    def track_wrapper(self, job: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
        """Join one public create/resume continuation.

        Factory dispose awaits its settlement; the caller still owns the
        result (the tracked observation swallows settlement, not the task).
        """
        task = asyncio.ensure_future(job)

        async def settled() -> None:
            try:
                await task
            except BaseException:
                pass

        self.track_startup(asyncio.ensure_future(settled()))
        return task

    def _inactive_future(self) -> asyncio.Future[None]:
        if self._inactive is None:
            self._inactive = asyncio.get_event_loop().create_future()
        return self._inactive

    async def wait_while_active(self, job: Awaitable[None]) -> None:
        """Resolve ``job``, or stop waiting when factory teardown begins."""
        task = (
            job
            if isinstance(job, asyncio.Future)
            else asyncio.ensure_future(_await_value(job))
        )
        await asyncio.wait(
            {task, self._inactive_future()}, return_when=asyncio.FIRST_COMPLETED
        )

    async def dispose(self) -> None:
        self._accepting = False
        self._teardown.abort(RuntimeError('agent loop is not active'))
        inactive = self._inactive_future()
        if not inactive.done():
            inactive.set_result(None)
        await asyncio.gather(
            *(dispose() for dispose in list(self._live_agents)),
            *self._startup_tasks,
        )


@dataclass
class _PreparedAgent:
    """Prepared-but-unpublished agent resources sharing one memoized teardown."""

    agent: ReactLoopAgent
    #: Aborts when the factory unloads, the caller cancels, or teardown
    #: begins — ends any setup await.
    signal: LoopAbortSignal
    #: Enter registries, announce, notify session-start, and start the machine.
    publish: Callable[[SessionStartSource], AgentHandle]
    #: Reverse teardown: stop the machine, unregister, unwind the scope.
    dispose: Callable[[], asyncio.Future[None]]


#: Alias for annotations inside ``AgentLoop``, whose ``Config`` class
#: attribute (the loader validator) shadows the dataclass name in class scope.
_ConfigData = Config


class AgentLoop(Service):
    """Concrete agent factory and driver service (``ctx.agentLoop``)."""

    inject: ClassVar[list[str]] = ['agents', 'sessions', 'llm', 'tools', 'systemPrompt']

    #: Cordis loader config hook: raw mapping -> validated ``Config``.
    Config = staticmethod(_resolve_config)

    def __init__(
        self, ctx: Context, config: _ConfigData | dict[str, Any] | None = None
    ) -> None:
        super().__init__(ctx, 'agentLoop')
        resolved = _resolve_config(config)
        identities: ConfiguredAgentIdentities | None = ctx.get(
            CONFIGURED_AGENT_IDENTITIES_KEY
        )
        agents = _apply_launcher_identities(resolved.agents, identities)
        _validate_configured_agents(agents)
        #: Validated configuration owned by the agent-loop service.
        self.config = Config(
            max_parallel_tool_calls=resolved.max_parallel_tool_calls,
            agents=agents,
        )
        self._ownership = FactoryOwnership(ctx.fiber)
        #: Plain holder preventing re-tracing through a caller shadow.
        self._runtime: dict[str, Context] = {'ctx': ctx}
        #: Strong references to contained fire-and-forget tasks.
        self._background_tasks: set[asyncio.Future[Any]] = set()
        ctx.effect(self._dispose_ownership, 'agentLoop.transactions()')
        ctx.effect(self._register_factory, 'agentLoop.setFactory()')
        ctx.systemPrompt.variable(
            'provider',
            lambda context: (agent := getattr(context, 'agent', None))
            and agent.options.provider,
        )
        ctx.systemPrompt.variable(
            'model',
            lambda context: (agent := getattr(context, 'agent', None))
            and agent.options.model,
        )
        ctx.systemPrompt.variable(
            'cwd',
            lambda context: (agent := getattr(context, 'agent', None))
            and agent.session.header.get('cwd'),
        )

        for entry in self.config.agents:
            options = AgentOptions(
                provider=entry.provider,
                model=entry.model,
                max_tokens=entry.max_tokens,
            )
            meta = (
                CreateSessionMeta()
                if entry.cwd is None
                else CreateSessionMeta(cwd=entry.cwd)
            )
            if entry.resume_session_id is None or entry.resume_session_id == '':
                configured_id = entry.session_id or SessionId(
                    f'{entry.id}-session-{uuid4()}'
                )
                persistence = (
                    None
                    if entry.session_id is None
                    else ctx.get('sessionPersistence')
                )
                if persistence is None:
                    self.create(configured_id, options, meta)
                else:
                    startup = asyncio.ensure_future(
                        self._restore_or_create_configured(
                            ctx, persistence, configured_id, options, meta
                        )
                    )

                    def report(
                        error: BaseException,
                        _entry: ConfiguredAgent = entry,
                        _id: SessionId = configured_id,
                    ) -> None:
                        self._report_configured_startup_failure(
                            _entry.id, 'restore', _id, error
                        )

                    def observe(
                        task: asyncio.Future[Any], _report: Any = report
                    ) -> None:
                        if task.cancelled():
                            return
                        error = task.exception()
                        if error is not None:
                            _report(error)

                    startup.add_done_callback(observe)
                    self._ownership.track_startup(startup)
                continue
            resume_session_id = entry.resume_session_id

            def resume_effect(
                _resume_session_id: SessionId = resume_session_id,
                _options: AgentOptions = options,
                _entry_id: str = entry.id,
            ) -> Any:
                fiber = ctx.inject(
                    ['sessionPersistence'],
                    lambda child_ctx: self._start_configured_resume(
                        child_ctx, _resume_session_id, _options, _entry_id
                    ),
                )
                return fiber.dispose

            ctx.effect(resume_effect, f'agentLoop.resume({entry.id})')

    def _start_configured_resume(
        self,
        child_ctx: Context,
        resume_session_id: SessionId,
        options: AgentOptions,
        entry_id: str,
    ) -> None:
        """Begin a deferred config-driven resume once persistence is available."""
        async def run() -> None:
            try:
                await self._resume_with(
                    self.ctx,
                    child_ctx.get('sessionPersistence'),
                    ResumeAgentOptions(
                        resume_session_id=resume_session_id, agent_options=options
                    ),
                )
            except Exception as error:
                self._report_configured_startup_failure(
                    entry_id, 'resume', resume_session_id, error
                )

        self._schedule(run())

    def _schedule(self, awaitable: Any) -> None:
        """Run a contained fire-and-forget task, holding a strong reference."""
        task = asyncio.ensure_future(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _dispose_ownership(self) -> Callable[[], Awaitable[None]]:
        return self._ownership.dispose

    def _register_factory(self) -> Callable[[], Any]:
        # The exact single-shot effect disposer, identity-nested as this
        # effect's cleanup (the ``agents.register()`` convention).
        disposer: Callable[[], Any] = self.ctx.agents.set_factory(self)
        return disposer

    def _report_configured_startup_failure(
        self,
        config_id: str,
        action: Literal['restore', 'resume'],
        session_id: SessionId,
        error: BaseException,
    ) -> None:
        """Report a contained declarative-start failure to identity-bound consumers."""
        if not self._ownership.is_active():
            return
        self.ctx.logger.warn(
            f'agent "{config_id}": config-driven {action} of "{session_id}"'
            f' failed: {error_chain(error)}'
        )
        args: list[Any] = [
            'agent-loop/config-start-failed',
            {'sessionId': session_id, 'error': error},
        ]
        for callback in self.ctx.events.dispatch('emit', args):
            try:
                returned = callback(*args)
            except Exception as listener_error:
                self.ctx.logger.warn(
                    f'agent "{config_id}": config-start-failed listener threw:'
                    f' {error_chain(listener_error)}'
                )
                continue
            if inspect.isawaitable(returned):

                async def observe(
                    awaitable: Any = returned, _config_id: str = config_id
                ) -> None:
                    try:
                        await awaitable
                    except Exception as listener_error:
                        self.ctx.logger.warn(
                            f'agent "{_config_id}": config-start-failed listener'
                            f' rejected: {error_chain(listener_error)}'
                        )

                self._schedule(observe())

    async def _restore_or_create_configured(
        self,
        owner_ctx: Context,
        persistence: SessionPersistence,
        session_id: SessionId,
        agent_options: AgentOptions,
        meta: CreateSessionMeta,
    ) -> None:
        """Restore a materialized exact config identity, or create it on first use."""
        await self._wait_for_draining_configured_identity(owner_ctx, session_id)
        if not self._ownership.is_active():
            return
        try:
            await self._resume_with(
                owner_ctx,
                persistence,
                ResumeAgentOptions(
                    resume_session_id=session_id, agent_options=agent_options
                ),
            )
            return
        except Exception:
            if not self._ownership.is_active():
                return
            # A load is the per-id serialization barrier for eager
            # write-behind and lifecycle retirement. Only a genuinely absent
            # artifact falls back to first creation; corruption and backend
            # failures stay loud.
            exists = any(
                header['id'] == session_id
                for header in await maybe_await(persistence.list())
            )
            if exists:
                raise
        self.create(session_id, agent_options, meta)

    async def _wait_for_draining_configured_identity(
        self, owner_ctx: Context, session_id: SessionId
    ) -> None:
        """Wait for a draining same-id lifecycle to finish registry teardown."""
        # Only an id still occupying a registry needs waiting for; a live
        # healthy occupant is a collision the create/resume below surfaces.
        if (
            owner_ctx.agents.get(session_id) is None
            and owner_ctx.sessions.get(session_id) is None
        ):
            return
        released: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def check_released(*_args: Any) -> None:
            if (
                owner_ctx.agents.get(session_id) is None
                and owner_ctx.sessions.get(session_id) is None
                and not released.done()
            ):
                released.set_result(None)

        dispose_agent_listener = owner_ctx.on('agent/disposed', check_released)
        dispose_session_listener = owner_ctx.on('session/disposed', check_released)
        try:
            check_released()
            await self._ownership.wait_while_active(released)
        finally:
            dispose_agent_listener()
            dispose_session_listener()

    def _prepare(
        self,
        owner_ctx: Context,
        id: SessionId,
        options: AgentOptions,
        session: Session,
        caller_signal: AbortSignal | None = None,
    ) -> _PreparedAgent:
        """Construct the driver, scope, and one memoized reverse teardown.

        The teardown is registered with the factory and the owner fiber
        BEFORE publication, so a mid-setup unload rolls everything back;
        ``signal`` fuses caller cancellation with lifecycle teardown for
        setup awaits.
        """
        _assert_agent_options(options)
        owner_ctx.fiber.assert_active()
        # Every caller reaches _prepare() synchronously from a service method
        # whose Cordis dispatch already requires the live factory fiber, or
        # re-checks ownership itself after its awaits (resume's load barrier).
        if not self._ownership.is_active():
            raise RuntimeError('agent loop is not active')
        if caller_signal is not None and caller_signal.aborted:
            raise _abort_reason(caller_signal, id)
        loop_ctx = self._runtime['ctx']

        # Deactivation fuses three owners, each with its own reason: the
        # caller's cancellation signal, the owner fiber's unload, and factory
        # teardown. It is registered BEFORE any resource exists, over mutable
        # slots, so an unload arriving while the scope is still minting finds
        # a working disposer instead of a leak.
        abort = LoopAbortController()

        def on_caller_abort() -> None:
            reason = caller_signal.reason if caller_signal is not None else None
            if isinstance(reason, BaseException):
                abort.abort(reason)
            else:
                abort.abort(RuntimeError(f'agent "{id}" creation aborted'))

        def on_factory_teardown() -> None:
            abort.abort(self._ownership.signal.reason)

        if caller_signal is not None:
            caller_signal.add_listener(on_caller_abort)
        self._ownership.signal.add_listener(on_factory_teardown)

        machine: ReactLoopAgent | None = None
        detach_session: Callable[[], None] | None = None
        detach_agent: Callable[[], None] | None = None
        disposing: asyncio.Future[None] | None = None
        # Waiters for construction (normally none: construction is synchronous
        # and dispose can only run at a later await point).
        machine_ready: list[asyncio.Future[None]] = []

        async def do_dispose(owner_triggered: bool) -> None:
            # Reverse teardown, memoized so every racing owner awaits one
            # quiescence: stop the machine, leave the registries, unwind the
            # scope, release bookkeeping.
            abort.abort(RuntimeError(f'agent "{id}" lifecycle disposed'))
            if caller_signal is not None:
                caller_signal.remove_listener(on_caller_abort)
            self._ownership.signal.remove_listener(on_factory_teardown)
            try:
                # Disposal IS a disposed-cause cancel followed by quiescence.
                # New work sent after this point is the sender's bug — the
                # registries are about to drop the agent.
                if machine is None:
                    waiter = asyncio.get_running_loop().create_future()
                    machine_ready.append(waiter)
                    await waiter
                if machine is not None:
                    machine.cancel({'kind': 'disposed'})
                    await machine.when_idle()
                    await machine.scope.dispose()
            finally:
                try:
                    if detach_agent is not None:
                        detach_agent()
                    if detach_session is not None:
                        detach_session()
                finally:
                    untrack()
                    if not owner_triggered:
                        await maybe_await(unfollow_owner())

        def dispose(owner_triggered: bool = False) -> asyncio.Future[None]:
            nonlocal disposing
            if disposing is None:
                disposing = asyncio.ensure_future(do_dispose(owner_triggered))
            return disposing

        untrack = self._ownership.track(dispose)
        try:
            def owner_effect() -> Callable[[], Any]:
                def on_owner_dispose() -> Any:
                    # Owner disposal owns the same quiescence boundary. Its
                    # teardown skips unregistering this already-running owner
                    # effect from inside itself.
                    if disposing is not None:
                        return None
                    abort.abort(
                        RuntimeError(
                            f'agent "{id}" setup aborted: owner disposed during setup'
                        )
                    )
                    return dispose(True)

                return on_owner_dispose

            unfollow_owner = owner_ctx.effect(
                owner_effect, f'agentLoop.lifecycle({id})'
            )
        except BaseException:
            untrack()
            if caller_signal is not None:
                caller_signal.remove_listener(on_caller_abort)
            self._ownership.signal.remove_listener(on_factory_teardown)
            raise

        def assert_live() -> None:
            if not abort.signal.aborted:
                return
            # Every fused abort source carries an exception reason.
            reason = abort.signal.reason
            if isinstance(reason, BaseException):
                raise reason
            raise RuntimeError(str(reason))

        try:
            agent = machine = ReactLoopAgent(loop_ctx, id, options, session)
            for waiter in machine_ready:
                if not waiter.done():
                    waiter.set_result(None)
            machine_ready.clear()
            assert_live()

            def publish(source: SessionStartSource) -> AgentHandle:
                nonlocal detach_session, detach_agent
                assert_live()
                detach_session = agent.ctx.sessions.enter(session)
                owner_agent = getattr(owner_ctx, 'agent', None)
                detach_agent = loop_ctx.agents.enter(agent, owner_agent)
                agent.ctx.sessions.announce(session)
                assert_live()
                loop_ctx.agents.announce(agent)
                assert_live()
                # A synchronous announce/session-start listener may have
                # started teardown; the machine is already live, so only the
                # liveness recheck is owed.
                emit_agent_event(
                    loop_ctx, agent, 'agent/session-start', {'source': source}
                )
                assert_live()
                return AgentHandle(agent=agent, dispose=dispose)

            return _PreparedAgent(
                agent=machine, signal=abort.signal, publish=publish, dispose=dispose
            )
        except BaseException:
            for waiter in machine_ready:
                if not waiter.done():
                    waiter.set_result(None)
            machine_ready.clear()
            # Memoized: rolls back whatever the construction established.
            dispose()
            raise

    def create(
        self,
        id: SessionId,
        options: AgentOptions | None = None,
        meta: CreateSessionMeta | None = None,
    ) -> Agent:
        """Create an agent and session under one caller-supplied identity.

        Owned by the accessing fiber. Constructor-driven config calls mint a
        fresh combined id before entering this boundary.
        """
        runtime_ctx = self._runtime['ctx']
        preparation = SessionPreparation.create(
            runtime_ctx.sessions.prepare(
                id, CreateSessionOptions(meta=meta if meta is not None else None)
            )
        )
        with preparation:
            prepared = self._prepare(
                self.ctx, id, options if options is not None else AgentOptions(),
                preparation.session,
            )
            try:
                return prepared.publish('startup').agent
            except BaseException:
                # Fire-and-forget reverse teardown (TS ``void dispose()``).
                prepared.dispose()
                raise

    async def create_agent(
        self, owner_ctx: Context, options: CreateAgentOptions
    ) -> AgentHandle:
        """Create an owned agent on a caller-supplied session id."""
        runtime_ctx = self._runtime['ctx']
        preparation = SessionPreparation.create(
            runtime_ctx.sessions.prepare(
                options.session_id,
                CreateSessionOptions(
                    seed=options.seed,
                    meta=options.meta if options.meta is not None else None,
                ),
            )
        )
        published = self._setup_and_publish(
            owner_ctx,
            options.session_id,
            preparation,
            options.agent_options
            if options.agent_options is not None
            else AgentOptions(),
            options.setup,
            options.signal,
            'startup',
        )
        return await self._ownership.track_wrapper(published)

    async def _setup_and_publish(
        self,
        owner_ctx: Context,
        id: SessionId,
        preparation: SessionPreparation,
        agent_options: AgentOptions,
        setup: AgentSetup | None,
        signal: AbortSignal | None,
        source: SessionStartSource,
    ) -> AgentHandle:
        """Prepare one agent around an acquired session, run setup, and publish it."""
        with preparation:
            session = preparation.session
            prepared = self._prepare(owner_ctx, id, agent_options, session, signal)
            try:
                commit: AgentSetupCommit | None = await race_abort(
                    setup(prepared.agent.ctx) if setup is not None else None,
                    prepared.signal,
                    id,
                )
                if commit is not None:
                    commit.commit()
                return prepared.publish(source)
            except BaseException:
                await prepared.dispose()
                raise

    async def resume(
        self, owner_ctx: Context, options: ResumeAgentOptions
    ) -> AgentHandle:
        """Resume an owned agent from the configured persistence service."""
        persistence = self._runtime['ctx'].get('sessionPersistence')
        if persistence is None:
            raise RuntimeError(
                'cannot resume: session persistence is not configured (load a'
                ' dsh-session-persistence backend)'
            )
        return await self._resume_with(owner_ctx, persistence, options)

    def _resume_with(
        self,
        owner_ctx: Context,
        persistence: SessionPersistence,
        options: ResumeAgentOptions,
    ) -> asyncio.Task[AgentHandle]:
        """Resume through an explicit persistence handle."""
        id = options.resume_session_id

        async def published() -> AgentHandle:
            # The load may outlive its owner: race it against caller
            # cancellation, owner-fiber unload, and factory teardown so a
            # never-settling backend cannot pin the identity.
            owner_abort = AbortController()

            def owner_effect() -> Callable[[], None]:
                def on_owner_dispose() -> None:
                    owner_abort.abort(
                        RuntimeError(
                            f'agent "{id}" setup aborted: owner disposed during setup'
                        )
                    )

                return on_owner_dispose

            unfollow_owner = owner_ctx.effect(
                owner_effect, f'agentLoop.resume-load({id})'
            )
            fused = AbortController()
            watching: list[tuple[AbortSignal, Callable[[], None]]] = []

            def fuse(source: AbortSignal) -> None:
                def relay() -> None:
                    fused.abort(source.reason)

                source.add_listener(relay)
                watching.append((source, relay))

            if options.signal is not None:
                fuse(options.signal)
            fuse(owner_abort.signal)
            fuse(self._ownership.signal)
            preparation: SessionPreparation | None = None
            try:
                try:
                    preparation = await race_abort_call(
                        lambda: persistence.prepare(id, fused.signal),
                        fused.signal,
                        id,
                        lambda abandoned: abandoned.close(),
                    )
                finally:
                    await maybe_await(unfollow_owner())
                    for source, relay in watching:
                        source.remove_listener(relay)
                owner_ctx.fiber.assert_active()
                if not self._ownership.is_active():
                    raise RuntimeError('agent loop is not active')
                return await self._setup_and_publish(
                    owner_ctx,
                    id,
                    preparation,
                    options.agent_options
                    if options.agent_options is not None
                    else AgentOptions(),
                    options.setup,
                    options.signal,
                    'resume',
                )
            finally:
                if preparation is not None:
                    preparation.close()

        return self._ownership.track_wrapper(published())


def _assert_agent_options(options: AgentOptions) -> None:
    """Reject an output-token cap that cannot ride the request wire exactly."""
    if options.max_tokens is not None and (
        not isinstance(options.max_tokens, int)
        or isinstance(options.max_tokens, bool)
        or options.max_tokens <= 0
    ):
        raise TypeError('agent maxTokens must be a positive safe integer')
