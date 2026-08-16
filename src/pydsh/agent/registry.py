"""Agent registry service (``ctx.agents``).

Registration, detachment, and lookup of live agents, plus the factory
delegation surface and the process-local initiator scope. Ported from
deepseek-harness ``packages/core/agent/src/index.ts`` (MIT). Not ported:
the ``typert`` lookup/context registrations (the RPC layer is not ported).

Initiator scope mapping: TypeScript's ``AsyncLocalStorage`` becomes two
per-instance :class:`contextvars.ContextVar` chains (the initiating agent and
the tracked-run chain). ``with_initiator`` preserves the operation's exact
synchronous value; an awaitable result is wrapped in a coroutine that
re-establishes the variables around the await and releases the tracked run on
settlement (ContextVar scoping cannot outlive the synchronous call the way
AsyncLocalStorage tracks a promise's continuations).

Lifecycle contract (identical to TS): ``register()`` records an
already-constructed agent and emits ``agent/created`` immediately and
``agent/disposed`` when the calling fiber is disposed, both through the
agent's scope carrier (``agent_carrier(agent)``) so the emits are
scope-filtered regardless of which context invoked ``register``. The
advanced ``enter()``/``announce()`` pair splits insertion from publication
for the async agent factory: an entered-but-unannounced agent can be rolled
back without inventing a disposal edge, and a detach requested from inside a
synchronous ``agent/created`` listener is deferred until the creation
dispatch unwinds.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol

from pycordis import Context, EffectDisposer, Fiber, FiberState, Service
from pycordis.utils import rebind, schedule_logged
from pydsh.session import CreateSessionMeta, SessionEvent

from .dispatch import ScopeCarrier, agent_carrier
from .runtime_types import (
    AbortSignal,
    Agent,
    AgentOptions,
    SessionId,
)

#: Thrown when create/resume is called before an agent factory is registered.
_NO_FACTORY_MESSAGE = 'no agent factory registered (load an agent-loop plugin)'
_NO_INITIATOR_MESSAGE = 'no initiating agent is active'
_DISPOSED_INITIATOR_MESSAGE = 'agent initiator scope is disposed'


class AgentSetupCommit(Protocol):
    """Synchronous finalizer returned by unpublished Agent setup.

    Validates the setup's contributions at the exact publication commit point.
    """

    def commit(self) -> None:
        """Validate and commit the prepared setup immediately before publication.

        :raises Exception: when publication must roll the unpublished agent back.
        """
        ...


#: Compose an unpublished agent scope and optionally return its publication
#: commit. Trusted same-process code: setup composes, it never drives.
AgentSetup = Callable[
    [Context],
    'AgentSetupCommit | Awaitable[AgentSetupCommit | None] | None',
]


@dataclass(frozen=True)
class CreateAgentOptions:
    """Options for creating an agent through the registry factory.

    The caller supplies the single live ``session_id`` shared by the agent
    registry and session log, plus optional session metadata/seed, agent
    options, unpublished setup, and a creation-only cancellation signal.
    """

    #: The live agent/session identity.
    session_id: SessionId
    #: Session creation metadata (validated absolute ``cwd``, fork lineage).
    meta: CreateSessionMeta | None = None
    #: Initial replay/fork history (a balanced completed-turn prefix).
    seed: Sequence[SessionEvent] | None = None
    #: Per-agent options (model, …).
    agent_options: AgentOptions | None = None
    #: Optional creation-only cancellation signal; detached before the
    #: returned handle becomes visible.
    signal: AbortSignal | None = None
    #: Creation-time composition of the agent's scoped world; awaited BEFORE
    #: either creation announcement, rollback-covered.
    setup: AgentSetup | None = None


@dataclass(frozen=True)
class ResumeAgentOptions:
    """Options for resuming an agent on a persisted session."""

    #: The persisted session id to load and use as the live identity.
    resume_session_id: SessionId
    #: Per-agent options (model, …).
    agent_options: AgentOptions | None = None
    #: Optional creation-only cancellation signal for persistence load/setup.
    signal: AbortSignal | None = None
    #: Resume-time composition of the agent's fresh scoped world.
    setup: AgentSetup | None = None


@dataclass(frozen=True)
class AgentHandle:
    """An owned agent plus its disposer, returned by create/resume.

    The disposer is a CAPABILITY: among consumers, only the holder can tear
    this agent down. ``dispose()`` stops the loop, awaits its exit,
    unregisters the agent, removes its session from the store, and finally
    unwinds its scoped world.
    """

    agent: Agent
    dispose: Callable[[], Awaitable[None]]


class AgentFactory(Protocol):
    """The agent-creation factory the loop implementation registers.

    Kept on the ``pydsh.agent`` interface so consumers program against
    ``ctx.agents`` without depending on the concrete ``pydsh.agent_loop``
    package.
    """

    async def create_agent(
        self, owner_ctx: Context, options: CreateAgentOptions
    ) -> AgentHandle:
        """Create and publish a new agent owned by ``owner_ctx``."""
        ...

    async def resume(
        self, owner_ctx: Context, options: ResumeAgentOptions
    ) -> AgentHandle:
        """Prepare a persisted session and resume an agent on it."""
        ...


@dataclass(frozen=True)
class _FactorySlot:
    """Plain holder preventing re-tracing the factory's dependency context."""

    target: AgentFactory


@dataclass
class _InitiatorRun:
    """One tracked boundary plus its inherited nesting chain."""

    active: bool
    parent: _InitiatorRun | None


@dataclass
class _AgentEntry:
    """All mutable lifecycle state for one exact registry entry."""

    id: SessionId
    agent: Agent
    #: Runtime creator-agent ownership; independent of durable session lineage.
    owner: Agent | None
    carrier: ScopeCarrier[Agent]
    announced: bool = False
    announcing: bool = False
    detach_requested: bool = False


def _warn_listener_rejected(
    ctx: Context,
    agent_id: SessionId,
    event: str,
    error: BaseException,
) -> None:
    ctx.logger.warn(f'agent "{agent_id}": {event} listener rejected: {error}')


def _warn_listener_threw(
    ctx: Context,
    agent_id: SessionId,
    event: str,
    error: BaseException,
) -> None:
    ctx.logger.warn(f'agent "{agent_id}": {event} listener threw: {error}')


class AgentRegistry(Service):
    """Agent service (``ctx.agents``): tracks live agents in registration order.

    Concrete creation and driving belong to the loop layer; this service only
    records already-constructed agents and pairs their lifecycle events.
    """

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, 'agents')
        self._store: dict[SessionId, _AgentEntry] = {}
        self._factory: _FactorySlot | None = None
        self._initiators: ContextVar[Agent | None] = ContextVar(
            'pydsh.agent.initiators', default=None
        )
        self._initiator_runs: ContextVar[_InitiatorRun | None] = ContextVar(
            'pydsh.agent.initiatorRuns', default=None
        )
        self._initiator_state: Literal['active', 'closing', 'disposed'] = 'active'
        self._active_initiator_runs = 0
        self._initiator_drain: asyncio.Future[None] | None = None
        self._initiator_disposal: asyncio.Future[None] | None = None
        # The ``ctx.agent`` DX accessor: defaults to ``None`` on every
        # context, so a plain plugin context reads cleanly instead of hitting
        # the unknown-property error. Each ``Agent.ctx`` shadows it with an
        # own meta property, so the accessor body never resolves a scope.
        ctx.accessor('agent', get=lambda _ctx, _error: None)
        ctx.on('internal/status', self._on_fiber_status)
        ctx.effect(self._initiator_lifecycle, 'agents.initiatorLifecycle()')

    def _initiator_lifecycle(self) -> Generator[Callable[[], Any], None, None]:
        # Disposers run in reverse yield order: close first, then drain.
        yield lambda: self._dispose_initiators()
        yield lambda: self._close_initiators()

    def _on_fiber_status(self, fiber: Fiber, *_args: Any) -> None:
        if fiber.state is FiberState.UNLOADING and self._has_lifecycle_ancestor(fiber):
            self._close_initiators()

    def current_initiator(self) -> Agent | None:
        """The agent that initiated the inherited asynchronous driver chain.

        Optional form for logging, tracing, metrics, or host attribution that
        also supports agentless calls.
        """
        self._assert_initiators_readable()
        return self._initiators.get()

    def require_initiator(self) -> Agent:
        """The initiating agent, failing when no initiator boundary is active."""
        agent = self.current_initiator()
        if agent is None:
            raise RuntimeError(_NO_INITIATOR_MESSAGE)
        return agent

    def with_initiator(self, agent: Agent, operation: Callable[[], Any]) -> Any:
        """Run an operation with one exact agent as its process-local initiator.

        The exact synchronous value is preserved; an awaitable result is
        wrapped so the boundary covers its full foreground lifetime.
        """
        return self._run_with_initiator(agent, operation)

    def without_initiator(self, operation: Callable[[], Any]) -> Any:
        """Run an operation inside a boundary that hides any inherited initiator."""
        return self._run_with_initiator(None, operation)

    def set_factory(self, factory: AgentFactory) -> EffectDisposer:
        """Register the agent-creation factory (the loop calls this effect-scoped).

        :returns: the exact Cordis effect disposer that clears the slot.
        """
        def effect() -> Callable[[], None]:
            if self._factory is not None:
                raise RuntimeError('an agent factory is already registered')
            self._factory = _FactorySlot(factory)
            return self._clear_factory

        return self.ctx.effect(effect, 'agents.setFactory()')

    def _clear_factory(self) -> None:
        self._factory = None

    def _require_factory(self) -> _FactorySlot:
        """The active creation factory."""
        if self._factory is None:
            raise RuntimeError(_NO_FACTORY_MESSAGE)
        return self._factory

    async def create(self, options: CreateAgentOptions) -> AgentHandle:
        """Create and publish a new agent through the registered factory.

        Distinct from :meth:`register` (which records an already-constructed
        agent): this constructs the agent and its session, bound to the
        accessing context's fiber and scope as the lifecycle owner.
        """
        owner_ctx = self.ctx
        slot = self._require_factory()
        # Rebind the factory to the accessing context so ownership follows
        # the caller (TS ``getTraceable``).
        receiver = rebind(owner_ctx, slot.target)
        return await receiver.create_agent(owner_ctx, options)

    async def resume(self, options: ResumeAgentOptions) -> AgentHandle:
        """Load a persisted session and resume an agent on it.

        The factory rejects when session persistence is not configured.
        """
        owner_ctx = self.ctx
        slot = self._require_factory()
        receiver = rebind(owner_ctx, slot.target)
        return await receiver.resume(owner_ctx, options)

    def _close_initiators(self) -> None:
        """Reject new initiator boundaries while inherited continuations drain."""
        if self._initiator_state == 'active':
            self._initiator_state = 'closing'

    def _dispose_initiators(self) -> asyncio.Future[None]:
        """Wait for returned-awaitable boundaries, then refuse further reads."""
        if self._initiator_disposal is None:
            self._initiator_disposal = asyncio.ensure_future(self._drain_initiators())
        return self._initiator_disposal

    async def _drain_initiators(self) -> None:
        self._close_initiators()
        self._release_reentrant_initiator_runs()
        if self._active_initiator_runs != 0:
            if self._initiator_drain is None:
                self._initiator_drain = asyncio.get_running_loop().create_future()
            await self._initiator_drain
        self._initiator_state = 'disposed'

    def _run_with_initiator(
        self, agent: Agent | None, operation: Callable[[], Any]
    ) -> Any:
        """Establish one tracked initiator or clearing boundary."""
        if self._initiator_state != 'active':
            raise RuntimeError(_DISPOSED_INITIATOR_MESSAGE)
        run = _InitiatorRun(active=True, parent=self._initiator_runs.get())
        self._active_initiator_runs += 1
        token_runs = self._initiator_runs.set(run)
        token_agent = self._initiators.set(agent)
        try:
            result = operation()
        except BaseException:
            self._initiator_runs.reset(token_runs)
            self._initiators.reset(token_agent)
            self._release_initiator_run(run)
            raise
        self._initiator_runs.reset(token_runs)
        self._initiators.reset(token_agent)
        if inspect.isawaitable(result):
            return self._bind_initiator_awaitable(agent, run, result)
        self._release_initiator_run(run)
        return result

    async def _bind_initiator_awaitable(
        self, agent: Agent | None, run: _InitiatorRun, awaitable: Any
    ) -> Any:
        """Re-establish the boundary around the await and release on settlement."""
        token_runs = self._initiator_runs.set(run)
        token_agent = self._initiators.set(agent)
        try:
            return await awaitable
        finally:
            self._initiator_runs.reset(token_runs)
            self._initiators.reset(token_agent)
            self._release_initiator_run(run)

    def _has_lifecycle_ancestor(self, candidate: Fiber) -> bool:
        """Whether one unloading fiber owns this service's lifecycle."""
        fiber = self.ctx.fiber
        while True:
            if fiber is candidate:
                return True
            parent_ctx = fiber.parent
            if parent_ctx is None:
                return False
            parent = parent_ctx.fiber
            if parent is fiber:
                return False
            fiber = parent

    def _assert_initiators_readable(self) -> None:
        if self._initiator_state == 'disposed':
            raise RuntimeError(_DISPOSED_INITIATOR_MESSAGE)

    def _release_reentrant_initiator_runs(self) -> None:
        """Exclude the boundary chain that initiated this teardown from its drain."""
        run = self._initiator_runs.get()
        while run is not None:
            self._release_initiator_run(run)
            run = run.parent

    def _release_initiator_run(self, run: _InitiatorRun) -> None:
        if not run.active:
            return
        run.active = False
        self._active_initiator_runs -= 1
        if self._active_initiator_runs != 0:
            return
        if self._initiator_drain is not None and not self._initiator_drain.done():
            self._initiator_drain.set_result(None)
        self._initiator_drain = None

    def register(self, agent: Agent) -> EffectDisposer:
        """Register a live agent.

        Emits ``agent/created`` now and ``agent/disposed`` on teardown.

        Raises if an agent with the same id is already registered. Returns
        the exact cordis effect disposer (single-shot): a composite effect
        that owns a teardown ORDER — the agent factory's lifecycle chain —
        must register this disposer at its own position so unregistration
        nests there; wrapping it would unregister the agent (and emit
        ``agent/disposed``) as a concurrent sibling while its final turn is
        still draining.
        """
        def effect() -> Generator[Callable[[], None], None, None]:
            # Generator effect mirroring the TS composite: the detach
            # disposer is collected BEFORE announce runs, so a synchronous
            # ``agent/created`` veto rolls the entry back (detach runs during
            # effect error cleanup) instead of leaking a live store entry.
            yield self.enter(agent, self._registering_owner())
            self.announce(agent)

        return self.ctx.effect(effect, 'agents.register()')

    def _registering_owner(self) -> Agent | None:
        """The agent association of the context that called ``register()``, if any."""
        owner = getattr(self.ctx, 'agent', None)
        return owner if owner is not None else None

    def enter(self, agent: Agent, owner: Agent | None) -> Callable[[], None]:
        """Insert an already-constructed agent without announcing it.

        This is the advanced ordered-lifecycle primitive used by the async
        agent factory; ordinary callers use ``register()``. Raises if the
        agent id does not match its session id, or if an agent with the same
        id is already registered — this is the authoritative collision
        boundary: concurrent create/resume operations may both prepare, but
        only one exact entry can publish.

        ``owner`` is the live agent whose scoped context created this agent,
        or ``None`` for a top-level runtime root (runtime ownership, not the
        resumed session's durable parent lineage).

        Returns an idempotent closure that removes this exact entry and emits
        ``agent/disposed`` with listener failures contained. When called from
        a synchronous ``agent/created`` listener, removal and disposal wait
        until that creation dispatch unwinds.
        """
        agent_id = agent.id
        session_id = agent.session.id
        if agent_id != session_id:
            raise ValueError(
                f'agent id "{agent_id}" does not match session id "{session_id}"'
            )
        carrier = agent_carrier(agent)
        if agent_id in self._store:
            raise ValueError(f'agent "{agent_id}" is already registered')
        entry = _AgentEntry(id=agent_id, agent=agent, owner=owner, carrier=carrier)
        self._store[agent_id] = entry
        entered = True

        def detach() -> None:
            nonlocal entered
            if not entered:
                return
            entered = False
            # Every callback reached by the creation dispatch must observe the
            # same live entry, and disposal must follow creation. A listener
            # may own this detach capability, so make that ordering structural.
            if entry.announcing:
                entry.detach_requested = True
                return
            self._detach_entered(entry)

        return detach

    def announce(self, agent: Agent) -> None:
        """Announce an agent previously inserted with ``enter()``.

        Emits ``agent/created`` through the entry's stable carrier. A
        synchronous listener failure vetoes publication and propagates (the
        caller rolls back); a returned-awaitable rejection happens after this
        synchronous boundary, so it is observed and logged instead of leaking
        an unhandled rejection. Raises if ``agent`` is not the exact live
        registry entry for its id, or its creation announcement already began
        (including a reentrant call from a creation listener).
        """
        entry = self._store.get(agent.id)
        if entry is None or entry.agent is not agent:
            raise ValueError(f'agent "{agent.id}" is not live in this registry')
        if entry.announced or entry.announcing:
            raise ValueError(f'agent "{entry.id}" was already announced')
        # Mark before dispatch so a listener cannot recursively create a
        # second lifecycle edge; detach still pairs a partially delivered
        # first edge.
        entry.announcing = True
        entry.announced = True
        args: list[Any] = [entry.carrier, 'agent/created', {'agent': entry.agent}]
        try:
            for callback in self.ctx.events.dispatch('emit', args):
                returned = callback(*args)
                if inspect.isawaitable(returned):
                    warn = partial(
                        _warn_listener_rejected, self.ctx, entry.id, 'agent/created'
                    )
                    schedule_logged(returned, warn)
        finally:
            entry.announcing = False
            if entry.detach_requested:
                self._detach_entered(entry)

    def get(self, agent_id: SessionId) -> Agent | None:
        """Look up a live agent, or ``None`` when no live agent has that id."""
        entry = self._store.get(agent_id)
        return entry.agent if entry is not None else None

    def is_owned_by(self, agent_id: SessionId, owner: Agent) -> bool:
        """Whether a live agent was created through one exact parent agent.

        The parent is identified by its scoped context. Runtime ownership is
        independent of durable session lineage and remains unambiguous when
        unrelated providers reuse an id.
        """
        entry = self._store.get(agent_id)
        return entry is not None and entry.owner is owner

    def list(self) -> builtins.list[Agent]:
        """All live agents, in registration order (a fresh list)."""
        return [entry.agent for entry in self._store.values()]

    def roots(self) -> builtins.list[Agent]:
        """All live top-level agents in registration order.

        A top-level agent was created without an owning agent context;
        durable session lineage does not affect this runtime relation, so a
        resumed fork may still be a root.
        """
        return [entry.agent for entry in self._store.values() if entry.owner is None]

    def _detach_entered(self, entry: _AgentEntry) -> None:
        """Remove one exact entered agent.

        Emits its paired disposal edge when it was announced.
        """
        entry.detach_requested = False
        # A stale capability can never delete a later same-id lifecycle; the
        # captured entry identity is the final boundary (enter() rejects
        # replacement while this single-shot detach capability is live).
        if self._store.get(entry.id) is not entry:
            return
        del self._store[entry.id]
        # An insertion rolled back before announce was never externally
        # created, so emitting disposed would invent an impossible lifecycle
        # edge. Marking happens before the created emit: if a later created
        # listener throws, earlier listeners may already have observed it and
        # must see disposal.
        if not entry.announced:
            return
        self._emit_disposed(entry)

    def _emit_disposed(self, entry: _AgentEntry) -> None:
        """Emit the paired disposal edge through the entry's stable carrier."""
        args: list[Any] = [entry.carrier, 'agent/disposed', {'agent': entry.agent}]
        for callback in self.ctx.events.dispatch('emit', args):
            try:
                returned = callback(*args)
            except Exception as error:
                _warn_listener_threw(self.ctx, entry.id, 'agent/disposed', error)
                continue
            if inspect.isawaitable(returned):
                warn = partial(
                    _warn_listener_rejected, self.ctx, entry.id, 'agent/disposed'
                )
                schedule_logged(returned, warn)
