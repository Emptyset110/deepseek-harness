"""Agent-scoped dispatch helpers over the shared pydsh scope carrier.

Ported from deepseek-harness ``packages/core/agent/src/dispatch.ts`` (MIT).
The fused dispatcher ``agent_events`` couples the agent subject to its scope
carrier, so the scope key and the payload's ``agent`` cannot diverge; repeat
dispatchers (the loop driver) build it once in the agent's constructor and
reuse it.

The scope carrier is ``pydsh.scope.scope_target``'s, exactly as in
TypeScript (``dsh-agent`` builds it from ``dsh-scope``): cordis'
``events.dispatch()`` consults the dispatch thisArg's ``_cordis_filter``
attribute (``Context.filter`` in TS), calling it with each listener's
registration context. The carrier preserves the base subject's filter,
admits listeners registered on an untagged context globally, and admits
listeners tagged with the dispatch key or any of its scope ancestors
(``pydsh.scope.bind_scope_parent``) — an enclosing scope's listeners receive
every descendant scope's events; a tag below the dispatch key stays
excluded.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import partial
from typing import Any, cast

from pycordis import Context
from pycordis.utils import schedule_logged
from pydsh.scope import (
    SCOPE_ATTR,
    ScopeCarrier,
    ScopeKey,
    scope_of,
    scope_target,
)
from pydsh.system_prompt import AssembleContext

from .runtime_types import Agent, AgentSubjectEvent

__all__ = [
    'SCOPE_ATTR',
    'AgentEventDispatch',
    'ScopeCarrier',
    'agent_carrier',
    'agent_events',
    'assemble_context_for',
    'emit_agent_event',
    'scope_of',
    'scope_target',
    'scoped_context',
]


def scoped_context(ctx: Context, key: Any) -> Context:
    """Derive a child context tagged with a scope key.

    Listeners registered through the tagged context receive events dispatched
    to a carrier built with the same key or a descendant of it; the loop tags
    each agent's scoped world with the agent itself.
    """
    return ctx.extend({SCOPE_ATTR: key})


def agent_carrier(agent: Agent) -> ScopeCarrier[Agent]:
    """Build the fused scope carrier for one agent subject.

    The carrier is a stateless routing object; callers that dispatch
    repeatedly for the same agent (the loop driver) build it once in the
    agent's constructor and reuse it, keeping hot-path dispatches
    allocation-free.
    """
    return scope_target(agent, agent)


def _warn_threw(ctx: Context, name: str, error: BaseException) -> None:
    ctx.logger.warn(f'agent event "{name}" listener threw: {error}')


def _warn_rejected(ctx: Context, name: str, error: BaseException) -> None:
    ctx.logger.warn(f'agent event "{name}" listener rejected: {error}')


class AgentEventDispatch:
    """The fused dispatcher returned by ``agent_events``.

    Each method dispatches the named agent-subject event with the agent's
    scope carrier as thisArg and the agent injected into the payload, so
    subject and scope key cannot diverge. Listeners receive
    ``(carrier, payload)`` — plus the ``next`` continuation for waterfalls.
    """

    def __init__(
        self,
        ctx: Context,
        agent: Agent,
        carrier: ScopeCarrier[Agent],
    ) -> None:
        self._ctx = ctx
        self._agent = agent
        self._carrier = carrier

    @property
    def carrier(self) -> ScopeCarrier[Agent]:
        """The scope carrier every dispatch of this dispatcher goes through."""
        return self._carrier

    def _fused(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # The dispatcher owns the subject injection: the payload fields come
        # first, so a payload that happens to carry an ``agent`` key can never
        # override the injected subject.
        return {**payload, 'agent': self._agent}

    def emit(self, name: AgentSubjectEvent, payload: Mapping[str, Any]) -> None:
        """Fire-and-forget notification in the agent's scope.

        Every matching listener is invoked; synchronous throws and
        returned-awaitable rejections are logged and contained per listener,
        so a notification cannot veto lifecycle progress or starve a later
        observer (unlike plain ``ctx.emit``, where one synchronous throw
        aborts the remaining listeners).
        """
        args: list[Any] = [self._carrier, name, self._fused(payload)]
        callbacks = self._ctx.events.dispatch('emit', args)
        for callback in callbacks:
            try:
                returned = callback(*args)
            except Exception as error:
                _warn_threw(self._ctx, name, error)
                continue
            if inspect.isawaitable(returned):
                schedule_logged(returned, partial(_warn_rejected, self._ctx, name))

    async def serial(self, name: AgentSubjectEvent, payload: Mapping[str, Any]) -> Any:
        """Awaited in-order dispatch (cordis ``serial``) in the agent's scope.

        Returns the serial chain's result (the first bail value, if any).
        """
        return await self._ctx.events.serial(self._carrier, name, self._fused(payload))

    def waterfall(
        self,
        name: AgentSubjectEvent,
        payload: Mapping[str, Any],
        *rest: Any,
    ) -> Any:
        """Around-middleware dispatch (cordis ``waterfall``) in the agent's scope.

        ``rest`` is exactly the event's arguments after the payload — the
        final element being the innermost ``next`` (the default the listener
        chain wraps). A listener that returns without calling ``next()``
        short-circuits the chain. Returns the outermost listener's return
        value synchronously; an async listener's result stays a coroutine the
        caller awaits.
        """
        fused = self._fused(payload)
        return self._ctx.events.waterfall(self._carrier, name, fused, *rest)


def agent_events(
    ctx: Context,
    agent: Agent,
    carrier: ScopeCarrier[Agent] | None = None,
) -> AgentEventDispatch:
    """Build a dispatcher that couples the agent subject to its scope carrier.

    ``carrier`` defaults to ``agent_carrier(agent)``; pass a
    constructor-built carrier to avoid rebuilding it for every dispatch.
    """
    if carrier is None:
        carrier = agent_carrier(agent)
    return AgentEventDispatch(ctx, agent, carrier)


def emit_agent_event(
    ctx: Context,
    agent: Agent,
    name: AgentSubjectEvent,
    payload: Mapping[str, Any],
) -> None:
    """Emit one contained agent notification.

    Use this one-shot form when no retained dispatcher is needed.
    """
    agent_events(ctx, agent).emit(name, payload)


def assemble_context_for(agent: Agent, signal: Any = None) -> AssembleContext:
    """Build the prompt assembly context with agent and scope set together.

    Ported from deepseek-harness ``packages/core/agent/src/dispatch.ts``
    (MIT). The agent is both the scope key and the merge-extensible ``agent``
    field, so agent-scoped prompt and tool contributions cannot be silently
    omitted.
    """
    context = AssembleContext(scope=cast(ScopeKey, agent), signal=signal)
    # Merge-extensible field (TS ``AssembleContext.agent``): declared by the
    # agent layer, attached dynamically like any plugin-defined field.
    context.agent = agent  # type: ignore[attr-defined]
    return context
