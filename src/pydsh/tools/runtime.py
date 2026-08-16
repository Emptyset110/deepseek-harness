"""Tool registry and the pre/guard/around/post/result execution pipeline.

Ported from deepseek-harness packages/core/tools (MIT), ``src/index.ts``,
restricted to the ``ToolRuntime`` kernel. Excluded with their owning layers:
Code Mode (``code-mode.ts``: the reserved ``run_code`` presentation
transport, ``mode: 'code'/'both'``, SDK prompt sections, sub-dispatch
logging), the SDK type renderers (``ts-types.ts`` / ``py-types.ts``), the
presentation vocabulary (``presentation.ts`` and the ``presentCall`` /
``presentResult`` definition callbacks), and the concrete tool
implementations. ``Config.mode`` accepts only ``'native'``; the other modes
fail loud at construction until Code Mode is ported. The ``run_code`` name
stays reserved so a later Code Mode port cannot collide with an existing
registration, and the mode-collapse denial path (a model-direct call under
``code``) is omitted with the modes that produce it.

Event surface (TypeScript declaration merging has no Python equivalent):

- ``tools/pre-execute`` (waterfall): ``(carrier, exec, next)`` ->
  :data:`PreToolDecision`. Allow, deny, or ask before dispatch; ``next()``
  delegates to allow, and a listener that returns without calling it vetoes
  the rest of the chain. Missing approval support turns ``ask`` into
  denial. Dispatch is scope-filtered by ``exec.agent``: agent-scoped
  listeners receive only that agent's (and its descendants') calls.
- ``tools/execute`` (waterfall): ``(carrier, exec, next)`` ->
  :data:`ToolExecutionResult`. Around-dispatch for timeout/retry/metrics;
  wrappers may replace only ``exec.signal`` (the registry re-fuses the
  original caller signal before the body, so replacement cannot detach
  caller cancellation). Same scope filtering.
- ``tools/post-execute`` (waterfall): ``(carrier, exec, result, next)`` ->
  :data:`PostToolDecision`. Accept, replace, enrich, or block a normalized
  dispatch result; thrown tools still reach this waterfall as errors. Same
  scope filtering.
- ``tools/result`` (emit): ``(carrier, exec, result)`` observing the final
  frozen outcome. Listener failures are contained (logged, never
  propagated). Same scope filtering.
- ``tools/change`` (emit, no arguments): a tool was registered or
  unregistered, or a scoped restriction changed. Deliberately UNFILTERED:
  a global change concerns every agent's next prompt assembly.

Python mappings:

- TS plain-object executions become :class:`ToolExecution`, a
  registry-owned object that wrappers may mutate only in its ``signal``
  slot; it freezes before ``tools/result`` observers run (mutating a frozen
  execution raises), standing in for ``Object.freeze``.
- TS plain-object results become frozen dataclasses with identity
  semantics: the registry's canonical-result map keys them like the TS
  ``WeakMap``. JSON-valued fields (``content``, ``meta``, ``value``) are
  detached snapshots treated as read-only by convention (``deepFreeze``).
- ``AbortSignal``/``AbortController`` are the DOM-primitive ports from
  :mod:`pydsh.subprocess.types`.
- Waterfall/emit listeners receive the scope carrier built by
  ``scope_target()`` as their first argument (JavaScript ``this``); the
  subject stays in the payload.
- ``additionalContexts`` are typed :class:`UserMessage` objects, not JSON:
  unlike the TS materialization they are copied, not snapshotted.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import weakref
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, Protocol, cast

from pycordis import Context, EffectDisposer, Service, ValidationError
from pycordis.utils import maybe_await, schedule_logged
from pydsh.agent import Agent, UserMessage
from pydsh.approval import ApprovalRequest
from pydsh.llm import CallId, ContentBlock, HarnessError
from pydsh.scope import (
    AnonymousEntries,
    NamedEntries,
    ScopedLayers,
    ScopeKey,
    scope_of,
    scope_target,
)
from pydsh.session.json import SNAPSHOT_FAILED, JsonValue, snapshot_json_value
from pydsh.subprocess.types import AbortController, AbortSignal
from pydsh.system_prompt import ToolProviderResult, ToolSchema

from .json_schema import (
    JsonSchemaNode,
    assert_supported_json_schema,
    validate_json_schema_value,
)

__all__ = [
    'RUN_CODE_NAME',
    'TOOL_ABORTED',
    'TOOL_ABORTED_BEFORE_DISPATCH',
    'Config',
    'DispatchFinalResult',
    'DispatchPostResult',
    'PostToolAccept',
    'PostToolBlock',
    'PostToolDecision',
    'PreToolAllow',
    'PreToolAsk',
    'PreToolDecision',
    'PreToolDeny',
    'ScheduledDispatch',
    'ScheduledFinalResult',
    'ScheduledPostResult',
    'ScheduledToolDispatch',
    'ScheduledToolPreparation',
    'ToolDefinition',
    'ToolDispatchExecution',
    'ToolErrorInfo',
    'ToolExecution',
    'ToolExecutionFailure',
    'ToolExecutionInput',
    'ToolExecutionMode',
    'ToolExecutionResult',
    'ToolExecutionSuccess',
    'ToolExecutionToken',
    'ToolFailure',
    'ToolGuard',
    'ToolNotFoundError',
    'ToolOutputDefinition',
    'ToolOutputError',
    'ToolPresentationMode',
    'ToolRestriction',
    'ToolRunContext',
    'ToolRuntime',
    'ToolRuntimeScheduler',
]

#: Name reserved for the Code Mode presentation transport. Registration and
#: restrictions reject it unconditionally so a later Code Mode port cannot
#: collide with an existing tool.
RUN_CODE_NAME = 'run_code'

#: Canonical error code for cancellation after a tool body was invoked.
TOOL_ABORTED = 'ABORTED'

#: Canonical error code for cancellation before a tool body was invoked.
TOOL_ABORTED_BEFORE_DISPATCH = 'ABORTED_BEFORE_DISPATCH'


class _Unset:
    """Sentinel distinguishing an absent optional field from JSON ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return 'UNSET'


_UNSET: Final = _Unset()


class ToolNotFoundError(HarnessError):
    """Raised (internally) when the model requests an unregistered tool.

    Carries ``code: 'UNKNOWN_TOOL'`` so an unknown-tool failure is as
    routable as a tool-thrown one.
    """

    def __init__(self, tool_name: str, reachable_from: str | None = None) -> None:
        super().__init__(
            f'unknown tool "{tool_name}"'
            if reachable_from is None
            else f'unknown tool "{tool_name}": {reachable_from}',
            'UNKNOWN_TOOL',
        )


class ToolOutputError(HarnessError):
    """A tool body or post-policy value violated its declared output."""

    def __init__(self, tool_name: str, violations: list[str]) -> None:
        super().__init__(
            f'tool "{tool_name}" returned invalid output: {"; ".join(violations)}',
            'INVALID_TOOL_OUTPUT',
        )
        #: Schema/value violations in validation order.
        self.violations = violations


@dataclass(frozen=True)
class ToolErrorInfo:
    """Structured error metadata for a failed tool call."""

    name: str
    code: str


@dataclass(frozen=True)
class ToolFailure:
    """Canonical failure detail; internal routing information is optional."""

    #: Human-readable failure message without the ``Error: `` envelope.
    message: str
    #: Internal error class/code used by policy and durable diagnostics.
    info: ToolErrorInfo | None = None


@dataclass(frozen=True, eq=False)
class ToolExecutionSuccess:
    """Successful canonical tool execution, including its model projection.

    Identity-hashed (TS object identity): the registry's canonical-result
    map keys results by identity to recognize its own normalizations.
    """

    #: Execution-local canonical value; deliberately execution-local only.
    value: JsonValue
    #: The final model-facing content.
    content: list[ContentBlock]
    is_error: Literal[False] = False
    #: The tool-private presentation payload projected by its output
    #: declaration; ``_Unset`` (absent) when no projector ran.
    meta: Any = _UNSET
    #: Deferred contexts ferried to the loop's active-batch FIFO.
    additional_contexts: tuple[UserMessage, ...] = ()
    #: The agent loop stops after committing this successful result batch.
    concludes_turn: bool = False


@dataclass(frozen=True, eq=False)
class ToolExecutionFailure:
    """Failed canonical tool execution; failures never carry a value."""

    error: ToolFailure
    #: The rendered error text (or post-policy feedback).
    content: list[ContentBlock]
    is_error: Literal[True] = True
    meta: Any = _UNSET
    additional_contexts: tuple[UserMessage, ...] = ()


#: The discriminated, execution-local outcome of one tool call.
ToolExecutionResult = ToolExecutionSuccess | ToolExecutionFailure


class ToolExecutionToken:
    """Opaque call identity: it permits correlation without exposing state."""

    __slots__ = ()


class ToolExecution:
    """One pending tool call inside the registry pipeline (registry-owned).

    Callers describe a call with :class:`ToolExecutionInput`; the runtime
    mints this object, deep-snapshots its arguments, and assigns its
    correlation token. Around-dispatch wrappers may replace ``signal`` for
    their delegated lifetime (the registry fuses any replacement with the
    captured caller signal before the tool body); every other attribute is
    read-only by contract. The registry freezes the object before
    ``tools/result`` observers run — mutation then raises.
    """

    __slots__ = (
        '__weakref__',
        '_concluded',
        '_deferred_contexts',
        '_frozen',
        'agent',
        'arguments',
        'call_id',
        'name',
        'parent',
        'root_call_id',
        'signal',
        'token',
    )

    token: ToolExecutionToken
    call_id: CallId
    #: Root model-requested call, resolved for root and nested executions.
    root_call_id: CallId
    name: str
    #: Losslessly snapshotted model arguments (read-only by convention).
    arguments: Any
    #: Cancellation signal visible to the next wrapper or tool body.
    signal: AbortSignal
    #: The agent on whose behalf the call runs (also the scope routing key).
    agent: Agent | None
    #: Opaque token of the enclosing transport execution, when one exists.
    parent: ToolExecutionToken | None
    #: Registry-owned deferred contexts (TS: a WeakMap keyed by execution).
    _deferred_contexts: list[UserMessage]
    #: Registry-owned turn-conclusion marker (TS: a WeakSet membership).
    _concluded: bool
    #: Set once the execution is published to ``tools/result`` observers.
    _frozen: bool

    def __init__(
        self,
        *,
        token: ToolExecutionToken,
        call_id: CallId,
        root_call_id: CallId,
        name: str,
        arguments: Any,
        signal: AbortSignal,
        agent: Agent | None = None,
        parent: ToolExecutionToken | None = None,
    ) -> None:
        object.__setattr__(self, '_frozen', False)
        self.token = token
        self.call_id = call_id
        self.root_call_id = root_call_id
        self.name = name
        self.arguments = arguments
        self.signal = signal
        self.agent = agent
        self.parent = parent
        self._deferred_contexts: list[UserMessage] = []
        self._concluded = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._frozen:
            raise AttributeError(
                'tool execution is frozen: the registry froze it before '
                'publishing the final result'
            )
        object.__setattr__(self, name, value)

    def defer_context(self, context: UserMessage) -> None:
        """Defer one context until this tool's final result reaches the loop.

        Contexts retain their individual source and are emitted in call
        order; a post-execute ``block`` discards them.
        """
        self._deferred_contexts.append(context)

    def conclude_turn(self) -> None:
        """Mark a successful final result as terminal for the agent turn."""
        self._concluded = True

    def _freeze(self) -> None:
        object.__setattr__(self, '_frozen', True)

    @property
    def _is_frozen(self) -> bool:
        return self._frozen


#: The runtime context handed to a tool implementation after acceptance.
#: Identical to :class:`ToolExecution`; the TS subtype adds exactly the
#: ``deferContext``/``concludeTurn`` methods this class already carries.
ToolRunContext = ToolExecution

#: The around-dispatch view of an execution: wrappers may replace
#: ``signal``. Identical runtime object; the TS type only widens mutability.
ToolDispatchExecution = ToolExecution


@dataclass(frozen=True)
class ToolExecutionInput:
    """Caller-supplied description of one tool call.

    The registry adds the correlation token to form the pipeline
    :class:`ToolExecution`; callers do not choose that token.
    """

    call_id: CallId
    name: str
    #: Losslessly JSON-serializable parsed arguments.
    arguments: Any
    #: Required caller-owned cancellation for this invocation.
    signal: AbortSignal
    #: Root model-requested call owning this execution tree; callers omit it
    #: for a root execution.
    root_call_id: CallId | None = None
    #: The agent on whose behalf the call runs (set by the agent loop).
    agent: Agent | None = None
    #: Opaque token of the enclosing transport execution, when one exists.
    parent: ToolExecutionToken | None = None


@dataclass
class ToolOutputDefinition:
    """Tool-owned canonical output contract used after the body returns."""

    #: Raw supported JSON Schema enforced against every successful value.
    schema: JsonSchemaNode
    #: Pure projection from validated arguments and value to model content.
    render: Callable[[Any, JsonValue], list[ContentBlock]]
    #: Pure replayable presentation projection, computed only for top-level
    #: calls (no ``parent`` token).
    presentation_meta: Callable[[Any, JsonValue], JsonValue] | None = None


@dataclass
class ToolDefinition:
    """A registered tool: its schema plus the execution function."""

    name: str
    description: str
    #: Raw JSON Schema for the arguments, in the enforced subset.
    parameters: dict[str, Any]
    #: Mandatory canonical output declaration.
    output: ToolOutputDefinition
    #: Run one accepted call and return only its canonical lossless-JSON
    #: value. Async work must observe or forward ``exec.signal``.
    execute: Callable[[Any, ToolExecution], Awaitable[Any]]
    #: Synchronous last-mile transform for model-facing content, snapshotted
    #: when execution starts and invoked exactly once for every normalized
    #: outcome, immediately before final materialization. Returning ``None``
    #: preserves the content. The callback must be total and must not throw.
    finalize_content: Callable[
        [ToolExecution, ToolExecutionResult], list[ContentBlock] | None
    ] | None = None
    #: Cooperative tool-call timeout budget in milliseconds; omit for no
    #: deadline. Never sent to the model — ``schemas()`` whitelists only
    #: name/description/parameters.
    timeout_ms: float | None = None
    #: Pure synchronous classifier for overlap with sibling tool calls. Only
    #: an exact ``True`` return opts in; omission, exceptions, and other
    #: returns are exclusive. Never model-visible.
    is_concurrency_safe: Callable[[Any], bool] | None = None


@dataclass(frozen=True)
class PreToolAllow:
    """Pre-dispatch decision: run the call."""

    kind: Literal['allow'] = 'allow'


@dataclass(frozen=True)
class PreToolDeny:
    """Pre-dispatch decision: materialize an error with this reason."""

    reason: str
    kind: Literal['deny'] = 'deny'


@dataclass(frozen=True)
class PreToolAsk:
    """Pre-dispatch decision: run only after the approval seam allows once.

    Missing approval support turns ``ask`` into denial.
    """

    reason: str | None = None
    kind: Literal['ask'] = 'ask'


#: Pre-dispatch decision. Input rewriting is excluded because arguments are
#: already logged and presented.
PreToolDecision = PreToolAllow | PreToolDeny | PreToolAsk


@dataclass(frozen=True)
class PostToolAccept:
    """Post-dispatch decision: accept the result, optionally replacing it.

    ``content`` replaces the model-facing projection directly; ``value``
    re-runs the owning output contract (re-validation and re-render).
    Setting both raises; setting ``value`` on a failed result raises.
    """

    content: list[ContentBlock] | None = None
    value: Any = _UNSET
    additional_contexts: Sequence[UserMessage] | None = None
    kind: Literal['accept'] = 'accept'


@dataclass(frozen=True)
class PostToolBlock:
    """Post-dispatch decision: turn corrective feedback into an error result."""

    feedback: list[ContentBlock]
    additional_contexts: Sequence[UserMessage] | None = None
    kind: Literal['block'] = 'block'


#: Post-dispatch decision.
PostToolDecision = PostToolAccept | PostToolBlock

#: Scheduling mode for one pending call. ``parallel`` may overlap with
#: siblings; ``exclusive`` runs alone and forms an ordering barrier. (TS:
#: ``{ kind: 'parallel' } | { kind: 'exclusive' }``.)
ToolExecutionMode = Literal['parallel', 'exclusive']

#: A monotonic execution guard evaluated after every ``tools/pre-execute``
#: listener and before the tool body. Returning a reason denies the call;
#: returning ``None`` leaves it unchanged. Guards have no allow result, so
#: listener ordering cannot turn a denial back into permission.
ToolGuard = Callable[[ToolExecution], str | None]

#: How the registry presents its tools to the model. Only ``'native'`` is
#: supported by this port (Code Mode is not ported).
ToolPresentationMode = Literal['native', 'code', 'both']


@dataclass(frozen=True)
class ToolRestriction:
    """Per-scope filter over global tools.

    Restrictions intersect and do not affect scoped registrations or the
    reserved Code Mode transport.
    """

    #: Global tool names that stay visible; everything else is removed.
    allow: Sequence[str] | None = None
    #: Global tool names removed from visibility.
    deny: Sequence[str] | None = None


@dataclass(frozen=True)
class ScheduledDispatch:
    """Scheduler preparation outcome: run the around-dispatch/body stage."""

    exec: ToolExecution
    kind: Literal['dispatch'] = 'dispatch'


@dataclass(frozen=True)
class ScheduledPostResult:
    """Scheduler outcome: a result that still receives post-execute."""

    exec: ToolExecution
    result: ToolExecutionResult
    kind: Literal['post-result'] = 'post-result'


@dataclass(frozen=True)
class ScheduledFinalResult:
    """Scheduler outcome: a result that bypasses post-execute."""

    exec: ToolExecution
    result: ToolExecutionResult
    kind: Literal['final-result'] = 'final-result'


#: Scheduler-only result after ordered pre-execute and guards.
ScheduledToolPreparation = (
    ScheduledDispatch | ScheduledPostResult | ScheduledFinalResult
)


@dataclass(frozen=True)
class DispatchPostResult:
    """Scheduler dispatch outcome: a result that still receives post-execute."""

    result: ToolExecutionResult
    kind: Literal['post-result'] = 'post-result'


@dataclass(frozen=True)
class DispatchFinalResult:
    """Scheduler dispatch outcome already matching execute failure semantics."""

    result: ToolExecutionResult
    kind: Literal['final-result'] = 'final-result'


#: Scheduler-only dispatch result.
ScheduledToolDispatch = DispatchPostResult | DispatchFinalResult


class ToolRuntimeScheduler(Protocol):
    """Staged scheduler view keeping pre/post policy ordered.

    Ordinary callers use :meth:`ToolRuntime.execute`; this view exists for
    the agent loop's parallel scheduler and is not a plugin extension point.
    """

    async def prepare(self, exec: ToolExecutionInput) -> ScheduledToolPreparation:
        """Materialize input, run the ordered pre-execute/guard gate."""
        ...

    async def dispatch(self, exec: ToolExecution) -> ScheduledToolDispatch:
        """Run only the around-dispatch/body stage."""
        ...

    async def finalize(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Run post-execute and content finalization, then materialize/notify."""
        ...

    def finish(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Run content finalization, then materialize/notify (no post-execute)."""
        ...


@dataclass(frozen=True)
class Config:
    """Plugin config: how the registered tools are presented to the model."""

    #: Model presentation. Only ``'native'`` (every visible schema is sent)
    #: is supported; ``'code'``/``'both'`` require the unported Code Mode
    #: transport and fail validation loudly.
    mode: ToolPresentationMode = 'native'


#: Alias for annotations inside ``ToolRuntime``, whose ``Config`` class
#: attribute (the loader validator) shadows the dataclass name in class scope.
_ConfigData = Config


def _resolve_config(config: Config | dict[str, Any] | None) -> Config:
    """Validate a raw plugin config mapping into a ``Config``.

    Installed as ``ToolRuntime.Config`` so the Cordis loader applies it;
    also accepts an already-built ``Config`` for direct construction.
    """
    if config is None:
        resolved = Config()
    elif isinstance(config, Config):
        resolved = config
    elif isinstance(config, dict):
        unknown = sorted(set(config) - {'mode'})
        if unknown:
            raise ValidationError(
                f'unknown tools config keys: {", ".join(unknown)}'
            )
        resolved = Config(mode=config.get('mode', 'native'))
    else:
        raise ValidationError(f'invalid tools config: {type(config).__name__}')
    if resolved.mode not in ('native', 'code', 'both'):
        raise ValidationError(
            f'invalid tools mode: {resolved.mode!r} (expected native/code/both)'
        )
    if resolved.mode != 'native':
        raise ValidationError(
            f'tools mode "{resolved.mode}" requires the Code Mode '
            'presentation transport (packages/core/tools code-mode.ts), which '
            'the Python rewrite has not ported yet; use mode "native"'
        )
    return resolved


class ToolLayer:
    """One scope's complete tool-registry contribution."""

    def __init__(self, scope: ScopeKey | None) -> None:
        hint = (
            ' (for a per-agent variant, register through that agent\'s '
            '`agent.ctx` instead)'
            if scope is None
            else ' in this scope'
        )
        self.tools: NamedEntries[ToolDefinition] = NamedEntries(
            lambda name: ValueError(f'tool "{name}" is already registered{hint}')
        )
        self.restrictions: AnonymousEntries[_CompiledToolRestriction] = (
            AnonymousEntries()
        )
        self.guards: AnonymousEntries[ToolGuard] = AnonymousEntries()

    def is_empty(self) -> bool:
        """Whether every contribution table in this layer is empty."""
        return (
            self.tools.is_empty()
            and self.restrictions.is_empty()
            and self.guards.is_empty()
        )

    def admits(self, name: str) -> bool:
        """Whether every compiled restriction in this layer admits a name."""
        for filter_ in self.restrictions.values():
            if (filter_.allow is not None and name not in filter_.allow) or (
                filter_.deny is not None and name in filter_.deny
            ):
                return False
        return True

    def guard_reason(self, exec: ToolExecution) -> str | None:
        """First monotonic denial from this layer's live guard registrations."""
        for guard in self.guards.values():
            reason = guard(exec)
            if reason is not None:
                return reason
        return None


@dataclass(frozen=True)
class _CompiledToolRestriction:
    """One restriction compiled at registration for repeated lookup."""

    allow: frozenset[str] | None = None
    deny: frozenset[str] | None = None


@dataclass(frozen=True)
class _ToolView:
    """One scope's complete registry view, derived in one layer traversal."""

    #: Visible definitions after restrictions and scoped shadowing.
    visible: dict[str, ToolDefinition]
    #: Pre-restriction capability names used by prompt-order validation.
    known_names: list[str]
    #: Current global names that a scoped restriction may name.
    restrictable_names: set[str]


@dataclass(frozen=True)
class _ToolAskResolution:
    """Approval decision plus whether the approval channel reported cancellation."""

    decision: PreToolAllow | PreToolDeny
    approval_cancelled: bool


@dataclass
class _ToolCancellationState:
    """Caller cancellation and dispatch state kept outside the wrapper view."""

    caller_signal: AbortSignal
    body_invoked: bool = False


@dataclass(frozen=True)
class _ReadyExecution:
    """Internal ``create_execution`` outcome for a call entering policy."""

    exec: ToolExecution


@dataclass(frozen=True)
class _FusedToolSignal:
    """One dispatch-scoped fused signal plus listener cleanup."""

    signal: AbortSignal
    dispose: Callable[[], None]


def _fuse_tool_signals(caller: AbortSignal, wrapper: AbortSignal) -> _FusedToolSignal:
    """Fuse caller and wrapper cancellation with dispatch-scoped relaying.

    Keeping the relay dispatch-scoped removes its listeners when the work
    settles.
    """
    if caller is wrapper:
        return _FusedToolSignal(caller, lambda: None)

    controller = AbortController()
    listening = False

    def dispose() -> None:
        nonlocal listening
        if not listening:
            return
        listening = False
        caller.remove_listener(abort_from_caller)
        wrapper.remove_listener(abort_from_wrapper)

    def abort_from(source: AbortSignal) -> None:
        controller.abort(source.reason)
        dispose()

    def abort_from_caller() -> None:
        abort_from(caller)

    def abort_from_wrapper() -> None:
        abort_from(wrapper)

    if wrapper.aborted:
        abort_from_wrapper()
    elif caller.aborted:
        abort_from_caller()
    else:
        listening = True
        caller.add_listener(abort_from_caller)
        wrapper.add_listener(abort_from_wrapper)
    return _FusedToolSignal(controller.signal, dispose)


def _error_message(error: Any) -> str:
    """Best-effort human-readable message from an arbitrary thrown value.

    Exception instances use ``str()``; non-Exception objects with a string
    ``message`` attribute use it too; everything else is stringified.
    """
    try:
        if isinstance(error, BaseException):
            return str(error)
        message = getattr(error, 'message', None)
        if isinstance(message, str):
            return message
        return str(error)
    except Exception:
        # Error normalization is the outermost safety boundary, so its
        # fallback must itself be total.
        return '<unprintable thrown value>'


def _error_info(error: Any) -> ToolErrorInfo | None:
    """Structured ``{name, code}`` for a thrown HarnessError, else None."""
    if isinstance(error, HarnessError):
        return ToolErrorInfo(type(error).__name__, error.code)
    return None


def _tool_error_result(error: Any) -> ToolExecutionFailure:
    info = _error_info(error)
    message = _error_message(error)
    return ToolExecutionFailure(
        error=ToolFailure(message, info),
        content=[{'type': 'text', 'text': f'Error: {message}'}],
    )


def _tool_aborted_result(
    prior: ToolExecutionResult | None = None,
) -> ToolExecutionResult:
    """Canonical result when cancellation supersedes success after invocation."""
    return ToolExecutionFailure(
        error=ToolFailure(
            'tool call aborted',
            ToolErrorInfo('AbortError', TOOL_ABORTED),
        ),
        content=[{'type': 'text', 'text': 'Error: tool call aborted'}],
        additional_contexts=prior.additional_contexts if prior is not None else (),
    )


def _tool_aborted_before_dispatch_result(
    prior: ToolExecutionResult | None = None,
) -> ToolExecutionResult:
    """Canonical result when cancellation prevents tool body invocation."""
    return ToolExecutionFailure(
        error=ToolFailure(
            'tool call aborted before dispatch',
            ToolErrorInfo('AbortError', TOOL_ABORTED_BEFORE_DISPATCH),
        ),
        content=[{'type': 'text', 'text': 'Error: tool call aborted before dispatch'}],
        additional_contexts=prior.additional_contexts if prior is not None else (),
    )


def _failure_message_from_content(content: list[ContentBlock]) -> str:
    """Derive one failure message from policy feedback without changing it."""
    text = '\n'.join(
        block['text'] if block['type'] == 'text' else f'[{block["type"]} content]'
        for block in content
    )
    return text if len(text) > 0 else 'tool result blocked by post-execute policy'


def _snapshot_tool_value(tool_name: str, candidate: Any) -> JsonValue:
    """Snapshot one body or policy value into the canonical failure class."""
    detached = snapshot_json_value(candidate)
    if detached is SNAPSHOT_FAILED:
        raise ToolOutputError(tool_name, ['value is not lossless JSON'])
    return cast(JsonValue, detached)


def _projection_error(
    tool_name: str,
    projector: Literal['render', 'presentationMeta'],
    error: Any,
) -> ToolOutputError:
    """Convert one projector exception into the canonical invalid-output failure."""
    return ToolOutputError(
        tool_name,
        [f'output.{projector} failed: {_error_message(error)}'],
    )


def _snapshot_projection(
    tool_name: str,
    projector: Literal['render', 'presentationMeta'],
    candidate: Any,
) -> Any:
    """Snapshot one projector result before durable-result materialization."""
    detached = snapshot_json_value(candidate)
    if detached is SNAPSHOT_FAILED:
        raise ToolOutputError(
            tool_name, [f'output.{projector} returned non-lossless JSON']
        )
    return detached


class _RuntimeScheduler:
    """Bound staged view handed to the loop's parallel scheduler.

    Python mapping of the TS ``TOOL_RUNTIME_SCHEDULER`` symbol-keyed member.
    """

    def __init__(self, runtime: ToolRuntime) -> None:
        self._runtime = runtime

    async def prepare(self, exec: ToolExecutionInput) -> ScheduledToolPreparation:
        return await self._runtime._prepare_scheduled_execution(exec)

    async def dispatch(self, exec: ToolExecution) -> ScheduledToolDispatch:
        return await self._runtime._dispatch_scheduled_execution(exec)

    async def finalize(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        return await self._runtime._finalize_scheduled_execution(exec, result)

    def finish(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        return self._runtime._finish_scheduled_execution(exec, result)


class ToolRuntime(Service):
    """Tool registry and execution pipeline.

    Scoped registrations shadow globals; one visibility resolver feeds
    presentation, lookup, and dispatch. Registered through an agent's
    ``agent.ctx``, contributions are owned by that scope and removed when
    its fiber unloads.
    """

    inject: ClassVar[list[str]] = ['systemPrompt']

    #: Cordis loader config hook: raw mapping -> validated ``Config``.
    Config = staticmethod(_resolve_config)

    def __init__(
        self,
        ctx: Context,
        config: _ConfigData | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ctx, 'tools')
        _resolve_config(config)
        self._layers: ScopedLayers[ToolLayer] = ScopedLayers(
            ToolLayer,
            lambda: self.ctx.emit('tools/change'),
        )
        #: Original caller cancellation, kept outside the wrapper-mutable
        #: execution object.
        self._cancellation_states: weakref.WeakKeyDictionary[
            ToolExecution, _ToolCancellationState
        ] = weakref.WeakKeyDictionary()
        #: Definition-owned final content transform snapshotted before policy.
        self._content_finalizers: weakref.WeakKeyDictionary[
            ToolExecution, Callable[
                [ToolExecution, ToolExecutionResult], list[ContentBlock] | None
            ] | None
        ] = weakref.WeakKeyDictionary()
        #: Registry-normalized results and the dispatch that validated each.
        self._canonical_results: weakref.WeakKeyDictionary[
            ToolExecutionResult, ToolExecutionToken
        ] = weakref.WeakKeyDictionary()
        #: Internal staged view consumed by the agent loop's scheduler.
        self.scheduler: ToolRuntimeScheduler = _RuntimeScheduler(self)
        system_prompt = ctx.get('systemPrompt')
        system_prompt.tools(lambda context: self._wire_schemas(context.scope))

    # -- registration -----------------------------------------------------

    def register(self, definition: ToolDefinition) -> EffectDisposer:
        """Register globally or in the calling agent scope.

        Scoped tools shadow globals; duplicates within one layer and the
        reserved ``run_code`` name fail.

        :param definition: tool schema, execution, and optional
            finalization callbacks.
        :return: the exact disposer that unregisters the tool.
        """
        name = definition.name
        output = definition.output
        if (
            output is None
            or not callable(getattr(output, 'render', None))
            or (
                getattr(output, 'presentation_meta', None) is not None
                and not callable(output.presentation_meta)
            )
        ):
            raise TypeError(
                f'tool "{name}" must declare output '
                '{ schema, render, presentationMeta? }'
            )
        assert_supported_json_schema(output.schema)
        timeout_ms = definition.timeout_ms
        if timeout_ms is not None and (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, (int, float))
            or not math.isfinite(timeout_ms)
            or timeout_ms <= 0
        ):
            raise TypeError(
                f'tool "{name}" timeoutMs must be a positive finite number'
            )
        if name == RUN_CODE_NAME:
            raise ValueError(
                f'tool name "{RUN_CODE_NAME}" is reserved for the Code Mode '
                'presentation transport and cannot be registered or shadowed'
            )
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.tools.insert(name, definition),
            'tools.register()',
        )

    def restrict(self, filter: ToolRestriction) -> EffectDisposer:
        """Restrict global tools for the calling agent scope.

        Empty filters, unknown names, scope-local names, and reserved
        transport names fail. Restrictions intersect; scoped registrations
        remain visible.

        :param filter: global-tool mask: ``allow`` (keep only) and/or
            ``deny`` (remove).
        :return: the exact disposer that lifts this restriction.
        """
        scope = scope_of(self.ctx)
        if scope is None:
            raise ValueError(
                'tools.restrict() requires a scoped context (agent.ctx): a '
                'context-global restriction would mask every agent — deny '
                'the tool for the intended agent instead'
            )
        allow, deny = filter.allow, filter.deny
        if allow is None and deny is None:
            raise ValueError(
                'tools.restrict({}) is a no-op: pass `allow` and/or `deny` '
                '(an empty filter is almost always a materialized-empty-config '
                'bug)'
            )
        named = [*(allow or []), *(deny or [])]
        if RUN_CODE_NAME in named:
            raise ValueError(
                'tools.restrict() cannot name reserved Code Mode presentation '
                f'transport "{RUN_CODE_NAME}"; restrict end-capability tools '
                'instead'
            )
        known = self._view(scope).restrictable_names
        unknown = [name for name in named if name not in known]
        if unknown:
            plural = 's' if len(unknown) > 1 else ''
            listed = ', '.join(f'"{name}"' for name in unknown)
            raise ValueError(
                f'tools.restrict() names unknown global tool{plural} {listed}; '
                f'known global tools: {", ".join(sorted(known)) or "(none)"}'
            )
        compiled = _CompiledToolRestriction(
            frozenset(allow) if allow is not None else None,
            frozenset(deny) if deny is not None else None,
        )
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.restrictions.append(compiled),
            'tools.restrict()',
        )

    def guard(self, guard: ToolGuard) -> EffectDisposer:
        """Register a monotonic guard after the ``tools/pre-execute`` waterfall.

        A plain-context guard applies globally; one registered through
        ``agent.ctx`` applies only to that agent. Any matching guard may
        deny by returning a reason; no guard can force-allow a call another
        guard denied. Guard changes do not fire ``tools/change`` (they do
        not change the visible tool set).

        :param guard: synchronous check; a returned string denies the call.
        :return: the exact disposer that unregisters the guard.
        """
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.guards.append(guard),
            'tools.guard()',
            notify=False,
        )

    # -- lookup and presentation ------------------------------------------

    def get(self, name: str, scope: ScopeKey | None = None) -> ToolDefinition | None:
        """Look up a tool as one scope sees it.

        Scoped shadows global; a restricted-away global reads as absent.

        :param name: the tool name as registered.
        :param scope: the viewing scope (the agent); omitted = global view.
        :return: the definition the scope resolves, or None when none is
            visible.
        """
        return self._view(scope).visible.get(name)

    def schemas(self, scope: ScopeKey | None = None) -> list[ToolSchema]:
        """Project visible definitions onto the model-facing schema fields.

        Execution and finalization callbacks are excluded; parameters are
        deep-cloned.

        :param scope: the viewing scope (the agent); omitted = global view.
        :return: one deep-cloned schema per visible tool.
        """
        return [
            self._schema_of(definition, detach=True)
            for definition in self._view(scope).visible.values()
        ]

    def execution_mode(self, exec: ToolExecutionInput) -> ToolExecutionMode:
        """Classify a pending call through the caller's visible definition.

        Only an exact ``True`` is parallel; unknown, hidden, undeclared,
        invalid, or throwing classifiers are exclusive.

        :param exec: call name, parsed arguments, and optional agent scope.
        :return: the fail-closed scheduling mode.
        """
        tool = self.get(exec.name, exec.agent)
        if tool is None or tool.is_concurrency_safe is None:
            return 'exclusive'
        try:
            concurrency_safe: Any = tool.is_concurrency_safe(exec.arguments)
            return 'parallel' if concurrency_safe is True else 'exclusive'
        except Exception:
            return 'exclusive'

    # -- the pipeline ------------------------------------------------------

    async def execute(self, exec: ToolExecutionInput) -> ToolExecutionResult:
        """Run one call through the complete pipeline.

        Pre-policy, guards, around-dispatch, post-policy, definition-owned
        content finalization, and final notification. Tool and listener
        failures resolve as materialized error results; an invisible tool
        reports ``UNKNOWN_TOOL``. Cancellation arriving after entry and
        before final materialization skips a not-yet-started body with
        ``ABORTED_BEFORE_DISPATCH`` or replaces a successful started outcome
        with ``ABORTED``; already-started work is still drained.

        :param exec: the typed same-process call input.
        :return: the materialized final result.
        """
        return cast(
            ToolExecutionResult,
            await self._prepare_execution(exec, self._complete_scheduled_execution),
        )

    async def _prepare_scheduled_execution(
        self,
        input_: ToolExecutionInput,
    ) -> ScheduledToolPreparation:
        """Run the ordered pre-execute and guard stages for the scheduler."""
        return cast(
            ScheduledToolPreparation,
            await self._prepare_execution(input_, lambda prepared: prepared),
        )

    async def _complete_scheduled_execution(
        self,
        prepared: ScheduledToolPreparation,
    ) -> ToolExecutionResult:
        if isinstance(prepared, ScheduledDispatch):
            dispatched = await self._dispatch_scheduled_execution(prepared.exec)
            if isinstance(dispatched, DispatchPostResult):
                return await self._finalize_scheduled_execution(
                    prepared.exec, dispatched.result,
                )
            return self._finish_scheduled_execution(prepared.exec, dispatched.result)
        if isinstance(prepared, ScheduledPostResult):
            return await self._finalize_scheduled_execution(
                prepared.exec, prepared.result,
            )
        return self._finish_scheduled_execution(prepared.exec, prepared.result)

    def _create_execution(
        self,
        input_: ToolExecutionInput,
    ) -> ScheduledToolPreparation | _ReadyExecution:
        """Materialize one input into a live execution (or an early failure).

        The finalizer is captured BEFORE argument materialization: the
        ``finalize_content`` contract snapshots the callback when the call
        starts. Arguments cross one lossless-JSON materialization boundary
        before policy.
        """
        token = ToolExecutionToken()
        call_id = input_.call_id
        root_call_id = (
            input_.root_call_id if input_.root_call_id is not None else call_id
        )
        visible = self.get(input_.name, input_.agent)
        captured_finalizer = (
            visible.finalize_content if visible is not None else None
        )
        try:
            detached = snapshot_json_value(input_.arguments)
            if detached is SNAPSHOT_FAILED:
                raise TypeError(
                    'tool execution arguments must be losslessly JSON-serializable'
                )
            execution = ToolExecution(
                token=token,
                call_id=call_id,
                root_call_id=root_call_id,
                name=input_.name,
                arguments=detached,
                signal=input_.signal,
                agent=input_.agent,
                parent=input_.parent,
            )
            self._content_finalizers[execution] = captured_finalizer
            self._cancellation_states[execution] = _ToolCancellationState(
                input_.signal,
            )
            return _ReadyExecution(execution)
        except Exception as error:
            execution = ToolExecution(
                token=token,
                call_id=call_id,
                root_call_id=root_call_id,
                name=input_.name,
                arguments=None,
                signal=input_.signal,
                agent=input_.agent,
                parent=input_.parent,
            )
            self._content_finalizers[execution] = captured_finalizer
            return ScheduledFinalResult(execution, _tool_error_result(error))

    async def _prepare_execution(
        self,
        input_: ToolExecutionInput,
        next_: Callable[[ScheduledToolPreparation], Any],
    ) -> Any:
        """Run pre-execute and guards, then hand the next stage to ``next_``."""
        created = self._create_execution(input_)
        if not isinstance(created, _ReadyExecution):
            return await maybe_await(next_(created))
        exec = created.exec
        if self._caller_cancelled(exec):
            return await maybe_await(next_(
                ScheduledFinalResult(exec, _tool_aborted_before_dispatch_result()),
            ))
        async def allow(*_args: Any) -> PreToolDecision:
            return PreToolAllow()

        try:
            carrier = scope_target(self, exec.agent)
            gate: PreToolDecision = await maybe_await(self.ctx.waterfall(
                carrier,
                'tools/pre-execute',
                exec,
                allow,
            ))
            if isinstance(gate, PreToolAsk):
                resolution = await self._service_ask(exec, gate)
            else:
                resolution = _ToolAskResolution(gate, False)
            decision = resolution.decision
            if self._caller_cancelled(exec) and resolution.approval_cancelled:
                return await maybe_await(next_(
                    ScheduledPostResult(exec, _tool_aborted_before_dispatch_result()),
                ))
            denial_reason = (
                self._guard_reason(exec)
                if decision.kind == 'allow'
                else decision.reason
            )
            if denial_reason is not None:
                return await maybe_await(next_(ScheduledPostResult(
                    exec,
                    self._materialize_final_result(ToolExecutionFailure(
                        error=ToolFailure(denial_reason),
                        content=[{'type': 'text', 'text': f'Error: {denial_reason}'}],
                    )),
                )))
            if self._caller_cancelled(exec):
                return await maybe_await(next_(
                    ScheduledPostResult(exec, _tool_aborted_before_dispatch_result()),
                ))
            return await maybe_await(next_(ScheduledDispatch(exec)))
        except Exception as error:
            return await maybe_await(next_(
                ScheduledFinalResult(exec, _tool_error_result(error)),
            ))

    def _caller_cancelled(self, exec: ToolExecution) -> bool:
        """Whether the original caller signal is currently aborted."""
        state = self._cancellation_states.get(exec)
        if state is None:
            raise RuntimeError(
                'tool registry scheduler invariant violated: missing '
                'cancellation state'
            )
        return state.caller_signal.aborted

    def _cancellation_result(
        self,
        exec: ToolExecution,
        prior: ToolExecutionResult | None = None,
    ) -> ToolExecutionResult:
        """Canonical cancellation outcome selected by whether the body started."""
        state = self._cancellation_states.get(exec)
        if state is None:
            raise RuntimeError(
                'tool registry scheduler invariant violated: missing '
                'cancellation state'
            )
        if state.body_invoked:
            return _tool_aborted_result(prior)
        return _tool_aborted_before_dispatch_result(prior)

    async def _dispatch_tool_body(self, exec: ToolExecution) -> ToolExecutionResult:
        """Dispatch the body with the caller signal fused into replacements.

        Cancellation never abandons the body: a started coroutine reaches
        quiescence before its outcome becomes ``ABORTED``.
        """
        state = self._cancellation_states.get(exec)
        if state is None:
            raise RuntimeError(
                'tool registry scheduler invariant violated: missing '
                'cancellation state'
            )
        wrapper_signal = exec.signal
        fused = _fuse_tool_signals(state.caller_signal, wrapper_signal)
        signal = fused.signal

        if signal.aborted:
            fused.dispose()
            return _tool_aborted_before_dispatch_result()
        exec.signal = signal
        try:
            tool = self.get(exec.name, exec.agent)
            if tool is None:
                raise ToolNotFoundError(exec.name)
            state.body_invoked = True
            returned = await maybe_await(tool.execute(exec.arguments, exec))
            result = self._create_success_result(exec, tool, returned)
            return _tool_aborted_result(result) if signal.aborted else result
        except Exception as error:
            return _tool_error_result(error)
        finally:
            fused.dispose()
            exec.signal = wrapper_signal

    async def _dispatch_scheduled_execution(
        self,
        exec: ToolExecution,
    ) -> ScheduledToolDispatch:
        """Run around-dispatch and the tool body.

        Tool and unknown-tool failures still receive post-execute; pipeline
        failures are already final.
        """
        try:
            carrier = scope_target(self, exec.agent)
            result: ToolExecutionResult = await maybe_await(self.ctx.waterfall(
                carrier,
                'tools/execute',
                exec,
                lambda *_args: self._dispatch_tool_body(exec),
            ))
            normalized = self._normalize_dispatch_result(exec, result)
            deferred = exec._deferred_contexts
            if deferred:
                normalized = self._mark_canonical(exec, dataclasses.replace(
                    normalized,
                    additional_contexts=(
                        *deferred,
                        *normalized.additional_contexts,
                    ),
                ))
            if self._caller_cancelled(exec) and not normalized.is_error:
                normalized = self._cancellation_result(exec, normalized)
            return DispatchPostResult(normalized)
        except Exception as error:
            return DispatchFinalResult(_tool_error_result(error))

    async def _finalize_scheduled_execution(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Run ordered post-execute, then finish (materialize and notify)."""
        try:
            post_result = await self._post_execute(exec, result)
            if self._caller_cancelled(exec) and not post_result.is_error:
                post_result = self._cancellation_result(exec, post_result)
            return self._finish_scheduled_execution(exec, post_result)
        except Exception as error:
            return self._finish_scheduled_execution(exec, _tool_error_result(error))

    def _finish_scheduled_execution(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Materialize, apply definition-owned finalization, and notify."""
        try:
            materialized = self._materialize_final_result(result)
        except Exception as error:
            materialized = self._materialize_final_result(_tool_error_result(error))
        try:
            final = self._materialize_final_result(
                self._apply_final_content(exec, materialized),
            )
        except Exception as error:
            final = self._materialize_final_result(_tool_error_result(error))
        self._notify_result(exec, final)
        return final

    def _apply_final_content(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Apply the snapshotted tool-owned content transform."""
        finalize_content = self._content_finalizers.get(exec)
        if finalize_content is None:
            return result
        content = finalize_content(exec, result)
        if content is None:
            return result
        return dataclasses.replace(result, content=content)

    def _notify_result(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> None:
        """Notify observers without exposing a mutation or error channel."""
        # Freeze the registry's live object before observers receive its
        # readonly view.
        exec._freeze()
        tool_name, call_id = exec.name, exec.call_id

        def report_failure(error: BaseException) -> None:
            self.ctx.logger.warn(
                f'tool "{tool_name}" ({call_id}): tools/result observer '
                f'failed: {_error_message(error)}'
            )

        args: list[Any] = [
            scope_target(self, exec.agent),
            'tools/result',
            exec,
            result,
        ]
        callbacks = self.ctx.events.dispatch('emit', args)
        for callback in callbacks:
            try:
                returned: Any = callback(*args)
            except Exception as error:
                report_failure(error)
                continue
            if inspect.isawaitable(returned):
                schedule_logged(returned, report_failure)

    async def _service_ask(
        self,
        exec: ToolExecution,
        ask: PreToolAsk,
    ) -> _ToolAskResolution:
        """Resolve an ``ask`` decision to allow/deny through the approval seam.

        The seam is consumed opportunistically with ``ctx.get('approval')``:
        a deployment that composes no approval service degrades to deny, an
        unmount mid-session degrades the same way on the next ask, and an
        agent-less execution also degrades (no session to audit to, no UI to
        route to). Otherwise the outcome maps one-to-one — ``allowed-once``
        proceeds; the three non-grants deny with distinct reasons.
        """
        approval = self.ctx.get('approval')
        if approval is None:
            return _ToolAskResolution(
                PreToolDeny(
                    ask.reason
                    if ask.reason is not None
                    else f'tool "{exec.name}" requires approval (not yet supported)'
                ),
                False,
            )
        if exec.agent is None:
            return _ToolAskResolution(
                PreToolDeny(
                    f'tool "{exec.name}" requires approval, but the call has '
                    'no agent to route it through'
                ),
                False,
            )
        request = ApprovalRequest(
            agent=exec.agent,
            tool_name=exec.name,
            call_id=exec.call_id,
            reason=ask.reason,
            signal=exec.signal,
        )
        outcome = await maybe_await(approval.request(request))
        if outcome == 'allowed-once':
            return _ToolAskResolution(PreToolAllow(), False)
        if outcome == 'rejected':
            return _ToolAskResolution(
                PreToolDeny(f'the user rejected tool "{exec.name}"'), False,
            )
        if outcome == 'cancelled':
            return _ToolAskResolution(
                PreToolDeny(f'approval for tool "{exec.name}" was cancelled'),
                True,
            )
        if outcome == 'unavailable':
            return _ToolAskResolution(
                PreToolDeny(
                    f'tool "{exec.name}" requires approval, but no approval '
                    'channel is available'
                ),
                False,
            )
        raise ValueError(f'unexpected approval outcome: {outcome!r}')

    async def _post_execute(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Run ``tools/post-execute`` over ``result`` and apply its decision.

        ``accept`` keeps the call successful (replacing ``content`` when
        given, or re-running the output contract for ``value``); ``block``
        turns it into an error whose content is the corrective ``feedback``.
        Either decision may attach ``additional_contexts``. Context deferred
        by the tool body survives an accepted result but is discarded when
        the call is blocked. Runs inside the outer containment (a throwing
        listener becomes an error result).
        """
        async def accept(*_args: Any) -> PostToolDecision:
            return PostToolAccept()

        decision: PostToolDecision = await maybe_await(self.ctx.waterfall(
            scope_target(self, exec.agent),
            'tools/post-execute',
            exec,
            result,
            accept,
        ))
        decision_contexts = tuple(decision.additional_contexts or ())
        if isinstance(decision, PostToolBlock):
            message = _failure_message_from_content(decision.feedback)
            return self._mark_canonical(exec, ToolExecutionFailure(
                error=ToolFailure(message),
                content=decision.feedback,
                additional_contexts=decision_contexts,
            ))
        has_value = decision.value is not _UNSET
        if decision.content is not None and has_value:
            raise TypeError(
                'tools/post-execute accept decision cannot replace both '
                'value and content'
            )
        additional_contexts = (*result.additional_contexts, *decision_contexts)
        if has_value:
            if result.is_error:
                raise TypeError(
                    'tools/post-execute cannot replace the value of a failed '
                    'result'
                )
            tool = self.get(exec.name, exec.agent)
            if tool is None:
                raise ToolNotFoundError(exec.name)
            replaced = self._create_success_result(exec, tool, decision.value)
            return self._mark_canonical(exec, dataclasses.replace(
                replaced,
                additional_contexts=additional_contexts,
            ))
        return self._mark_canonical(exec, dataclasses.replace(
            result,
            content=(
                decision.content
                if decision.content is not None
                else result.content
            ),
            additional_contexts=additional_contexts,
        ))

    # -- normalization and materialization ---------------------------------

    def _mark_canonical(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Mark one registry-normalized result canonical for its dispatch."""
        self._canonical_results[result] = exec.token
        return result

    def _create_success_result(
        self,
        exec: ToolExecution,
        tool: ToolDefinition,
        candidate: Any,
    ) -> ToolExecutionSuccess:
        """Snapshot, validate, render, and optionally project one body value."""
        detached = _snapshot_tool_value(tool.name, candidate)
        violations = validate_json_schema_value(tool.output.schema, detached, 'value')
        if violations:
            raise ToolOutputError(tool.name, violations)
        value = detached
        try:
            rendered = tool.output.render(exec.arguments, value)
        except Exception as error:
            raise _projection_error(tool.name, 'render', error) from error
        content = _snapshot_projection(tool.name, 'render', rendered)
        meta: Any = _UNSET
        if exec.parent is None and tool.output.presentation_meta is not None:
            try:
                projected = tool.output.presentation_meta(exec.arguments, value)
            except Exception as error:
                raise _projection_error(tool.name, 'presentationMeta', error) from error
            meta = _snapshot_projection(tool.name, 'presentationMeta', projected)
        concludes_turn = exec._concluded
        materialized = self._materialize_final_result(ToolExecutionSuccess(
            value=value,
            content=cast(list[ContentBlock], content),
            meta=meta,
            concludes_turn=concludes_turn,
        ))
        return cast(ToolExecutionSuccess, self._mark_canonical(exec, materialized))

    def _normalize_dispatch_result(
        self,
        exec: ToolExecution,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Normalize a wrapper's authored result through the output contract."""
        if isinstance(result, (ToolExecutionSuccess, ToolExecutionFailure)) and (
            self._canonical_results.get(result) is exec.token
        ):
            return result
        if result.is_error:
            return self._mark_canonical(exec, ToolExecutionFailure(
                error=result.error,
                content=result.content,
                meta=result.meta,
                additional_contexts=result.additional_contexts,
            ))
        tool = self.get(exec.name, exec.agent)
        if tool is None:
            raise ToolNotFoundError(exec.name)
        normalized = self._create_success_result(exec, tool, result.value)
        return self._mark_canonical(exec, dataclasses.replace(
            normalized,
            additional_contexts=result.additional_contexts,
        ))

    def _materialize_final_result(
        self,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Materialize the authoritative commit outcome before ``tools/result``.

        Detaches the JSON-facing fields (``content``, ``meta``); results
        are frozen dataclasses treated as read-only, standing in for the TS
        ``deepFreeze``. ``additional_contexts`` are typed ``UserMessage``
        objects, copied rather than snapshotted.
        """
        content = snapshot_json_value(result.content)
        if content is SNAPSHOT_FAILED:
            raise TypeError('tool result must be losslessly JSON-serializable')
        meta: Any = _UNSET
        if result.meta is not _UNSET:
            meta = snapshot_json_value(result.meta)
            if meta is SNAPSHOT_FAILED:
                raise TypeError('tool result must be losslessly JSON-serializable')
        if isinstance(result, ToolExecutionFailure):
            return ToolExecutionFailure(
                error=result.error,
                content=cast(list[ContentBlock], content),
                meta=meta,
                additional_contexts=tuple(result.additional_contexts),
            )
        return ToolExecutionSuccess(
            value=result.value,
            content=cast(list[ContentBlock], content),
            meta=meta,
            additional_contexts=tuple(result.additional_contexts),
            concludes_turn=result.concludes_turn,
        )

    # -- scope views --------------------------------------------------------

    def _wire_schemas(self, scope: ScopeKey | None) -> ToolProviderResult:
        """Build one scope's wire schemas and names for prompt-order validation."""
        view = self._view(scope)
        schemas = [
            self._schema_of(definition, detach=False)
            for definition in view.visible.values()
        ]
        return ToolProviderResult(schemas=schemas, known_names=view.known_names)

    def _schema_of(self, definition: ToolDefinition, detach: bool) -> ToolSchema:
        """Project one definition onto the model-facing schema fields."""
        parameters: Any = definition.parameters
        if detach:
            detached = snapshot_json_value(parameters)
            if detached is SNAPSHOT_FAILED:
                raise ValueError(
                    f'tool "{definition.name}" parameters must be lossless '
                    'JSON before schema projection'
                )
            parameters = detached
        return ToolSchema(
            name=definition.name,
            description=definition.description,
            parameters=parameters,
        )

    def _guard_reason(self, exec: ToolExecution) -> str | None:
        """First monotonic denial from the global then the scope chain's guards."""
        global_reason = self._layers.global_.guard_reason(exec)
        if global_reason is not None:
            return global_reason
        if exec.agent is None:
            return None
        for layer in self._layers.chain_layers(exec.agent):
            reason = layer.guard_reason(exec)
            if reason is not None:
                return reason
        return None

    def _view(self, scope: ScopeKey | None) -> _ToolView:
        """Resolve every registry fact one scope needs in one traversal.

        The visible map applies restrictions to the INHERITED surface, then
        the scope's own registrations; the other sets retain the
        pre-restriction facts needed by restriction and prompt-order
        validation. A restriction filters what a scope inherits — the global
        layer and every ancestor layer on its chain — and never what its OWN
        layer registers.
        """
        # Scope-chain layers, farthest ancestor first, the exact scope last.
        layers = self._layers.chain_layers(scope)
        # Chain-blind on purpose: the ONE layer whose registrations the scope
        # owns rather than inherits.
        own = self._layers.peek(scope)
        # Inherited surface, nearest ancestor last: a nearer scope's same-name
        # entry shadows a farther one, and the global layer is the farthest.
        inherited = dict(self._layers.global_.tools.entries())
        for layer in layers:
            if layer is own:
                continue
            for name, definition in layer.tools.entries():
                inherited[name] = definition
        visible: dict[str, ToolDefinition] = {}
        known: dict[str, None] = {}
        restrictable: set[str] = set()
        for name, definition in inherited.items():
            known[name] = None
            restrictable.add(name)
            # Restrictions intersect across the whole chain: any scope on it
            # may mask an inherited name for everything nested inside it.
            if all(layer.admits(name) for layer in layers):
                visible[name] = definition
        # The scope's own registrations last, shadowing an inherited name and
        # outside the filter above.
        if own is not None:
            for name, definition in own.tools.entries():
                known[name] = None
                visible[name] = definition
        return _ToolView(
            visible=visible,
            known_names=list(known),
            restrictable_names=restrictable,
        )
