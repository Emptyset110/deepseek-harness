"""Behavior tests for the pydsh.tools runtime (``ctx.tools``).

Ported-semantics tests for the Python port of deepseek-harness
``packages/core/tools/src/index.ts`` (MIT): registration lifecycle, the
pre-execute/guard/execute/post-execute pipeline, the approval ``ask``
bridge, cancellation, result normalization, and the package invariant.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import pytest

from pycordis import Context, LoggerLevel, ValidationError
from pydsh.agent import UserMessage
from pydsh.approval import ApprovalRequest
from pydsh.llm import CallId
from pydsh.scope import create_scope, scope_target
from pydsh.subprocess.types import AbortController
from pydsh.system_prompt import AssembleContext, SystemPrompt
from pydsh.tools import (
    RUN_CODE_NAME,
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    JsonSchemaError,
    PostToolAccept,
    PostToolBlock,
    PreToolAllow,
    PreToolAsk,
    PreToolDeny,
    ToolDefinition,
    ToolErrorInfo,
    ToolExecutionInput,
    ToolExecutionSuccess,
    ToolFailure,
    ToolNotFoundError,
    ToolOutputDefinition,
    ToolRestriction,
    ToolRuntime,
    invariant,
)


def text(value: str) -> dict:
    return {'type': 'text', 'text': value}


def make_definition(
    name: str = 'tool_a',
    *,
    execute: Any = None,
    output_schema: Any = None,
    render: Any = None,
    presentation_meta: Any = None,
    finalize_content: Any = None,
    is_concurrency_safe: Any = None,
    timeout_ms: Any = None,
    parameters: Any = None,
) -> ToolDefinition:
    async def default_execute(args: Any, exec: Any) -> Any:
        return 'ok'

    return ToolDefinition(
        name=name,
        description=f'{name} description',
        parameters=(
            parameters if parameters is not None
            else {'type': 'object', 'properties': {}}
        ),
        output=ToolOutputDefinition(
            schema=output_schema if output_schema is not None else {'type': 'string'},
            render=(
                render if render is not None
                else lambda args, value: [text(str(value))]
            ),
            presentation_meta=presentation_meta,
        ),
        execute=execute if execute is not None else default_execute,
        finalize_content=finalize_content,
        timeout_ms=timeout_ms,
        is_concurrency_safe=is_concurrency_safe,
    )


def make_input(
    name: str = 'tool_a',
    *,
    args: Any = None,
    signal: Any = None,
    agent: Any = None,
    parent: Any = None,
    call_id: str = 'call-1',
) -> ToolExecutionInput:
    return ToolExecutionInput(
        call_id=CallId(call_id),
        name=name,
        arguments=args if args is not None else {},
        signal=signal if signal is not None else AbortController().signal,
        agent=agent,
        parent=parent,
    )


@pytest.fixture
async def ctx() -> AsyncGenerator[Context]:
    context = Context()
    await context.plugin(SystemPrompt)
    await context.plugin(ToolRuntime)
    yield context


def tools(ctx: Context) -> Any:
    return ctx.get('tools')


class WarnRecorder:
    """Exporter capturing warn records (the default level gate drops warns)."""

    levels: ClassVar[dict[str, LoggerLevel]] = {'default': LoggerLevel.WARN}

    def __init__(self) -> None:
        self.messages: list[str] = []

    def export(self, message: Any) -> None:
        if message.type == 'warn':
            self.messages.append(str(message.args[0]))


def record_warns(ctx: Context) -> list[str]:
    recorder = WarnRecorder()
    ctx.logger.exporter(recorder)
    return recorder.messages


# --- registration lifecycle ------------------------------------------------------


async def test_register_and_execute_success(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    assert result.value == 'ok'
    assert result.content == [text('ok')]


async def test_disposer_unregisters_and_notifies(ctx: Context) -> None:
    changes: list[str] = []
    ctx.on('tools/change', lambda: changes.append('change'))
    dispose = tools(ctx).register(make_definition())
    assert tools(ctx).get('tool_a') is not None
    dispose()
    assert tools(ctx).get('tool_a') is None
    assert changes == ['change', 'change']


async def test_duplicate_registration_messages(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    with pytest.raises(ValueError) as exc_info:
        tools(ctx).register(make_definition())
    assert 'tool "tool_a" is already registered' in str(exc_info.value)
    assert 'agent.ctx' in str(exc_info.value)

    scope = create_scope(ctx, object())
    scoped_tools = scope.ctx.get('tools')
    scoped_tools.register(make_definition('scoped_tool'))
    with pytest.raises(ValueError, match='already registered in this scope'):
        scoped_tools.register(make_definition('scoped_tool'))
    # a scoped registration may shadow a global name
    scoped_tools.register(make_definition())


async def test_reserved_run_code_name(ctx: Context) -> None:
    with pytest.raises(ValueError, match='reserved for the Code Mode'):
        tools(ctx).register(make_definition(RUN_CODE_NAME))


async def test_register_validates_output_and_timeout(ctx: Context) -> None:
    with pytest.raises(JsonSchemaError) as exc_info:
        tools(ctx).register(make_definition(output_schema={'type': 'date'}))
    assert exc_info.value.code == 'UNSUPPORTED_SCHEMA'
    with pytest.raises(TypeError, match='timeoutMs must be a positive finite'):
        tools(ctx).register(make_definition(timeout_ms=0))
    definition = make_definition()
    definition.output.render = None  # type: ignore[assignment]
    with pytest.raises(TypeError, match='must declare output'):
        tools(ctx).register(definition)


async def test_scoped_fiber_disposal_removes_tools(ctx: Context) -> None:
    key = object()
    scope = create_scope(ctx, key)
    scope.ctx.get('tools').register(make_definition('scoped_tool'))
    tools(ctx).register(make_definition())
    assert tools(ctx).get('scoped_tool', key) is not None
    assert tools(ctx).get('scoped_tool') is None

    await scope.dispose()
    assert tools(ctx).get('scoped_tool', key) is None
    assert tools(ctx).get('tool_a') is not None


async def test_scoped_tool_shadows_global(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    key = object()
    scope = create_scope(ctx, key)

    async def scoped_execute(args: Any, exec: Any) -> Any:
        return 'scoped'

    scope.ctx.get('tools').register(make_definition(execute=scoped_execute))
    result = await tools(ctx).execute(make_input(agent=key))
    assert result.value == 'scoped'
    root_result = await tools(ctx).execute(make_input())
    assert root_result.value == 'ok'


# --- restrict --------------------------------------------------------------------


async def test_restrict_requires_scoped_context(ctx: Context) -> None:
    with pytest.raises(ValueError, match='requires a scoped context'):
        tools(ctx).restrict(ToolRestriction(deny=['tool_a']))


async def test_restrict_rejects_empty_and_unknown(ctx: Context) -> None:
    scope = create_scope(ctx, object())
    scoped = scope.ctx.get('tools')
    with pytest.raises(ValueError, match=r'tools\.restrict\(\{\}\) is a no-op'):
        scoped.restrict(ToolRestriction())
    tools(ctx).register(make_definition())
    with pytest.raises(ValueError) as exc_info:
        scoped.restrict(ToolRestriction(deny=['nope']))
    assert 'names unknown global tool "nope"' in str(exc_info.value)
    assert 'known global tools: tool_a' in str(exc_info.value)
    with pytest.raises(ValueError, match='reserved Code Mode'):
        scoped.restrict(ToolRestriction(allow=[RUN_CODE_NAME]))


async def test_restrict_filters_inherited_but_not_own(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    tools(ctx).register(make_definition('tool_b'))
    key = object()
    scope = create_scope(ctx, key)
    scoped = scope.ctx.get('tools')
    scoped.register(make_definition('own_tool'))
    lift = scoped.restrict(ToolRestriction(deny=['tool_a']))

    assert tools(ctx).get('tool_a', key) is None
    assert tools(ctx).get('tool_b', key) is not None
    # a restriction never masks the scope's OWN registrations
    assert tools(ctx).get('own_tool', key) is not None
    # the global view is untouched
    assert tools(ctx).get('tool_a') is not None

    lift()
    assert tools(ctx).get('tool_a', key) is not None


async def test_restrictions_intersect_across_the_chain(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    tools(ctx).register(make_definition('tool_b'))
    key_a, key_b = object(), object()
    scope_a = create_scope(ctx, key_a)
    scope_b = create_scope(scope_a.ctx, key_b, parent=key_a)
    scope_a.ctx.get('tools').restrict(ToolRestriction(allow=['tool_a', 'tool_b']))
    scope_b.ctx.get('tools').restrict(ToolRestriction(deny=['tool_b']))

    assert tools(ctx).get('tool_a', key_b) is not None
    assert tools(ctx).get('tool_b', key_b) is None


# --- guard -----------------------------------------------------------------------


async def test_guard_denies_monotonically(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    seen: list[str] = []
    ctx.on('tools/pre-execute', lambda carrier, exec, next_: PreToolAllow())

    def guard(exec: Any) -> str | None:
        seen.append(exec.name)
        return 'guarded away'

    remove = tools(ctx).guard(guard)
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.content == [text('Error: guarded away')]
    assert result.error == ToolFailure('guarded away')
    assert seen == ['tool_a']

    remove()
    result = await tools(ctx).execute(make_input(call_id='call-2'))
    assert not result.is_error


async def test_guard_does_not_fire_change(ctx: Context) -> None:
    changes: list[str] = []
    ctx.on('tools/change', lambda: changes.append('change'))
    tools(ctx).guard(lambda exec: None)
    assert changes == []


async def test_guards_not_evaluated_after_policy_denial(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on('tools/pre-execute', lambda carrier, exec, next_: PreToolDeny('no'))
    called: list[str] = []
    tools(ctx).guard(lambda exec: called.append('guard') or 'x')
    result = await tools(ctx).execute(make_input())
    assert result.error.message == 'no'
    assert called == []


async def test_scoped_guard_applies_to_its_agent_only(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    key = object()
    scope = create_scope(ctx, key)
    scope.ctx.get('tools').guard(lambda exec: 'scoped denial')

    denied = await tools(ctx).execute(make_input(agent=key))
    assert denied.error.message == 'scoped denial'
    allowed = await tools(ctx).execute(make_input(call_id='call-2'))
    assert not allowed.is_error


# --- tools/pre-execute ---------------------------------------------------------


async def test_pre_execute_default_allows(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    hits: list[str] = []

    async def listener(carrier: Any, exec: Any, next_: Any) -> Any:
        hits.append(exec.name)
        return await next_()

    ctx.on('tools/pre-execute', listener)
    result = await tools(ctx).execute(make_input())
    assert hits == ['tool_a']
    assert not result.is_error


async def test_pre_execute_deny_short_circuits(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    calls: list[str] = []
    ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_: PreToolDeny('denied by policy'),
    )
    ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_: calls.append('second'),
    )
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.content == [text('Error: denied by policy')]
    assert result.error.message == 'denied by policy'
    assert result.error.info is None
    # a listener that does not call next() vetoes the rest of the chain
    assert calls == []


async def test_pre_execute_listener_failure_is_contained(ctx: Context) -> None:
    tools(ctx).register(make_definition())

    def broken(carrier: Any, exec: Any, next_: Any) -> Any:
        raise ValueError('policy bug')

    ctx.on('tools/pre-execute', broken)
    post_calls: list[str] = []
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: post_calls.append('post'),
    )
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.content == [text('Error: policy bug')]
    # pipeline failures are final: post-execute does not run
    assert post_calls == []


async def test_pre_execute_scoped_filtering(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    key_a, key_b = object(), object()
    scope_a = create_scope(ctx, key_a)
    create_scope(ctx, key_b)
    hits: list[str] = []
    scope_a.ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_: hits.append(exec.call_id) or PreToolAllow(),
    )

    await tools(ctx).execute(make_input(agent=key_a, call_id='a1'))
    await tools(ctx).execute(make_input(agent=key_b, call_id='b1'))
    await tools(ctx).execute(make_input(call_id='root'))
    assert hits == ['a1']


# --- ask -> approval bridge -----------------------------------------------------


class FakeApproval:
    """Stand-in approval service recording requests and returning an outcome."""

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.requests: list[ApprovalRequest] = []

    async def request(self, req: ApprovalRequest) -> str:
        self.requests.append(req)
        return self.outcome


def ask_listener(carrier: Any, exec: Any, next_: Any) -> Any:
    return PreToolAsk()


async def test_ask_without_approval_service_denies(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on('tools/pre-execute', ask_listener)
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.error.message == (
        'tool "tool_a" requires approval (not yet supported)'
    )


async def test_ask_uses_custom_reason_when_unsupported(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_: PreToolAsk('dangerous command'),
    )
    result = await tools(ctx).execute(make_input())
    assert result.error.message == 'dangerous command'


async def test_ask_without_agent_denies(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    approval = FakeApproval('allowed-once')
    ctx.provide('approval', approval)
    ctx.on('tools/pre-execute', ask_listener)
    result = await tools(ctx).execute(make_input())
    assert result.error.message == (
        'tool "tool_a" requires approval, but the call has no agent to route '
        'it through'
    )
    assert approval.requests == []


@pytest.mark.parametrize('outcome', ['allowed-once'])
async def test_ask_allowed_once_executes(ctx: Context, outcome: str) -> None:
    tools(ctx).register(make_definition())
    approval = FakeApproval(outcome)
    ctx.provide('approval', approval)
    agent = object()
    ctx.on('tools/pre-execute', lambda carrier, exec, next_: PreToolAsk('may I?'))
    result = await tools(ctx).execute(make_input(agent=agent))
    assert not result.is_error
    assert result.value == 'ok'
    request = approval.requests[0]
    assert request.agent is agent
    assert request.tool_name == 'tool_a'
    assert request.call_id == CallId('call-1')
    assert request.reason == 'may I?'


@pytest.mark.parametrize(
    ('outcome', 'message'),
    [
        ('rejected', 'the user rejected tool "tool_a"'),
        ('cancelled', 'approval for tool "tool_a" was cancelled'),
        (
            'unavailable',
            'tool "tool_a" requires approval, but no approval channel is available',
        ),
    ],
)
async def test_ask_non_grants_deny(
    ctx: Context, outcome: str, message: str,
) -> None:
    tools(ctx).register(make_definition())
    ctx.provide('approval', FakeApproval(outcome))
    ctx.on('tools/pre-execute', ask_listener)
    result = await tools(ctx).execute(make_input(agent=object()))
    assert result.is_error
    assert result.error.message == message


# --- tools/execute wrappers ------------------------------------------------------


async def test_execute_wrapper_authored_result_is_normalized(ctx: Context) -> None:
    rendered: list[Any] = []

    def render(args: Any, value: Any) -> list:
        rendered.append(value)
        return [text(f'rendered:{value}')]

    tools(ctx).register(make_definition(render=render))

    async def wrapper(carrier: Any, exec: Any, next_: Any) -> Any:
        await next_()  # drain the body, then replace the outcome
        return ToolExecutionSuccess(value='wrapped', content=[text('ignored')])

    ctx.on('tools/execute', wrapper)
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    assert result.value == 'wrapped'
    # wrapper-authored values re-run the owning output contract (re-render)
    assert rendered == ['ok', 'wrapped']
    assert result.content == [text('rendered:wrapped')]


async def test_execute_wrapper_authored_invalid_value_fails(ctx: Context) -> None:
    tools(ctx).register(make_definition())

    async def wrapper(carrier: Any, exec: Any, next_: Any) -> Any:
        await next_()
        return ToolExecutionSuccess(value=42, content=[])

    ctx.on('tools/execute', wrapper)
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.error.info == ToolErrorInfo('ToolOutputError', 'INVALID_TOOL_OUTPUT')
    assert result.content == [
        text('Error: tool "tool_a" returned invalid output: "value" must be a string')
    ]


async def test_execute_wrapper_signal_is_fused_and_restored(ctx: Context) -> None:
    caller = AbortController()
    wrapper_controller = AbortController()
    observed: list[bool] = []
    captured: list[Any] = []

    async def wrapper(carrier: Any, exec: Any, next_: Any) -> Any:
        exec.signal = wrapper_controller.signal
        return await next_()

    async def body(args: Any, exec: Any) -> Any:
        caller.abort()
        # the body observes the fused signal: caller cancellation propagates
        observed.append(exec.signal.aborted)
        captured.append(exec)
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    ctx.on('tools/execute', wrapper)
    result = await tools(ctx).execute(make_input(signal=caller.signal))
    assert observed == [True]
    assert result.is_error
    assert result.error.info == ToolErrorInfo('AbortError', TOOL_ABORTED)
    # the wrapper replacement is restored after the body settles
    assert captured[0].signal is wrapper_controller.signal


async def test_execute_wrapper_replacement_alone_does_not_abort(ctx: Context) -> None:
    wrapper_controller = AbortController()
    observed: list[bool] = []

    async def wrapper(carrier: Any, exec: Any, next_: Any) -> Any:
        exec.signal = wrapper_controller.signal
        return await next_()

    async def body(args: Any, exec: Any) -> Any:
        observed.append(exec.signal.aborted)
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    ctx.on('tools/execute', wrapper)
    result = await tools(ctx).execute(make_input())
    assert observed == [False]
    assert not result.is_error


# --- tools/post-execute ----------------------------------------------------------


async def test_post_execute_accepts_by_default(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    seen: list[Any] = []

    async def listener(carrier: Any, exec: Any, result: Any, next_: Any) -> Any:
        seen.append(result.value)
        return await next_()

    ctx.on('tools/post-execute', listener)
    result = await tools(ctx).execute(make_input())
    assert seen == ['ok']
    assert not result.is_error


async def test_post_execute_replace_content(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolAccept(content=[text('replaced')]),
    )
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    assert result.content == [text('replaced')]
    assert result.value == 'ok'


async def test_post_execute_replace_value_revalidates(ctx: Context) -> None:
    rendered: list[Any] = []
    tools(ctx).register(make_definition(
        render=lambda args, value: rendered.append(value) or [text(str(value))],
    ))
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolAccept(value='policy'),
    )
    result = await tools(ctx).execute(make_input())
    assert result.value == 'policy'
    assert rendered == ['ok', 'policy']
    assert result.content == [text('policy')]


async def test_post_execute_rejects_value_plus_content(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolAccept(
            content=[text('x')], value='y',
        ),
    )
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert 'cannot replace both value and content' in result.error.message


async def test_post_execute_cannot_replace_failed_value(ctx: Context) -> None:
    async def failing(args: Any, exec: Any) -> Any:
        raise ValueError('boom')

    tools(ctx).register(make_definition(execute=failing))
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolAccept(value='rescue'),
    )
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert 'cannot replace the value of a failed result' in result.error.message


async def test_post_execute_block(ctx: Context) -> None:
    async def body(args: Any, exec: Any) -> Any:
        exec.defer_context(UserMessage(id='m1', content='deferred'))
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolBlock(
            feedback=[text('try again differently')],
            additional_contexts=[UserMessage(id='m2', content='policy context')],
        ),
    )
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.content == [text('try again differently')]
    assert result.error.message == 'try again differently'
    # a block discards body-deferred contexts; only the decision's survive
    assert [m.id for m in result.additional_contexts] == ['m2']


async def test_post_execute_additional_contexts_merge(ctx: Context) -> None:
    async def body(args: Any, exec: Any) -> Any:
        exec.defer_context(UserMessage(id='body', content='b'))
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    ctx.on(
        'tools/post-execute',
        lambda carrier, exec, result, next_: PostToolAccept(
            additional_contexts=[UserMessage(id='policy', content='p')],
        ),
    )
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    # body-deferred contexts come first, then the decision's
    assert [m.id for m in result.additional_contexts] == ['body', 'policy']


async def test_post_execute_observes_tool_errors(ctx: Context) -> None:
    async def failing(args: Any, exec: Any) -> Any:
        raise ValueError('tool exploded')

    tools(ctx).register(make_definition(execute=failing))
    seen: list[bool] = []

    async def listener(carrier: Any, exec: Any, result: Any, next_: Any) -> Any:
        seen.append(result.is_error)
        return await next_()

    ctx.on('tools/post-execute', listener)
    result = await tools(ctx).execute(make_input())
    assert seen == [True]
    assert result.is_error
    assert result.content == [text('Error: tool exploded')]
    assert result.error.message == 'tool exploded'
    assert result.error.info is None


# --- deferContext / concludeTurn -------------------------------------------------


async def test_conclude_turn_marks_success(ctx: Context) -> None:
    async def body(args: Any, exec: Any) -> Any:
        exec.conclude_turn()
        return 'done'

    tools(ctx).register(make_definition(execute=body))
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    assert result.concludes_turn is True


# --- cancellation ----------------------------------------------------------------


async def test_pre_aborted_signal_skips_the_body(ctx: Context) -> None:
    controller = AbortController()
    called: list[str] = []

    async def body(args: Any, exec: Any) -> Any:
        called.append('body')
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    controller.abort()
    result = await tools(ctx).execute(make_input(signal=controller.signal))
    assert called == []
    assert result.is_error
    assert result.error.message == 'tool call aborted before dispatch'
    assert result.error.info == ToolErrorInfo(
        'AbortError', TOOL_ABORTED_BEFORE_DISPATCH,
    )
    assert result.content == [text('Error: tool call aborted before dispatch')]


async def test_abort_after_body_started_replaces_success(ctx: Context) -> None:
    controller = AbortController()
    drained: list[str] = []

    async def body(args: Any, exec: Any) -> Any:
        controller.abort()
        await asyncio.sleep(0)  # started work still drains to quiescence
        drained.append('settled')
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    result = await tools(ctx).execute(make_input(signal=controller.signal))
    assert drained == ['settled']
    assert result.is_error
    assert result.error.info == ToolErrorInfo('AbortError', TOOL_ABORTED)


async def test_abort_preserves_deferred_contexts(ctx: Context) -> None:
    controller = AbortController()

    async def body(args: Any, exec: Any) -> Any:
        exec.defer_context(UserMessage(id='kept', content='k'))
        controller.abort()
        return 'ok'

    tools(ctx).register(make_definition(execute=body))
    result = await tools(ctx).execute(make_input(signal=controller.signal))
    assert result.is_error
    assert [m.id for m in result.additional_contexts] == ['kept']


# --- error shapes ----------------------------------------------------------------


async def test_unknown_tool(ctx: Context) -> None:
    result = await tools(ctx).execute(make_input('missing'))
    assert result.is_error
    assert result.content == [text('Error: unknown tool "missing"')]
    assert result.error.message == 'unknown tool "missing"'
    assert result.error.info == ToolErrorInfo('ToolNotFoundError', 'UNKNOWN_TOOL')
    error = ToolNotFoundError('missing')
    assert error.code == 'UNKNOWN_TOOL'


async def test_harness_error_info_is_routed(ctx: Context) -> None:
    from pydsh.tools import ToolOutputError

    async def body(args: Any, exec: Any) -> Any:
        raise ToolOutputError('tool_a', ['deliberate'])

    tools(ctx).register(make_definition(execute=body))
    result = await tools(ctx).execute(make_input())
    assert result.error.info == ToolErrorInfo('ToolOutputError', 'INVALID_TOOL_OUTPUT')


async def test_non_lossless_arguments_fail_before_policy(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    pre_calls: list[str] = []
    ctx.on('tools/pre-execute', lambda carrier, exec, next_: pre_calls.append('pre'))
    result = await tools(ctx).execute(make_input(args={'bad': object()}))
    assert result.is_error
    assert result.error.message == (
        'tool execution arguments must be losslessly JSON-serializable'
    )
    assert pre_calls == []


async def test_output_schema_violation(ctx: Context) -> None:
    async def body(args: Any, exec: Any) -> Any:
        return 42

    tools(ctx).register(make_definition(execute=body))
    result = await tools(ctx).execute(make_input())
    assert result.is_error
    assert result.error.info == ToolErrorInfo('ToolOutputError', 'INVALID_TOOL_OUTPUT')
    assert result.error.message == (
        'tool "tool_a" returned invalid output: "value" must be a string'
    )


async def test_render_and_meta_projection_failures(ctx: Context) -> None:
    def bad_render(args: Any, value: Any) -> list:
        raise RuntimeError('render exploded')

    tools(ctx).register(make_definition(render=bad_render))
    result = await tools(ctx).execute(make_input())
    assert result.error.message == (
        'tool "tool_a" returned invalid output: output.render failed: '
        'render exploded'
    )

    def bad_meta(args: Any, value: Any) -> Any:
        raise RuntimeError('meta exploded')

    tools(ctx).register(make_definition('tool_b', presentation_meta=bad_meta))
    result = await tools(ctx).execute(make_input('tool_b'))
    assert result.error.message == (
        'tool "tool_b" returned invalid output: output.presentationMeta '
        'failed: meta exploded'
    )


async def test_presentation_meta_only_for_top_level(ctx: Context) -> None:
    meta_calls: list[Any] = []

    def meta(args: Any, value: Any) -> Any:
        meta_calls.append(value)
        return {'summary': value}

    tools(ctx).register(make_definition('inner', presentation_meta=meta))

    async def outer_body(args: Any, exec: Any) -> Any:
        nested = await tools(ctx).execute(make_input(
            'inner', call_id='call-inner', parent=exec.token,
        ))
        return nested.value

    tools(ctx).register(make_definition('outer', execute=outer_body))
    await tools(ctx).execute(make_input('outer'))
    # the nested sub-dispatch (parent token set) does not project meta
    assert meta_calls == []
    inner_direct = await tools(ctx).execute(make_input('inner', call_id='call-2'))
    assert inner_direct.meta == {'summary': 'ok'}
    assert meta_calls == ['ok']


# --- finalizeContent -------------------------------------------------------------


async def test_finalize_content_applies_to_success_and_failure(ctx: Context) -> None:
    def finalize(exec: Any, result: Any) -> list:
        return [text(f'finalized:{result.is_error}')]

    async def failing(args: Any, exec: Any) -> Any:
        raise ValueError('nope')

    tools(ctx).register(make_definition(finalize_content=finalize))
    tools(ctx).register(make_definition(
        'tool_b', execute=failing, finalize_content=finalize,
    ))

    ok = await tools(ctx).execute(make_input())
    assert ok.content == [text('finalized:False')]
    failed = await tools(ctx).execute(make_input('tool_b'))
    assert failed.content == [text('finalized:True')]


async def test_finalize_content_none_preserves(ctx: Context) -> None:
    tools(ctx).register(make_definition(finalize_content=lambda exec, result: None))
    result = await tools(ctx).execute(make_input())
    assert result.content == [text('ok')]


# --- tools/result ----------------------------------------------------------------


async def test_result_observer_receives_frozen_execution(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    observed: list[Any] = []
    ctx.on(
        'tools/result',
        lambda carrier, exec, result: observed.append((exec, result)),
    )
    result = await tools(ctx).execute(make_input())
    exec, reported = observed[0]
    assert reported is result
    with pytest.raises(AttributeError, match='frozen'):
        exec.name = 'mutated'


async def test_result_observer_failure_is_contained(ctx: Context) -> None:
    warns = record_warns(ctx)
    tools(ctx).register(make_definition())

    def broken(carrier: Any, exec: Any, result: Any) -> None:
        raise ValueError('observer bug')

    ctx.on('tools/result', broken)
    result = await tools(ctx).execute(make_input())
    assert not result.is_error
    assert warns == [
        'tool "tool_a" (call-1): tools/result observer failed: observer bug',
    ]


# --- schemas / presentation to the model -----------------------------------------


async def test_schemas_project_model_facing_fields(ctx: Context) -> None:
    definition = make_definition(timeout_ms=5000)
    tools(ctx).register(definition)
    schemas = tools(ctx).schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema.name == 'tool_a'
    assert schema.description == 'tool_a description'
    assert schema.parameters == {'type': 'object', 'properties': {}}
    # projected schemas are detached copies
    schema.parameters['type'] = 'mutated'
    assert tools(ctx).schemas()[0].parameters == {
        'type': 'object', 'properties': {},
    }


async def test_system_prompt_assembly_includes_visible_tools(ctx: Context) -> None:
    sp = ctx.get('systemPrompt')
    dispose_b = tools(ctx).register(make_definition('b_tool'))
    tools(ctx).register(make_definition('a_tool'))

    assembly = await sp.assemble()
    assert [tool.name for tool in assembly.tools] == ['a_tool', 'b_tool']

    key = object()
    scope = create_scope(ctx, key)
    scope.ctx.get('tools').restrict(ToolRestriction(deny=['a_tool']))
    scoped_assembly = await sp.assemble(AssembleContext(scope=key))
    assert [tool.name for tool in scoped_assembly.tools] == ['b_tool']

    dispose_b()
    assembly = await sp.assemble()
    assert [tool.name for tool in assembly.tools] == ['a_tool']


# --- execution_mode --------------------------------------------------------------


async def test_execution_mode_classification(ctx: Context) -> None:
    tools(ctx).register(make_definition(
        'parallel_tool', is_concurrency_safe=lambda args: True,
    ))
    tools(ctx).register(make_definition(
        'exclusive_tool', is_concurrency_safe=lambda args: False,
    ))
    tools(ctx).register(make_definition(
        'throwing_tool', is_concurrency_safe=lambda args: 1 / 0,
    ))
    tools(ctx).register(make_definition('undeclared_tool'))

    assert tools(ctx).execution_mode(make_input('parallel_tool')) == 'parallel'
    assert tools(ctx).execution_mode(make_input('exclusive_tool')) == 'exclusive'
    assert tools(ctx).execution_mode(make_input('throwing_tool')) == 'exclusive'
    assert tools(ctx).execution_mode(make_input('undeclared_tool')) == 'exclusive'
    assert tools(ctx).execution_mode(make_input('missing')) == 'exclusive'


# --- scheduler stages ------------------------------------------------------------


async def test_scheduler_stages_match_execute(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    scheduler = tools(ctx).scheduler
    prepared = await scheduler.prepare(make_input())
    assert prepared.kind == 'dispatch'
    dispatched = await scheduler.dispatch(prepared.exec)
    assert dispatched.kind == 'post-result'
    final = await scheduler.finalize(prepared.exec, dispatched.result)
    assert not final.is_error
    assert final.value == 'ok'


async def test_scheduler_prepare_denial_is_post_result(ctx: Context) -> None:
    tools(ctx).register(make_definition())
    ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_: PreToolDeny('staged denial'),
    )
    prepared = await tools(ctx).scheduler.prepare(make_input())
    assert prepared.kind == 'post-result'
    final = await tools(ctx).scheduler.finalize(prepared.exec, prepared.result)
    assert final.is_error
    assert final.error.message == 'staged denial'


# --- config ----------------------------------------------------------------------


async def test_config_validation() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime, {'mode': 'native'})

    fiber = ctx.plugin(ToolRuntime, {'bogus': 1})
    with pytest.raises(ValidationError, match='unknown tools config keys'):
        await fiber


async def test_non_native_mode_fails_loud() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    with pytest.raises(ValidationError, match='Code Mode'):
        await ctx.plugin(ToolRuntime, {'mode': 'code'})
    with pytest.raises(ValidationError, match='Code Mode'):
        await ctx.plugin(ToolRuntime, {'mode': 'both'})


# --- invariant -------------------------------------------------------------------


async def test_invariant_accepts_a_healthy_pipeline(ctx: Context) -> None:
    failures: list[str] = []
    invariant.install(ctx, failures.append)
    tools(ctx).register(make_definition())

    async def failing(args: Any, exec: Any) -> Any:
        raise ValueError('boom')

    tools(ctx).register(make_definition('tool_b', execute=failing))
    ctx.on('tools/pre-execute', lambda carrier, exec, next_: (
        PreToolDeny('no') if exec.name == 'tool_b' else PreToolAllow()
    ))

    await tools(ctx).execute(make_input())
    await tools(ctx).execute(make_input('tool_b'))
    assert failures == []


async def test_invariant_flags_stage_violations(ctx: Context) -> None:
    failures: list[str] = []
    invariant.install(ctx, failures.append)
    tools(ctx).register(make_definition())

    captured: list[Any] = []
    ctx.on(
        'tools/pre-execute',
        lambda carrier, exec, next_=None: captured.append(exec),
    )
    await tools(ctx).execute(make_input())
    await tools(ctx).execute(make_input(call_id='call-2'))
    exec_one, exec_two = captured

    carrier = scope_target(tools(ctx), None)
    # tools/execute without a preceding tools/pre-execute stage
    ctx.emit(carrier, 'tools/execute', exec_one)
    assert failures == ['tools/execute must follow tools/pre-execute']

    # a repeated pre-execute stage for one execution
    failures.clear()
    ctx.emit(carrier, 'tools/pre-execute', exec_two)
    ctx.emit(carrier, 'tools/pre-execute', exec_two)
    assert failures == ['tools/pre-execute repeated for one execution']
