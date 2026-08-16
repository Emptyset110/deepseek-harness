"""Behavior tests for the agent loop (``ctx.agentLoop`` + ReactLoopAgent).

Ported-semantics tests for the Python port of deepseek-harness
``packages/core/agent-loop`` (MIT). A fake adapter registered on the real
``pydsh.llm`` runtime drives the real ReAct loop end to end — no network.
Covers: single-turn text completion, the tool-call loop (results fed back,
parallel/serial scheduling), pre-step reject and message replacement, the
request waterfall, request-error retry decisions, cancel interruption,
teardown convergence from both owners, and resume without persistence.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any
from uuid import uuid4

import pytest

from pycordis import Context, ValidationError
from pydsh.agent import (
    AgentOptions,
    AgentRegistry,
    CreateAgentOptions,
    PreStepEnter,
    PreStepReject,
    RequestErrorRetry,
    ResumeAgentOptions,
)
from pydsh.agent_loop import AgentLoop
from pydsh.agent_loop.invariant import install as install_invariant
from pydsh.llm import (
    GenerateOptions,
    LlmAdapter,
    LlmRuntime,
    ResolvedNormalRetryPolicy,
    StreamChunk,
    is_agent_loop_request,
    mark_agent_loop_request,
)
from pydsh.session import Session, SessionId, SessionStore
from pydsh.system_prompt import PromptContext, SystemPrompt
from pydsh.tools import ToolDefinition, ToolOutputDefinition, ToolRuntime

FAKE_PROVIDER = 'fake'
FAKE_MODEL = 'fake-model'


# --- scripted fake adapter ---------------------------------------------------


def user_message(text: str, id: str | None = None) -> dict[str, Any]:
    return {
        'id': id or f'msg-{uuid4()}',
        'role': 'user',
        'content': [{'type': 'text', 'text': text}],
        'source': {'kind': 'user'},
    }


def text_chunks(text: str, reason: dict[str, Any] | None = None) -> list[StreamChunk]:
    return [
        {'type': 'block-start', 'index': 0, 'blockType': 'text'},
        {'type': 'text-delta', 'index': 0, 'text': text},
        {'type': 'block-end', 'index': 0, 'block': {'type': 'text', 'text': text}},
        {'type': 'usage', 'usage': {'inputTokens': 3, 'outputTokens': 2}},
        {'type': 'finish', 'reason': reason or {'kind': 'stop'}},
    ]


def tool_call_chunks(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> list[StreamChunk]:
    chunks: list[StreamChunk] = []
    for index, (call_id, name, arguments) in enumerate(calls):
        raw = json.dumps(arguments)
        chunks.extend([
            {'type': 'block-start', 'index': index, 'blockType': 'tool-call'},
            {
                'type': 'tool-call-delta',
                'index': index,
                'id': call_id,
                'name': name,
                'argumentsDelta': raw,
            },
            {
                'type': 'block-end',
                'index': index,
                'block': {
                    'type': 'tool-call',
                    'id': call_id,
                    'name': name,
                    'arguments': raw,
                },
            },
        ])
    chunks.append({'type': 'finish', 'reason': {'kind': 'tool-calls'}})
    return chunks


ScriptEntry = Any  # list[StreamChunk] | BaseException | (options) -> AsyncIterable


class FakeAdapter(LlmAdapter):
    """Scripted adapter: each ``stream()`` call pops the next script entry."""

    def __init__(self, script: list[ScriptEntry]) -> None:
        self.script = script
        self.requests: list[GenerateOptions] = []

    def stream(self, options: GenerateOptions) -> AsyncIterable[StreamChunk]:
        self.requests.append(options)
        entry: ScriptEntry = (
            self.script.pop(0) if self.script else text_chunks('<empty>')
        )

        async def gen() -> AsyncGenerator[StreamChunk]:
            if isinstance(entry, BaseException):
                raise entry
            if callable(entry):
                async for chunk in entry(options):
                    yield chunk
                return
            for chunk in entry:
                yield chunk

        return gen()


def make_tool(
    name: str,
    *,
    execute: Any = None,
    is_concurrency_safe: Any = None,
) -> ToolDefinition:
    async def default_execute(args: Any, exec: Any) -> Any:
        return 'ok'

    return ToolDefinition(
        name=name,
        description=f'{name} description',
        parameters={'type': 'object', 'properties': {}},
        output=ToolOutputDefinition(
            schema={'type': 'string'},
            render=lambda args, value: [{'type': 'text', 'text': str(value)}],
        ),
        execute=execute if execute is not None else default_execute,
        is_concurrency_safe=is_concurrency_safe,
    )


async def make_app(
    script: list[ScriptEntry],
    config: dict[str, Any] | None = None,
) -> tuple[Context, FakeAdapter, AgentLoop]:
    """Compose the real service spine with the fake adapter and loop plugin."""
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(LlmRuntime)
    SessionStore(ctx)
    AgentRegistry(ctx)
    await ctx.plugin(ToolRuntime)
    adapter = FakeAdapter(script)
    ctx.get('llm').register_adapter([FAKE_PROVIDER], adapter)
    await ctx.plugin(AgentLoop, config)
    loop: AgentLoop = ctx.get('agentLoop')
    return ctx, adapter, loop


def make_agent(loop: AgentLoop, id: str = 's1') -> Any:
    return loop.create(
        SessionId(id), AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL)
    )


def event_types(session: Session) -> list[str]:
    return [event['type'] for event in session.events]


# --- single-turn completion ---------------------------------------------------


async def test_single_turn_text_completion() -> None:
    ctx, adapter, loop = await make_app([text_chunks('hello world')])
    agent = make_agent(loop)
    statuses: list[str] = []
    ctx.on('agent/status', lambda carrier, payload: statuses.append(payload['status']))

    agent.followup(user_message('hi', id='m1'))
    assert agent.status == 'running'
    await agent.when_idle()

    assert agent.status == 'idle'
    assert statuses == ['running', 'idle']
    types = event_types(agent.session)
    # The durable inbox splice precedes the turn boundary it woke.
    assert types[:2] == ['agent/inbox/spliced', 'turn/start']
    assert 'request/header' in types
    assert 'request/context' in types
    assert types.count('request/header') == 1
    assert 'step/start' in types
    assert 'step/end' in types
    assert types[-1] == 'turn/end'
    turn_end = agent.session.events[-1]
    assert turn_end['data']['reason'] == {'kind': 'completed'}

    # The request the adapter saw is the marked loop-built request.
    request = adapter.requests[0]
    assert is_agent_loop_request(request)
    assert request['provider'] == FAKE_PROVIDER
    assert request['model'] == FAKE_MODEL
    assert request['sessionId'] == SessionId('s1')
    # Model-visible ⟺ logged: the request messages equal the logged
    # derivation at request time (user/message is logged before the call).
    assert request['messages'][0]['id'] == 'm1'
    assert request['system']

    derived = agent.session.derive_messages()
    assert [message['role'] for message in derived] == ['user', 'assistant']
    assert derived[-1]['content'] == [{'type': 'text', 'text': 'hello world'}]
    assert derived[-1]['source']['kind'] == 'model'


async def test_header_not_relogged_when_unchanged_across_steps() -> None:
    ctx, adapter, loop = await make_app([
        tool_call_chunks([('c1', 'echo', {'value': 'x'})]),
        text_chunks('done'),
    ])
    ctx.get('tools').register(make_tool('echo'))
    agent = make_agent(loop)
    agent.followup(user_message('run a tool'))
    await agent.when_idle()

    # Same route, system, and tools in both steps: one initial header only.
    assert event_types(agent.session).count('request/header') == 1
    assert len(adapter.requests) == 2


# --- tool-call loop -----------------------------------------------------------


async def test_tool_call_loop_feeds_results_back() -> None:
    executed: list[Any] = []

    async def echo(args: Any, exec: Any) -> Any:
        executed.append(args)
        return f"echo:{args['value']}"

    ctx, adapter, loop = await make_app([
        tool_call_chunks([('c1', 'echo', {'value': 'x'})]),
        text_chunks('finished'),
    ])
    ctx.get('tools').register(make_tool('echo', execute=echo))
    agent = make_agent(loop)
    agent.followup(user_message('please echo'))
    await agent.when_idle()

    assert executed == [{'value': 'x'}]
    types = event_types(agent.session)
    assert 'tool/call' in types
    assert 'tool/result' in types
    assert types[-1] == 'turn/end'
    assert agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}

    # The tool result is model-visible ⟺ logged: it reaches the follow-up
    # request through the durable derivation.
    result_messages = [
        message
        for message in adapter.requests[1]['messages']
        if message['source'].get('kind') == 'tool'
    ]
    assert len(result_messages) == 1
    block = result_messages[0]['content'][0]
    assert block['type'] == 'tool-result'
    assert block['toolCallId'] == 'c1'
    assert block['content'] == [{'type': 'text', 'text': 'echo:x'}]
    assert block['isError'] is False


async def test_parallel_and_exclusive_scheduling() -> None:
    started: list[str] = []
    finished: list[str] = []

    def make_execute(name: str) -> Any:
        async def execute(args: Any, exec: Any) -> Any:
            started.append(name)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            finished.append(name)
            return name

        return execute

    ctx, _adapter, loop = await make_app([
        tool_call_chunks([
            ('c1', 'p1', {}),
            ('c2', 'gate', {}),
            ('c3', 'p2', {}),
        ]),
        text_chunks('done'),
    ])
    tools = ctx.get('tools')
    tools.register(
        make_tool('p1', execute=make_execute('p1'), is_concurrency_safe=lambda a: True)
    )
    tools.register(make_tool('gate', execute=make_execute('gate')))
    tools.register(
        make_tool('p2', execute=make_execute('p2'), is_concurrency_safe=lambda a: True)
    )
    agent = make_agent(loop)
    agent.followup(user_message('go'))
    await agent.when_idle()

    # p1 starts first; the exclusive gate is a barrier: it starts only after
    # p1 finished, and p2 waits for the gate.
    assert started == ['p1', 'gate', 'p2']
    assert finished == ['p1', 'gate', 'p2']
    results = [
        event for event in agent.session.events if event['type'] == 'tool/result'
    ]
    assert [event['data']['message']['source']['callId'] for event in results] == [
        'c1',
        'c2',
        'c3',
    ]


async def test_parallel_pool_runs_calls_concurrently() -> None:
    running = 0
    max_running = 0

    async def execute(args: Any, exec: Any) -> Any:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        running -= 1
        return 'ok'

    ctx, _adapter, loop = await make_app([
        tool_call_chunks([('c1', 'p', {}), ('c2', 'p', {}), ('c3', 'p', {})]),
        text_chunks('done'),
    ])
    ctx.get('tools').register(
        make_tool('p', execute=execute, is_concurrency_safe=lambda a: True)
    )
    agent = make_agent(loop)
    agent.followup(user_message('go'))
    await agent.when_idle()

    assert max_running == 3
    assert agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}


# --- pre-step waterfall -------------------------------------------------------


async def test_pre_step_reject_blocks_the_turn() -> None:
    ctx, adapter, loop = await make_app([text_chunks('never used')])
    ctx.on(
        'agent/pre-step', lambda carrier, payload, next_: PreStepReject()
    )
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert adapter.requests == []
    types = event_types(agent.session)
    assert 'step/start' not in types
    assert types[-1] == 'turn/end'
    assert agent.session.events[-1]['data']['reason'] == {'kind': 'blocked'}


async def test_pre_step_replacement_messages_are_logged() -> None:
    ctx, adapter, loop = await make_app([text_chunks('ok')])
    replacement = user_message('replaced', id='m-repl')

    def rewrite(carrier: Any, payload: dict[str, Any], next_: Any) -> Any:
        return PreStepEnter(messages=[replacement])

    ctx.on('agent/pre-step', rewrite)
    agent = make_agent(loop)
    agent.followup(user_message('original', id='m-orig'))
    await agent.when_idle()

    # The claimed original entered the inbox claim events; the replacement is
    # what the step logged and the model saw.
    assert adapter.requests[0]['messages'][0]['id'] == 'm-repl'
    user_events = [
        event for event in agent.session.events if event['type'] == 'user/message'
    ]
    assert [event['data']['id'] for event in user_events] == ['m-repl']


# --- agent/request waterfall ---------------------------------------------------


async def test_request_waterfall_replaces_call_config() -> None:
    ctx, adapter, loop = await make_app([text_chunks('ok')])

    async def tune(carrier: Any, payload: dict[str, Any], next_: Any) -> Any:
        config = await next_()
        assert config['provider'] == FAKE_PROVIDER
        return {**config, 'temperature': 0.5}

    ctx.on('agent/request', tune)
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert adapter.requests[0]['temperature'] == 0.5
    header_events = [
        event for event in agent.session.events if event['type'] == 'request/header'
    ]
    assert header_events[0]['data']['header']['config']['temperature'] == 0.5


async def test_request_without_route_fails_loud() -> None:
    ctx, adapter, loop = await make_app([text_chunks('ok')])
    agent = loop.create(SessionId('s1'), AgentOptions())
    errors: list[Any] = []
    ctx.on('agent/error', lambda carrier, payload: errors.append(payload))

    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert adapter.requests == []
    assert len(errors) == 1
    assert 'no provider/model' in str(errors[0]['error'])
    reason = agent.session.events[-1]['data']['reason']
    assert reason['kind'] == 'error'
    assert reason['error']['code'] == 'UNKNOWN'


# --- request-error waterfall ---------------------------------------------------


async def test_request_error_retry_then_success() -> None:
    ctx, adapter, loop = await make_app([
        [{'type': 'finish', 'reason': {'kind': 'error', 'failure': {
            'message': 'server exploded', 'code': 'SERVER',
        }}}],
        text_chunks('recovered'),
    ])
    decisions: list[dict[str, Any]] = []

    def on_error(carrier: Any, payload: dict[str, Any], next_: Any) -> Any:
        decisions.append(payload)
        return RequestErrorRetry()

    ctx.on('agent/request-error', on_error)
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert len(adapter.requests) == 2
    assert len(decisions) == 1
    assert decisions[0]['failure']['code'] == 'SERVER'
    assert decisions[0]['provider'] == FAKE_PROVIDER
    assert isinstance(decisions[0]['retry_policy'], ResolvedNormalRetryPolicy)
    reason = agent.session.events[-1]['data']['reason']
    assert reason == {'kind': 'completed'}
    assert agent.session.derive_messages()[-1]['content'] == [
        {'type': 'text', 'text': 'recovered'}
    ]


async def test_request_error_terminal_without_listener() -> None:
    ctx, adapter, loop = await make_app([
        [{'type': 'finish', 'reason': {'kind': 'error', 'failure': {
            'message': 'boom', 'code': 'SERVER',
        }}}],
    ])
    errors: list[Any] = []
    ctx.on('agent/error', lambda carrier, payload: errors.append(payload))
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert len(adapter.requests) == 1
    assert len(errors) == 1
    reason = agent.session.events[-1]['data']['reason']
    assert reason['kind'] == 'error'
    assert reason['error'] == {'message': 'boom', 'code': 'SERVER'}


async def test_adapter_raise_normalizes_to_request_error() -> None:
    _ctx, _adapter, loop = await make_app([ValueError('transport down')])
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    reason = agent.session.events[-1]['data']['reason']
    assert reason['kind'] == 'error'
    assert reason['error']['message'] == 'transport down'
    assert reason['error']['code'] == 'UNKNOWN'


# --- cancellation -------------------------------------------------------------


async def test_cancel_interrupts_active_turn() -> None:
    started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    release = asyncio.Event()

    async def blocking_stream(options: GenerateOptions) -> AsyncGenerator[StreamChunk]:
        yield {'type': 'text-delta', 'index': 0, 'text': 'partial'}
        if not started.done():
            started.set_result(None)
        await release.wait()

    _ctx, _adapter, loop = await make_app([blocking_stream])
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await started

    agent.cancel({'kind': 'user'})
    release.set()
    await agent.when_idle()

    assert agent.status == 'idle'
    reason = agent.session.events[-1]['data']['reason']
    assert reason['kind'] == 'aborted'
    assert reason['reason'] == {'kind': 'user'}
    # The partial chunk was logged; no assistant message was committed.
    types = event_types(agent.session)
    assert 'assistant/chunk' in types
    assert 'assistant/message' not in types


async def test_cancel_clears_pending_followup() -> None:
    _ctx, adapter, loop = await make_app([text_chunks('first'), text_chunks('second')])
    agent = make_agent(loop)
    agent.followup(user_message('one'))
    agent.followup(user_message('two'))
    agent.cancel({'kind': 'user'})
    await agent.when_idle()

    # The cancel landed before the driver ran and discarded both queued
    # follow-ups: the aborted driver opens no turn at all.
    assert event_types(agent.session).count('turn/start') == 0
    assert adapter.requests == []
    canceled = [
        event
        for event in agent.session.events
        if event['data'].get('outcome') == 'canceled'
    ]
    assert len(canceled) == 1  # one next-turn clearing splice


# --- max-tokens stickiness ------------------------------------------------------


async def test_max_tokens_turn_ending() -> None:
    _ctx, _adapter, loop = await make_app([
        text_chunks('truncated', reason={'kind': 'max-tokens'}),
    ])
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    reason = agent.session.events[-1]['data']['reason']
    assert reason == {'kind': 'max-tokens'}


# --- teardown convergence -------------------------------------------------------


def _owner_plugin(ctx: Context, config: Any = None) -> None:
    """No-op owner plugin whose fiber structurally owns created agents."""


async def test_teardown_via_owner_fiber_dispose() -> None:
    ctx, _adapter, _loop = await make_app([text_chunks('ok')])
    owner_fiber = ctx.plugin(_owner_plugin)
    handle = await owner_fiber.ctx.get('agents').create(
        CreateAgentOptions(
            session_id=SessionId('owned'),
            agent_options=AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL),
        )
    )
    agents = ctx.get('agents')
    sessions = ctx.get('sessions')
    assert agents.get(SessionId('owned')) is handle.agent
    assert sessions.get(SessionId('owned')) is not None

    handle.agent.followup(user_message('hi'))
    await handle.agent.when_idle()

    result = owner_fiber.dispose()
    if asyncio.isfuture(result) or asyncio.iscoroutine(result):
        await result

    assert agents.get(SessionId('owned')) is None
    assert sessions.get(SessionId('owned')) is None


async def test_teardown_via_handle_dispose() -> None:
    ctx, _adapter, _loop = await make_app([text_chunks('ok')])
    owner_fiber = ctx.plugin(_owner_plugin)
    handle = await owner_fiber.ctx.get('agents').create(
        CreateAgentOptions(
            session_id=SessionId('owned'),
            agent_options=AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL),
        )
    )
    agents = ctx.get('agents')
    sessions = ctx.get('sessions')

    handle.agent.followup(user_message('hi'))
    await handle.agent.when_idle()
    await handle.dispose()

    assert agents.get(SessionId('owned')) is None
    assert sessions.get(SessionId('owned')) is None
    # Disposal is memoized: a second call awaits the same quiescence.
    await handle.dispose()


async def test_owner_dispose_converges_with_handle_dispose() -> None:
    """Both owners racing land on the same memoized reverse teardown."""
    ctx, _adapter, _loop = await make_app([text_chunks('ok')])
    owner_fiber = ctx.plugin(_owner_plugin)
    handle = await owner_fiber.ctx.get('agents').create(
        CreateAgentOptions(
            session_id=SessionId('owned'),
            agent_options=AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL),
        )
    )
    agents = ctx.get('agents')

    disposal = asyncio.ensure_future(handle.dispose())
    result = owner_fiber.dispose()
    if asyncio.isfuture(result) or asyncio.iscoroutine(result):
        await result
    await disposal

    assert agents.get(SessionId('owned')) is None


async def test_owner_dispose_interrupts_active_turn() -> None:
    """Owner unload cancels the live turn and converges to the same quiescence."""
    started: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def blocking_stream(options: GenerateOptions) -> AsyncGenerator[StreamChunk]:
        yield {'type': 'text-delta', 'index': 0, 'text': 'partial'}
        if not started.done():
            started.set_result(None)
        # An adapter honors the request signal: abort ends the stream.
        await options['signal'].wait()

    ctx, _adapter, _loop = await make_app([blocking_stream])
    owner_fiber = ctx.plugin(_owner_plugin)
    handle = await owner_fiber.ctx.get('agents').create(
        CreateAgentOptions(
            session_id=SessionId('owned'),
            agent_options=AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL),
        )
    )
    handle.agent.followup(user_message('hi'))
    await started
    assert handle.agent.status == 'running'

    result = owner_fiber.dispose()
    if asyncio.isfuture(result) or asyncio.iscoroutine(result):
        await result

    agents = ctx.get('agents')
    assert agents.get(SessionId('owned')) is None
    reason = handle.agent.session.events[-1]['data']['reason']
    assert reason['kind'] == 'aborted'
    assert reason['reason'] == {'kind': 'disposed'}


# --- turn-stopping serial ---------------------------------------------------------


async def test_turn_stopping_listener_can_steer_another_step() -> None:
    ctx, adapter, loop = await make_app([
        text_chunks('first answer'),
        text_chunks('second answer'),
    ])
    steered: list[int] = []

    async def on_turn_stopping(carrier: Any, payload: dict[str, Any]) -> None:
        turn: int = payload['turn']
        steered.append(turn)
        if len(steered) == 1:
            payload['agent'].steer(user_message('one more thing', id='m-steer'))

    ctx.on('agent/turn-stopping', on_turn_stopping)
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    # The steered message reopened the step boundary within the same turn:
    # two steps, one turn, completed.
    types = event_types(agent.session)
    assert types.count('step/start') == 2
    assert types.count('turn/start') == 1
    assert len(adapter.requests) == 2
    second_messages = adapter.requests[1]['messages']
    assert [m['id'] for m in second_messages][-1] == 'm-steer'
    assert second_messages[1]['role'] == 'assistant'
    assert agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}


# --- resume without persistence ---------------------------------------------------

async def test_resume_requires_session_persistence() -> None:
    ctx, _adapter, _loop = await make_app([])
    with pytest.raises(RuntimeError, match='session persistence is not configured'):
        await ctx.get('agents').resume(
            ResumeAgentOptions(resume_session_id=SessionId('gone'))
        )


async def test_create_via_registry_factory() -> None:
    """The registry delegates to the loop factory with the caller as owner."""
    ctx, _adapter, _loop = await make_app([text_chunks('ok')])
    handle = await ctx.get('agents').create(
        CreateAgentOptions(
            session_id=SessionId('via-registry'),
            agent_options=AgentOptions(provider=FAKE_PROVIDER, model=FAKE_MODEL),
        )
    )
    assert handle.agent.id == SessionId('via-registry')
    handle.agent.followup(user_message('hi'))
    await handle.agent.when_idle()
    assert handle.agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}
    await handle.dispose()


async def test_create_without_factory_fails() -> None:
    ctx = Context()
    AgentRegistry(ctx)
    with pytest.raises(RuntimeError, match='no agent factory registered'):
        await ctx.get('agents').create(
            CreateAgentOptions(session_id=SessionId('x'))
        )


# --- config validation -----------------------------------------------------------


async def test_config_rejects_invalid_parallel_cap() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(LlmRuntime)
    SessionStore(ctx)
    AgentRegistry(ctx)
    await ctx.plugin(ToolRuntime)
    with pytest.raises(ValidationError, match='positive integer'):
        await ctx.plugin(AgentLoop, {'max_parallel_tool_calls': 0})


async def test_config_rejects_conflicting_identities() -> None:
    ctx = Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(LlmRuntime)
    SessionStore(ctx)
    AgentRegistry(ctx)
    await ctx.plugin(ToolRuntime)
    with pytest.raises(ValidationError, match='mutually exclusive'):
        await ctx.plugin(AgentLoop, {
            'agents': [{
                'id': 'a',
                'session_id': 's1',
                'resume_session_id': 's2',
            }],
        })


async def test_config_starts_declarative_agent() -> None:
    ctx, _adapter, _loop = await make_app(
        [text_chunks('configured')],
        {
            'agents': [{
                'id': 'main',
                'provider': FAKE_PROVIDER,
                'model': FAKE_MODEL,
            }],
        },
    )
    agents = ctx.get('agents').list()
    assert len(agents) == 1
    assert agents[0].id.startswith('main-session-')
    agents[0].followup(user_message('hi'))
    await agents[0].when_idle()
    assert agents[0].session.events[-1]['data']['reason'] == {'kind': 'completed'}


# --- request-reconstruction invariant -------------------------------------------


async def test_invariant_passes_a_wellformed_turn() -> None:
    ctx, _adapter, loop = await make_app([text_chunks('ok')])
    failures: list[str] = []
    install_invariant(ctx, failures.append)
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    assert failures == []


async def test_invariant_catches_log_divergence() -> None:
    ctx, _adapter, loop = await make_app([text_chunks('ok'), text_chunks('extra')])
    failures: list[str] = []
    install_invariant(ctx, failures.append)
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()
    assert failures == []

    # A hand-built request marked as loop-built but diverging from the log.
    bogus = GenerateOptions({
        'provider': FAKE_PROVIDER,
        'model': FAKE_MODEL,
        'messages': [],
        'sessionId': agent.id,
    })
    mark_agent_loop_request(bogus)
    async for _chunk in ctx.get('llm').stream(bogus):
        pass
    assert any('log-reconstruction desync' in failure for failure in failures)


# --- runtime context snapshot -----------------------------------------------------


async def test_runtime_context_snapshot_enters_step() -> None:
    ctx, adapter, loop = await make_app([text_chunks('ok')])
    ctx.get('systemPrompt').context(
        PromptContext(name='test:fact', order=0, text='the sky is blue')
    )
    agent = make_agent(loop)
    agent.followup(user_message('hi'))
    await agent.when_idle()

    # The runtime-context snapshot is a logged, model-visible user message.
    messages = adapter.requests[0]['messages']
    snapshot = [m for m in messages if m['source'].get('form') == 'snapshot']
    assert len(snapshot) == 1
    assert 'the sky is blue' in snapshot[0]['content'][0]['text']
    logged = [
        event
        for event in agent.session.events
        if event['type'] == 'user/message'
    ]
    assert any(event['data']['id'] == snapshot[0]['id'] for event in logged)
