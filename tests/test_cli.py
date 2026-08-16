"""End-to-end smoke tests for the minimal CLI (``pydsh.cli``).

A scripted fake adapter registered on the real ``LlmRuntime`` drives the
fully assembled plugin tree (SystemPrompt → LlmRuntime → adapter →
SessionStore/AgentRegistry → ToolRuntime → ApprovalService + stdin answerer →
AgentLoop) through one conversation with a tool call — no network. The stdin
approval answerer is fed via an injected reader; CLI parsing and the missing
API-key error path go through ``pydsh.cli.main``.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

import pytest

from pydsh import cli
from pydsh.cli import CliError
from pydsh.llm import GenerateOptions, LlmAdapter, StreamChunk
from pydsh.tools import PreToolAsk, ToolDefinition, ToolOutputDefinition

FAKE_PROVIDER = 'fake'
FAKE_MODEL = 'fake-model'


# --- scripted fake adapter (same shape as tests/test_agent_loop.py) ---------


def text_chunks(text: str) -> list[StreamChunk]:
    return [
        {'type': 'block-start', 'index': 0, 'blockType': 'text'},
        {'type': 'text-delta', 'index': 0, 'text': text},
        {'type': 'block-end', 'index': 0, 'block': {'type': 'text', 'text': text}},
        {'type': 'usage', 'usage': {'inputTokens': 3, 'outputTokens': 2}},
        {'type': 'finish', 'reason': {'kind': 'stop'}},
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


class FakeAdapter(LlmAdapter):
    """Scripted adapter: each ``stream()`` call pops the next script entry."""

    def __init__(self, script: list[list[StreamChunk]]) -> None:
        self.script = script
        self.requests: list[GenerateOptions] = []

    def stream(self, options: GenerateOptions) -> AsyncIterable[StreamChunk]:
        self.requests.append(options)
        entry = self.script.pop(0) if self.script else text_chunks('<empty>')

        async def gen() -> AsyncGenerator[StreamChunk]:
            for chunk in entry:
                yield chunk

        return gen()


def make_tool(name: str, execute: Any = None) -> ToolDefinition:
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
    )


async def make_app(
    script: list[list[StreamChunk]],
    out: io.StringIO | None = None,
    **kwargs: Any,
) -> tuple[cli.CliApp, FakeAdapter, io.StringIO]:
    adapter = FakeAdapter(script)
    stream = out if out is not None else io.StringIO()
    app = await cli.assemble(
        cli.Context(),
        provider=FAKE_PROVIDER,
        model=FAKE_MODEL,
        adapters=[([FAKE_PROVIDER], adapter)],
        out=stream,
        **kwargs,
    )
    return app, adapter, stream


def event_types(app: cli.CliApp) -> list[str]:
    return [event['type'] for event in app.agent.session.events]


# --- end-to-end smoke ---------------------------------------------------------


async def test_end_to_end_tool_call_smoke() -> None:
    executed: list[Any] = []

    async def echo(args: Any, exec: Any) -> Any:
        executed.append(args)
        return f"echo:{args['value']}"

    app, adapter, out = await make_app([
        tool_call_chunks([('c1', 'echo', {'value': 'x'})]),
        text_chunks('finished'),
    ])
    app.ctx.get('tools').register(make_tool('echo', execute=echo))
    statuses: list[str] = []
    app.ctx.on(
        'agent/status', lambda carrier, payload: statuses.append(payload['status'])
    )

    app.agent.followup(cli._user_message('please echo'))
    await app.agent.when_idle()

    # The tool ran and the model's text streamed to the output.
    assert executed == [{'value': 'x'}]
    rendered = out.getvalue()
    assert '[tool call] echo {"value": "x"}' in rendered
    assert '[tool result] c1 ok: echo:x' in rendered
    assert 'finished' in rendered

    # Event sequence: one turn, tool call before its result, clean completion.
    types = event_types(app)
    assert types[:2] == ['agent/inbox/spliced', 'turn/start']
    assert 'assistant/chunk' in types
    assert types.index('tool/call') < types.index('tool/result')
    assert types[-1] == 'turn/end'
    assert app.agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}
    assert statuses == ['running', 'idle']
    assert len(adapter.requests) == 2

    # Ordered teardown unwinds the whole tree.
    await app.dispose()
    assert app.disposed
    assert all(fiber.uid is None for fiber in app.fibers)
    await app.dispose()  # idempotent


# --- REPL ----------------------------------------------------------------------


async def test_repl_round_trip_then_exit() -> None:
    app, adapter, out = await make_app([text_chunks('hi there')])
    lines = iter(['hello', '/exit'])
    await cli.run_repl(app, read=lambda prompt: next(lines), out=out)

    rendered = out.getvalue()
    assert 'hi there' in rendered
    assert len(adapter.requests) == 1
    assert adapter.requests[0]['messages'][0]['content'] == [
        {'type': 'text', 'text': 'hello'}
    ]
    await app.dispose()


async def test_repl_eof_exits_cleanly() -> None:
    app, adapter, out = await make_app([])

    def eof(prompt: str) -> str:
        raise EOFError

    await cli.run_repl(app, read=eof, out=out)
    assert adapter.requests == []
    await app.dispose()


async def test_repl_blank_lines_are_ignored() -> None:
    app, adapter, _out = await make_app([])
    lines = iter(['', '   ', '/exit'])
    await cli.run_repl(app, read=lambda prompt: next(lines))
    assert adapter.requests == []
    await app.dispose()


# --- approval answerer ----------------------------------------------------------


def _gate_danger(carrier: Any, exec: Any, next_: Any) -> Any:
    if exec.name == 'danger':
        return PreToolAsk(reason='it is dangerous')
    return next_()


async def test_approval_answerer_yes_allows_once() -> None:
    executed: list[Any] = []

    async def danger(args: Any, exec: Any) -> Any:
        executed.append(args)
        return 'did it'

    app, adapter, out = await make_app(
        [
            tool_call_chunks([('c1', 'danger', {})]),
            text_chunks('done'),
        ],
        approval_reader=lambda prompt: 'y',
    )
    app.ctx.get('tools').register(make_tool('danger', execute=danger))
    app.ctx.on('tools/pre-execute', _gate_danger)

    app.agent.followup(cli._user_message('do the dangerous thing'))
    await app.agent.when_idle()

    assert executed == [{}]
    rendered = out.getvalue()
    assert '[approval] tool "danger" requests permission.' in rendered
    assert 'reason: it is dangerous' in rendered

    decided = [
        event['data']
        for event in app.agent.session.events
        if event['type'] == 'approval/decided'
    ]
    assert [entry['outcome'] for entry in decided] == ['allowed-once']
    assert len(adapter.requests) == 2
    await app.dispose()


async def test_approval_answerer_no_rejects() -> None:
    executed: list[Any] = []

    async def danger(args: Any, exec: Any) -> Any:
        executed.append(args)
        return 'did it'

    app, _adapter, out = await make_app(
        [
            tool_call_chunks([('c1', 'danger', {})]),
            text_chunks('understood'),
        ],
        approval_reader=lambda prompt: 'n',
    )
    app.ctx.get('tools').register(make_tool('danger', execute=danger))
    app.ctx.on('tools/pre-execute', _gate_danger)

    app.agent.followup(cli._user_message('do the dangerous thing'))
    await app.agent.when_idle()

    assert executed == []
    decided = [
        event['data']
        for event in app.agent.session.events
        if event['type'] == 'approval/decided'
    ]
    assert [entry['outcome'] for entry in decided] == ['rejected']
    # The rejection reached the model as a tool error, and the turn completed.
    assert '[tool result] c1 error:' in out.getvalue()
    assert app.agent.session.events[-1]['type'] == 'turn/end'
    await app.dispose()


async def test_approval_answerer_eof_rejects() -> None:
    def eof(prompt: str) -> str:
        raise EOFError

    app, _adapter, _out = await make_app(
        [
            tool_call_chunks([('c1', 'danger', {})]),
            text_chunks('understood'),
        ],
        approval_reader=eof,
    )
    app.ctx.get('tools').register(make_tool('danger'))
    app.ctx.on('tools/pre-execute', _gate_danger)

    app.agent.followup(cli._user_message('try'))
    await app.agent.when_idle()

    decided = [
        event['data']
        for event in app.agent.session.events
        if event['type'] == 'approval/decided'
    ]
    assert [entry['outcome'] for entry in decided] == ['rejected']
    await app.dispose()


# --- assembly seams ---------------------------------------------------------------


async def test_unknown_provider_fails_loud() -> None:
    with pytest.raises(CliError, match='unknown provider "bogus"'):
        await cli.assemble(cli.Context(), provider='bogus', model='m')


async def test_missing_api_key_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cli.DEEPSEEK_API_KEY_ENV, raising=False)
    with pytest.raises(CliError, match=cli.DEEPSEEK_API_KEY_ENV):
        await cli.assemble(
            cli.Context(), provider=cli.DEEPSEEK_PROVIDER, model='deepseek-v4-flash'
        )


async def test_extra_plugins_mount_before_agent_loop() -> None:
    """One more tool plugin is one more ``extra_plugins`` entry."""
    mounted: list[str] = []

    def tool_plugin(ctx: Any, _config: Any = None) -> None:
        ctx.get('tools').register(make_tool('extra'))
        mounted.append('extra')

    app, _adapter, _out = await make_app(
        [tool_call_chunks([('c1', 'extra', {})]), text_chunks('done')],
        extra_plugins=[(tool_plugin, None)],
    )
    app.agent.followup(cli._user_message('use the extra tool'))
    await app.agent.when_idle()

    assert mounted == ['extra']
    types = event_types(app)
    assert 'tool/result' in types
    assert app.agent.session.events[-1]['data']['reason'] == {'kind': 'completed'}
    await app.dispose()


# --- argument parsing and entry-point error paths ---------------------------------


def test_parse_args_defaults() -> None:
    args = cli.build_parser().parse_args([])
    assert args.provider == cli.DEEPSEEK_PROVIDER
    assert args.model == cli.DEFAULT_MODEL
    assert args.cwd is None
    assert args.session_id is None


def test_parse_args_explicit() -> None:
    args = cli.build_parser().parse_args([
        '--provider', 'deepseek-official',
        '--model', 'deepseek-v4-pro',
        '--cwd', '/tmp',
        '--session-id', 's1',
    ])
    assert args.model == 'deepseek-v4-pro'
    assert args.cwd == '/tmp'
    assert args.session_id == 's1'


def test_main_without_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(cli.DEEPSEEK_API_KEY_ENV, raising=False)
    rc = cli.main([])
    assert rc == 2
    assert cli.DEEPSEEK_API_KEY_ENV in capsys.readouterr().err


def test_main_unknown_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(cli.DEEPSEEK_API_KEY_ENV, raising=False)
    rc = cli.main(['--provider', 'bogus'])
    assert rc == 2
    assert 'unknown provider "bogus"' in capsys.readouterr().err


async def test_end_to_end_bash_tool_smoke() -> None:
    """The production bash tool stack runs a real subprocess end to end."""
    app, adapter, out = await make_app(
        [
            tool_call_chunks([
                ('c1', 'bash', {'command': 'echo hi-from-bash', 'description': 't'}),
            ]),
            text_chunks('done'),
        ],
        extra_plugins=cli.bash_tool_plugins(),
    )

    app.agent.followup(cli._user_message('run echo'))
    await app.agent.when_idle()

    rendered = out.getvalue()
    assert '[tool call] bash' in rendered
    assert 'hi-from-bash' in rendered
    assert 'done' in rendered
    assert len(adapter.requests) == 2

    await app.dispose()
    assert all(fiber.uid is None for fiber in app.fibers)
