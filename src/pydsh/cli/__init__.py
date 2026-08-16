"""Minimal terminal CLI: assembles the full pydsh plugin tree and drives one
agent conversation from the terminal.

This is NOT a port of the TypeScript ``apps/cli`` (its web UI / ACP / bundle
system is out of scope). It is the smallest entry point that wires every
ported capability layer together with the cordis plugin/lifecycle machinery:

    SystemPrompt → LlmRuntime → provider adapter → SessionStore →
    AgentRegistry → ToolRuntime → ApprovalService + stdin answerer →
    extra tool plugins → AgentLoop

(load order mirrors ``tests/test_agent_loop.py``'s service spine; the answerer
listens on the ``approval/request`` waterfall, mapping a terminal y/n answer
to the outcome). ``CliApp.dispose()`` unloads the plugin fibers in reverse
order, so Ctrl-C / exit tears the whole tree down in an orderly fashion.

Extension seam: one more tool plugin is one more entry in
``extra_plugins``. The production entry point mounts the bash tool stack
(``LocalSubprocessRuntime`` → ``LocalBashExecutor`` → ``pydsh.tool_bash``)
this way; tests substitute their own tool plugins.

Test seam: ``assemble()`` accepts pre-built ``adapters`` registered on the
real ``LlmRuntime``, so tests drive the whole tree with a scripted fake
adapter and never touch the network.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, TextIO
from uuid import uuid4

from pycordis import Context, Fiber
from pydsh.agent import Agent, AgentOptions, AgentRegistry, Message
from pydsh.agent_loop import AgentLoop
from pydsh.approval import ApprovalOutcome, ApprovalRequest, ApprovalService
from pydsh.llm import LlmAdapter, LlmRuntime
from pydsh.session import (
    CreateSessionMeta,
    Session,
    SessionEvent,
    SessionId,
    SessionStore,
)
from pydsh.system_prompt import SystemPrompt
from pydsh.tools import ToolRuntime


def bash_tool_plugins() -> list[tuple[Any, Any]]:
    """The bash tool stack: subprocess backend → shell executor → bash tool."""
    from pydsh import tool_bash
    from pydsh.shell.local import LocalBashExecutor
    from pydsh.subprocess.local import LocalSubprocessRuntime

    return [
        (LocalSubprocessRuntime, None),
        (LocalBashExecutor, None),
        (tool_bash, None),
    ]

__all__ = [
    'DEEPSEEK_API_KEY_ENV',
    'DEEPSEEK_PROVIDER',
    'DEFAULT_MODEL',
    'AdapterSpec',
    'CliApp',
    'CliError',
    'assemble',
    'bash_tool_plugins',
    'build_parser',
    'main',
    'make_stdin_answerer',
    'run',
    'run_repl',
]

#: The provider route owned by the ``pydsh.llm_deepseek`` plugin.
DEEPSEEK_PROVIDER = 'deepseek-official'
#: Environment variable the DeepSeek adapter resolves its API key from.
DEEPSEEK_API_KEY_ENV = 'DEEPSEEK_API_KEY'
#: Default model for the DeepSeek provider route.
DEFAULT_MODEL = 'deepseek-v4-flash'

#: ``(provider routes, adapter)`` pair registered on the real LlmRuntime.
AdapterSpec = tuple[list[str], LlmAdapter]

#: Reads one input line (``builtins.input``-shaped: prompt in, line out).
InputReader = Callable[[str], str]


class CliError(Exception):
    """A user-facing CLI failure (bad provider, missing credentials, ...)."""


@dataclass
class CliApp:
    """The assembled application: root context, its agent, and load tracking.

    ``fibers`` records every mounted plugin fiber in load order;
    :meth:`dispose` unloads them in reverse.
    """

    ctx: Context
    agent: Agent
    fibers: list[Fiber] = field(default_factory=list)
    disposed: bool = False

    async def dispose(self) -> None:
        """Unload the whole plugin tree in reverse load order, exactly once."""
        if self.disposed:
            return
        self.disposed = True
        for fiber in reversed(self.fibers):
            result = fiber.dispose()
            if inspect.isawaitable(result):
                await result


def _registries_plugin(ctx: Context, _config: Any = None) -> None:
    """Mount the session store and agent registry services."""
    SessionStore(ctx)
    AgentRegistry(ctx)


class _AdaptersPlugin:
    """Object plugin registering pre-built adapters (the test seam) on
    ``ctx.llm``. Service access from a plugin context requires the inject
    declaration."""

    name = 'cli-adapters'
    inject: ClassVar[list[str]] = ['llm']

    @staticmethod
    def apply(ctx: Context, specs: Any = None) -> None:
        for providers, adapter in specs or ():
            ctx.llm.register_adapter(list(providers), adapter)


def make_stdin_answerer(
    out: TextIO = sys.stdout,
    read: InputReader | None = None,
) -> Callable[[Context, Any], None]:
    """Build the terminal approval answerer as a cordis plugin.

    The plugin listens on the ``approval/request`` waterfall: it prints the
    question, reads one line from stdin (in a worker thread so the event loop
    stays live), and maps the answer to an outcome — ``y``/``yes`` grants
    ``'allowed-once'``; anything else, EOF, or an unreadable stdin rejects.
    Not calling ``next()`` makes this answerer own the question.
    """

    def read_line(prompt: str) -> str:
        if read is not None:
            return read(prompt)
        return input(prompt)

    async def answer(
        _carrier: Any, req: ApprovalRequest, _next: Callable[[], Any]
    ) -> ApprovalOutcome:
        reason = f'\nreason: {req.reason}' if req.reason else ''
        out.write(f'\n[approval] tool "{req.tool_name}" requests permission.{reason}\n')
        out.flush()
        try:
            reply = await asyncio.to_thread(read_line, 'approve? [y/N] ')
        except (EOFError, OSError):
            reply = ''
        return 'allowed-once' if reply.strip().lower() in ('y', 'yes') else 'rejected'

    def plugin(ctx: Context, _config: Any = None) -> None:
        ctx.on('approval/request', answer)

    return plugin


def _print_session_event(out: TextIO, session: Session, event: SessionEvent) -> None:
    """Render the model's stream and tool activity as session events land."""
    type_ = event['type']
    data = event['data']
    if type_ == 'assistant/chunk':
        chunk = data['chunk']
        if chunk['type'] == 'text-delta':
            out.write(chunk['text'])
            out.flush()
        elif chunk['type'] == 'block-end' and chunk['block']['type'] == 'text':
            out.write('\n')
            out.flush()
    elif type_ == 'tool/call':
        out.write(f"\n[tool call] {data['name']} {data['arguments']}\n")
        out.flush()
    elif type_ == 'tool/result':
        block = data['message']['content'][0]
        is_error = bool(block.get('isError'))
        summary = ''
        for part in block.get('content') or []:
            if part.get('type') == 'text':
                summary = part['text']
                break
        if len(summary) > 200:
            summary = summary[:200] + '...'
        status = 'error' if is_error else 'ok'
        out.write(f"[tool result] {block.get('toolCallId', '?')} {status}: {summary}\n")
        out.flush()


def _presentation_plugin(out: TextIO) -> Callable[[Context, Any], None]:
    """Plugin printing streamed text and tool summaries for every session."""

    def plugin(ctx: Context, _config: Any = None) -> None:
        ctx.on('session/event', lambda session, event: _print_session_event(
            out, session, event
        ))

    return plugin


async def assemble(
    ctx: Context,
    *,
    provider: str,
    model: str,
    cwd: str | None = None,
    session_id: str | None = None,
    adapters: Sequence[AdapterSpec] | None = None,
    extra_plugins: Sequence[tuple[Any, Any]] = (),
    out: TextIO = sys.stdout,
    approval_reader: InputReader | None = None,
) -> CliApp:
    """Assemble the full plugin tree on ``ctx`` and create the CLI's agent.

    :param provider: provider route the agent's requests use. With
        ``adapters=None`` only :data:`DEEPSEEK_PROVIDER` is supported: the
        ``pydsh.llm_deepseek`` plugin is loaded and its API key must be in the
        environment.
    :param adapters: pre-built ``(routes, adapter)`` pairs registered instead
        of loading a provider plugin (the no-network test seam).
    :param extra_plugins: additional ``(plugin, config)`` pairs mounted after
        the tool runtime and approval answerer, before the agent loop — the
        one-line assembly point for another tool plugin.
    :param approval_reader: stdin reader for the approval answerer; ``None``
        reads via ``builtins.input``.
    :raises CliError: unknown provider, or a missing API key when the real
        DeepSeek provider is requested.
    """
    fibers: list[Fiber] = []
    fibers.append(await ctx.plugin(SystemPrompt))
    fibers.append(await ctx.plugin(LlmRuntime))
    if adapters is not None:
        fibers.append(await ctx.plugin(_AdaptersPlugin(), list(adapters)))
    elif provider == DEEPSEEK_PROVIDER:
        if not os.environ.get(DEEPSEEK_API_KEY_ENV):
            raise CliError(
                f'no API key for provider route "{DEEPSEEK_PROVIDER}"; export '
                f'{DEEPSEEK_API_KEY_ENV} in the launching environment'
            )
        from pydsh import llm_deepseek

        fibers.append(await ctx.plugin(llm_deepseek))
    else:
        raise CliError(
            f'unknown provider "{provider}" (supported: {DEEPSEEK_PROVIDER}); '
            'tests inject adapters through assemble(adapters=...)'
        )
    fibers.append(await ctx.plugin(_registries_plugin))
    fibers.append(await ctx.plugin(ToolRuntime))
    fibers.append(await ctx.plugin(ApprovalService))
    fibers.append(await ctx.plugin(make_stdin_answerer(out, approval_reader)))
    for plugin, config in extra_plugins:
        fibers.append(await ctx.plugin(plugin, config))
    fibers.append(await ctx.plugin(AgentLoop))

    loop: AgentLoop = ctx.get('agentLoop')
    sid = SessionId(session_id if session_id else f'cli-{uuid4().hex[:8]}')
    meta = CreateSessionMeta() if cwd is None else CreateSessionMeta(cwd=cwd)
    agent = loop.create(sid, AgentOptions(provider=provider, model=model), meta)
    app = CliApp(ctx=ctx, agent=agent, fibers=fibers)
    # The printer rides a plugin fiber so disposal unwinds it with the tree.
    app.fibers.append(await ctx.plugin(_presentation_plugin(out)))
    return app


def _user_message(text: str) -> Message:
    return {
        'id': f'cli-msg-{uuid4().hex[:8]}',
        'role': 'user',
        'content': [{'type': 'text', 'text': text}],
        'source': {'kind': 'user'},
    }


async def run_repl(
    app: CliApp,
    *,
    read: InputReader | None = None,
    out: TextIO = sys.stdout,
) -> None:
    """Read user input, drive the agent, print its stream; ``/exit`` or EOF
    quits. ``read`` is the injectable line reader (defaults to ``input``)."""

    def read_line(prompt: str) -> str:
        if read is not None:
            return read(prompt)
        return input(prompt)

    agent = app.agent
    while True:
        try:
            line = await asyncio.to_thread(read_line, 'you> ')
        except (EOFError, OSError):
            out.write('\n')
            break
        text = line.strip()
        if not text:
            continue
        if text in ('/exit', '/quit'):
            break
        try:
            agent.followup(_user_message(text))
            await agent.when_idle()
        except Exception as error:
            out.write(f'[error] {error}\n')
            out.flush()


def build_parser() -> argparse.ArgumentParser:
    """The CLI argument parser (kept separate for testing)."""
    parser = argparse.ArgumentParser(
        prog='pydsh',
        description='Minimal terminal agent CLI for the pydsh rewrite.',
    )
    parser.add_argument(
        '--provider',
        default=DEEPSEEK_PROVIDER,
        help=f'provider route (default: {DEEPSEEK_PROVIDER})',
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'model id (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--cwd',
        default=None,
        help='workspace directory recorded on the session (default: none)',
    )
    parser.add_argument(
        '--session-id',
        default=None,
        help='session id (default: a fresh cli-<random> id)',
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    """Assemble the app, run the REPL, and dispose the tree on the way out."""
    app = await assemble(
        Context(),
        provider=args.provider,
        model=args.model,
        cwd=args.cwd,
        session_id=args.session_id,
        extra_plugins=bash_tool_plugins(),
    )
    try:
        await run_repl(app)
    finally:
        await app.dispose()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.provider == DEEPSEEK_PROVIDER and not os.environ.get(DEEPSEEK_API_KEY_ENV):
        print(
            f'pydsh: error: {DEEPSEEK_API_KEY_ENV} is not set; export it '
            'before launching (the DeepSeek adapter resolves its key from the '
            'environment).',
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except CliError as error:
        print(f'pydsh: error: {error}', file=sys.stderr)
        return 2
