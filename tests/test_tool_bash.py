"""Behavior tests for the model-facing ``bash`` tool: output rendering, the
background outcome mapping, and the registered tool's execution pipeline
(validation, foreground results, timeout/abort, background gating) over real
bash subprocesses.

Ported-semantics tests for the deepseek-harness shell port (MIT); sources:
packages/shell/tool-bash/src (render.ts, background.ts, index.ts).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest

import pydsh.tool_bash as tool_bash
from pycordis import Context
from pydsh.llm import CallId
from pydsh.shell import CollectedOutput, ShellRunResult
from pydsh.shell.local import LocalBashExecutor
from pydsh.shell.types import ShellProcessRead, ShellSandboxInfo
from pydsh.subprocess import AbortController
from pydsh.subprocess.local import LocalSubprocessRuntime
from pydsh.system_prompt import SystemPrompt
from pydsh.tool_bash import (
    apply,  # noqa: F401  (re-exported plugin entry point)
    bash_description,
    validate_bash_args,
    validate_escalation_args,
)
from pydsh.tool_bash import background as background_mod
from pydsh.tool_bash.render import (
    render_process_read,
    render_result,
    sandbox_denial_marker,
)
from pydsh.tools import (
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    ToolExecutionFailure,
    ToolExecutionInput,
    ToolExecutionSuccess,
    ToolRuntime,
)


def collected(
    text: str, truncated: bool = False, spill: str | None = None
) -> CollectedOutput:
    return CollectedOutput(text=text, truncated=truncated, spill_path=spill)


_TMP_REALPATH = os.path.realpath("/tmp")


def result(
    exit_code: int | None = 0,
    *,
    signal: str | None = None,
    timed_out: bool = False,
    aborted: bool = False,
    timeout_ms: float = 120_000,
    stdout: CollectedOutput | None = None,
    stderr: CollectedOutput | None = None,
    sandbox: ShellSandboxInfo | None = None,
) -> ShellRunResult:
    return ShellRunResult(
        exit_code=exit_code,
        signal=signal,
        timed_out=timed_out,
        aborted=aborted,
        timeout_ms=timeout_ms,
        stdout=stdout if stdout is not None else collected(""),
        stderr=stderr if stderr is not None else collected(""),
        sandbox=sandbox,
    )


class TestRenderResult:
    def test_clean_exit(self) -> None:
        assert render_result(result(stdout=collected("hello\n"))) == "hello\n"

    def test_nonzero_exit_marker(self) -> None:
        assert (
            render_result(result(3, stdout=collected("boom\n")))
            == "boom\n[exit code: 3]"
        )

    def test_stderr_section(self) -> None:
        text = render_result(
            result(stdout=collected("out\n"), stderr=collected("err\n"))
        )
        assert text == "out\n[stderr]\nerr\n"

    def test_stderr_separator_added_when_stdout_lacks_newline(self) -> None:
        text = render_result(result(stdout=collected("out"), stderr=collected("err\n")))
        assert text == "out\n[stderr]\nerr\n"

    def test_no_output_placeholder(self) -> None:
        assert render_result(result()) == "(no output)"

    def test_timeout_marker(self) -> None:
        text = render_result(
            result(None, signal="SIGTERM", timed_out=True, timeout_ms=100)
        )
        assert text == (
            "(no output)\n[timed out after 100ms]\n[killed by signal: SIGTERM]"
        )

    def test_signal_beats_exit_code(self) -> None:
        text = render_result(result(0, signal="SIGKILL"))
        assert text == "(no output)\n[killed by signal: SIGKILL]"

    def test_truncation_notice_with_spill(self) -> None:
        text = render_result(
            result(stdout=collected("tail", truncated=True, spill="/tmp/spill"))
        )
        assert text == "tail\n[output truncated; full output: /tmp/spill]"

    def test_truncation_notice_without_spill(self) -> None:
        text = render_result(result(stdout=collected("tail", truncated=True)))
        assert text == "tail\n[output truncated; full output: (unavailable)]"

    def test_sandbox_denial_without_escalation(self) -> None:
        sandbox = ShellSandboxInfo(mode="read-only", denied=True)
        text = render_result(result(1, sandbox=sandbox))
        assert sandbox_denial_marker("read-only") in text
        assert "escalation available" not in text
        assert text.endswith("[exit code: 1]")

    def test_sandbox_denial_with_escalation_hint(self) -> None:
        sandbox = ShellSandboxInfo(mode="read-only", denied=True)
        text = render_result(
            result(1, sandbox=sandbox), ["workspace-write", "danger-full-access"]
        )
        assert "escalation available" in text
        # The exit marker stays last for parse_exit_status.
        assert text.endswith("[exit code: 1]")


class TestRenderProcessRead:
    def test_plain_delta(self) -> None:
        read = ShellProcessRead(delta="new\n", lossy=False)
        assert render_process_read(read) == "new\n"

    def test_lossy_notice_with_paths(self) -> None:
        read = ShellProcessRead(delta="tail", lossy=True, stdout_spill_path="/tmp/out")
        text = render_process_read(read)
        assert text == (
            "tail\n[some output was dropped from memory; full output: /tmp/out]"
        )

    def test_lossy_notice_without_paths(self) -> None:
        read = ShellProcessRead(delta="", lossy=True)
        text = render_process_read(read)
        assert "(unavailable)" in text

    def test_runner_failed_notice(self) -> None:
        read = ShellProcessRead(delta="", lossy=False)
        sandbox = ShellSandboxInfo(
            mode="workspace-write", denied=False, runner_failed=True
        )
        text = render_process_read(read, sandbox)
        assert "the sandbox runner itself failed" in text


class _FakeProc:
    def __init__(self, status: str, exit_code: int | None, signal: str | None) -> None:
        self.status = status
        self.exit_code = exit_code
        self.signal = signal


class TestProcessOutcome:
    def test_completed_reports_exit_code(self) -> None:
        outcome = background_mod.process_outcome(_FakeProc("completed", 7, None))  # type: ignore[arg-type]
        assert outcome.status == "completed"
        assert outcome.detail == "exit code: 7"

    def test_killed_reports_signal(self) -> None:
        outcome = background_mod.process_outcome(_FakeProc("killed", None, "SIGTERM"))  # type: ignore[arg-type]
        assert outcome.status == "killed"
        assert outcome.detail == "signal: SIGTERM"

    def test_killed_without_signal(self) -> None:
        outcome = background_mod.process_outcome(_FakeProc("killed", None, None))  # type: ignore[arg-type]
        assert outcome.detail == "killed before exit"


class TestArgValidation:
    def _valid(self) -> dict[str, Any]:
        return {"command": "true", "description": "Do nothing"}

    def test_blank_command_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid command"):
            validate_bash_args({**self._valid(), "command": "   "})

    def test_blank_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid description"):
            validate_bash_args({**self._valid(), "description": ""})

    @pytest.mark.parametrize("bad", [0, -5])
    def test_nonpositive_timeout_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="invalid timeoutMs"):
            validate_bash_args({**self._valid(), "timeoutMs": bad})

    def test_escalation_pairing(self) -> None:
        with pytest.raises(ValueError, match="requires a justification"):
            validate_escalation_args("workspace-write", None)
        with pytest.raises(ValueError, match="only valid together"):
            validate_escalation_args(None, "because")
        with pytest.raises(ValueError, match="non-empty sentence"):
            validate_escalation_args("workspace-write", "  ")
        validate_escalation_args("workspace-write", "Need to write build output")
        validate_escalation_args(None, None)


class TestDescription:
    def test_background_enabled(self) -> None:
        text = bash_description(True, [])
        assert "run_in_background" in text
        assert "$DSH_*" in text

    def test_background_disabled(self) -> None:
        text = bash_description(False, [])
        assert "Background execution is not available" in text

    def test_escalation_guidance_only_with_modes(self) -> None:
        assert "sandbox_permissions" not in bash_description(True, [])
        text = bash_description(True, ["workspace-write"])
        assert "sandbox_permissions" in text


@pytest.fixture
async def ctx() -> AsyncGenerator[Context]:
    context = Context()
    await context.plugin(SystemPrompt)
    await context.plugin(ToolRuntime)
    subprocess_fiber = context.plugin(LocalSubprocessRuntime)
    await subprocess_fiber
    shell_fiber = context.plugin(LocalBashExecutor)
    await shell_fiber
    tool_fiber = context.plugin(tool_bash)
    await tool_fiber
    yield context
    await tool_fiber.dispose()
    await shell_fiber.dispose()
    await subprocess_fiber.dispose()


def make_input(
    args: dict[str, Any],
    *,
    signal: Any = None,
    call_id: str = "call-1",
) -> ToolExecutionInput:
    return ToolExecutionInput(
        call_id=CallId(call_id),
        name="bash",
        arguments=args,
        signal=signal if signal is not None else AbortController().signal,
    )


def success_text(result: Any) -> str:
    assert isinstance(result, ToolExecutionSuccess)
    block = result.content[0]
    assert block["type"] == "text"
    return block["text"]


def failure_message(result: Any) -> str:
    assert isinstance(result, ToolExecutionFailure)
    return result.error.message


class TestBashTool:
    async def test_tool_is_registered(self, ctx: Context) -> None:
        assert ctx.get("tools").get("bash") is not None

    async def test_foreground_success(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input({"command": "echo hello", "description": "Say hello"})
        )
        assert isinstance(result, ToolExecutionSuccess)
        value = result.value
        assert value["kind"] == "foreground"
        assert value["exitCode"] == 0
        assert value["signal"] is None
        assert value["timedOut"] is False
        assert value["aborted"] is False
        assert value["stdout"] == {"text": "hello\n", "truncated": False}
        assert success_text(result) == "hello\n"

    async def test_nonzero_exit_is_reported_not_errored(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input({"command": "false", "description": "Fail with exit 1"})
        )
        assert isinstance(result, ToolExecutionSuccess)
        assert result.value["exitCode"] == 1
        assert success_text(result).endswith("[exit code: 1]")

    async def test_stderr_renders_marked_section(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input({"command": "echo err >&2", "description": "Write to stderr"})
        )
        assert "[stderr]\nerr\n" in success_text(result)

    async def test_workdir_parameter(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input(
                {
                    "command": "pwd",
                    "description": "Print working directory",
                    "workdir": "/tmp",
                }
            )
        )
        assert success_text(result).strip() == _TMP_REALPATH

    async def test_timeout_parameter(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input(
                {
                    "command": "sleep 30",
                    "description": "Sleep past the timeout",
                    "timeoutMs": 100,
                }
            )
        )
        assert isinstance(result, ToolExecutionSuccess)
        assert result.value["timedOut"] is True
        assert result.value["timeoutMs"] == 100
        text = success_text(result)
        assert "[timed out after 100ms]" in text
        assert "[killed by signal:" in text

    async def test_abort_during_run(self, ctx: Context) -> None:
        controller = AbortController()
        task = asyncio.ensure_future(
            ctx.get("tools").execute(
                make_input(
                    {"command": "sleep 30", "description": "Sleep forever"},
                    signal=controller.signal,
                )
            )
        )
        await asyncio.sleep(0.1)
        controller.abort()
        result = await task
        message = failure_message(result)
        assert result.error.info is not None
        assert result.error.info.code in (
            TOOL_ABORTED,
            TOOL_ABORTED_BEFORE_DISPATCH,
        )
        assert "aborted" in message

    async def test_missing_required_arg_fails(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(make_input({"command": "true"}))
        assert "invalid arguments" in failure_message(result)

    async def test_blank_command_fails(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input({"command": "  ", "description": "Blank command"})
        )
        assert "invalid command" in failure_message(result)

    async def test_background_without_jobs_capability(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input(
                {
                    "command": "sleep 1",
                    "description": "Background sleep",
                    "run_in_background": True,
                }
            )
        )
        assert "background jobs unavailable" in failure_message(result)

    async def test_sandbox_permissions_unavailable(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input(
                {
                    "command": "true",
                    "description": "Escalate without sandbox",
                    "sandbox_permissions": "workspace-write",
                    "justification": "Need wider access for this command",
                }
            )
        )
        assert "not available in this composition" in failure_message(result)

    async def test_escalation_pairing_enforced(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input(
                {
                    "command": "true",
                    "description": "Escalate without justification",
                    "sandbox_permissions": "workspace-write",
                }
            )
        )
        assert "requires a justification" in failure_message(result)


class TestBashToolBackgroundDisabled:
    @pytest.fixture
    async def ctx(self) -> AsyncGenerator[Context]:
        context = Context()
        await context.plugin(SystemPrompt)
        await context.plugin(ToolRuntime)
        subprocess_fiber = context.plugin(LocalSubprocessRuntime)
        await subprocess_fiber
        shell_fiber = context.plugin(LocalBashExecutor)
        await shell_fiber
        tool_fiber = context.plugin(tool_bash, {"enableRunInBackground": False})
        await tool_fiber
        yield context
        await tool_fiber.dispose()
        await shell_fiber.dispose()
        await subprocess_fiber.dispose()

    async def test_background_call_rejected(self, ctx: Context) -> None:
        tools = ctx.get("tools")
        definition = tools.get("bash")
        assert "run_in_background" not in definition.parameters["properties"]
        result = await tools.execute(
            make_input(
                {
                    "command": "sleep 1",
                    "description": "Background sleep",
                    "run_in_background": True,
                }
            )
        )
        assert "run_in_background is disabled" in failure_message(result)

    async def test_foreground_still_works(self, ctx: Context) -> None:
        result = await ctx.get("tools").execute(
            make_input({"command": "echo ok", "description": "Say ok"})
        )
        assert success_text(result) == "ok\n"
