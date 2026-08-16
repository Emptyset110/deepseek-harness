"""Behavior tests for the local bash executor (``LocalBashExecutor``), using
real ``bash`` child processes (echo, sleep, false, pwd).

Ported-semantics tests for the deepseek-harness shell port (MIT); source:
packages/shell/bash-local/tests (resolve/run/start semantics).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from pycordis import Context
from pydsh.shell import ShellExecRequest
from pydsh.shell.local import (
    ENV_OVERRIDES,
    LocalBashConfig,
    LocalBashExecutor,
    assert_serviceable_bash_config,
)
from pydsh.subprocess import AbortController
from pydsh.subprocess.local import LocalSubprocessRuntime


def _read_text(path: str) -> str:
    with open(path) as file:
        return file.read()


@pytest.fixture
async def ctx() -> AsyncGenerator[Context]:
    context = Context()
    subprocess_fiber = context.plugin(LocalSubprocessRuntime)
    await subprocess_fiber
    shell_fiber = context.plugin(LocalBashExecutor)
    await shell_fiber
    yield context
    await shell_fiber.dispose()
    await subprocess_fiber.dispose()


def shell(ctx: Context) -> LocalBashExecutor:
    return ctx.get("shell")


async def run(ctx: Context, command: str, **kwargs: Any) -> Any:
    executor = shell(ctx)
    return await executor.run(
        executor.resolve(ShellExecRequest(command=command, **kwargs))
    )


class TestConfig:
    def test_defaults(self) -> None:
        config = LocalBashExecutor.Config(None)
        assert config.cwd is None
        assert config.timeout_ms == 120_000
        assert config.max_timeout_ms == 600_000
        assert config.max_output_bytes == 64_000
        assert config.grace_ms == 3_000

    def test_mapping_overrides(self) -> None:
        config = LocalBashExecutor.Config({"timeout_ms": 5_000, "cwd": "/tmp"})
        assert config.timeout_ms == 5_000
        assert config.cwd == "/tmp"

    @pytest.mark.parametrize(
        "field",
        [
            "timeout_ms",
            "max_timeout_ms",
            "max_output_bytes",
            "max_spill_bytes",
            "grace_ms",
        ],
    )
    async def test_nonpositive_field_fails_loud(self, field: str) -> None:
        context = Context()
        await context.plugin(LocalSubprocessRuntime)
        with pytest.raises(ValueError, match=field):
            await context.plugin(LocalBashExecutor, {field: 0})

    def test_grace_beyond_timer_bound_rejected(self) -> None:
        with pytest.raises(ValueError, match="grace_ms"):
            assert_serviceable_bash_config(LocalBashConfig(grace_ms=2_147_483_648))


class TestResolve:
    def test_fills_defaults(self, ctx: Context) -> None:
        spec = shell(ctx).resolve(ShellExecRequest(command="true"))
        assert spec.command == "true"
        assert spec.workdir == os.getcwd()
        assert spec.timeout_ms == 120_000
        assert spec.stdout_max_bytes == 64_000

    def test_caps_timeout(self, ctx: Context) -> None:
        spec = shell(ctx).resolve(
            ShellExecRequest(command="true", timeout_ms=999_999_999)
        )
        assert spec.timeout_ms == 600_000

    def test_invalid_timeout_raises(self, ctx: Context) -> None:
        with pytest.raises(ValueError, match=r"request\.timeout_ms"):
            shell(ctx).resolve(ShellExecRequest(command="true", timeout_ms=0))

    def test_carries_optional_fields_verbatim(self, ctx: Context) -> None:
        controller = AbortController()
        spec = shell(ctx).resolve(
            ShellExecRequest(
                command="cat",
                workdir="/tmp",
                stdin="payload",
                env={"A": "1"},
                dsh_env={"DSH_X": "2"},
                signal=controller.signal,
            )
        )
        assert spec.workdir == "/tmp"
        assert spec.stdin == "payload"
        assert spec.env == {"A": "1"}
        assert spec.dsh_env == {"DSH_X": "2"}
        assert spec.signal is controller.signal


class TestRun:
    async def test_success_captures_stdout(self, ctx: Context) -> None:
        result = await run(ctx, "echo hello")
        assert result.exit_code == 0
        assert result.signal is None
        assert result.stdout.text == "hello\n"
        assert not result.stdout.truncated
        assert result.stderr.text == ""
        assert not result.timed_out
        assert not result.aborted
        assert result.timeout_ms == 120_000
        assert result.sandbox is None

    async def test_nonzero_exit_resolves(self, ctx: Context) -> None:
        # A non-zero exit is NOT an infrastructure failure.
        result = await run(ctx, "false")
        assert result.exit_code == 1
        result = await run(ctx, "exit 42")
        assert result.exit_code == 42

    async def test_captures_stderr(self, ctx: Context) -> None:
        result = await run(ctx, "echo out; echo err >&2")
        assert result.stdout.text == "out\n"
        assert result.stderr.text == "err\n"

    async def test_runs_in_requested_workdir(self, ctx: Context) -> None:
        result = await run(ctx, "pwd", workdir="/tmp")
        assert result.stdout.text.strip() == "/tmp"

    async def test_timeout_kills_and_classifies(self, ctx: Context) -> None:
        result = await run(ctx, "sleep 30", timeout_ms=100)
        assert result.timed_out
        assert not result.aborted
        assert result.signal in ("SIGTERM", "SIGKILL")
        assert result.exit_code is None
        assert result.timeout_ms == 100

    async def test_abort_kills_and_classifies(self, ctx: Context) -> None:
        controller = AbortController()
        executor = shell(ctx)
        spec = executor.resolve(
            ShellExecRequest(command="sleep 30", signal=controller.signal)
        )
        task = asyncio.ensure_future(executor.run(spec))
        await asyncio.sleep(0.1)
        controller.abort()
        result = await task
        assert result.aborted
        assert not result.timed_out

    async def test_writes_stdin_and_closes(self, ctx: Context) -> None:
        result = await run(ctx, "cat", stdin="piped in")
        assert result.exit_code == 0
        assert result.stdout.text == "piped in"

    async def test_model_friendly_env_overrides(self, ctx: Context) -> None:
        result = await run(ctx, 'echo "TERM=$TERM NO_COLOR=$NO_COLOR PAGER=$PAGER"')
        assert result.stdout.text == "TERM=dumb NO_COLOR=1 PAGER=cat\n"

    async def test_caller_env_beats_overrides(self, ctx: Context) -> None:
        result = await run(ctx, 'echo "$TERM"', env={"TERM": "xterm"})
        assert result.stdout.text == "xterm\n"

    async def test_dsh_env_beats_caller_env(self, ctx: Context) -> None:
        result = await run(
            ctx,
            'echo "$DSH_SESSION"',
            env={"DSH_SESSION": "caller"},
            dsh_env={"DSH_SESSION": "managed"},
        )
        assert result.stdout.text == "managed\n"

    async def test_infrastructure_failure_raises(self, ctx: Context) -> None:
        executor = shell(ctx)
        with pytest.raises(FileNotFoundError):
            await executor.run(
                executor.resolve(
                    ShellExecRequest(
                        command="true", workdir="/nonexistent-dir-dsh-test"
                    )
                )
            )

    async def test_stdout_budget_is_respected(self, ctx: Context) -> None:
        result = await run(
            ctx,
            'printf "x%.0s" $(seq 1 20000)',
            stdout_max_bytes=1_000,
        )
        assert result.stdout.truncated
        assert len(result.stdout.text.encode()) <= 1_000
        # The spill file holds the complete stream.
        assert result.stdout.spill_path is not None
        assert len(_read_text(result.stdout.spill_path)) == 20_000


class TestStart:
    async def test_background_echo_settles_completed(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(ShellExecRequest(command="echo bg"))
        )
        assert proc.status == "running"
        await proc.done
        assert proc.status == "completed"
        assert proc.exit_code == 0
        read = proc.read_output()
        assert "bg" in read.delta

    async def test_reads_are_incremental(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(
                ShellExecRequest(command="echo first; sleep 0.2; echo second")
            )
        )
        await proc.done
        first = proc.read_output()
        assert "first" in first.delta and "second" in first.delta
        second = proc.read_output()
        assert second.delta == ""
        assert not second.lossy

    async def test_stderr_merges_marked(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(ShellExecRequest(command="echo out; echo err >&2"))
        )
        await proc.done
        read = proc.read_output()
        assert "out" in read.delta
        assert "[stderr]\nerr" in read.delta

    async def test_no_timeout_applies(self, ctx: Context) -> None:
        executor = shell(ctx)
        # The spec's timeout_ms is ignored for background processes.
        proc = await executor.start(
            executor.resolve(
                ShellExecRequest(command="sleep 0.3; echo slow", timeout_ms=50)
            )
        )
        await proc.done
        assert proc.status == "completed"
        assert "slow" in proc.read_output().delta

    async def test_kill_terminates(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(ShellExecRequest(command="sleep 30"))
        )
        assert proc.kill() is True
        assert proc.status == "killed"
        # Idempotent once finished.
        await proc.done
        assert proc.kill() is False
        assert proc.exit_code is None
        assert proc.signal in ("SIGTERM", "SIGKILL")

    async def test_nonzero_exit_completes(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(ShellExecRequest(command="exit 7"))
        )
        await proc.done
        assert proc.status == "completed"
        assert proc.exit_code == 7

    async def test_caller_abort_marks_killed(self, ctx: Context) -> None:
        controller = AbortController()
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(
                ShellExecRequest(command="sleep 30", signal=controller.signal)
            )
        )
        controller.abort()
        await proc.done
        assert proc.status == "killed"

    async def test_env_overrides_apply(self, ctx: Context) -> None:
        executor = shell(ctx)
        proc = await executor.start(
            executor.resolve(ShellExecRequest(command='echo "$GIT_PAGER"'))
        )
        await proc.done
        assert proc.read_output().delta == f"{ENV_OVERRIDES['GIT_PAGER']}\n"
