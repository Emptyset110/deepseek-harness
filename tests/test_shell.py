"""Behavior tests for the shell Service Definition: the exit-status marker
contract (``parse_exit_status``), the timeout arithmetic port
(``clamp_timeout``/``deadline``/``timeout_of``), and the abstract
``ShellExecutor`` registration shape.

Ported-semantics tests for the deepseek-harness shell port (MIT); sources:
packages/shell/shell/src/render.ts and packages/util/timeout/src/index.ts.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pycordis import Context, Service
from pydsh.shell import (
    ShellExecutor,
    ShellProcess,
    ShellRunResult,
    parse_exit_status,
)
from pydsh.shell.timeout import (
    MAX_TIMER_DELAY_MS,
    TimeoutReason,
    clamp_timeout,
    deadline,
    timeout_of,
)
from pydsh.shell.types import ShellExecRequest, ShellExecSpec
from pydsh.subprocess import AbortController


class TestParseExitStatus:
    def test_clean_exit_without_marker(self) -> None:
        parsed = parse_exit_status("hello\n")
        assert parsed.body == "hello\n"
        assert parsed.exit_code == 0
        assert parsed.signal is None

    def test_nonzero_exit_marker_is_split_off(self) -> None:
        parsed = parse_exit_status("boom\n[exit code: 3]")
        assert parsed.body == "boom"
        assert parsed.exit_code == 3
        assert parsed.signal is None

    def test_signal_marker_is_split_off(self) -> None:
        parsed = parse_exit_status("out\n[killed by signal: SIGTERM]")
        assert parsed.body == "out"
        assert parsed.exit_code is None
        assert parsed.signal == "SIGTERM"

    def test_marker_requires_leading_newline_and_end(self) -> None:
        # Output merely ending with marker-like text is left alone.
        parsed = parse_exit_status("see [exit code: 1]")
        assert parsed.exit_code == 0
        assert parsed.body == "see [exit code: 1]"
        parsed = parse_exit_status("[exit code: 1]\ntrailing")
        assert parsed.exit_code == 0

    def test_timeout_marker_stays_in_body(self) -> None:
        # Only the exit pill is consumed; other markers carry facts no pill
        # shows.
        parsed = parse_exit_status("x\n[timed out after 100ms]\n[exit code: 1]")
        assert parsed.body == "x\n[timed out after 100ms]"
        assert parsed.exit_code == 1


class TestClampTimeout:
    def test_absent_uses_default(self) -> None:
        assert clamp_timeout(None, 120_000, 600_000) == 120_000

    def test_requested_is_capped(self) -> None:
        assert clamp_timeout(999_999, 120_000, 600_000) == 600_000

    def test_requested_below_cap_wins(self) -> None:
        assert clamp_timeout(5_000, 120_000, 600_000) == 5_000

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    def test_invalid_requested_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="positive finite"):
            clamp_timeout(bad, 120_000, 600_000, "test.timeout_ms")


class TestDeadline:
    async def test_timeout_fires_with_coded_reason(self) -> None:
        with deadline(None, 30, "BASH_TIMEOUT") as d:
            assert not d.signal.aborted
            await asyncio.sleep(0.15)
            assert d.signal.aborted
            reason = timeout_of(d.signal, "BASH_TIMEOUT")
            assert reason is not None
            assert reason.timeout_ms == 30

    async def test_upstream_abort_wins_first_cause(self) -> None:
        upstream = AbortController()
        with deadline(upstream.signal, 60_000, "BASH_TIMEOUT") as d:
            upstream.abort("caller cancelled")
            assert d.signal.aborted
            # An upstream win is NOT this executor's timeout.
            assert timeout_of(d.signal, "BASH_TIMEOUT") is None
            assert d.signal.reason == "caller cancelled"

    async def test_already_aborted_upstream_aborts_immediately(self) -> None:
        upstream = AbortController()
        upstream.abort()
        with deadline(upstream.signal, 60_000, "BASH_TIMEOUT") as d:
            assert d.signal.aborted

    async def test_no_timeout_sentinel_forwards_upstream(self) -> None:
        upstream = AbortController()
        with deadline(upstream.signal, 0, "BASH_TIMEOUT") as d:
            assert d.signal is upstream.signal

    async def test_close_clears_the_timer(self) -> None:
        with deadline(None, 50, "BASH_TIMEOUT") as d:
            pass
        await asyncio.sleep(0.1)
        assert not d.signal.aborted

    async def test_invalid_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="no greater than"):
            deadline(None, MAX_TIMER_DELAY_MS + 1, "BASH_TIMEOUT")

    def test_timeout_of_matches_code(self) -> None:
        reason = TimeoutReason("BASH_TIMEOUT", 100)
        carrier: Any = type("Carrier", (), {"reason": reason})()
        assert timeout_of(carrier) is reason
        assert timeout_of(carrier, "BASH_TIMEOUT") is reason
        assert timeout_of(carrier, "OTHER") is None
        assert timeout_of(AbortController().signal) is None


class _DummyExecutor(ShellExecutor):
    def resolve(self, request: ShellExecRequest) -> ShellExecSpec:
        return ShellExecSpec(
            command=request.command,
            workdir=request.workdir or "/",
            timeout_ms=request.timeout_ms or 1000,
            stdout_max_bytes=request.stdout_max_bytes or 64_000,
        )

    async def run(self, spec: ShellExecSpec) -> ShellRunResult:
        raise NotImplementedError

    async def start(self, spec: ShellExecSpec) -> ShellProcess:
        raise NotImplementedError


class TestShellExecutor:
    async def test_subclass_registers_as_ctx_shell(self) -> None:
        ctx = Context()
        fiber = ctx.plugin(_DummyExecutor)
        await fiber
        # ctx.get returns a per-access ServiceView; assert behavior, not
        # identity.
        service = ctx.get("shell")
        assert service.name == "shell"
        spec = service.resolve(ShellExecRequest(command="true"))
        assert spec.command == "true"
        await fiber.dispose()

    async def test_sandbox_mode_defaults_to_none(self) -> None:
        ctx = Context()
        fiber = ctx.plugin(_DummyExecutor)
        await fiber
        assert ctx.get("shell").sandbox_mode is None
        await fiber.dispose()

    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ShellExecutor(Context())  # type: ignore[abstract]

    async def test_duplicate_registration_fails(self) -> None:
        ctx = Context()
        fiber = ctx.plugin(_DummyExecutor)
        await fiber
        with pytest.raises(RuntimeError):
            await ctx.plugin(_DummyExecutor)
        await fiber.dispose()


def test_service_base_registration_name() -> None:
    assert issubclass(ShellExecutor, Service)
