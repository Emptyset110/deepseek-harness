"""Local Service Provider for the bash capability seam over the subprocess
capability seam. Public commands run as ``bash -c`` in a managed process
group spawned through ``ctx.subprocess``; subclasses may reuse the same
mechanics with an explicit argv. This executor owns command defaulting,
deadlines and cause classification, the model-friendly terminal environment,
and the model-facing stdout/stderr merge for background reads. Execution
policy belongs in ``tools/pre-execute`` or a sandboxing executor.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/bash-local/src/index.ts``.

Python mappings:

- ``ctx.subprocess.spawn`` is a coroutine in the Python port and raises
  spawn-level failures directly, so :meth:`LocalBashExecutor.start` is a
  coroutine too and the TS "spawn failure settles the handle as killed with
  the note on the read path" branch is unreachable here — an infrastructure
  failure raises from ``start``/``run`` before a handle exists.
- The settings-section overlay (``installSettingsSection``) is dropped with
  the unported settings capability: the construction config is the single
  source, read through :attr:`LocalBashExecutor.config`.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pycordis import Context
from pydsh.subprocess import (
    MAX_TIMER_DELAY_MS,
    AbortSignal,
    CollectedOutput,
    SpillLimit,
    StdinData,
    SubprocessCollect,
    SubprocessHandle,
    SubprocessOutputReader,
    SubprocessSpawnSpec,
    SubprocessStdio,
)

from . import ShellExecutor
from .timeout import clamp_timeout, deadline, timeout_of
from .types import (
    ShellExecRequest,
    ShellExecSpec,
    ShellProcess,
    ShellProcessRead,
    ShellProcessStatus,
    ShellRunResult,
    ShellSandboxInfo,
)

__all__ = [
    "DEFAULT_GRACE_MS",
    "DEFAULT_MAX_SPILL_BYTES",
    "ENV_OVERRIDES",
    "LocalBashConfig",
    "LocalBashExecutor",
    "assert_serviceable_bash_config",
]

#: Model-friendly environment overrides: disable colors, pagers, and
#: interactive terminal features that would garble tool output (the same set
#: Codex hardcodes; Claude Code achieves it via TERM=dumb). Merged first
#: into the spawn's explicit env, so a trusted caller's own entry still
#: wins; the subprocess service applies its credential scrub independently.
ENV_OVERRIDES: dict[str, str] = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
}

#: Default SIGTERM→SIGKILL grace period (the ``grace_ms`` config; matches
#: OpenCode's 3s).
DEFAULT_GRACE_MS = 3_000

#: Default per-stream spill cap (the ``max_spill_bytes`` config).
DEFAULT_MAX_SPILL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class LocalBashConfig:
    """Resolved plugin config (every field defaulted except ``cwd``)."""

    #: Default working directory for commands (default: ``os.getcwd()``).
    cwd: str | None = None
    #: Default foreground timeout in milliseconds.
    timeout_ms: float = 120_000
    #: Upper bound for per-call timeout overrides.
    max_timeout_ms: float = 600_000
    #: Per-stream in-memory output cap; overflow spills to a temp file.
    max_output_bytes: int = 64_000
    #: Per-stream spill-file cap; larger streams retain only their in-memory
    #: tail.
    max_spill_bytes: int = DEFAULT_MAX_SPILL_BYTES
    #: Grace period for kill escalation and inherited pipes; at most
    #: ``MAX_TIMER_DELAY_MS``.
    grace_ms: float = DEFAULT_GRACE_MS


def _assert_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"bash-local: {name} must be a positive finite number")


def assert_serviceable_bash_config(config: LocalBashConfig) -> None:
    """Reject a resolved config this executor could not run with. The field
    types express neither "positive and finite" nor the timer bound
    ``grace_ms`` has to fit, so a bad value is refused where it is written
    instead of failing at the next command.

    :param config: the resolved config.
    :raises ValueError: naming the field that cannot be used.
    """
    _assert_positive_finite("timeout_ms", config.timeout_ms)
    _assert_positive_finite("max_timeout_ms", config.max_timeout_ms)
    _assert_positive_finite("max_output_bytes", config.max_output_bytes)
    _assert_positive_finite("max_spill_bytes", config.max_spill_bytes)
    _assert_positive_finite("grace_ms", config.grace_ms)
    if config.grace_ms > MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"bash-local: grace_ms must be no greater than {MAX_TIMER_DELAY_MS}"
        )


def _resolve_config(
    config: Mapping[str, Any] | LocalBashConfig | None,
) -> LocalBashConfig:
    """Cordis loader config hook: raw mapping -> validated resolved config."""
    if isinstance(config, LocalBashConfig):
        resolved = config
    else:
        raw = dict(config) if config is not None else {}
        resolved = LocalBashConfig(
            cwd=raw.get("cwd"),
            timeout_ms=raw.get("timeout_ms", LocalBashConfig.timeout_ms),
            max_timeout_ms=raw.get("max_timeout_ms", LocalBashConfig.max_timeout_ms),
            max_output_bytes=raw.get(
                "max_output_bytes", LocalBashConfig.max_output_bytes
            ),
            max_spill_bytes=raw.get("max_spill_bytes", LocalBashConfig.max_spill_bytes),
            grace_ms=raw.get("grace_ms", LocalBashConfig.grace_ms),
        )
    assert_serviceable_bash_config(resolved)
    return resolved


def _final_output(reader: SubprocessOutputReader) -> CollectedOutput:
    """Project a settled collect-mode reader into the final CollectedOutput
    shape."""
    read = reader.read_from(0)
    return CollectedOutput(
        text=read.text, truncated=read.lossy, spill_path=read.spill_path
    )


class _LocalShellProcess:
    """Background process handle over one live subprocess handle."""

    def __init__(
        self, running: SubprocessHandle, caller_signal: AbortSignal | None
    ) -> None:
        self.status: ShellProcessStatus = "running"
        self.exit_code: int | None = None
        self.signal: str | None = None
        self.sandbox: ShellSandboxInfo | None = None
        self.done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._running = running
        self._caller_signal = caller_signal
        self._stdout_offset = 0
        self._stderr_offset = 0

    def settle(self) -> None:
        """Stamp exit facts once the underlying process closes."""
        outcome = self._running.done.result()
        # Any signal termination is killed, including a command signaling
        # itself, as is a run cut short by the caller's abort signal.
        if self.status == "running":
            caller_aborted = (
                self._caller_signal is not None and self._caller_signal.aborted
            )
            self.status = (
                "killed"
                if caller_aborted or outcome.signal is not None
                else "completed"
            )
        self.exit_code = outcome.exit_code
        self.signal = outcome.signal
        self.done.set_result(None)

    def read_output(self) -> ShellProcessRead:
        collected = LocalBashExecutor._collected(self._running)
        out = collected[0].read_from(self._stdout_offset)
        err = collected[1].read_from(self._stderr_offset)
        self._stdout_offset = out.next_offset
        self._stderr_offset = err.next_offset
        # Single newline between sections: stdout chunks usually end with
        # one already; add it only when missing.
        separator = "\n" if out.text and not out.text.endswith("\n") else ""
        delta = out.text + (f"{separator}[stderr]\n{err.text}" if err.text else "")
        return ShellProcessRead(
            delta=delta,
            lossy=out.lossy or err.lossy,
            stdout_spill_path=out.spill_path,
            stderr_spill_path=err.spill_path,
        )

    def kill(self) -> bool:
        if self.status != "running":
            return False
        self.status = "killed"
        self._running.terminate()
        return True


class LocalBashExecutor(ShellExecutor):
    """Local bash executor over ``ctx.subprocess``. Bounded output, spill
    files, and process-group SIGTERM→SIGKILL escalation are the subprocess
    service's mechanics; this executor supplies their configured budgets per
    spawn, so a still-running background process stays managed (killed and
    joined at composition teardown) even across an executor reload.
    """

    inject: ClassVar[list[str]] = ["subprocess"]

    #: Cordis loader config hook: raw mapping -> validated
    #: :class:`LocalBashConfig`.
    Config = staticmethod(_resolve_config)

    def __init__(
        self,
        ctx: Context,
        config: Mapping[str, Any] | LocalBashConfig | None = None,
    ) -> None:
        super().__init__(ctx)
        # The loader applies Config before construction; a direct
        # constructor call resolves here.
        self._config = (
            config if isinstance(config, LocalBashConfig) else _resolve_config(config)
        )

    @property
    def config(self) -> LocalBashConfig:
        """The validated construction config (the single source — the TS
        settings-section overlay is not ported)."""
        return self._config

    def resolve(self, request: ShellExecRequest) -> ShellExecSpec:
        """Resolve a request into a fully-specified spec: fill ``workdir``
        from ``config.cwd`` (else ``os.getcwd()``), and ``timeout_ms`` from
        ``config.timeout_ms``, capped at ``config.max_timeout_ms``. The tool
        layer calls this before :meth:`run`/:meth:`start`, so those methods
        receive explicit values and never re-default.
        """
        config = self.config
        timeout_ms = clamp_timeout(
            request.timeout_ms,
            config.timeout_ms,
            config.max_timeout_ms,
            "bash-local: request.timeout_ms",
        )
        stdout_max_bytes = (
            request.stdout_max_bytes
            if request.stdout_max_bytes is not None
            else config.max_output_bytes
        )
        _assert_positive_finite("request.stdout_max_bytes", stdout_max_bytes)
        return ShellExecSpec(
            command=request.command,
            workdir=request.workdir or config.cwd or os.getcwd(),
            timeout_ms=timeout_ms,
            stdout_max_bytes=stdout_max_bytes,
            signal=request.signal,
            # Carry stdin/ordinary env/trusted dsh_env through verbatim —
            # optional, no config default. The subprocess service owns the
            # scrub and merge order.
            stdin=request.stdin,
            env=request.env,
            dsh_env=request.dsh_env,
            # Carry a sandbox policy through verbatim: this executor never
            # confines, so the field is inert here (the seam contract) — a
            # sandboxing subclass overrides resolve() to stamp its default
            # instead.
            sandbox_policy=request.sandbox_policy,
        )

    def _spawn_spec(
        self,
        spec: ShellExecSpec,
        argv: Sequence[str],
        stdout_max_bytes: int,
        signal: AbortSignal | None,
    ) -> SubprocessSpawnSpec:
        """Map one resolved bash spec and explicit argv onto a
        fully-specified subprocess spawn."""
        config = self.config

        def collect(max_bytes: int) -> SubprocessCollect:
            return SubprocessCollect(max_bytes, SpillLimit(config.max_spill_bytes))

        return SubprocessSpawnSpec(
            argv=argv,
            cwd=spec.workdir,
            stdio=SubprocessStdio(
                stdin=StdinData(spec.stdin) if spec.stdin is not None else "ignore",
                stdout=collect(stdout_max_bytes),
                stderr=collect(config.max_output_bytes),
            ),
            grace_ms=config.grace_ms,
            signal=signal,
            # One explicit env map for the seam, layered so the trusted
            # dsh_env snapshot beats both the caller's env and the terminal
            # overrides; the subprocess service merges the whole map after
            # its ambient scrub.
            env={**ENV_OVERRIDES, **(spec.env or {}), **(spec.dsh_env or {})},
        )

    @staticmethod
    def _collected(
        handle: SubprocessHandle,
    ) -> tuple[SubprocessOutputReader, SubprocessOutputReader]:
        """The collect-mode readers the executor itself requested (present
        by construction)."""
        stdout, stderr = handle.collected.stdout, handle.collected.stderr
        if stdout is None or stderr is None:
            raise RuntimeError(
                "bash-local: subprocess implementation dropped a requested"
                " collect stream"
            )
        return stdout, stderr

    def _subprocess(self) -> Any:
        """The injected subprocess service, read through ``ctx.get``: a
        service reached through a caller's context view (``ServiceView``)
        cannot use property access for its own injections — the view binds
        ``self.ctx`` to the caller's context, which never declared
        ``subprocess``.
        """
        return self.ctx.get("subprocess")

    async def run(self, spec: ShellExecSpec) -> ShellRunResult:
        return await self.run_argv(spec, ["bash", "-c", spec.command])

    async def run_argv(
        self, spec: ShellExecSpec, argv: Sequence[str]
    ) -> ShellRunResult:
        """Run an explicit argv with the foreground lifecycle, environment,
        output, timeout, and cancellation semantics of this executor.
        Subclasses use this after replacing the public command's shell argv
        at an execution boundary.

        :param spec: resolved execution settings and caller-owned command
            metadata.
        :param argv: exact executable and arguments to hand to
            ``ctx.subprocess``.
        :return: the settled foreground result with collected output and
            cause facts.
        """
        # One deadline combines timeout and upstream cancellation; closing
        # clears its timer.
        with deadline(spec.signal, spec.timeout_ms, "BASH_TIMEOUT") as d:
            handle = await self._subprocess().spawn(
                self._spawn_spec(spec, argv, spec.stdout_max_bytes, d.signal)
            )
            outcome = await handle.done
            collected = self._collected(handle)
            # Only this executor's timeout reason counts as timed_out; outer
            # deadlines count as aborts.
            timed_out = timeout_of(d.signal, "BASH_TIMEOUT") is not None
            aborted = d.signal.aborted and not timed_out
            return ShellRunResult(
                exit_code=outcome.exit_code,
                signal=outcome.signal,
                timed_out=timed_out,
                aborted=aborted,
                timeout_ms=spec.timeout_ms,
                stdout=_final_output(collected[0]),
                stderr=_final_output(collected[1]),
            )

    async def start(self, spec: ShellExecSpec) -> ShellProcess:
        return await self.start_argv(spec, ["bash", "-c", spec.command])

    async def start_argv(
        self, spec: ShellExecSpec, argv: Sequence[str]
    ) -> ShellProcess:
        """Start an explicit argv with the background lifecycle,
        environment, output, cancellation, and process-tree ownership
        semantics of this executor. Subclasses use this after replacing the
        public command's shell argv at an execution boundary.

        :param spec: resolved execution settings and caller-owned command
            metadata.
        :param argv: exact executable and arguments to hand to
            ``ctx.subprocess``.
        :return: the live background handle.
        """
        # Background runs ignore timeout_ms; callers stop them through
        # kill() or spec.signal.
        running = await self._subprocess().spawn(
            self._spawn_spec(spec, argv, self.config.max_output_bytes, spec.signal)
        )
        proc = _LocalShellProcess(running, spec.signal)

        def settle(_done: asyncio.Future[Any]) -> None:
            proc.settle()

        running.done.add_done_callback(settle)
        return proc
