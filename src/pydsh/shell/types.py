"""Execution types for the bash executor seam: the request/spec split, the
foreground run result, and the background process handle. Background job
semantics (ids, ownership, polling, notices) belong to the unported
``dsh-jobs`` capability; this seam exposes only process handles. The
captured-output vocabulary is owned by the subprocess seam and re-exported
here so bash consumers keep one import root.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/shell/src/types.ts``.

Python mappings:

- ``NodeJS.Signals`` becomes a plain ``str`` signal name.
- The TS-only template-literal ``DshEnvironment`` mapping is a plain
  ``Mapping[str, str]``; executors still merge it last so managed ``DSH_*``
  facts cannot be displaced by ordinary ``env`` entries.
- ``SandboxExecutionPolicy``/``SandboxMode`` belong to the unported sandbox
  capability; ``sandbox_policy`` is typed ``Any`` and carried verbatim (the
  seam contract), and sandbox modes are plain strings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydsh.subprocess import DSH_ENV_PREFIX, AbortSignal, CollectedOutput

__all__ = [
    "DSH_ENV_PREFIX",
    "CollectedOutput",
    "ShellExecRequest",
    "ShellExecSpec",
    "ShellProcess",
    "ShellProcessRead",
    "ShellProcessStatus",
    "ShellRunResult",
    "ShellSandboxInfo",
]


@dataclass(frozen=True)
class ShellSandboxInfo:
    """Sandbox facts for one run, present iff a sandboxing executor handled
    it. Facts are reported independently of process exit status so callers can
    distinguish command failures from policy denials and runner failures."""

    #: The mode the command actually ran under.
    mode: str
    #: Whether the sandbox denied a file operation.
    denied: bool
    #: How completely the selected runner enforced the requested mode.
    enforcement: str | None = None
    #: Whether the sandbox runner failed before the command could run.
    runner_failed: bool | None = None


@dataclass(frozen=True)
class ShellExecRequest:
    """A caller's execution REQUEST: ``workdir`` and ``timeout_ms`` are
    optional and filled by :meth:`ShellExecutor.resolve` from the
    implementation's config. This is the model-/plugin-facing shape; pass it
    to ``resolve()`` to obtain a fully-resolved :class:`ShellExecSpec`."""

    command: str
    #: Working directory override (default: implementation-configured).
    workdir: str | None = None
    #: Timeout override in milliseconds (implementations cap it).
    timeout_ms: float | None = None
    #: Foreground stdout capture budget in bytes. Absent uses the executor's
    #: default output cap. Trusted in-process consumers use this when they
    #: must parse complete stdout up to their own bounded limit; the
    #: model-facing bash tool does not expose it as a parameter.
    stdout_max_bytes: int | None = None
    #: Abort signal — implementations kill the command when it fires.
    signal: AbortSignal | None = None
    #: Text to write to the command's stdin, then close it. Absent leaves
    #: stdin closed/empty (the default for model-driven tool calls). Set by
    #: in-process plugins (e.g. hook bridges writing a JSON payload); the
    #: model-facing bash tool does not expose it as a parameter (a model that
    #: needs stdin uses shell syntax like a heredoc or a pipe).
    stdin: str | None = None
    #: Ordinary environment entries for the command, merged after the
    #: credential scrub. Managed facts belong in ``dsh_env``, which merges
    #: after this map, so an entry here can never displace one.
    env: Mapping[str, str] | None = None
    #: Harness-owned ``DSH_*`` variables for this execution. Executors
    #: discard ambient ``DSH_*`` entries before merging this snapshot last,
    #: so an unavailable current fact cannot inherit a stale value from the
    #: harness process and a caller ``env`` entry cannot displace a managed
    #: one.
    dsh_env: Mapping[str, str] | None = None
    #: Fully resolved per-call sandbox policy; sandboxing executors default
    #: it. Owned by the unported sandbox capability and carried verbatim.
    sandbox_policy: Any = None


@dataclass(frozen=True)
class ShellExecSpec:
    """A resolved execution spec. :meth:`ShellExecutor.resolve` fills and
    caps the required fields; :meth:`ShellExecutor.start` ignores
    ``timeout_ms`` because background processes have no executor timeout."""

    command: str
    workdir: str
    timeout_ms: float
    #: Resolved foreground stdout capture budget in bytes. ``run()`` uses it
    #: for stdout; background jobs and stderr keep the executor's own output
    #: cap.
    stdout_max_bytes: int
    #: Abort signal — implementations kill the command when it fires.
    signal: AbortSignal | None = None
    #: Text to write to stdin before closing it; absent means no stdin.
    stdin: str | None = None
    #: Ordinary environment entries carried through from
    #: ``ShellExecRequest.env``; ``dsh_env`` still merges after them.
    env: Mapping[str, str] | None = None
    #: Managed ``DSH_*`` snapshot; merges after ``env``.
    dsh_env: Mapping[str, str] | None = None
    #: Resolved sandbox policy; ignored by executors that do not confine.
    sandbox_policy: Any = None


@dataclass(frozen=True)
class ShellRunResult:
    """The outcome of one completed (or killed) foreground run."""

    #: Exit code; ``None`` when the process died from a signal.
    exit_code: int | None
    #: Terminating signal (e.g. ``'SIGTERM'``); ``None`` on normal exit.
    signal: str | None
    #: True when the executor's own timeout was the FIRST cause to cut the
    #: command short. Mutually exclusive with ``aborted``: one fused deadline
    #: drives both the timeout and the caller's cancellation, so a timeout
    #: and an abort racing before process close report the single
    #: first-abort cause, not both.
    timed_out: bool
    #: True when the caller's abort signal was the FIRST cause to kill the
    #: command (and it was not the executor's own timeout). Mutually
    #: exclusive with ``timed_out``.
    aborted: bool
    #: The effective timeout applied to this run (after defaulting/capping).
    timeout_ms: float
    stdout: CollectedOutput
    stderr: CollectedOutput
    #: Sandbox execution facts, absent for an unsandboxed executor.
    sandbox: ShellSandboxInfo | None = None


#: Lifecycle of a background process.
ShellProcessStatus = Literal["running", "completed", "killed"]


@dataclass(frozen=True)
class ShellProcessRead:
    """One incremental :meth:`ShellProcess.read_output` read."""

    #: Output produced since the previous read (stderr in a marked section).
    delta: str
    #: True when truncation dropped unread bytes the delta cannot include.
    lossy: bool
    #: Full stdout spill file, when stdout truncation occurred and a safe
    #: path is available.
    stdout_spill_path: str | None = None
    #: Full stderr spill file, when stderr truncation occurred and a safe
    #: path is available.
    stderr_spill_path: str | None = None


class ShellProcess(Protocol):
    """A background process handle returned by :meth:`ShellExecutor.start`.
    It is the only access path; buffered output remains readable after exit.
    Composition teardown (the subprocess service's disposal) kills running
    processes and awaits ``done``; an executor-only reload leaves them
    running.
    """

    #: Process lifecycle state (settled exactly once).
    status: ShellProcessStatus
    #: Exit code once finished (``None`` = killed by signal / still running).
    exit_code: int | None
    #: Terminating signal name, when signal-killed.
    signal: str | None
    #: Resolves when the underlying process closes. Never raises: the Python
    #: subprocess seam raises spawn-level failures from ``start()`` itself
    #: (before a handle exists), so by the time a handle exists the process
    #: is live.
    done: asyncio.Future[None]
    #: Sandbox facts, stamped once a confined process settles.
    sandbox: ShellSandboxInfo | None

    def read_output(self) -> ShellProcessRead:
        """Read output produced since the previous read (consuming —
        consecutive reads never re-deliver). Reads that lost data flag
        ``lossy`` and point at full-stream spill files when available.
        """
        ...

    def kill(self) -> bool:
        """Kill the process group. Returns False when it had already
        finished (no-op); idempotent.
        """
        ...
