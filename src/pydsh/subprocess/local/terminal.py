"""Local terminal-process implementation for the subprocess seam.

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess-local/src/terminal.ts``.

SIMPLIFIED PORT — no PTY. The TS implementation allocates a real terminal via
node-pty; core spawn semantics do not depend on a PTY, so this port runs the
terminal process on pipes instead (no third-party pty dependency):

- stdin is a pipe; stdout and stderr are merged into one byte stream
  (``stderr=STDOUT``), approximating a terminal's combined output;
- ``rows``/``cols`` on the spec are accepted but ignored (no window size);
- without a controlling tty there is no foreground process group:
  ``inspect_foreground`` always returns ``None`` and ``signal_foreground``
  raises — consumers needing job control must wait for a real PTY backend;
- descendant identity tracking (start-time-verified process identities from
  the TS inspector) is replaced by whole-process-group escalation.

The seam's termination contract is kept: ``terminate()`` is idempotent,
escalates SIGTERM → ``grace_ms`` → SIGKILL on the whole process group, and
settles only once no write, inspection, or signal call remains in flight.
"""

from __future__ import annotations

import asyncio
import os
from signal import SIGKILL, SIGTERM

from ..types import (
    SubprocessOutcome,
    SubprocessTerminalForeground,
    SubprocessTerminalHandle,
    SubprocessTerminalSignal,
)
from .spawn import _outcome_of, kill_group

#: Liveness-poll cadence for whole-group quiescence waits, in seconds.
_GROUP_POLL_SECONDS = 0.025


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalTerminalHandle(SubprocessTerminalHandle):
    """A local terminal whose process-session ownership stays below the pipe
    transport; see the module docstring for the simplifications relative to
    the node-pty implementation.
    """

    def __init__(self, proc: asyncio.subprocess.Process, grace_ms: float) -> None:
        self._proc = proc
        self._grace_ms = grace_ms
        self.pid = proc.pid
        if proc.stdout is None:  # pragma: no cover - spawn always pipes stdout
            raise RuntimeError('terminal process must be spawned with piped stdout')
        self.output = proc.stdout
        self.done: asyncio.Future[SubprocessOutcome] = asyncio.ensure_future(
            self._wait()
        )
        self._cleanup: asyncio.Task[None] | None = None
        self._exited = False

    async def _wait(self) -> SubprocessOutcome:
        returncode = await self._proc.wait()
        self._exited = True
        return _outcome_of(returncode)

    async def write(self, data: str) -> None:
        if self._exited:
            raise RuntimeError('terminal process has exited')
        if self._proc.stdin is None:  # pragma: no cover - spawn always pipes stdin
            raise RuntimeError('terminal process must be spawned with piped stdin')
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()

    async def inspect_foreground(self) -> SubprocessTerminalForeground | None:
        """Always ``None``: a pipe-based terminal has no controlling tty and
        therefore no resolvable foreground process group (simplified port)."""
        return None

    async def signal_foreground(self, signal: SubprocessTerminalSignal) -> int:
        """Raises: foreground-group signalling requires a controlling
        terminal, which this pipe-based port does not provide."""
        raise RuntimeError(
            'foreground-group signalling requires a controlling terminal;'
            ' this port runs terminals on pipes'
        )

    async def terminate(self) -> None:
        if self._cleanup is None:
            self._cleanup = asyncio.ensure_future(self._close_once())
        await asyncio.shield(self._cleanup)

    def terminate_for_host_exit(self) -> None:
        """Force-terminate the process group synchronously during host exit.
        Does not claim quiescence and does not replace terminate()."""
        kill_group(self.pid, SIGKILL)

    async def _close_once(self) -> None:
        grace = self._grace_ms / 1000
        if not self._exited:
            kill_group(self.pid, SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(self.done), timeout=grace)
            except TimeoutError:
                pass
        if not self._exited:
            kill_group(self.pid, SIGKILL)
            try:
                await asyncio.wait_for(asyncio.shield(self.done), timeout=grace)
            except TimeoutError:
                pass
        if not self._exited:
            raise RuntimeError(f'terminal cleanup failed; surviving pid: {self.pid}')
        # Await whole-group quiescence: a session member that outlived the
        # top-level process must not survive teardown unnoticed. Group
        # liveness has no event source; polling killpg(0) is the mechanism.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace
        while _group_alive(self.pid) and loop.time() < deadline:
            remaining = max(0.001, deadline - loop.time())
            await asyncio.sleep(min(_GROUP_POLL_SECONDS, remaining))
        if _group_alive(self.pid):
            raise RuntimeError(
                f'terminal cleanup failed; process group of pid {self.pid} still live'
            )
