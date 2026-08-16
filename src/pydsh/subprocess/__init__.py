"""Service Definition for the subprocess capability seam (``ctx.subprocess``):
execution-world executable lookup, fully specified managed process trees with
raw or collected stdio, and one terminal-process primitive. Command
defaulting, shell semantics, deadlines, protocol framing, terminal readiness,
and presentation belong to consumers. The local implementation lives in
:mod:`pydsh.subprocess.local`.

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess/src/index.ts``.

Python mapping note: the TS ``spawn(spec)`` is synchronous and reports
spawn-level failures (bad cwd, missing executable, invalid argv/grace,
pre-aborted signal) by rejecting ``handle.done``. asyncio creates processes
asynchronously, so this port makes ``spawn`` a coroutine that raises those
failures directly. The resolve/reject contract is unchanged: a process that
started — including non-zero exits and signal deaths — never raises; it
settles ``handle.done`` with a :class:`SubprocessOutcome`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pycordis import Context, Service

from .types import (
    DSH_ENV_PREFIX,
    MAX_TIMER_DELAY_MS,
    SENSITIVE_ENV_PATTERN,
    AbortController,
    AbortError,
    AbortSignal,
    CollectedOutput,
    SpillLimit,
    StdinData,
    SubprocessCollect,
    SubprocessCollectedOutputs,
    SubprocessHandle,
    SubprocessOutcome,
    SubprocessOutputMode,
    SubprocessOutputRead,
    SubprocessOutputReader,
    SubprocessSpawnSpec,
    SubprocessStdinMode,
    SubprocessStdio,
    SubprocessTerminalForeground,
    SubprocessTerminalHandle,
    SubprocessTerminalSignal,
    SubprocessTerminalSpawnSpec,
)

__all__ = [
    'DSH_ENV_PREFIX',
    'MAX_TIMER_DELAY_MS',
    'SENSITIVE_ENV_PATTERN',
    'AbortController',
    'AbortError',
    'AbortSignal',
    'CollectedOutput',
    'SpillLimit',
    'StdinData',
    'SubprocessCollect',
    'SubprocessCollectedOutputs',
    'SubprocessHandle',
    'SubprocessOutcome',
    'SubprocessOutputMode',
    'SubprocessOutputRead',
    'SubprocessOutputReader',
    'SubprocessRuntime',
    'SubprocessSpawnSpec',
    'SubprocessStdinMode',
    'SubprocessStdio',
    'SubprocessTerminalForeground',
    'SubprocessTerminalHandle',
    'SubprocessTerminalSignal',
    'SubprocessTerminalSpawnSpec',
    'scrubbed_parent_env',
]


def scrubbed_parent_env() -> dict[str, str]:
    """The ambient parent environment minus credential-shaped names and minus
    all ``DSH_*`` names — the canonical base every harness child starts from.
    ``PATH``, ``HOME``, locale, and proxy variables survive, so child CLIs run
    normally; harness identity never leaks implicitly (a deliberately forwarded
    credential or current ``DSH_*`` fact goes through the spec's explicit
    ``env``, which merges after this scrub). Both scrubs match
    case-insensitively, mirroring Windows' case-insensitive environment names.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not SENSITIVE_ENV_PATTERN.search(key)
        and not key.upper().startswith(DSH_ENV_PREFIX)
    }


class SubprocessRuntime(Service, ABC):
    """Abstract subprocess service. Subclass, implement :meth:`spawn`, and
    load the subclass as a plugin — it registers as ``ctx.subprocess`` (one
    implementation per context; loading a second throws, which is cordis'
    standard duplicate-service behavior).

    Implementations must honor these semantics:

    - Executable paths belong to one execution world shared with the mounted
      filesystem provider.
    - :meth:`spawn` returns a live handle once the process started; ``done``
      settles at process close with exit facts. Spawn-level failures raise
      from :meth:`spawn` itself (see module docstring for the mapping).
    - Collect-mode readers are offset-based and non-consuming, so independent
      readers never consume one another's output; lossy reads report
      truncation and the spill file holding the complete stream when one
      exists. Piped streams are handed to the caller raw and never buffered
      here.
    - ``SubprocessHandle.terminate`` (and the spec's abort signal) escalates
      SIGTERM→grace→SIGKILL — the only termination verb — tree-scoped.
      ``SubprocessHandle.wait_for_exit`` observes whole-tree liveness, so a
      consumer-owned teardown ladder can hold each tier on real quiescence.
    - Disposal of the service terminates all still-running managed processes
      and awaits their exit.
    - :meth:`spawn_terminal` owns terminal allocation, text transport,
      foreground groups, signalling, and whole-session quiescence behind one
      awaited termination method.
    """

    def __init__(self, ctx: Context, config: Any = None) -> None:
        super().__init__(ctx, 'subprocess')

    @abstractmethod
    async def resolve_executable(
        self,
        command: str,
        env: Mapping[str, str] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        """Resolve one configured executable in this provider's execution
        world. Absolute paths are verified; bare names use the provider's
        scrubbed PATH plus explicit environment overrides. Relative paths
        containing separators are rejected: the resolution base is undefined,
        so providers fail loud instead of guessing.

        Returns a canonical executable path.
        """
        ...

    @abstractmethod
    async def spawn(self, spec: SubprocessSpawnSpec) -> SubprocessHandle:
        """Start one managed child process from a fully-specified spec; this
        seam applies no defaults. Raises for spawn-level failures; a started
        process reports every outcome — non-zero exits included — through
        ``handle.done``.
        """
        ...

    @abstractmethod
    async def spawn_terminal(
        self,
        spec: SubprocessTerminalSpawnSpec,
    ) -> SubprocessTerminalHandle:
        """Allocate a real terminal and start one owned process session. This
        is the only non-pipe process primitive: implementations own terminal
        byte I/O, foreground groups, signals, and complete session-tree
        cleanup.
        """
        ...
