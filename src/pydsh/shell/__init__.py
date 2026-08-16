"""Service Definition for the ``ctx.shell`` capability seam, covering
foreground commands and background process handles. Job ids, ownership,
polling, and notices belong to the (unported) jobs capability, keeping
executors independent of sessions.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/shell/src/index.ts``.

Python mappings:

- :meth:`ShellExecutor.start` is a coroutine: the Python subprocess seam
  creates processes asynchronously and raises spawn-level failures from
  ``spawn`` itself, so a background spawn failure raises from ``start``
  rather than settling the handle as ``killed`` with the error on stderr
  (that TS path cannot occur once a handle exists).
- The settings-backed config swap (``SHELL_SETTINGS_NAMESPACE`` /
  ``installSettingsSection``) is dropped with the unported settings
  capability; an executor's construction config is its single source.
- The TS declaration merging that types ``ctx.shell`` has no Python
  equivalent; read the service through ``ctx.shell`` after injecting
  ``shell``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pycordis import Context, Service

from .render import ParsedExitStatus, parse_exit_status
from .types import (
    DSH_ENV_PREFIX,
    CollectedOutput,
    ShellExecRequest,
    ShellExecSpec,
    ShellProcess,
    ShellProcessRead,
    ShellProcessStatus,
    ShellRunResult,
    ShellSandboxInfo,
)

__all__ = [
    "DSH_ENV_PREFIX",
    "CollectedOutput",
    "ParsedExitStatus",
    "ShellExecRequest",
    "ShellExecSpec",
    "ShellExecutor",
    "ShellProcess",
    "ShellProcessRead",
    "ShellProcessStatus",
    "ShellRunResult",
    "ShellSandboxInfo",
    "parse_exit_status",
]


class ShellExecutor(Service, ABC):
    """Abstract bash execution service. Subclass, implement the abstract
    methods, and load the subclass as a plugin — it registers as
    ``ctx.shell`` (one implementation per context; loading a second raises,
    which is cordis' standard duplicate-service behavior).

    Implementations must honor these semantics:

    - :meth:`run` raises only for infrastructure failures. Nonzero exits,
      timeout kills, and abort kills resolve with a :class:`ShellRunResult`.
    - :meth:`start` returns a live handle; no timeout applies to background
      processes. ``done`` settles at process close and never raises.
    - ``ShellProcess.read_output`` is incremental: consecutive reads never
      repeat output. Lossy reads report truncation and available spill
      files.
    - A still-running background process is stopped and awaited when its
      owning composition tears down. With the subprocess seam that boundary
      is ``ctx.subprocess`` disposal, so a background process survives an
      executor-only reload.
    """

    def __init__(self, ctx: Context, config: Any = None) -> None:
        super().__init__(ctx, "shell")

    @property
    def sandbox_mode(self) -> str | None:
        """The sandbox mode this executor applies by default, or ``None``
        when it does not sandbox commands."""
        return None

    @abstractmethod
    def resolve(self, request: ShellExecRequest) -> ShellExecSpec:
        """Apply implementation-owned defaults and caps to a request before
        execution.

        :param request: the caller's request; omitted fields get this
            implementation's defaults, capped fields are clamped.
        :return: the fully-specified spec to hand to :meth:`run`/
            :meth:`start`.
        """
        ...

    @abstractmethod
    async def run(self, spec: ShellExecSpec) -> ShellRunResult:
        """Run a command in the foreground; resolves when it finishes.

        :param spec: a resolved spec from :meth:`resolve`, never a raw
            request.
        :return: the outcome; nonzero exits, timeout kills, and abort kills
            resolve with a descriptive result rather than raise.
        """
        ...

    @abstractmethod
    async def start(self, spec: ShellExecSpec) -> ShellProcess:
        """Start a background process and return its handle once live.

        :param spec: a resolved spec from :meth:`resolve`, never a raw
            request.
        :return: the live process handle (reads, kill, quiescence future).
        """
        ...
