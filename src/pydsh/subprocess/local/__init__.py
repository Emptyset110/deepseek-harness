"""Local Service Provider for the subprocess capability seam. Each spawn is a
detached process tree with the spec's per-stream stdio dispositions. Normal
disposal terminates and joins live trees; interpreter exit force-stops any
trees the service still owns. It has no config: every disposition and limit
arrives on the spec, so the deployment-varying choices stay with the caller's
config (the bash executor's, the LSP host's, …).

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess-local/src/index.ts``.

Python mappings:
- ``process.on('exit')`` host-exit finalization becomes an ``atexit`` hook
  registered by the teardown effect.
- The TS ``internals`` test hooks (platform/taskkill overrides) are dropped
  together with the Windows routing; ``spill_dir`` remains as the
  deterministic-spill test hook.
- The PTY terminal is replaced by a pipe-based one; see
  :mod:`pydsh.subprocess.local.terminal` for the simplifications.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import subprocess
from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from pycordis import Context

from .. import SubprocessRuntime
from ..types import (
    AbortSignal,
    SubprocessHandle,
    SubprocessSpawnSpec,
    SubprocessTerminalHandle,
    SubprocessTerminalSpawnSpec,
)
from .spawn import LocalSubprocessHandle, child_env, spawn_subprocess
from .terminal import LocalTerminalHandle


def _is_executable_file(candidate: str) -> bool:
    """One PATH-candidate probe; a stat failure is simply a miss — the final
    miss receives one stable error from the caller."""
    try:
        return os.path.isfile(candidate) and os.access(candidate, os.X_OK)
    except OSError:
        return False


class LocalSubprocessRuntime(SubprocessRuntime):
    """Local subprocess service: detached process trees, Node-shaped stdio
    dispositions (raw pipes, inherit, bounded tail-keep collection with spill
    files), credential-scrubbed environment, and tree-scoped signalling with
    SIGTERM→grace→SIGKILL escalation, plus synchronous final termination
    during interpreter exit.
    """

    #: Test hook: directory for spill files (defaults to the OS temp dir).
    spill_dir: str | None

    def __init__(self, ctx: Context, config: Any = None) -> None:
        #: Live handles retained for normal disposal and host-exit
        #: finalization.
        self._live: set[LocalSubprocessHandle] = set()
        #: Live terminals retained through normal quiescence or host-exit
        #: finalization.
        self._terminals: set[LocalTerminalHandle] = set()
        self.spill_dir = None
        super().__init__(ctx, config)
        ctx.effect(self._register_teardown, 'local subprocess teardown')

    def _register_teardown(self) -> Callable[[], Coroutine[Any, Any, None]]:
        atexit.register(self._terminate_for_host_exit)

        async def dispose() -> None:
            try:
                await self._dispose_managed_processes()
            finally:
                atexit.unregister(self._terminate_for_host_exit)

        return dispose

    def _terminate_for_host_exit(self) -> None:
        for handle in self._live:
            try:
                handle.terminate_for_host_exit()
            except Exception:
                # Host exit cannot await or report one target; continue with
                # the rest.
                pass
        for terminal in self._terminals:
            try:
                terminal.terminate_for_host_exit()
            except Exception:
                # One terminal must not prevent final termination of another.
                pass

    async def _dispose_managed_processes(self) -> None:
        # Terminate (escalating), then await WHOLE-TREE exit — not just the
        # direct child's settlement — so even a TERM-trapping descendant
        # cannot outlive the fiber.
        pending: list[Coroutine[Any, Any, None]] = []
        for handle in self._live:
            handle.terminate()
            pending.append(self._join_handle(handle))
        for terminal in self._terminals:
            pending.append(terminal.terminate())
        results = await asyncio.gather(*pending, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]
        if failures:
            self._terminate_for_host_exit()
        self._live.clear()
        self._terminals.clear()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup('local subprocess teardown failed', failures)

    @staticmethod
    async def _join_handle(handle: LocalSubprocessHandle) -> None:
        await handle.done
        await handle.wait_for_exit()

    async def resolve_executable(
        self,
        command: str,
        env: Mapping[str, str] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        if not command:
            raise ValueError('subprocess-local: executable must be non-empty')
        if signal is not None:
            signal.throw_if_aborted()
        environment = child_env(env)
        if os.path.isabs(command):
            candidates = [command]
        elif os.sep in command or (os.altsep is not None and os.altsep in command):
            raise ValueError(
                f'subprocess-local: command {command!r} is a relative path;'
                ' use an absolute path or a bare PATH name'
            )
        else:
            candidates = [
                os.path.join(directory, command)
                for directory in environment.get('PATH', '').split(os.pathsep)
                if directory
            ]
        for candidate in candidates:
            if signal is not None:
                signal.throw_if_aborted()
            if _is_executable_file(candidate):
                return candidate
        if os.path.isabs(command):
            raise FileNotFoundError(
                f'subprocess-local: command {command!r} is not an executable file'
            )
        raise FileNotFoundError(
            f'subprocess-local: command {command!r} was not found on PATH'
        )

    async def spawn(self, spec: SubprocessSpawnSpec) -> SubprocessHandle:
        handle = await spawn_subprocess(spec, spill_dir=self.spill_dir)
        self._live.add(handle)
        # Release ownership only once the whole TREE is gone, not at
        # direct-child settlement — a TERM-trapping helper that outlives the
        # leader must stay owned so teardown can still escalate it. For the
        # common no-survivor case wait_for_exit resolves immediately after
        # settlement.
        handle.done.add_done_callback(
            lambda _done: asyncio.ensure_future(self._release(handle))
        )
        return handle

    async def _release(self, handle: LocalSubprocessHandle) -> None:
        await handle.wait_for_exit()
        self._live.discard(handle)

    async def spawn_terminal(
        self,
        spec: SubprocessTerminalSpawnSpec,
    ) -> SubprocessTerminalHandle:
        if not spec.argv or not spec.argv[0]:
            raise ValueError('subprocess-local: terminal argv must contain a program')
        if spec.signal is not None:
            spec.signal.throw_if_aborted()
        program, *args = spec.argv
        proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            cwd=spec.cwd,
            env=child_env(spec.env),
            stdin=subprocess.PIPE,
            # Merged stdout+stderr approximates a terminal's combined output
            # in this pipe-based port.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle = LocalTerminalHandle(proc, spec.grace_ms)
        self._terminals.add(handle)
        handle.done.add_done_callback(
            lambda _done: asyncio.ensure_future(self._release_terminal(handle))
        )
        return handle

    async def _release_terminal(self, handle: LocalTerminalHandle) -> None:
        try:
            await handle.terminate()
        finally:
            self._terminals.discard(handle)


__all__ = ['LocalSubprocessRuntime']
