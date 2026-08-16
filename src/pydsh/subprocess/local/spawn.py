"""Process plumbing for the local subprocess service: detached process-tree
spawn with per-stream stdio dispositions, tail-keep collection with spill
files, POSIX process-group signalling, and the SIGTERM→SIGKILL escalation.
This layer reacts to an abort signal; callers own deadlines, teardown ladders,
and cause classification.

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess-local/src/spawn.ts``.

Simplifications:
- Only POSIX is supported: tree termination signals the detached process
  group (``start_new_session=True``); the Windows ``taskkill /T`` routing and
  its injectable platform knobs are not ported.
- Node's ``close`` event settles after ALL pipes close; here ``done`` settles
  at process exit plus a grace-bounded drain of the harness-COLLECTED pipes.
  Caller-owned ``'pipe'`` streams close independently and never gate ``done``.
"""

from __future__ import annotations

import asyncio
import math
import os
import secrets
import subprocess
import sys
import tempfile
from collections.abc import Coroutine, Mapping
from signal import SIGKILL, SIGTERM, Signals
from typing import Any

from .. import scrubbed_parent_env
from ..types import (
    MAX_TIMER_DELAY_MS,
    AbortError,
    AbortSignal,
    CollectedOutput,
    StdinData,
    SubprocessCollect,
    SubprocessCollectedOutputs,
    SubprocessHandle,
    SubprocessOutcome,
    SubprocessOutputMode,
    SubprocessOutputRead,
    SubprocessSpawnSpec,
)
from .process_inspector import linux_process_group_has_live_members

#: Liveness-poll cadence for tree-exit waits, in seconds.
_TREE_POLL_SECONDS = 0.015

_spill_counter = 0
_default_spill_dir: str | None = None


def _private_spill_dir() -> str:
    """The default spill location: a private (0700) per-process directory
    under the OS tmpdir, created lazily. Predictable world-readable paths
    would let other local users read command output or pre-create symlinks."""
    global _default_spill_dir
    if _default_spill_dir is None:
        _default_spill_dir = tempfile.mkdtemp(prefix='dsh-subprocess-')
    return _default_spill_dir


def child_env(extra: Mapping[str, str | None] | None = None) -> dict[str, str]:
    """Build a child environment: explicit caller entries override the
    scrubbed parent base. A string deliberately restores or overrides an
    entry; an explicit ``None`` tombstone removes an ordinary ambient entry.
    (The TS Windows case-insensitive key merge is not ported; this backend is
    POSIX-only.)
    """
    env = scrubbed_parent_env()
    for key, value in (extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


class OutputCollector:
    """Collects one stream with a bounded in-memory tail. With a spill cap,
    on first overflow a spill file is created and every chunk (including those
    already collected) is appended there while the full stream remains within
    the cap; without one, only the in-memory tail is ever retained (the
    diagnostic-tail shape — a language server's stderr).

    Tail-keep rationale (pi/OpenCode): errors and final results cluster at
    the end of command output; the spill file covers the head.

    Implements the :class:`SubprocessOutputReader` protocol.
    """

    def __init__(
        self,
        max_bytes: int,
        max_spill_bytes: int | None,
        label: str,
        spill_dir: str,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_spill_bytes = max_spill_bytes
        self._label = label
        self._spill_dir = spill_dir
        self._chunks: list[bytes] = []
        self._bytes = 0
        self._dropped = False
        self._spill_fd: int | None = None
        self._spill_file: str | None = None
        self._spill_disabled = max_spill_bytes is None
        #: Total bytes ever pushed (not just retained).
        self._total = 0

    def push(self, chunk: bytes) -> None:
        """Ingest one stream chunk, counting it toward the whole-stream total.
        On first overflow of the in-memory cap a spill file is opened (when
        spilling is enabled) and every chunk, already-collected ones included,
        is appended there from then on; the in-memory tail then drops bytes
        from its head until it fits the cap again — byte-exact at the cap, so
        a diagnostic tail holds the LAST ``max_bytes`` regardless of chunking.
        """
        self._total += len(chunk)
        overflows = self._bytes + len(chunk) > self._max_bytes
        if not self._spill_disabled and (overflows or self._spill_fd is not None):
            self._spill_all(chunk)
        self._chunks.append(chunk)
        self._bytes += len(chunk)
        while self._bytes > self._max_bytes:
            head = self._chunks[0]
            excess = self._bytes - self._max_bytes
            if len(head) <= excess:
                self._chunks.pop(0)
                self._bytes -= len(head)
            else:
                self._chunks[0] = head[excess:]
                self._bytes -= excess
            self._dropped = True

    def _spill_all(self, chunk: bytes) -> None:
        """Open the spill file lazily and append ``chunk`` (and any prior
        chunks once)."""
        global _spill_counter
        if self._max_spill_bytes is not None and self._total > self._max_spill_bytes:
            self._discard_spill()
            return
        if self._spill_fd is None:
            # Random suffix + O_EXCL + owner-only mode: defeats spill-path
            # prediction and symlink planting in shared tmp dirs.
            _spill_counter += 1
            self._spill_file = os.path.join(
                self._spill_dir,
                f'dsh-subprocess-{os.getpid()}-{_spill_counter}'
                f'-{secrets.token_hex(6)}-{self._label}.log',
            )
            self._spill_fd = os.open(
                self._spill_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            for prior in self._chunks:
                _write_all(self._spill_fd, prior)
        _write_all(self._spill_fd, chunk)

    def _discard_spill(self) -> None:
        """Stop spilling and remove the file once it can no longer hold the
        complete stream."""
        fd, file = self._spill_fd, self._spill_file
        self._spill_fd = None
        self._spill_file = None
        self._spill_disabled = True
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                # Retain the descriptor so seal can retry the failed close.
                self._spill_fd = fd
        if file is not None:
            try:
                os.unlink(file)
            except OSError:
                # A failed unlink leaves at most max_spill_bytes behind, never
                # an unbounded file.
                pass

    def read_from(self, from_byte: int) -> SubprocessOutputRead:
        """Incremental read in whole-stream byte coordinates: returns
        everything pushed since ``from_byte``. When ``from_byte`` has already
        slid out of the in-memory tail window, the read is ``lossy`` — it
        returns the whole retained tail and the gap is only recoverable from
        the spill file.
        """
        window_start = self._total - self._bytes
        buffer = b''.join(self._chunks)
        lossy = from_byte < window_start
        slice_ = buffer if lossy else buffer[from_byte - window_start:]
        return SubprocessOutputRead(
            text=slice_.decode('utf-8', errors='replace'),
            next_offset=self._total,
            lossy=lossy,
            spill_path=self._spill_file,
        )

    def seal(self) -> None:
        """Close the spill file once the stream has ended. A failed close
        (delayed writeback fault) stops advertising the spill path — the file
        may be missing its tail — while every in-memory read keeps working.
        Idempotent; the spawn path seals both collectors at settlement so
        reads after exit never point at a still-open file.
        """
        if self._spill_fd is None:
            return
        try:
            os.close(self._spill_fd)
        except OSError:
            # A delayed writeback failure makes the spill unreliable; keep the
            # in-memory result but stop advertising that file.
            self._spill_file = None
        self._spill_fd = None

    def finalize(self) -> CollectedOutput:
        """Seal the spill file and return the final output: tail text,
        truncation flag, and the spill path when intact."""
        self.seal()
        return CollectedOutput(
            text=b''.join(self._chunks).decode('utf-8', errors='replace'),
            truncated=self._dropped,
            spill_path=self._spill_file,
        )


def kill_group(pid: int, sig: Signals) -> None:
    """Send ``sig`` to a detached POSIX process group. Never raises: delivery
    races process exit and may run in a timer callback, so failures are
    contained and a non-positive pid is a no-op.
    """
    if pid <= 0:
        return
    try:
        os.killpg(pid, sig)
    except OSError:
        pass


def _outcome_of(returncode: int) -> SubprocessOutcome:
    """Map asyncio's returncode vocabulary onto Node's close-event facts:
    a negative returncode means the process died from signal ``-returncode``."""
    if returncode < 0:
        try:
            name = Signals(-returncode).name
        except ValueError:
            name = f'SIG{-returncode}'
        return SubprocessOutcome(exit_code=None, signal=name)
    return SubprocessOutcome(exit_code=returncode, signal=None)


class LocalSubprocessHandle(SubprocessHandle):
    """The local backend's :class:`SubprocessHandle`: an asyncio child
    process rooted in its own detached process group, with collector reader
    tasks, the terminate escalation, and a whole-tree exit observer.
    """

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        spec: SubprocessSpawnSpec,
        spill_dir: str,
    ) -> None:
        self._proc = proc
        self._spec = spec
        self.pid = proc.pid

        stdin_mode = spec.stdio.stdin
        out_mode = spec.stdio.stdout
        err_mode = spec.stdio.stderr
        self.stdin = proc.stdin if stdin_mode == 'pipe' else None
        self.stdout = proc.stdout if out_mode == 'pipe' else None
        self.stderr = proc.stderr if err_mode == 'pipe' else None

        self._stdout_collector = self._make_collector(out_mode, 'stdout', spill_dir)
        self._stderr_collector = self._make_collector(err_mode, 'stderr', spill_dir)
        self.collected = SubprocessCollectedOutputs(
            stdout=self._stdout_collector, stderr=self._stderr_collector
        )
        self.done: asyncio.Future[SubprocessOutcome] = (
            asyncio.get_running_loop().create_future()
        )

        self._settled = False
        self._tree_exit_observed = False
        self._observer: asyncio.Task[None] | None = None
        self._grace_handle: asyncio.TimerHandle | None = None
        self._background: set[asyncio.Task[None]] = set()
        self._readers: list[asyncio.Task[None]] = []
        if self._stdout_collector is not None and proc.stdout is not None:
            self._readers.append(
                asyncio.ensure_future(
                    self._collect(proc.stdout, self._stdout_collector)
                )
            )
        if self._stderr_collector is not None and proc.stderr is not None:
            self._readers.append(
                asyncio.ensure_future(
                    self._collect(proc.stderr, self._stderr_collector)
                )
            )

        # Batch stdin is written and closed up front; process exit and
        # captured output remain authoritative, so write errors (EPIPE) are
        # best-effort.
        if isinstance(stdin_mode, StdinData) and proc.stdin is not None:
            try:
                proc.stdin.write(stdin_mode.data.encode())
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self._track(self._wait_stdin_closed())

        # The caller owns timeout classification; this layer only reacts to
        # abort.
        if spec.signal is not None:
            spec.signal.add_listener(self.terminate)

        self._track(self._settle())

    def _track(self, coro: Coroutine[Any, Any, None]) -> None:
        """Schedule a fire-and-forget task, keeping a reference until it
        settles (the loop only weakly references tasks)."""
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @staticmethod
    def _make_collector(
        mode: SubprocessOutputMode, label: str, spill_dir: str
    ) -> OutputCollector | None:
        if not isinstance(mode, SubprocessCollect):
            return None
        spill = mode.spill.max_bytes if mode.spill is not None else None
        return OutputCollector(mode.max_bytes, spill, label, spill_dir)

    @staticmethod
    async def _collect(
        reader: asyncio.StreamReader, collector: OutputCollector
    ) -> None:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            collector.push(chunk)

    async def _wait_stdin_closed(self) -> None:
        stdin = self._proc.stdin
        if stdin is None:
            return
        try:
            await stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            # The child exited without reading; outcome rides on exit/output.
            pass

    async def _wait_child_exit(self) -> None:
        """Direct-child exit, independent of pipe closure. asyncio's
        ``Process.wait()`` only settles once ALL pipes disconnect (a surviving
        descendant holding an inherited descriptor would block it), so exit is
        detected by polling ``returncode``, which the child watcher sets at
        process death.
        """
        while self._proc.returncode is None:  # noqa: ASYNC110
            await asyncio.sleep(_TREE_POLL_SECONDS)

    async def _settle(self) -> None:
        """Resolve ``done`` at process close: runtime exit plus a
        grace-bounded drain of the harness-collected pipes (a surviving
        descendant that inherited a pipe must not hold the outcome open).
        """
        await self._wait_child_exit()
        self._settled = True
        returncode = self._proc.returncode
        assert returncode is not None
        if self._readers:
            _done, pending = await asyncio.wait(
                self._readers, timeout=self._spec.grace_ms / 1000
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._stdout_collector is not None:
            self._stdout_collector.seal()
        if self._stderr_collector is not None:
            self._stderr_collector.seal()
        if self._spec.signal is not None:
            self._spec.signal.remove_listener(self.terminate)
        self.done.set_result(_outcome_of(returncode))

    def _tree_alive(self) -> bool:
        """Whether the detached tree's root (or POSIX group) is still
        alive."""
        if self._tree_exit_observed or self.pid <= 0:
            return False
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # A group containing only unreaped zombies still answers killpg(0),
        # but it can execute no work and cannot be signalled into quiescence.
        # Only inspect after direct-child settlement so live-process polls
        # remain a syscall rather than repeated process-table scans.
        if (
            self._settled
            and sys.platform == 'linux'
            and linux_process_group_has_live_members(self.pid) is False
        ):
            return False
        return True

    def _observe_tree_exit(self) -> asyncio.Task[None]:
        """Start or reuse the handle's single whole-tree exit observer. The
        first confirmed absence is a permanent no-more-signals boundary: it
        cancels a pending escalation before this process-group id can be
        reused.
        """
        if self._observer is None:
            self._observer = asyncio.ensure_future(self._watch_tree())
        return self._observer

    async def _watch_tree(self) -> None:
        # Process-group liveness has no event source; polling killpg(0) is
        # the mechanism (same 15ms cadence as the TS sleepTick).
        while self._tree_alive():  # noqa: ASYNC110
            await asyncio.sleep(_TREE_POLL_SECONDS)
        self._tree_exit_observed = True
        if self._grace_handle is not None:
            self._grace_handle.cancel()
            self._grace_handle = None

    def _kill(self, sig: Signals) -> None:
        """The escalation's tier primitive (not on the handle — terminate()
        is the only consumer-facing termination verb). Guards on TREE
        liveness, not outcome settlement: a TERM-trapping helper can outlive
        the settled direct child and must stay signalable, while a fully-dead
        tree (possible pid reuse) must not be re-signalled by a later tier.
        """
        if not self._tree_alive():
            return
        try:
            os.killpg(self.pid, sig)
        except OSError:
            # Fall back to the direct child when the group signal fails.
            try:
                self._proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                # The direct child already exited; teardown stays idempotent.
                pass

    def terminate(self) -> None:
        if self._tree_exit_observed or self._grace_handle is not None:
            return
        # Observe from the first termination tier onward, even when inherited
        # pipes delay ``done`` and no consumer has begun its own teardown
        # wait.
        self._observe_tree_exit()
        if self._tree_exit_observed:
            return
        self._kill(SIGTERM)
        # The escalation must survive direct-child settlement — the leader
        # dying does not mean the tree died — so settle does not clear this
        # timer, and ``_kill`` re-probes tree liveness before force-killing.
        # Self-bounds at grace_ms.
        self._grace_handle = asyncio.get_running_loop().call_later(
            self._spec.grace_ms / 1000, self._kill, SIGKILL
        )

    def terminate_for_host_exit(self) -> None:
        """Force-terminate the current tree synchronously without starting
        timers or waits. Used by the owning service during host exit and as
        the last fallback after failed normal disposal; intentionally absent
        from the public seam.
        """
        self._kill(SIGKILL)

    async def wait_for_exit(self, signal: AbortSignal | None = None) -> bool:
        observed = self._observe_tree_exit()
        if self._tree_exit_observed:
            return True
        if signal is not None and signal.aborted:
            return False
        if signal is None:
            await asyncio.shield(observed)
            return True
        aborted: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        def on_abort() -> None:
            if not aborted.done():
                aborted.set_result(False)

        signal.add_listener(on_abort)
        try:
            waiters: set[asyncio.Future[Any]] = {observed, aborted}
            done, _pending = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED
            )
            return observed in done
        finally:
            signal.remove_listener(on_abort)


async def spawn_subprocess(
    spec: SubprocessSpawnSpec, *, spill_dir: str | None = None
) -> LocalSubprocessHandle:
    """Spawn one isolated detached process tree with the spec's per-stream
    stdio dispositions. Runtime exits settle ``done`` as
    :class:`SubprocessOutcome`; spawn-level failures (invalid argv/grace,
    pre-aborted signal, missing executable or cwd) raise from this coroutine
    — the Python mapping of TS rejecting ``done`` for spawn-level failures.
    """
    grace_ms = spec.grace_ms
    if not math.isfinite(grace_ms) or grace_ms <= 0 or grace_ms > MAX_TIMER_DELAY_MS:
        raise ValueError(
            'subprocess grace_ms must be a positive finite number'
            f' no greater than {MAX_TIMER_DELAY_MS}'
        )
    if spec.signal is not None and spec.signal.aborted:
        reason = spec.signal.reason
        reason_text = 'aborted' if reason is None else str(reason)
        raise AbortError(f'aborted before spawn: {reason_text}')
    if not spec.argv or not spec.argv[0]:
        raise ValueError('invalid argv: expected a non-empty program name at argv[0]')
    program, *args = spec.argv

    stdin_mode = spec.stdio.stdin
    out_mode = spec.stdio.stdout
    err_mode = spec.stdio.stderr
    proc = await asyncio.create_subprocess_exec(
        program,
        *args,
        cwd=spec.cwd,
        env=child_env(spec.env),
        stdin=subprocess.DEVNULL if stdin_mode == 'ignore' else subprocess.PIPE,
        stdout=None if out_mode == 'inherit' else subprocess.PIPE,
        stderr=None if err_mode == 'inherit' else subprocess.PIPE,
        # A new session gives teardown a tree root on POSIX (its own process
        # group).
        start_new_session=True,
    )
    return LocalSubprocessHandle(proc, spec, spill_dir or _private_spill_dir())
