"""Vocabulary for the subprocess Service Definition: fully-specified spawn
requests with per-stream stdio modes, bounded collected output with spill
recovery, raw piped streams, and tree-scoped termination. Command defaulting,
shell semantics, protocol framing, and presentation belong to consumers.

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess/src/types.ts``.

Python mappings:
- ``AbortSignal``/``AbortController`` stand in for the DOM primitives the TS
  spec carries; a listener registered on an already-aborted signal fires
  immediately, closing the check-then-register race the same way.
- ``stdin: { data }`` becomes :class:`StdinData`; the ``{ maxBytes, spill }``
  collect mode becomes :class:`SubprocessCollect` with :class:`SpillLimit`.
- Streams are ``asyncio`` objects: ``StreamWriter`` for stdin, ``StreamReader``
  for stdout/stderr/terminal output.
- The TS-only template-literal types ``DshEnvironmentKey``/``DshEnvironment``
  have no Python equivalent and are not ported.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

#: Namespace prefix reserved for harness-managed child environment facts.
DSH_ENV_PREFIX = 'DSH_'

#: Credential-shaped environment names are not forwarded to children.
SENSITIVE_ENV_PATTERN = re.compile(r'KEY|PASSWORD|SECRET|TOKEN', re.IGNORECASE)

#: Upper bound for ``grace_ms``, mirroring Node's maximum timer delay
#: (``MAX_TIMER_DELAY_MS`` from ``@deepseek-ai/dsh-timeout``).
MAX_TIMER_DELAY_MS = 2_147_483_647


class AbortError(Exception):
    """Raised by ``AbortSignal.throw_if_aborted`` and pre-aborted spawns."""


class AbortSignal:
    """Minimal port of the DOM ``AbortSignal`` carried by spawn specs.

    The caller owns deadlines and cause classification; the subprocess seam
    only reacts to the abort by starting the terminate escalation.
    """

    def __init__(self) -> None:
        self.aborted = False
        self.reason: Any = None
        self._listeners: list[Callable[[], None]] = []

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Register an abort listener; fires immediately if already aborted."""
        if self.aborted:
            listener()
            return
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        """Remove a previously registered listener; unknown listeners no-op."""
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def throw_if_aborted(self) -> None:
        """Raise :class:`AbortError` carrying the abort reason when aborted."""
        if self.aborted:
            reason = self.reason
            raise AbortError('aborted' if reason is None else str(reason))

    def _abort(self, reason: Any = None) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.reason = reason
        listeners, self._listeners = self._listeners, []
        for listener in listeners:
            listener()


class AbortController:
    """Owner side of an :class:`AbortSignal`, mirroring the DOM pair."""

    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: Any = None) -> None:
        """Fire the signal once with an optional reason; repeats are no-ops."""
        self.signal._abort(reason)


@dataclass(frozen=True)
class CollectedOutput:
    """One captured stream: the (possibly truncated) text plus recovery info."""

    #: Collected text — the TAIL of the stream when truncated.
    text: str
    #: True when bytes were dropped from ``text``.
    truncated: bool
    #: Path to a file holding the COMPLETE stream, when truncated and intact.
    spill_path: str | None = None


@dataclass(frozen=True)
class StdinData:
    """Batch stdin disposition: write ``data`` to the child and close."""

    data: str


#: stdin disposition. ``'ignore'`` leaves fd 0 on ``/dev/null``; ``'pipe'``
#: exposes ``SubprocessHandle.stdin`` for the caller's ongoing protocol writes;
#: :class:`StdinData` writes the bytes and closes (the batch shape).
SubprocessStdinMode = Literal['ignore', 'pipe'] | StdinData


@dataclass(frozen=True)
class SpillLimit:
    """Whole-stream byte cap for a spill file; a larger stream discards its
    now-incomplete spill."""

    max_bytes: int


@dataclass(frozen=True)
class SubprocessCollect:
    """Bounded in-memory collection for one output stream, with an optional
    full-stream spill file. Omitting ``spill`` keeps only the in-memory tail —
    the diagnostic-tail shape (a language server's stderr); including it makes
    the complete stream recoverable up to its cap (the bash tool shape).
    """

    #: In-memory cap in bytes; overflow keeps the TAIL.
    max_bytes: int
    #: Full-stream spill file limit; absent disables spilling entirely.
    spill: SpillLimit | None = None


#: stdout/stderr disposition. ``'pipe'`` exposes the raw ``StreamReader`` for
#: the caller's protocol decoding; ``'inherit'`` passes the parent's descriptor
#: through; a :class:`SubprocessCollect` buffers boundedly with offset reads.
SubprocessOutputMode = Literal['pipe', 'inherit'] | SubprocessCollect


@dataclass(frozen=True)
class SubprocessStdio:
    """Per-stream stdio dispositions, all explicit — this seam applies no
    defaults."""

    stdin: SubprocessStdinMode
    stdout: SubprocessOutputMode
    stderr: SubprocessOutputMode


@dataclass(frozen=True)
class SubprocessSpawnSpec:
    """A fully-specified spawn request. This seam applies no defaults: every
    disposition, limit, and directory is explicit, so the caller's own config —
    not a hidden subprocess-service default — decides them.
    """

    #: Executable and arguments; ``argv[0]`` is the program. Never
    #: shell-interpreted here.
    argv: Sequence[str]
    #: Working directory for the child.
    cwd: str
    #: Per-stream stdio dispositions.
    stdio: SubprocessStdio
    #: Positive finite grace period in milliseconds, no greater than
    #: ``MAX_TIMER_DELAY_MS``, for the ``terminate`` escalation and for
    #: draining still-open collected pipes after the process exits (an
    #: inherited descriptor held by a surviving descendant cannot hold the
    #: outcome open indefinitely).
    grace_ms: float
    #: Abort signal — starts the terminate escalation on the process tree when
    #: it fires. The caller owns deadlines and cause classification; this seam
    #: only reacts to the abort.
    signal: AbortSignal | None = None
    #: Explicit environment entries merged onto the implementation's scrubbed
    #: parent base, with no namespace validation. A string is a deliberate
    #: caller opt-in; ``None`` is a tombstone that removes an ordinary ambient
    #: entry from the child.
    env: Mapping[str, str | None] | None = None


@dataclass(frozen=True)
class SubprocessOutcome:
    """Exit facts of one closed process. Deliberately carries NO timeout or
    cancellation classification (the caller reads the signal it owns to
    classify causes) and NO output: collected streams stay readable through
    ``SubprocessHandle.collected`` after settlement.
    """

    #: Exit code; ``None`` when the process died from a signal.
    exit_code: int | None
    #: Terminating signal name (e.g. ``'SIGTERM'``); ``None`` on normal exit.
    signal: str | None


@dataclass(frozen=True)
class SubprocessOutputRead:
    """One incremental ``SubprocessOutputReader.read_from`` read."""

    #: Stream text from the requested offset (the whole retained tail when
    #: lossy).
    text: str
    #: Whole-stream offset to resume from on the next read.
    next_offset: int
    #: True when the requested offset slid out of the in-memory tail window.
    lossy: bool
    #: Path to the full-stream spill file, when one was created and remains
    #: intact.
    spill_path: str | None = None


class SubprocessOutputReader(Protocol):
    """Cursor-free incremental access to one collected output stream. Offsets
    are whole-stream byte coordinates owned by the caller, so independent
    readers cannot consume one another's output; ``read_from(0)`` after
    settlement is the batch result (``lossy`` then means the in-memory tail
    lost its head).
    """

    def read_from(self, from_byte: int) -> SubprocessOutputRead:
        """Read everything captured since ``from_byte`` (a prior read's
        ``next_offset``; 0 for the first read). When that offset has slid out
        of the in-memory tail window the read is ``lossy`` — it returns the
        whole retained tail and the gap is only recoverable from the spill
        file.
        """
        ...


@dataclass(frozen=True)
class SubprocessCollectedOutputs:
    """Offset-based readers for the streams spawned in collect mode; each is
    present iff the corresponding stream is a :class:`SubprocessCollect`.
    """

    stdout: SubprocessOutputReader | None = None
    stderr: SubprocessOutputReader | None = None


class SubprocessHandle(Protocol):
    """A live child process rooted in its own process tree. Collected output
    remains readable after exit; piped streams belong to the caller.

    Termination is tree-scoped: POSIX signals the detached process group
    (falling back to the direct child when the group is gone), so helper
    processes cannot outlive the handle unnoticed.
    """

    #: Process id (tree root).
    pid: int
    #: The child's stdin, present iff spawned with ``stdin: 'pipe'``.
    stdin: asyncio.StreamWriter | None
    #: The child's raw stdout, present iff spawned with ``stdout: 'pipe'``.
    stdout: asyncio.StreamReader | None
    #: The child's raw stderr, present iff spawned with ``stderr: 'pipe'``.
    stderr: asyncio.StreamReader | None
    #: Offset-based readers for collect-mode streams (readable after exit).
    collected: SubprocessCollectedOutputs
    #: Settles at process close with exit facts. Never rejects in this port:
    #: spawn-level failures raise from ``SubprocessRuntime.spawn`` itself
    #: (asyncio creates processes asynchronously), so by the time a handle
    #: exists the process is live.
    done: asyncio.Future[SubprocessOutcome]

    def terminate(self) -> None:
        """Begin the SIGTERM → ``grace_ms`` → SIGKILL escalation on the
        process tree — the seam's only termination verb. Idempotent, a no-op
        once the tree is gone (the pid may be reused), and also triggered by
        the spec's abort signal.
        """
        ...

    async def wait_for_exit(self, signal: AbortSignal | None = None) -> bool:
        """Wait until the process tree has exited — the tree, not just the
        direct child, so a still-running helper is observable before teardown
        returns. Returns ``True`` when the tree exited, ``False`` when
        ``signal`` aborted first.
        """
        ...


#: Signals supported by the terminal-process primitive.
SubprocessTerminalSignal = Literal['SIGINT', 'SIGTERM', 'SIGKILL', 'SIGTSTP', 'SIGHUP']


@dataclass(frozen=True)
class SubprocessTerminalSpawnSpec:
    """A fully specified terminal-process spawn."""

    #: Executable and arguments; ``argv[0]`` is the program.
    argv: Sequence[str]
    #: Working directory in this subprocess provider's execution world.
    cwd: str
    #: Initial terminal row count. Ignored by the pipe-based local port.
    rows: int
    #: Initial terminal column count. Ignored by the pipe-based local port.
    cols: int
    #: TERM-to-KILL cleanup grace for the complete terminal session.
    grace_ms: float
    #: Explicit environment layered after the provider's ambient scrub.
    env: Mapping[str, str] | None = None
    #: Cancellation of terminal allocation; a published handle owns its later
    #: lifetime.
    signal: AbortSignal | None = None


@dataclass(frozen=True)
class SubprocessTerminalForeground:
    """Current foreground process-group facts for one terminal."""

    #: Foreground process-group id published by the terminal driver.
    process_group_id: int
    #: Whether the provider can currently prove that group is waiting on
    #: terminal input.
    input_waiting: bool


class SubprocessTerminalHandle(Protocol):
    """One live terminal process and its owned OS session. Terminal
    allocation, foreground-group inspection/signalling, and session-tree
    cleanup are one deep subprocess primitive because none can be reconstructed
    from ordinary piped stdio without substrate-specific process control.
    """

    #: Top-level terminal process id.
    pid: int
    #: Terminal output bytes in delivery order; ends after queued output when
    #: the terminal exits.
    output: asyncio.StreamReader
    #: Settles when the top-level process exits.
    done: asyncio.Future[SubprocessOutcome]

    async def write(self, data: str) -> None:
        """Write text to the terminal input, without implicit newline
        conversion."""
        ...

    async def inspect_foreground(self) -> SubprocessTerminalForeground | None:
        """Inspect the current foreground process group, or ``None`` when no
        foreground group can be resolved."""
        ...

    async def signal_foreground(self, signal: SubprocessTerminalSignal) -> int:
        """Deliver a signal to the current foreground process group; returns
        the exact group id that received it.
        """
        ...

    async def terminate(self) -> None:
        """Idempotently terminate every terminal-session member the provider
        can still observe and await quiescence. After settlement, no write,
        inspection, or signal call remains in flight.
        """
        ...
