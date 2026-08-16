"""Shared timeout arithmetic, signal fusion, and classification — the slice
of ``dsh-timeout`` the shell executor needs: optional-hint clamping, a fused
upstream/timeout deadline, and timeout-cause recovery. The library only
notifies through abort signals; the executor owns the mechanism that stops
its work and translates timeout reasons into public outcomes.

Ported from deepseek-harness packages/util (MIT),
``packages/util/timeout/src/index.ts`` (``clampTimeout``, ``deadline``,
``timeoutOf``, ``TimeoutReason``; the idle watchdog is not needed here).

Python mappings:

- ``setTimeout``/``clearTimeout`` become ``loop.call_later`` handles;
  ``deadline`` therefore requires a running event loop.
- ``AbortSignal.any`` first-cause fusion becomes a fused
  :class:`~pydsh.subprocess.types.AbortController` that adopts the reason of
  whichever source aborts first (the fused controller's abort is idempotent,
  so the loser cannot overwrite the cause).
- The TS ``using`` disposal becomes a context manager / ``close()``.
"""

from __future__ import annotations

import asyncio
import math
from types import TracebackType

from pydsh.subprocess import MAX_TIMER_DELAY_MS, AbortController, AbortSignal

__all__ = [
    "MAX_TIMER_DELAY_MS",
    "Deadline",
    "TimeoutReason",
    "clamp_timeout",
    "deadline",
    "timeout_of",
]


class TimeoutReason(Exception):
    """Internal abort reason carrying a capability-owned code and elapsed
    deadline. Providers translate it through :func:`timeout_of` before
    returning to callers."""

    def __init__(self, code: str, timeout_ms: float) -> None:
        super().__init__(f"{code} after {timeout_ms}ms")
        #: Capability-owned timeout code (e.g. ``BASH_TIMEOUT``).
        self.code = code
        #: The deadline that elapsed, in milliseconds.
        self.timeout_ms = timeout_ms


def _assert_timer_delay(timeout_ms: float, name: str) -> None:
    if (
        not math.isfinite(timeout_ms)
        or timeout_ms <= 0
        or timeout_ms > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            f"{name} must be a positive finite number"
            f" no greater than {MAX_TIMER_DELAY_MS}"
        )


def clamp_timeout(
    requested: float | None,
    default: float,
    max_: float,
    name: str = "timeoutMs",
) -> float:
    """Validate a caller's optional timeout hint, use the backend default,
    then cap it. Supplied values must be positive and finite; zero is not a
    public disable-timeout sentinel.

    :param requested: the caller's optional hint; validated when present.
    :param default: the backend default applied when ``requested`` is absent.
    :param max_: the backend upper bound the result is capped to.
    :param name: field name used in the raised message (so the caller sees
        which input was bad).
    :return: the effective timeout in milliseconds:
        ``min(requested ?? default, max)``.
    """
    if requested is not None and (not math.isfinite(requested) or requested <= 0):
        raise ValueError(f"{name} must be a positive finite number")
    return min(requested if requested is not None else default, max_)


class Deadline:
    """A deadline signal plus the cleanup that clears its timer."""

    #: Aborts on upstream cancellation OR on timeout (the timeout carries a
    #: :class:`TimeoutReason`).
    signal: AbortSignal

    def __init__(
        self,
        upstream: AbortSignal | None,
        timeout_ms: float,
        code: str,
    ) -> None:
        self._timer: asyncio.TimerHandle | None = None
        self._fused: AbortController | None = None
        if timeout_ms <= 0:
            # No timeout (background work): forward only the upstream signal,
            # or a never-aborting one when there is no upstream.
            self.signal = upstream if upstream is not None else AbortController().signal
            return
        _assert_timer_delay(timeout_ms, "deadline timeout_ms")
        timer = AbortController()
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(
            timeout_ms / 1000.0, timer.abort, TimeoutReason(code, timeout_ms)
        )
        if upstream is None:
            self.signal = timer.signal
            return
        # First-cause fusion: the fused controller adopts the reason of
        # whichever source aborts FIRST, so a race resolves to a single
        # cause — timeout_of() reads TimeoutReason only when the timeout won,
        # and upstream-wins leaves an ordinary abort reason.
        fused = AbortController()
        upstream.add_listener(lambda: fused.abort(upstream.reason))
        timer.signal.add_listener(lambda: fused.abort(timer.signal.reason))
        self._fused = fused
        self.signal = fused.signal

    def close(self) -> None:
        """Clear the timer. Safe to call once; the context manager calls it
        at scope exit."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def __enter__(self) -> Deadline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def deadline(
    upstream: AbortSignal | None,
    timeout_ms: float,
    code: str,
) -> Deadline:
    """Fuse upstream cancellation with an identifiable timeout.
    ``timeout_ms <= 0`` is the internal no-timer sentinel; ``close()``
    clears an armed timer. The signal only notifies, so callers must stop
    their own work.

    :param upstream: the caller's cancellation signal, if any, fused into
        the result.
    :param timeout_ms: deadline in milliseconds; ``<= 0`` means "no timeout"
        (arm no timer).
    :param code: capability-owned code stamped onto the timeout's
        :class:`TimeoutReason`.
    :return: the fused :class:`Deadline` (signal + timer cleanup).
    """
    return Deadline(upstream, timeout_ms, code)


def timeout_of(
    x: AbortSignal | object,
    code: str | None = None,
) -> TimeoutReason | None:
    """Recover a timeout reason from a reason-bearing object. Supplying
    ``code`` distinguishes this deadline from a nested upstream deadline; a
    foreign code follows the ordinary cancellation path.

    :param x: an :class:`AbortSignal` or any reason carrier (e.g. a caught
        abort error).
    :param code: when provided, only a :class:`TimeoutReason` with this
        exact ``code`` matches.
    :return: the matching :class:`TimeoutReason`, else ``None``.
    """
    reason = getattr(x, "reason", None)
    if not isinstance(reason, TimeoutReason):
        return None
    return reason if code is None or reason.code == code else None
