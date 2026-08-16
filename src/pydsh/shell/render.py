"""Shared rendering helper for the shell tools: the exit-status marker
contract the tools' renderers emit and the presentation layer parses back.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/shell/src/render.ts``.

Python mapping: the TS intersection-type result becomes a frozen dataclass
carrying both optional facts (``signal is not None`` marks a kill; otherwise
``exit_code`` holds the terminal code, ``0`` when no marker was present).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ParsedExitStatus", "parse_exit_status"]

_SIGNAL_MARKER = re.compile(r"\n\[killed by signal: ([^\]\n]+)\]$")
_EXIT_MARKER = re.compile(r"\n\[exit code: (\d+)\]$")


@dataclass(frozen=True)
class ParsedExitStatus:
    """The exit status recovered from a rendered result, with the output
    body that status was split off from."""

    #: The marker-free output body.
    body: str
    #: The terminal exit code; ``None`` when the command was signal-killed.
    exit_code: int | None
    #: The terminating signal name; ``None`` unless killed.
    signal: str | None = None


def parse_exit_status(text: str) -> ParsedExitStatus:
    """Split a rendered shell-tool result string into its output body and
    the structured exit status — the inverse of the ``[exit code: N]`` /
    ``[killed by signal: X]`` markers the shell tools' renderers append. A
    killed marker yields ``signal``; otherwise a non-zero marker yields
    ``exit_code``; absent both means a clean exit 0.

    The consumed marker is removed from ``body`` because a terminal
    presentation shows the exit status as its own pill: leaving the marker
    in the output would render the exit twice. Other markers (timeout,
    sandbox denial) carry facts no pill shows, so they stay in the body.

    Replay only retains the rendered content text, not the original
    ``ShellRunResult``, so terminal presentation must recover the exit pill
    here. Requiring a leading newline and the end of the string keeps
    ordinary output that merely ends with marker-like text from matching
    unless the final line is indistinguishable from a real marker.

    :param text: rendered model-facing shell-tool result.
    :return: the marker-free body plus the recovered terminal exit code or
        signal.
    """
    signal = _SIGNAL_MARKER.search(text)
    if signal is not None:
        return ParsedExitStatus(
            body=text[: signal.start()], exit_code=None, signal=signal.group(1)
        )
    exit_ = _EXIT_MARKER.search(text)
    if exit_ is not None:
        return ParsedExitStatus(
            body=text[: exit_.start()], exit_code=int(exit_.group(1))
        )
    return ParsedExitStatus(body=text, exit_code=0)
