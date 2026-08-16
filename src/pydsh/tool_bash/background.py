"""Generic-task adaptation for background bash process handles.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/tool-bash/src/background.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydsh.shell import ShellProcess

__all__ = ["ProcessOutcome", "process_outcome"]


@dataclass(frozen=True)
class ProcessOutcome:
    """The generic task-outcome vocabulary of one settled background bash
    process (TS: a plain ``{ status, detail }`` object)."""

    status: Literal["completed", "killed"]
    detail: str


def process_outcome(proc: ShellProcess) -> ProcessOutcome:
    """Map a settled background process onto the generic task-outcome
    vocabulary: ``killed`` stays ``killed`` (detail: the signal when one is
    known), everything else is ``completed`` with the exit code as detail. A
    nonzero command exit is reported, not failed, exactly like the
    foreground rendering.

    :param proc: the settled process handle.
    :return: the outcome for the jobs registration.
    """
    if proc.status == "killed":
        return ProcessOutcome(
            status="killed",
            detail=(
                f"signal: {proc.signal}"
                if proc.signal is not None
                else "killed before exit"
            ),
        )
    return ProcessOutcome(
        status="completed", detail=f"exit code: {proc.exit_code or 0}"
    )
