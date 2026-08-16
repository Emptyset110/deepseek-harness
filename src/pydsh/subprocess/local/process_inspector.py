"""Linux ``/proc`` process-table inspection used by spawn teardown.

Ported from deepseek-harness packages/subprocess (MIT),
``packages/subprocess/subprocess-local/src/process-inspector.ts``.

Simplification: only the parts the core spawn cleanup needs are ported —
``/proc/<pid>/stat`` parsing and the zombie-only process-group probe. The
foreground-group, syscall-wait, and tree/session enumeration used by the TS
PTY terminal are omitted: this port's terminal runs on pipes and never
inspects foreground groups.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_ZOMBIE_STATE = re.compile(r'^[ZXx]$')
_NUMERIC = re.compile(r'^\d+$')


@dataclass(frozen=True)
class ProcStat:
    """Fields used from one parsed ``/proc/<pid>/stat`` line."""

    pid: int
    parent_pid: int
    pgrp: int
    session: int
    state: str
    tpgid: int
    started: str


def parse_proc_stat(text: str) -> ProcStat | None:
    """Parse fields used from Linux ``/proc/<pid>/stat``, including
    parenthesized comm text. Returns ``None`` for malformed input."""
    open_paren = text.find('(')
    close_paren = text.rfind(')')
    if open_paren <= 0 or close_paren <= open_paren:
        return None
    try:
        pid = int(text[:open_paren].strip())
        rest = text[close_paren + 2:].split()
        state = rest[0]
        parent_pid = int(rest[1])
        pgrp = int(rest[2])
        session = int(rest[3])
        tpgid = int(rest[5])
        started = rest[19]
    except (IndexError, ValueError):
        return None
    if len(state) != 1:
        return None
    return ProcStat(
        pid=pid,
        parent_pid=parent_pid,
        pgrp=pgrp,
        session=session,
        state=state,
        tpgid=tpgid,
        started=started,
    )


def _read_stat(pid: int) -> ProcStat | None:
    try:
        with open(f'/proc/{pid}/stat') as file:
            return parse_proc_stat(file.read())
    except OSError:
        return None


def linux_process_group_has_live_members(process_group_id: int) -> bool | None:
    """Report whether a Linux process group has an executing member.
    ``False`` means the group contains only zombie/dead entries; ``None``
    means the process table could not prove either outcome.
    """
    try:
        entries = os.listdir('/proc')
    except OSError:
        return None
    matched = False
    for entry in entries:
        if not _NUMERIC.match(entry):
            continue
        stat = _read_stat(int(entry))
        if stat is None or stat.pgrp != process_group_id:
            continue
        matched = True
        if not _ZOMBIE_STATE.match(stat.state):
            return True
    return False if matched else None
