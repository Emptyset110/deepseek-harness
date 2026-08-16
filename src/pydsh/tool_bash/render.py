"""Model-facing result rendering for the bash tool.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/tool-bash/src/render.ts``.

The sandbox denial/escalation marker vocabulary (``sandboxDenialMarker`` /
``escalationHintMarker``) belongs to the unported sandbox capability
(``packages/sandbox/sandbox/src/escalation.ts``); it is duplicated here
verbatim so the rendered marker contract stays stable until that capability
is ported.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydsh.shell import (
    CollectedOutput,
    ParsedExitStatus,
    ShellProcessRead,
    ShellRunResult,
    ShellSandboxInfo,
    parse_exit_status,
)

__all__ = [
    "ParsedExitStatus",
    "escalation_hint_marker",
    "parse_exit_status",
    "render_process_read",
    "render_result",
    "sandbox_denial_marker",
]


def sandbox_denial_marker(mode: str) -> str:
    """The model-facing denial marker, exactly as the model sees it
    (borrowed from the unported sandbox capability)."""
    return f"[sandbox: file access denied under {mode} mode]"


def escalation_hint_marker(subject: str) -> str:
    """The same-turn escalation hint that rides a denial when the
    composition advertises the escalation fields (borrowed from the
    unported sandbox capability)."""
    return (
        "[sandbox: escalation available — retry this exact"
        f" {subject} once with sandbox_permissions (the narrowest wider mode"
        " that suffices) + justification; the approval prompt asks the user]"
    )


def _fmt_number(value: float) -> str:
    """Format a millisecond count like a JS number (no trailing ``.0``)."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _stream_text(output: CollectedOutput) -> str:
    """Append the truncation notice (with the full-output spill path) to a
    stream's text."""
    if not output.truncated:
        return output.text
    spill = output.spill_path if output.spill_path is not None else "(unavailable)"
    return f"{output.text}\n[output truncated; full output: {spill}]"


def render_result(
    result: ShellRunResult,
    escalation_modes: Sequence[str] = (),
) -> str:
    """Shape one finished run into the text the model sees: stdout, then a
    marked stderr section, then exit-status markers. Non-zero exits are
    reported, not errored — the model decides how to react; only
    infrastructure failures (spawn errors, aborts) surface as isError
    results.

    :param result: the completed foreground run from the executor.
    :param escalation_modes: the escalation targets this composition
        advertises; non-empty adds the same-turn escalation hint after a
        denial marker (default empty: no hint).
    :return: the model-facing text: output body (or ``(no output)``), then
        any timeout/signal/exit markers, each on its own line.
    """
    out = _stream_text(result.stdout)
    err = _stream_text(result.stderr)

    body = out
    if err:
        # Single newline between sections (stdout usually ends with one
        # already).
        if body and not body.endswith("\n"):
            body += "\n"
        body += f"[stderr]\n{err}"
    if not body:
        body = "(no output)"

    markers: list[str] = []
    # Keep the exit marker last because parse_exit_status anchors there.
    if result.sandbox is not None and result.sandbox.denied:
        markers.append(sandbox_denial_marker(result.sandbox.mode))
        # Hint only when the composition exposes escalation, before the
        # final exit marker.
        if escalation_modes:
            markers.append(escalation_hint_marker("command"))
    # A command may trap SIGTERM and exit 0 after timeout; still report
    # interruption.
    if result.timed_out:
        markers.append(f"[timed out after {_fmt_number(result.timeout_ms)}ms]")
    if result.signal is not None:
        markers.append(f"[killed by signal: {result.signal}]")
    elif result.exit_code != 0:
        markers.append(f"[exit code: {result.exit_code}]")
    if not markers:
        return body

    if not body.endswith("\n"):
        body += "\n"
    return body + "\n".join(markers)


def render_process_read(
    read: ShellProcessRead,
    sandbox: ShellSandboxInfo | None = None,
    escalation_modes: Sequence[str] = (),
) -> str:
    """Shape one background-process read into the ``job_output`` delta the
    model sees: the incremental delta, plus the lossy-read notice (with
    full-stream spill paths) when in-memory truncation dropped unread bytes.
    Empty-delta rendering (``(no new output)``) is the generic job
    controller's job.

    :param read: one incremental read from the process handle.
    :param sandbox: settled sandbox facts, when this was a confined process.
    :param escalation_modes: escalation targets advertised by this
        composition.
    :return: the delta text with any loss or sandbox notice appended.
    """
    notices: list[str] = []
    if read.lossy:
        paths = [
            path
            for path in (read.stdout_spill_path, read.stderr_spill_path)
            if path is not None
        ]
        notices.append(
            "[some output was dropped from memory; full output:"
            f" {', '.join(paths) if paths else '(unavailable)'}]"
        )
    if sandbox is not None and sandbox.runner_failed:
        notices.append(
            f"[sandbox: the sandbox runner itself failed under {sandbox.mode}"
            " mode — the command did not run; this is a sandbox problem, not"
            " a command failure]"
        )
    elif sandbox is not None and sandbox.denied:
        notices.append(sandbox_denial_marker(sandbox.mode))
        if escalation_modes:
            notices.append(escalation_hint_marker("command"))
    if not notices:
        return read.delta
    separator = "\n" if read.delta and not read.delta.endswith("\n") else ""
    return read.delta + separator + "\n".join(notices)
