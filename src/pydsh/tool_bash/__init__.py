"""Model-facing Consumer of the ``ctx.shell`` capability seam: the ``bash``
tool. Background calls register process handles with ``ctx.jobs`` when the
(unported) jobs capability is present; their work uses job cancellation
rather than the tool-call signal after an id is returned.

Ported from deepseek-harness packages/shell (MIT),
``packages/shell/tool-bash/src/index.ts``.

Python mappings and scope reductions:

- The presentation callbacks (``presentCall``/``presentResult``) belong to
  the unported presentation layer and are dropped, matching the
  :mod:`pydsh.tools` boundary.
- ``ctx.shellEnv`` (shell-env capability) is unported; the tool reads it
  optionally through ``ctx.get('shellEnv')`` and otherwise sends no
  ``dsh_env`` snapshot.
- The sandbox capability is unported: ``ctx.sandboxPolicy`` never resolves,
  so a confining executor fails this plugin's load (the TS composition
  guard) and ``sandbox_permissions`` always reports "not available in this
  composition". The escalation argument pairing is still validated; the
  approval-backed escalation sequence arrives with the sandbox port.
- ``ctx.jobs`` is unported; the background branch fails loud with the same
  guidance text until it lands. The jobs registration callback mirrors the
  TS shape (``cancel``/``done``/``read_output``), with ``done`` an awaitable
  resolving to :func:`~pydsh.tool_bash.background.process_outcome`'s value.
- ``AbortSignal`` is the port from :mod:`pydsh.subprocess.types`; the TS
  ``error.name = 'AbortError'`` tag has no Python equivalent — the
  ``TOOL_ABORTED`` code carries the routing fact.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

from pycordis import Context
from pydsh.llm import HarnessError
from pydsh.session.json import JsonValue
from pydsh.shell import (
    DSH_ENV_PREFIX,
    CollectedOutput,
    ShellExecRequest,
    ShellRunResult,
)
from pydsh.shell.types import ShellSandboxInfo
from pydsh.system_prompt import PromptSection
from pydsh.tools import (
    TOOL_ABORTED,
    DefineToolOutput,
    ToolExecution,
    define_tool,
)
from pydsh.tools.schema import ParameterSchemaSpec, ValueSchemaSpec

from .background import process_outcome
from .render import render_process_read, render_result

__all__ = [
    "Config",
    "apply",
    "bash_description",
    "inject",
    "name",
    "validate_bash_args",
    "validate_escalation_args",
]

#: Cordis plugin metadata.
name = "tool-bash"
inject = ["tools", "shell", "systemPrompt"]


def _resolve_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Cordis loader config hook: raw mapping -> validated config."""
    raw = dict(config) if config is not None else {}
    enabled = raw.get("enableRunInBackground", True)
    if not isinstance(enabled, bool):
        raise TypeError("tool-bash: enableRunInBackground must be a boolean")
    return {"enableRunInBackground": enabled}


#: Plugin config schema hook (the single ``enableRunInBackground`` toggle,
#: default true).
Config = _resolve_config


def validate_escalation_args(
    sandbox_permissions: str | None,
    justification: str | None,
) -> None:
    """Validate the escalation argument pairing a tool schema cannot
    express: ``sandbox_permissions`` and ``justification`` travel together —
    an approval prompt without a reason, or a reason driving nothing, is a
    malformed ask — and the justification must be a non-empty sentence.
    (Borrowed from the unported sandbox capability, which owns this rule for
    both enforcing families.)

    :param sandbox_permissions: the raw ``sandbox_permissions`` argument, if
        given.
    :param justification: the raw ``justification`` argument, if given.
    """
    if sandbox_permissions is not None and justification is None:
        raise ValueError(
            "invalid escalation: sandbox_permissions requires a justification"
        )
    if justification is not None and sandbox_permissions is None:
        raise ValueError(
            "invalid escalation: justification is only valid together with"
            " sandbox_permissions"
        )
    if justification is not None and justification.strip() == "":
        raise ValueError("invalid justification: expected a non-empty sentence")


def validate_bash_args(args: dict[str, Any]) -> None:
    """Validate the value constraints the parameter schema cannot express."""
    command = args["command"]
    if command.strip() == "":
        raise ValueError("invalid command: expected a non-empty string")
    if args["description"].strip() == "":
        raise ValueError("invalid description: expected a non-empty string")
    timeout_ms = args.get("timeoutMs")
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or timeout_ms <= 0
    ):
        raise ValueError(
            f"invalid timeoutMs: expected a positive number, got {timeout_ms!r}"
        )
    # The escalation pairing (sandbox_permissions ⇔ justification,
    # non-empty) is the shared rule both enforcing families validate
    # identically.
    validate_escalation_args(args.get("sandbox_permissions"), args.get("justification"))


def bash_description(background_enabled: bool, escalation_modes: list[str]) -> str:
    """The model-facing tool description for this composition."""
    background = (
        "Set `run_in_background: true` for long-running commands: the call"
        " returns a job id immediately; read its output with `job_output` and"
        " stop it with `job_kill`."
        if background_enabled
        else "Background execution is not available; long-running commands"
        " must finish within the timeout."
    )
    base = (
        "Execute a bash command (`bash -c`) and return its stdout/stderr. "
        "Each call runs in a fresh shell: no state (cwd, variables,"
        " functions) persists between calls — pass `workdir` instead of using"
        " `cd`. Non-zero exits are reported as `[exit code: N]`. Current"
        f" harness environment facts are exposed through managed `${DSH_ENV_PREFIX}*`"
        " variables; inspect them when needed. Commands may run under a file"
        " sandbox; a blocked file operation is reported as `[sandbox: file"
        " access denied under <mode> mode]` — a policy denial, not a bug in"
        " the command; do not retry another way. Long output is truncated to"
        " its tail; the full output is saved to a file whose path is reported"
        f" when available. {background}"
    )
    if not escalation_modes:
        return base
    return (
        base + " Attempting a command the sandbox may deny is safe and"
        " expected: run it and read the marker rather than assuming the"
        " denial. When a command is denied and a wider mode would let it"
        " succeed, escalate immediately in the same turn — the one sanctioned"
        " exception to a denial: retry the exact same command once with"
        " `sandbox_permissions` (the narrowest wider mode that suffices) plus"
        " a one-sentence `justification`. Do not detour through chat to ask"
        " permission first — the approval prompt raised by that retry is how"
        " the user consents. If the session states approval prompts are"
        " disabled, there is no exception: a denial is final — do not set"
        " `sandbox_permissions`. Never escalate speculatively: ground the"
        " request in a real denial — normally the one this command just hit;"
        " escalating up front is fine only when this session already denied"
        " the same access. A rejected escalation is final for that command —"
        " stop and explain, never work around it — but it does not forbid"
        " attempting or escalating other commands later."
    )


def _canonical_path(path: str) -> str:
    """The filesystem identity of one path (the unported sandbox
    capability's ``canonicalPath``): symlinks resolved; a missing or
    unreadable path survives as given — the conservative outcome."""
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def _resolve_workdir(
    model_workdir: str | None,
    exec: ToolExecution,
    policy_workspace_root: str | None = None,
) -> str | None:
    """Resolve an explicit workdir first, making a relative one
    session-workspace-relative; otherwise use the filesystem identity of the
    session cwd and leave executor defaulting as the fallback. A resolved
    sandbox-policy root wins so workdir and confinement use the exact same
    per-call identity."""
    header_cwd: str | None = None
    if exec.agent is not None:
        header_cwd = exec.agent.session.header.get("cwd")
    session_cwd = (
        policy_workspace_root
        if policy_workspace_root is not None
        else (_canonical_path(header_cwd) if header_cwd is not None else None)
    )
    if model_workdir is None:
        return session_cwd
    if session_cwd is not None and not os.path.isabs(model_workdir):
        return os.path.join(session_cwd, model_workdir)
    return model_workdir


def _canonical_bash_result(result: ShellRunResult) -> dict[str, Any]:
    """Detach the executor result from service types into plain JSON data."""

    def output(stream: Any) -> dict[str, Any]:
        value: dict[str, Any] = {
            "text": stream.text,
            "truncated": stream.truncated,
        }
        if stream.spill_path is not None:
            value["spillPath"] = stream.spill_path
        return value

    detached: dict[str, Any] = {
        "exitCode": result.exit_code,
        "signal": result.signal,
        "timedOut": result.timed_out,
        "aborted": result.aborted,
        "timeoutMs": result.timeout_ms,
        "stdout": output(result.stdout),
        "stderr": output(result.stderr),
    }
    if result.sandbox is not None:
        sandbox: dict[str, Any] = {
            "mode": result.sandbox.mode,
            "denied": result.sandbox.denied,
        }
        if result.sandbox.enforcement is not None:
            sandbox["enforcement"] = result.sandbox.enforcement
        if result.sandbox.runner_failed is not None:
            sandbox["runnerFailed"] = result.sandbox.runner_failed
        detached["sandbox"] = sandbox
    return detached


def _collected_from_value(stream: Mapping[str, Any]) -> CollectedOutput:
    return CollectedOutput(
        text=stream["text"],
        truncated=stream["truncated"],
        spill_path=stream.get("spillPath"),
    )


def _run_result_from_value(value: Mapping[str, Any]) -> ShellRunResult:
    """Reconstitute a validated foreground output value as a
    :class:`ShellRunResult` for the shared renderer."""
    sandbox = value.get("sandbox")
    return ShellRunResult(
        exit_code=value["exitCode"],
        signal=value["signal"],
        timed_out=value["timedOut"],
        aborted=value["aborted"],
        timeout_ms=value["timeoutMs"],
        stdout=_collected_from_value(value["stdout"]),
        stderr=_collected_from_value(value["stderr"]),
        sandbox=(
            ShellSandboxInfo(
                mode=sandbox["mode"],
                denied=sandbox["denied"],
                enforcement=sandbox.get("enforcement"),
                runner_failed=sandbox.get("runnerFailed"),
            )
            if sandbox is not None
            else None
        ),
    )


_BACKGROUND_OUTPUT_PROPERTIES: dict[str, Any] = {
    "kind": {"type": "string", "required": True, "const": "background"},
    "jobId": {"type": "string", "required": True},
}

_STREAM_OUTPUT_PROPERTIES: dict[str, Any] = {
    "text": {"type": "string", "required": True},
    "truncated": {"type": "boolean", "required": True},
    "spillPath": {"type": "string"},
}

#: Canonical output schema shared by the foreground/background union. The
#: literal is cast wholesale: ``ParameterSchemaSpec`` TypedDicts cannot
#: express the value-spec union keys, so the runtime compiler is the
#: authority (as documented in :mod:`pydsh.tools.schema`).
_OUTPUT_SCHEMA = cast(
    ValueSchemaSpec,
    {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": _BACKGROUND_OUTPUT_PROPERTIES,
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {
                        "type": "string",
                        "required": True,
                        "const": "foreground",
                    },
                    "exitCode": {
                        "required": True,
                        "oneOf": [{"type": "integer"}, {"type": "null"}],
                    },
                    "signal": {
                        "required": True,
                        "oneOf": [{"type": "string"}, {"type": "null"}],
                    },
                    "timedOut": {"type": "boolean", "required": True},
                    "aborted": {"type": "boolean", "required": True},
                    "timeoutMs": {"type": "number", "required": True},
                    "stdout": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": True,
                        "properties": _STREAM_OUTPUT_PROPERTIES,
                    },
                    "stderr": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": True,
                        "properties": _STREAM_OUTPUT_PROPERTIES,
                    },
                    "sandbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "mode": {"type": "string", "required": True},
                            "denied": {"type": "boolean", "required": True},
                            "enforcement": {"type": "string"},
                            "runnerFailed": {"type": "boolean"},
                        },
                    },
                },
            },
        ],
    },
)


def apply(ctx: Context, config: Mapping[str, Any] | None = None) -> None:
    """Register the ``bash`` tool on ``ctx.tools``."""
    resolved = _resolve_config(config)
    background_enabled: bool = resolved["enableRunInBackground"]
    default_mode = ctx.shell.sandbox_mode
    escalation_modes: list[str] = []
    if default_mode is not None:
        # The sandbox capability is unported: a confining executor cannot be
        # served its standing policy, so the composition fails at load (the
        # TS guard against a split composition).
        raise RuntimeError(
            "tool-bash: the mounted bash executor confines but"
            " ctx.sandboxPolicy is missing"
        )

    # Cross-call guidance belongs in the prompt rather than one-call schema
    # prose.
    ctx.systemPrompt.section(
        PromptSection(
            name="tool:bash",
            order=105,
            text="Check the [exit code: N] marker on every bash result;"
            " investigate failures before moving on.",
        )
    )

    async def execute(args: dict[str, Any], exec: ToolExecution) -> Any:
        validate_bash_args(args)
        # Description is display metadata; workdir defaults to the caller's
        # session.
        if args.get("sandbox_permissions") is not None:
            # Unreachable with a confining executor (the composition guard
            # above fails the load first); a non-confining composition has
            # nothing to escalate.
            raise ValueError(
                "sandbox_permissions is not available in this composition"
                " (no sandboxing executor to escalate)"
            )
        workdir = _resolve_workdir(args.get("workdir"), exec)
        shell_env = ctx.get("shellEnv")
        dsh_env = shell_env.collect(exec) if shell_env is not None else None
        request = ShellExecRequest(
            command=args["command"],
            workdir=workdir,
            timeout_ms=args.get("timeoutMs"),
            dsh_env=dsh_env,
        )
        if args.get("run_in_background") is True:
            # Undeclared keys are allowed, so schema omission also needs
            # enforcement.
            if not background_enabled:
                raise ValueError(
                    "run_in_background is disabled for this deployment"
                    " (enableRunInBackground: false)"
                )
            jobs = ctx.get("jobs")
            if jobs is None:
                raise ValueError(
                    "background jobs unavailable: load the jobs capability"
                    " (dsh-jobs + dsh-tool-jobs)"
                )
            # The caller owns cancellation until ctx.jobs commits detached
            # ownership.
            if exec.signal.aborted:
                raise HarnessError("tool call aborted", TOOL_ABORTED)

            async def run() -> Any:
                proc = await ctx.shell.start(ctx.shell.resolve(request))

                async def done() -> Any:
                    await proc.done
                    return process_outcome(proc)

                return {
                    "cancel": lambda: proc.kill(),
                    "done": done(),
                    "read_output": lambda: render_process_read(
                        proc.read_output(), proc.sandbox, escalation_modes
                    ),
                }

            job_id = jobs.start(
                {
                    "kind": "bash",
                    "label": args["command"],
                    **({"owner": exec.agent} if exec.agent is not None else {}),
                    "run": run,
                }
            )
            return {"kind": "background", "jobId": job_id}
        result = await ctx.shell.run(
            ctx.shell.resolve(
                ShellExecRequest(
                    command=request.command,
                    workdir=request.workdir,
                    timeout_ms=request.timeout_ms,
                    dsh_env=request.dsh_env,
                    signal=exec.signal,
                )
            )
        )
        if result.aborted:
            raise HarnessError("tool call aborted", TOOL_ABORTED)
        return {"kind": "foreground", **_canonical_bash_result(result)}

    def render(_args: Any, value: JsonValue) -> list[Any]:
        assert isinstance(value, dict)
        if value["kind"] == "background":
            return [
                {"type": "text", "text": f"started background job {value['jobId']}"}
            ]
        return [
            {
                "type": "text",
                "text": render_result(_run_result_from_value(value), escalation_modes),
            }
        ]

    ctx.tools.register(
        define_tool(
            name="bash",
            description=bash_description(background_enabled, escalation_modes),
            parameters=cast(
                ParameterSchemaSpec,
                {
                    "command": {
                        "type": "string",
                        "required": True,
                        "description": "The bash command to execute.",
                    },
                    "description": {
                        "type": "string",
                        "required": True,
                        "description": (
                            "Clear, concise description of what this command does"
                            " in active voice, 5-10 words (shown in the UI)."
                            ' Examples: "ls" → "List files in current directory";'
                            ' "git status" → "Show working tree status"; "npm'
                            ' install" → "Install package dependencies".'
                        ),
                    },
                    "timeoutMs": {
                        "type": "number",
                        "description": (
                            "Timeout in milliseconds. The executor applies its"
                            " configured default and cap, and kills the command"
                            " on expiry."
                        ),
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Working directory for this command. Defaults to the"
                            " session workspace; a relative path is resolved"
                            " against it."
                        ),
                    },
                    **(
                        {
                            "run_in_background": {
                                "type": "boolean",
                                "description": (
                                    "Run in the background and return a job id"
                                    " immediately (collect with job_output, stop"
                                    " with job_kill). No timeout applies."
                                ),
                            },
                        }
                        if background_enabled
                        else {}
                    ),
                },
            ),
            output=DefineToolOutput(schema=_OUTPUT_SCHEMA, render=render),
            execute=execute,
        )
    )
