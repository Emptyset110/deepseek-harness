"""Package-owned tool-pipeline invariants.

Ported from deepseek-harness packages/core/tools (MIT), ``src/invariant.ts``,
reduced to what the Python rewrite can check: the monotonic pipeline stage
ordering (``tools/pre-execute`` before ``tools/execute`` /
``tools/post-execute``, one ``tools/result`` per execution) and the
final-snapshot publication contract (frozen execution, non-empty identity).
The TypeScript companion additionally checks ``Object.isFrozen`` on the
result (frozen dataclasses make that structural here) and the
``tool/code-dispatch*`` session-event enclosure, which belongs to the
unported Code Mode layer.

The TypeScript companion registers through the ``dsh-invariants`` service,
which the Python rewrite does not have; ``install()`` takes the failure
reporter directly, same as :mod:`pydsh.system_prompt.invariant`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any, Literal, cast

from pycordis import Context, EffectDisposer

from .runtime import ToolExecution

#: Invariant failure reporter: one call per violated invariant.
InvariantFailure = Callable[[str], None]

_ToolStage = Literal['pre', 'execute', 'post']


def install(ctx: Context, fail: InvariantFailure) -> EffectDisposer:
    """Install monotonic pipeline and final-snapshot checks.

    Listens on ``internal/dispatch`` (global), so it observes every scope's
    tool-pipeline dispatches without participating in their filtering.

    :param ctx: context owning the listener registration.
    :param fail: reporter called once per violated invariant.
    :return: the listener's disposer.
    """
    stages: weakref.WeakKeyDictionary[ToolExecution, _ToolStage] = (
        weakref.WeakKeyDictionary()
    )

    def validate_result(exec: ToolExecution, result: Any) -> None:
        if not exec._is_frozen:
            fail('tools/result execution must be frozen before publication')
        if not exec.name or not str(exec.call_id):
            fail('tools/result execution must carry non-empty name and callId')

    def listener(
        mode: str,
        event_name: str,
        args: list[Any],
        this_arg: Any = None,
    ) -> None:
        if event_name == 'tools/pre-execute':
            exec = args[0]
            if exec in stages:
                fail('tools/pre-execute repeated for one execution')
            stages[exec] = 'pre'
            return
        if event_name == 'tools/execute':
            exec = args[0]
            if stages.get(exec) != 'pre':
                fail('tools/execute must follow tools/pre-execute')
            stages[exec] = 'execute'
            return
        if event_name == 'tools/post-execute':
            exec = args[0]
            previous = stages.get(exec)
            if previous not in ('pre', 'execute'):
                fail(
                    'tools/post-execute must follow tools/pre-execute or '
                    'tools/execute'
                )
            stages[exec] = 'post'
            return
        if event_name != 'tools/result':
            return
        exec, result = args[0], args[1]
        validate_result(exec, result)
        stages.pop(exec, None)

    return cast(EffectDisposer, ctx.on('internal/dispatch', listener, {'global': True}))
