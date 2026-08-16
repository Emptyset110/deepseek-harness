"""Package-owned prompt-assembly invariants.

Ported from deepseek-harness packages/core/system-prompt/invariant (MIT).

The TypeScript companion registers through the ``dsh-invariants`` service,
which the Python rewrite does not have; ``install()`` here takes the failure
reporter directly and keeps the same waterfall placement (global, prepended,
validating the authoritative value after ``next()``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, cast

from pycordis import Context, EffectDisposer
from pycordis.utils import maybe_await

from . import PromptAssembly

#: Invariant failure reporter: one call per violated invariant.
InvariantFailure = Callable[[str], None]

_VARIABLE_NAME = re.compile(r'^[a-z][a-z0-9_]*$')


def validate_assembly(assembly: PromptAssembly, fail: InvariantFailure) -> None:
    """Validate the authoritative assembly returned by the waterfall.

    :param assembly: the post-waterfall assembly to check.
    :param fail: reporter called once per violated invariant.
    """
    section_names: set[str] = set()
    for section in assembly.sections:
        if not section.name:
            fail('assembled section names must be non-empty')
        if section.name in section_names:
            fail(
                f'assembled section name {json.dumps(section.name)} is duplicated'
            )
        section_names.add(section.name)
        if not isinstance(section.text, str):
            fail(
                f'assembled section {json.dumps(section.name)} '
                'text must be a string'
            )

    context_names: set[str] = set()
    for context in assembly.contexts:
        if not context.name:
            fail('assembled context names must be non-empty')
        if context.name in context_names:
            fail(
                f'assembled context name {json.dumps(context.name)} '
                'is duplicated'
            )
        context_names.add(context.name)
        if not isinstance(context.text, str):
            fail(
                f'assembled context {json.dumps(context.name)} '
                'text must be a string'
            )

    for tool in assembly.tools:
        if not tool.name:
            fail('assembled tool names must be non-empty')

    for name, value in assembly.variables.items():
        if not _VARIABLE_NAME.match(name):
            fail(f'assembled variable name {json.dumps(name)} is invalid')
        if value is not None and not isinstance(value, str):
            fail(
                f'assembled variable {json.dumps(name)} '
                'must be a string or None'
            )


def install(ctx: Context, fail: InvariantFailure) -> EffectDisposer:
    """Install validation around the authoritative assembly waterfall result.

    The listener runs prepended and global, so it observes the final value
    of every scope's ``system-prompt/assemble`` waterfall.

    :param ctx: context owning the listener registration.
    :param fail: reporter called once per violated invariant.
    :return: the listener's disposer.
    """
    async def listener(
        carrier: Any,
        assembly: PromptAssembly,
        context: Any,
        next_: Callable[[], Any],
    ) -> PromptAssembly:
        assembled: PromptAssembly = await maybe_await(next_())
        validate_assembly(assembled, fail)
        return assembled

    return cast(
        EffectDisposer,
        ctx.on(
            'system-prompt/assemble',
            listener,
            {'global': True, 'prepend': True},
        ),
    )
