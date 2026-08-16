"""Behavior tests for the system-prompt assembly invariant companion.

Ported-semantics tests for the Python port of deepseek-harness
packages/core/system-prompt/invariant (MIT).
"""

from __future__ import annotations

from collections.abc import Callable

from pycordis import Context
from pydsh.system_prompt import (
    AssembleContext,
    AssembledSection,
    PromptAssembly,
    PromptSection,
    SystemPrompt,
    create_scope,
)
from pydsh.system_prompt.invariant import install, validate_assembly


def empty_assembly() -> PromptAssembly:
    return PromptAssembly(sections=[], contexts=[], tools=[], variables={})


async def test_validate_assembly_accepts_clean_input() -> None:
    failures: list[str] = []
    validate_assembly(empty_assembly(), failures.append)
    assert failures == []


async def test_validate_assembly_reports_each_violation() -> None:
    assembly = PromptAssembly(
        sections=[
            AssembledSection(name='', text='x'),
            AssembledSection(name='dup', text='a'),
            AssembledSection(name='dup', text='b'),
        ],
        contexts=[],
        tools=[],
        variables={'Bad Name': 'x', 'ok': None},
    )
    failures: list[str] = []
    validate_assembly(assembly, failures.append)
    assert failures == [
        'assembled section names must be non-empty',
        'assembled section name "dup" is duplicated',
        'assembled variable name "Bad Name" is invalid',
    ]


async def test_install_validates_post_waterfall_result() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    failures: list[str] = []
    install(ctx, failures.append)

    def bad_listener(
        carrier: object,
        assembly: PromptAssembly,
        context: object,
        next_: object,
    ) -> PromptAssembly:
        assembly.sections.append(
            AssembledSection(name='harness:identity', text='shadow')
        )
        return assembly

    ctx.on('system-prompt/assemble', bad_listener)
    assembly = await sp.assemble()
    # the installed listener is prepended, so it validates the final value
    # after the inner listener's mutation; the result itself passes through
    assert failures == ['assembled section name "harness:identity" is duplicated']
    assert [s.name for s in assembly.sections].count('harness:identity') == 2


async def test_install_observes_every_scope() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    failures: list[str] = []
    install(ctx, failures.append)

    def bad_listener(
        carrier: object,
        assembly: PromptAssembly,
        context: AssembleContext,
        next_: Callable[[], object],
    ) -> PromptAssembly:
        if context.scope is not None:
            assembly.sections.append(AssembledSection(name='', text='x'))
        return assembly

    # registered globally with no scope tag: admitted for every scope
    ctx.on('system-prompt/assemble', bad_listener)

    key = object()
    create_scope(ctx, key)
    await sp.assemble()
    assert failures == []
    await sp.assemble(AssembleContext(scope=key))
    assert failures == ['assembled section names must be non-empty']


async def test_install_disposer_removes_validation() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    failures: list[str] = []
    dispose = install(ctx, failures.append)
    sp.section(PromptSection(name='', order=10, text='x'))
    # registered sections are assembled verbatim; empty names are caught only
    # while the invariant listener is installed
    await sp.assemble()
    assert len(failures) == 1
    dispose()
    await sp.assemble()
    assert len(failures) == 1
