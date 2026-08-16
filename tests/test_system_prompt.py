"""Behavior tests for the system-prompt registry service.

Ported-semantics tests for the Python port of deepseek-harness
packages/core/system-prompt (MIT). Waterfall listeners receive the dispatch
carrier prepended as their first argument (JavaScript ``this``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from pycordis import Context, ValidationError
from pydsh.system_prompt import (
    PERSONA_SECTION,
    TOOL_ORDER_REST,
    AssembleContext,
    AssembledContext,
    AssembledSection,
    Config,
    ContextSnapshotSection,
    PromptAssembly,
    PromptContext,
    PromptSection,
    SystemPrompt,
    ToolProviderResult,
    ToolSchema,
    bind_scope_parent,
    create_scope,
    join_context_sections,
    render_context_sections,
    render_context_snapshot,
    render_prompt,
)

IDENTITY_TEXT = 'You are an AI agent powered by DeepSeek Harness.'


def make_service(config: Config | dict[str, object] | None = None) -> SystemPrompt:
    return SystemPrompt(Context(), config)


def tool(name: str, parameters: object = None) -> ToolSchema:
    return ToolSchema(name=name, description=f'{name} tool', parameters=parameters)


async def test_default_assembly() -> None:
    sp = make_service()
    assembly = await sp.assemble()
    assert [section.name for section in assembly.sections] == [
        'harness:identity',
        PERSONA_SECTION,
    ]
    assert assembly.contexts == []
    assert assembly.tools == []
    assert assembly.variables == {}
    # the empty default persona renders nothing
    assert render_prompt(assembly) == IDENTITY_TEXT


async def test_persona_config() -> None:
    sp = make_service(Config(persona='You are helpful.'))
    assembly = await sp.assemble()
    assert render_prompt(assembly) == f'{IDENTITY_TEXT}\n\nYou are helpful.'


async def test_include_harness_identity_false() -> None:
    sp = make_service(Config(include_harness_identity=False, persona='P'))
    assembly = await sp.assemble()
    assert [section.name for section in assembly.sections] == [PERSONA_SECTION]
    assert render_prompt(assembly) == 'P'


async def test_section_order_and_render() -> None:
    sp = make_service()
    sp.section(PromptSection(name='tools:guide', order=100, text='last'))
    sp.section(PromptSection(name='early', order=-200, text='first'))
    sp.section(PromptSection(name='empty', order=50, text=''))
    assembly = await sp.assemble()
    assert [section.name for section in assembly.sections] == [
        'early',
        'harness:identity',
        PERSONA_SECTION,
        'empty',
        'tools:guide',
    ]
    assert render_prompt(assembly) == f'first\n\n{IDENTITY_TEXT}\n\nlast'


async def test_duplicate_section_throws() -> None:
    sp = make_service()
    with pytest.raises(ValueError, match='already registered'):
        sp.section(PromptSection(name='harness:identity', order=1, text='x'))


async def test_non_finite_order_throws() -> None:
    sp = make_service()
    with pytest.raises(TypeError, match='finite number'):
        sp.section(PromptSection(name='nan', order=float('nan'), text='x'))
    with pytest.raises(TypeError, match='finite number'):
        sp.context(PromptContext(name='inf', order=float('inf'), text='x'))


async def test_registration_emits_change_and_disposes() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    changes: list[int] = []
    ctx.on('system-prompt/change', lambda: changes.append(1))
    dispose = sp.section(PromptSection(name='extra', order=10, text='x'))
    assert len(changes) == 1
    assembly = await sp.assemble()
    assert 'extra' in [section.name for section in assembly.sections]
    dispose()
    assert len(changes) == 2
    assembly = await sp.assemble()
    assert 'extra' not in [section.name for section in assembly.sections]


async def test_dynamic_text_receives_assemble_context() -> None:
    sp = make_service()
    seen: list[AssembleContext] = []
    sp.section(PromptSection(
        name='dynamic',
        order=10,
        text=lambda ctx: seen.append(ctx) or 'dyn',
    ))
    context = AssembleContext(scope=object(), signal='sig')
    assembly = await sp.assemble(context)
    assert seen == [context]
    assert render_prompt(assembly).endswith('dyn')


async def test_context_snapshot() -> None:
    sp = make_service()
    sp.context(PromptContext(name='b', order=20, text='second'))
    sp.context(PromptContext(name='a', order=10, text='first'))
    sp.context(PromptContext(name='empty', order=15, text=''))
    assembly = await sp.assemble()
    assert [c.name for c in assembly.contexts] == ['a', 'empty', 'b']
    sections = render_context_sections(assembly)
    assert sections == [
        ContextSnapshotSection(name='a', text='first'),
        ContextSnapshotSection(name='b', text='second'),
    ]
    assert render_context_snapshot(assembly) == (
        'Current runtime context. This snapshot supersedes earlier '
        'runtime-context snapshots.\n\nfirst\n\nsecond'
    )
    assert join_context_sections(sections) == render_context_snapshot(assembly)


async def test_join_context_sections_empty() -> None:
    assert join_context_sections([]) == ''
    sp = make_service()
    assert render_context_snapshot(await sp.assemble()) == ''


async def test_suppress_runtime_context() -> None:
    sp = make_service()
    sp.context(PromptContext(name='c', order=0, text='ctx'))
    dispose = sp.suppress_runtime_context()
    assembly = await sp.assemble()
    assert assembly.contexts == []
    dispose()
    assembly = await sp.assemble()
    assert [c.name for c in assembly.contexts] == ['c']


async def test_config_include_runtime_context_false() -> None:
    sp = make_service(Config(include_runtime_context=False))
    sp.context(PromptContext(name='c', order=0, text='ctx'))
    assembly = await sp.assemble()
    assert assembly.contexts == []


async def test_variable_interpolation() -> None:
    sp = make_service()
    sp.variable('user', lambda ctx: 'World')
    sp.section(PromptSection(name='greet', order=10, text='Hello {{user}}!'))
    assembly = await sp.assemble()
    assert assembly.variables == {'user': 'World'}
    assert render_prompt(assembly).endswith('Hello World!')


async def test_variable_invalid_name() -> None:
    sp = make_service()
    with pytest.raises(ValueError, match='invalid prompt variable name'):
        sp.variable('User', lambda ctx: 'x')
    with pytest.raises(ValueError, match='invalid prompt variable name'):
        sp.variable('', lambda ctx: 'x')


async def test_unknown_variable_error_lists_registered() -> None:
    sp = make_service()
    sp.variable('known', lambda ctx: 'x')
    sp.section(PromptSection(name='s', order=10, text='{{missing}}'))
    assembly = await sp.assemble()
    with pytest.raises(ValueError, match='registered variables: known'):
        render_prompt(assembly)


async def test_undefined_variable_value() -> None:
    sp = make_service()
    sp.variable('absent', lambda ctx: None)
    sp.section(PromptSection(name='s', order=10, text='{{absent}}'))
    assembly = await sp.assemble()
    with pytest.raises(ValueError, match='has no value for this assembly'):
        render_prompt(assembly)


async def test_malformed_references() -> None:
    sp = make_service()
    sp.section(PromptSection(name='empty-group', order=10, text='a {{}} b'))
    with pytest.raises(ValueError, match='malformed prompt variable reference'):
        render_prompt(await sp.assemble())

    sp2 = make_service()
    sp2.section(PromptSection(name='bad-name', order=10, text='{{not valid}}'))
    with pytest.raises(ValueError, match='malformed prompt variable reference'):
        render_prompt(await sp2.assemble())

    # a lone `{{` closed only by a much later `}}` is malformed too
    sp3 = make_service()
    sp3.section(PromptSection(name='broken', order=10, text='{{open ... }}'))
    with pytest.raises(ValueError, match='malformed prompt variable reference'):
        render_prompt(await sp3.assemble())


async def test_unclosed_braces_are_literal_prose() -> None:
    sp = make_service()
    sp.section(PromptSection(name='s', order=10, text='use {{ mustache'))
    assembly = await sp.assemble()
    assert render_prompt(assembly).endswith('use {{ mustache')


async def test_substituted_values_are_not_rescanned() -> None:
    sp = make_service()
    sp.variable('payload', lambda ctx: 'literal {{unregistered}}')
    sp.section(PromptSection(name='s', order=10, text='{{payload}}'))
    assembly = await sp.assemble()
    assert render_prompt(assembly).endswith('literal {{unregistered}}')


async def test_tools_collected_and_parameters_detached() -> None:
    sp = make_service()
    stored = tool('alpha', {'properties': {}})
    sp.tools(lambda ctx: ToolProviderResult(schemas=[stored, tool('beta')]))
    assembly = await sp.assemble()
    # default order is lexicographic
    assert [t.name for t in assembly.tools] == ['alpha', 'beta']
    # assembly owns a deep copy of the parameters
    assert assembly.tools[0].parameters == {'properties': {}}
    assert assembly.tools[0].parameters is not stored.parameters
    assembly.tools[0].parameters['properties']['x'] = 1
    assert stored.parameters == {'properties': {}}


async def test_tool_order_config_validation() -> None:
    with pytest.raises(ValueError, match='more than once'):
        make_service(Config(tool_order=['a', 'a', TOOL_ORDER_REST]))
    with pytest.raises(ValueError, match='rest entry'):
        make_service(Config(tool_order=['a']))
    with pytest.raises(ValueError, match='rest entry'):
        make_service(Config(tool_order=[]))


async def test_tool_order_applied() -> None:
    sp = make_service(Config(tool_order=['zeta', TOOL_ORDER_REST, 'alpha']))
    for name in ['alpha', 'mid', 'zeta', 'ant']:
        sp.tools(lambda ctx, _name=name: ToolProviderResult(schemas=[tool(_name)]))
    assembly = await sp.assemble()
    # unlisted tools sort lexicographically at the rest marker
    assert [t.name for t in assembly.tools] == ['zeta', 'ant', 'mid', 'alpha']


async def test_tool_order_unknown_name_fails_at_assembly() -> None:
    sp = make_service(Config(tool_order=['ghost', TOOL_ORDER_REST]))
    sp.tools(lambda ctx: ToolProviderResult(schemas=[tool('alpha')]))
    with pytest.raises(ValueError, match='unregistered tool "ghost"'):
        await sp.assemble()


async def test_tool_order_known_but_restricted_name_may_be_absent() -> None:
    sp = make_service(Config(tool_order=['hidden', TOOL_ORDER_REST]))
    sp.tools(lambda ctx: ToolProviderResult(schemas=[], known_names=['hidden']))
    assembly = await sp.assemble()
    assert assembly.tools == []


async def test_reserved_tool_name_fails() -> None:
    sp = make_service()
    sp.tools(lambda ctx: ToolProviderResult(schemas=[tool(TOOL_ORDER_REST)]))
    with pytest.raises(ValueError, match='reserved tool name'):
        await sp.assemble()


async def test_waterfall_transforms_assembly() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)

    def listener(
        carrier: object,
        assembly: PromptAssembly,
        context: AssembleContext,
        next_: object,
    ) -> PromptAssembly:
        assembly.sections.append(AssembledSection(name='added', text='extra'))
        return assembly

    ctx.on('system-prompt/assemble', listener)
    assembly = await sp.assemble()
    assert [s.name for s in assembly.sections][-1] == 'added'
    assert render_prompt(assembly).endswith('extra')


async def test_waterfall_veto_truncates_chain() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    calls: list[str] = []

    def first(
        carrier: object,
        assembly: PromptAssembly,
        context: object,
        next_: object,
    ) -> str:
        calls.append('first')
        # not calling next() truncates the chain; the return value is
        # authoritative
        return 'vetoed'

    def second(
        carrier: object,
        assembly: PromptAssembly,
        context: object,
        next_: object,
    ) -> None:
        calls.append('second')

    ctx.on('system-prompt/assemble', first)
    ctx.on('system-prompt/assemble', second)
    result = await sp.assemble()
    assert result == 'vetoed'
    assert calls == ['first']


async def test_complete_section_is_restored_after_waterfall() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    sp.section(PromptSection(name='override', order=10, text='ONLY', complete=True))

    def listener(
        carrier: object,
        assembly: PromptAssembly,
        context: object,
        next_: object,
    ) -> PromptAssembly:
        assembly.sections.append(AssembledSection(name='injected', text='x'))
        return assembly

    ctx.on('system-prompt/assemble', listener)
    assembly = await sp.assemble()
    # the waterfall ran (contexts/tools/variables resolved), but the complete
    # section is restored as the sole prompt section
    assert [(s.name, s.text) for s in assembly.sections] == [('override', 'ONLY')]
    assert render_prompt(assembly) == 'ONLY'


async def test_multiple_complete_sections_fail() -> None:
    sp = make_service()
    sp.section(PromptSection(name='c1', order=1, text='a', complete=True))
    sp.section(PromptSection(name='c2', order=2, text='b', complete=True))
    with pytest.raises(ValueError, match='multiple complete prompt sections'):
        await sp.assemble()


async def test_suppressed_contexts_stay_empty_after_waterfall() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    sp.context(PromptContext(name='c', order=0, text='ctx'))
    sp.suppress_runtime_context()

    def listener(
        carrier: object,
        assembly: PromptAssembly,
        context: object,
        next_: object,
    ) -> PromptAssembly:
        assembly.contexts.append(AssembledContext(name='late', text='x'))
        return assembly

    ctx.on('system-prompt/assemble', listener)
    assembly = await sp.assemble()
    assert assembly.contexts == []


async def test_plugin_config_path() -> None:
    ctx = Context()
    fiber = ctx.plugin(SystemPrompt, {'persona': 'from dict'})
    await fiber
    sp = ctx.get('systemPrompt')
    assert render_prompt(await sp.assemble()).endswith('from dict')


async def test_plugin_config_unknown_key_fails() -> None:
    ctx = Context()
    fiber = ctx.plugin(SystemPrompt, {'bogus': 1})
    with pytest.raises(ValidationError, match='unknown systemPrompt config'):
        await fiber


async def test_scoped_section_shadows_global() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx, Config(persona='global persona'))
    key = object()
    scope = create_scope(ctx, key)
    scoped = scope.ctx.get('systemPrompt')
    scoped.section(PromptSection(
        name=PERSONA_SECTION,
        order=0,
        text='scoped persona',
    ))
    scoped.section(PromptSection(name='scoped-only', order=50, text='S'))

    global_assembly = await sp.assemble()
    assert [s.name for s in global_assembly.sections] == [
        'harness:identity',
        PERSONA_SECTION,
    ]

    scoped_assembly = await sp.assemble(AssembleContext(scope=key))
    assert [(s.name, s.text) for s in scoped_assembly.sections] == [
        ('harness:identity', IDENTITY_TEXT),
        (PERSONA_SECTION, 'scoped persona'),
        ('scoped-only', 'S'),
    ]

    other = object()
    other_assembly = await sp.assemble(AssembleContext(scope=other))
    assert 'scoped-only' not in [s.name for s in other_assembly.sections]


async def test_scoped_variables_shadow_globals() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    sp.variable('who', lambda ctx: 'global')
    key = object()
    scope = create_scope(ctx, key)
    scope.ctx.get('systemPrompt').variable('who', lambda ctx: 'scoped')
    sp.section(PromptSection(name='s', order=10, text='{{who}}'))

    assert (await sp.assemble()).variables == {'who': 'global'}
    scoped = await sp.assemble(AssembleContext(scope=key))
    assert scoped.variables == {'who': 'scoped'}
    assert render_prompt(scoped).endswith('scoped')


async def test_scoped_waterfall_filtering() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    parent_key, child_key = object(), object()
    bind_scope_parent(child_key, parent_key)
    parent_scope = create_scope(ctx, parent_key)
    child_scope = create_scope(ctx, child_key)

    hits: list[str] = []

    def record(tag: str) -> Callable[..., PromptAssembly]:
        def listener(
            carrier: object,
            assembly: PromptAssembly,
            context: object,
            next_: Callable[[], PromptAssembly],
        ) -> PromptAssembly:
            hits.append(tag)
            return next_()

        return listener

    ctx.on('system-prompt/assemble', record('root'))
    parent_scope.ctx.on('system-prompt/assemble', record('parent'))
    child_scope.ctx.on('system-prompt/assemble', record('child'))

    await sp.assemble()
    assert hits == ['root']

    hits.clear()
    await sp.assemble(AssembleContext(scope=parent_key))
    assert hits == ['root', 'parent']

    hits.clear()
    await sp.assemble(AssembleContext(scope=child_key))
    # events flow up the chain: the ancestor listener observes descendants
    assert hits == ['root', 'parent', 'child']


async def test_scope_dispose_removes_registrations() -> None:
    ctx = Context()
    sp = SystemPrompt(ctx)
    key = object()
    scope = create_scope(ctx, key)
    scope.ctx.get('systemPrompt').section(
        PromptSection(name='scoped-only', order=50, text='S')
    )
    assembly = await sp.assemble(AssembleContext(scope=key))
    assert 'scoped-only' in [s.name for s in assembly.sections]
    await scope.dispose()
    assembly = await sp.assemble(AssembleContext(scope=key))
    assert 'scoped-only' not in [s.name for s in assembly.sections]


async def test_scoped_duplicate_error_message() -> None:
    ctx = Context()
    SystemPrompt(ctx)
    scope = create_scope(ctx, object())
    scoped = scope.ctx.get('systemPrompt')
    # a scoped layer may reuse a global name (shadowing), but not twice
    scoped.section(PromptSection(name='dup', order=1, text='a'))
    with pytest.raises(ValueError, match='already registered in this scope'):
        scoped.section(PromptSection(name='dup', order=2, text='b'))


async def test_scope_parent_rebind_and_cycle() -> None:
    key, parent_a, parent_b = object(), object(), object()
    binding = bind_scope_parent(key, parent_a)
    binding.rebind(parent_b)
    with pytest.raises(ValueError, match='already bound'):
        bind_scope_parent(key, parent_a)
    with pytest.raises(ValueError, match='cycle'):
        bind_scope_parent(parent_b, key)
