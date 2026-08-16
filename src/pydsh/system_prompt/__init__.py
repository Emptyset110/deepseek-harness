"""Registry for ordered system sections, contexts, tool schemas, and variables.

Ported from deepseek-harness packages/core/system-prompt (MIT).

Event surface (TypeScript declaration merging has no Python equivalent, so
the events are documented here):

- ``system-prompt/assemble`` (waterfall): expert waterfall over the
  assembled sections, contexts, tools, and variables. The dispatch carrier
  built by ``scope_target()`` is prepended as the listener's first argument
  where TypeScript binds ``this``, so listeners have the signature
  ``(carrier, assembly, context, next)`` and MUST call ``next()`` to
  delegate — returning without it truncates the chain. The returned value
  is authoritative, except that an effective complete section is restored
  afterwards as the sole prompt section. Dispatch is scope-filtered:
  listeners registered on a scoped context receive only that scope's (and
  its descendants') assemblies.
- ``system-prompt/change`` (emit): fired when any prompt provider changes.
  Unfiltered, because a global change affects every scope.

Type mappings: ``Record<string, string | undefined>`` becomes
``dict[str, str | None]``; ``ToolSchema`` / ``ContextSnapshotSection``
(``@deepseek-ai/dsh-llm`` types) are defined locally; zod config validation
becomes a plain ``Config`` dataclass plus a validator callable.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pycordis import Context, EffectDisposer, Service, ValidationError
from pycordis.utils import maybe_await
from pydsh.scope import (
    AnonymousEntries,
    NamedEntries,
    Scope,
    ScopedLayers,
    ScopeKey,
    bind_scope_parent,
    create_scope,
    scope_chain_of,
    scope_of,
    scope_parent_of,
    scope_target,
)

__all__ = [
    'PERSONA_ORDER',
    'PERSONA_SECTION',
    'TOOL_ORDER_REST',
    'AssembleContext',
    'AssembledContext',
    'AssembledSection',
    'Config',
    'ContextSnapshotSection',
    'PromptAssembly',
    'PromptContext',
    'PromptSection',
    'Scope',
    'ScopeKey',
    'SystemPrompt',
    'ToolProviderResult',
    'ToolSchema',
    'bind_scope_parent',
    'create_scope',
    'join_context_sections',
    'render_context_sections',
    'render_context_snapshot',
    'render_prompt',
    'scope_chain_of',
    'scope_of',
    'scope_parent_of',
    'scope_target',
]

#: The deployment persona's section name. Exported because a composition can
#: replace this slot — an agent preset shadows the deployment's persona with
#: its own — and both sides naming the same section is what makes the
#: replacement work rather than duplicate.
PERSONA_SECTION = 'deployment:persona'

#: Prompt order of the persona slot; the first section a model reads.
PERSONA_ORDER = 0

#: Reserved ``Config.tool_order`` marker for unlisted tools.
TOOL_ORDER_REST = '<unlisted-tools>'

#: Valid variable names: how they are written between the braces.
_VARIABLE_NAME = re.compile(r'^[a-z][a-z0-9_]*$')

#: A complete ``{{...}}`` reference group at the scan position.
_GROUP_AT = re.compile(r'\{\{([^{}]*)\}\}')


@dataclass
class AssembleContext:
    """Merge-extensible context for one prompt assembly.

    Plugins may attach further fields; the dataclass only fixes the two
    fields the registry itself reads.
    """

    #: Scope whose providers and waterfall listeners participate. When
    #: ``None``, only global providers and subject-less listeners participate.
    scope: ScopeKey | None = None
    #: Explicit control signal for the turn that requested this assembly, if
    #: any. It controls only this explicit request and must not be retained
    #: to control later turns. (TypeScript ``AbortSignal``; opaque here.)
    signal: Any | None = None


@dataclass(frozen=True)
class PromptSection:
    """One contributed section of the system prompt (registry input)."""

    #: Unique name — a duplicate registration raises.
    name: str
    #: Sections are concatenated in ascending order. Convention: ``-100`` is
    #: the harness identity, ``0`` the deployment persona, tool guidance uses
    #: 100-199; other negative orders also render before the persona.
    order: float
    #: Static text or a provider evaluated at each assembly with that
    #: assembly's ``AssembleContext``. The text may reference
    #: ``{{variable}}``s — interpolated later, by ``render_prompt()``.
    text: str | Callable[[AssembleContext], str]
    #: Treat this contribution as the complete system prompt. Assembly still
    #: runs the cooperative waterfall so tools, contexts, and variables can
    #: be resolved, then restores this exact section as the sole prompt
    #: section. More than one effective complete section makes assembly fail.
    complete: bool = False


@dataclass(frozen=True)
class PromptContext:
    """Dynamic model context materialized as a durable user-role snapshot."""

    #: Unique name — a duplicate registration raises.
    name: str
    #: Contexts are joined in ascending order.
    order: float
    #: Static text or a provider evaluated for each assembly. Empty text
    #: contributes nothing.
    text: str | Callable[[AssembleContext], str]


@dataclass
class AssembledSection:
    """One section of an assembly: ``PromptSection`` with its text resolved."""

    #: The contributing section's unique name.
    name: str
    #: The resolved (but not yet interpolated) section text.
    text: str


@dataclass
class AssembledContext:
    """One resolved dynamic context contribution."""

    #: The contributing context's unique name.
    name: str
    #: The resolved text before variable interpolation.
    text: str


@dataclass
class ToolSchema:
    """Model-facing tool schema (local stand-in for the dsh-llm type)."""

    name: str
    description: str
    #: The JSON-schema parameters object. ``assemble()`` deep-copies it, so
    #: one assembly can never mutate a provider's stored schema.
    parameters: Any


@dataclass
class ContextSnapshotSection:
    """One named, rendered contribution of the runtime-context snapshot."""

    name: str
    text: str


@dataclass(frozen=True)
class ToolProviderResult:
    """Tool schemas visible in one assembly and their pre-restriction name set."""

    #: The schemas this provider contributes to THIS assembly.
    schemas: Sequence[ToolSchema]
    #: The pre-restriction name universe for config validation (defaults to
    #: ``schemas``' names).
    known_names: Sequence[str] | None = None


@dataclass
class PromptAssembly:
    """Merge-extensible assembled model input.

    Sections and contexts remain uninterpolated until rendered; tools are
    already in canonical order. Plugins may attach further fields.
    """

    sections: list[AssembledSection]
    contexts: list[AssembledContext]
    tools: list[ToolSchema]
    variables: dict[str, str | None]


@dataclass(frozen=True)
class Config:
    """Plugin config: the deployment-authored fragment of the system prompt."""

    #: Include the fixed DeepSeek Harness identity before the deployment
    #: persona.
    include_harness_identity: bool = True
    #: Include dynamic runtime-context snapshots in model history.
    include_runtime_context: bool = True
    #: Deployment-wide order-0 persona template. A scoped section named
    #: ``deployment:persona`` shadows it; ``{{variable}}`` references are
    #: strict.
    persona: str = ''
    #: Model-facing tool names in order, with ``TOOL_ORDER_REST`` exactly
    #: once. Invalid fields fail at load and unknown names fail at assembly;
    #: known names hidden in one scope may be absent there. ``None`` means
    #: lexicographic order. (Omission is preserved because an explicit empty
    #: order lacks the rest marker.)
    tool_order: list[str] | None = None


_CONFIG_KEYS = frozenset({
    'include_harness_identity',
    'include_runtime_context',
    'persona',
    'tool_order',
})


def _resolve_config(config: Config | dict[str, Any] | None) -> Config:
    """Validate a raw plugin config mapping into a ``Config``.

    Installed as ``SystemPrompt.Config`` so the Cordis loader applies it;
    also accepts an already-built ``Config`` for direct construction.
    """
    if config is None:
        return Config()
    if isinstance(config, Config):
        return config
    if isinstance(config, dict):
        unknown = sorted(set(config) - _CONFIG_KEYS)
        if unknown:
            raise ValidationError(
                f'unknown systemPrompt config keys: {", ".join(unknown)}'
            )
        return Config(**config)
    raise ValidationError(
        f'invalid systemPrompt config: {type(config).__name__}'
    )


#: Alias for annotations inside ``SystemPrompt``, whose ``Config`` class
#: attribute (the loader validator) shadows the dataclass name in class scope.
_ConfigData = Config


def _validate_tool_order(tool_order: list[str] | None) -> list[str] | None:
    """Validate duplicate names and the required ``TOOL_ORDER_REST`` marker.

    Registered names are checked later because plugins have not loaded yet.
    """
    if tool_order is None:
        return None
    seen: set[str] = set()
    for name in tool_order:
        if name in seen:
            raise ValueError(f'toolOrder lists "{name}" more than once')
        seen.add(name)
    if TOOL_ORDER_REST not in seen:
        raise ValueError(
            f'toolOrder must contain the "{TOOL_ORDER_REST}" rest entry '
            '(where unlisted tools are inserted)'
        )
    return tool_order


def _order_tools(
    tools: list[ToolSchema],
    tool_order: list[str] | None,
    known_names: set[str],
) -> list[ToolSchema]:
    """Apply configured tool order, inserting unlisted tools at the rest marker.

    Unlisted tools are inserted lexicographically at ``TOOL_ORDER_REST``.
    Unknown configured names fail; known but restricted names may be absent.
    """
    if any(tool.name == TOOL_ORDER_REST for tool in tools):
        raise ValueError(
            f'tool provider returned reserved tool name "{TOOL_ORDER_REST}" '
            "(reserved for toolOrder's rest entry)"
        )
    if tool_order is None:
        return sorted(tools, key=lambda tool: tool.name)
    unknown = [
        name for name in tool_order
        if name != TOOL_ORDER_REST and name not in known_names
    ]
    if unknown:
        plural = 's' if len(unknown) > 1 else ''
        listed = ', '.join(f'"{name}"' for name in unknown)
        known = ', '.join(sorted(known_names)) or '(none)'
        raise ValueError(
            f'toolOrder lists unregistered tool{plural} {listed}; '
            f'known tools: {known}'
        )
    listed_names = set(tool_order)
    rest = sorted(
        (tool for tool in tools if tool.name not in listed_names),
        key=lambda tool: tool.name,
    )
    ordered: list[ToolSchema] = []
    for name in tool_order:
        if name == TOOL_ORDER_REST:
            ordered.extend(rest)
        else:
            ordered.extend(tool for tool in tools if tool.name == name)
    return ordered


def render_prompt(assembly: PromptAssembly) -> str:
    """Interpolate strict ``{{variable}}`` references and join the sections.

    Empty sections are dropped, the rest joined with blank lines. Malformed,
    unknown, or undefined references raise; a lone ``{{`` without any later
    ``}}`` is literal prose, and substituted values are not scanned again.

    :param assembly: the assembly whose sections and variables to render.
    :return: the rendered prompt, or ``''`` when all sections are empty.
    """
    return '\n\n'.join(
        text
        for section in assembly.sections
        if (text := _interpolate(section, assembly.variables, 'section'))
    )


def render_context_snapshot(assembly: PromptAssembly) -> str:
    """Render the complete dynamic context snapshot.

    :param assembly: the assembly whose contexts and variables to render.
    :return: the current full snapshot, or ``''`` when no context is active.
    """
    return join_context_sections(render_context_sections(assembly))


def join_context_sections(sections: Sequence[ContextSnapshotSection]) -> str:
    """The model-facing snapshot text for an already-rendered section list.

    A caller that also needs the sections renders them once and joins here,
    so a request does not interpolate every context twice.

    :param sections: sections from ``render_context_sections()``.
    :return: the current full snapshot, or ``''`` when no context is active.
    """
    body = '\n\n'.join(section.text for section in sections)
    if not body:
        return ''
    return (
        'Current runtime context. This snapshot supersedes earlier '
        f'runtime-context snapshots.\n\n{body}'
    )


def render_context_sections(assembly: PromptAssembly) -> list[ContextSnapshotSection]:
    """The same snapshot, kept as the named contributions it was assembled from.

    ``render_context_snapshot()`` joins these for the model; a consumer that
    presents the snapshot uses them to attribute each part to the subsystem
    that contributed it, without re-splitting the joined prose.

    :param assembly: the assembly whose contexts and variables to render.
    :return: one entry per contributing context that rendered to non-empty text.
    """
    return [
        ContextSnapshotSection(name=context.name, text=text)
        for context in assembly.contexts
        if (text := _interpolate(context, assembly.variables, 'context'))
    ]


def _interpolate(
    input_: AssembledSection | AssembledContext,
    variables: dict[str, str | None],
    kind: Literal['section', 'context'],
) -> str:
    """Interpolate one section or context, attributing diagnostics to its owner."""
    text = input_.text
    parts: list[str] = []
    last = 0
    open_ = text.find('{{')
    while open_ >= 0:
        group = _GROUP_AT.match(text, open_)
        if group is None:
            # A later closing brace makes this malformed; otherwise literal.
            if text.find('}}', open_ + 2) >= 0:
                raise ValueError(
                    'malformed prompt variable reference at '
                    f'"{text[open_:open_ + 16]}…" in {kind} "{input_.name}" '
                    '(references are complete simple {{name}} groups)'
                )
            parts.append(text[last:open_ + 2])
            last = open_ + 2
        else:
            # `{{}}` yields an empty name and follows the malformed path.
            name = group.group(1)
            if not _VARIABLE_NAME.match(name):
                raise ValueError(
                    f'malformed prompt variable reference "{{{{{name}}}}}" '
                    f'in {kind} "{input_.name}" (variable names match '
                    f'/{_VARIABLE_NAME.pattern}/)'
                )
            if name not in variables:
                known = ', '.join(variables) or '(none)'
                raise ValueError(
                    f'unknown prompt variable "{{{{{name}}}}}" in {kind} '
                    f'"{input_.name}"; registered variables: {known}'
                )
            value = variables[name]
            if value is None:
                raise ValueError(
                    f'prompt variable "{{{{{name}}}}}" has no value for this '
                    f'assembly ({kind} "{input_.name}")'
                )
            parts.append(text[last:open_])
            parts.append(value)
            last = open_ + len(group.group(0))
        open_ = text.find('{{', last)
    parts.append(text[last:])
    return ''.join(parts)


#: One tool-schema provider stored in a prompt layer.
ToolProvider = Callable[[AssembleContext], ToolProviderResult]

#: One prompt-variable provider stored in a prompt layer.
VariableProvider = Callable[[AssembleContext], str | None]


class PromptLayer:
    """All prompt registrations owned by one global or scoped layer."""

    def __init__(self, scope: ScopeKey | None) -> None:
        if scope is None:
            section_hint = (
                ' (for a per-agent override, register through that agent\'s '
                '`agent.ctx` instead)'
            )
            variable_hint = (
                ' (for a per-agent value, register through that agent\'s '
                '`agent.ctx` instead)'
            )
        else:
            section_hint = ' in this scope'
            variable_hint = ' in this scope'
        self.sections: NamedEntries[PromptSection] = NamedEntries(
            lambda name: ValueError(
                f'prompt section "{name}" is already registered{section_hint}'
            )
        )
        self.contexts: NamedEntries[PromptContext] = NamedEntries(
            lambda name: ValueError(
                f'prompt context "{name}" is already registered{section_hint}'
            )
        )
        self.runtime_context_suppressors: AnonymousEntries[bool] = (
            AnonymousEntries()
        )
        self.tool_providers: AnonymousEntries[ToolProvider] = AnonymousEntries()
        self.variables: NamedEntries[VariableProvider] = NamedEntries(
            lambda name: ValueError(
                f'prompt variable "{name}" is already registered{variable_hint}'
            )
        )

    def is_empty(self) -> bool:
        """Whether this layer owns no prompt registrations."""
        return (
            self.sections.is_empty()
            and self.contexts.is_empty()
            and self.runtime_context_suppressors.is_empty()
            and self.tool_providers.is_empty()
            and self.variables.is_empty()
        )


def _check_finite_order(kind: str, name: str, order: float) -> None:
    """Reject non-numeric and non-finite orders (TypeScript Number.isFinite)."""
    if isinstance(order, bool) or not isinstance(order, (int, float)):
        raise TypeError(f'prompt {kind} "{name}" order must be a finite number')
    if not math.isfinite(order):
        raise TypeError(f'prompt {kind} "{name}" order must be a finite number')


class SystemPrompt(Service):
    """Registry service for the prompt inputs assembled before each model step."""

    #: Cordis loader config hook: raw mapping -> validated ``Config``.
    Config = staticmethod(_resolve_config)

    def __init__(
        self,
        ctx: Context,
        config: _ConfigData | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ctx, 'systemPrompt')
        resolved = _resolve_config(config)
        self._layers: ScopedLayers[PromptLayer] = ScopedLayers(
            PromptLayer,
            lambda: self.ctx.emit('system-prompt/change'),
        )
        self._tool_order = _validate_tool_order(resolved.tool_order)
        # Keep harness-owned openers independent of the selected loop plugin.
        if resolved.include_harness_identity:
            self.section(PromptSection(
                name='harness:identity',
                order=-100,
                text='You are an AI agent powered by DeepSeek Harness.',
            ))
        self.section(PromptSection(
            name=PERSONA_SECTION,
            order=PERSONA_ORDER,
            text=resolved.persona,
        ))
        if not resolved.include_runtime_context:
            self.suppress_runtime_context()

    def section(self, section: PromptSection) -> EffectDisposer:
        """Register an ordered prompt section in the calling context's scope.

        A scoped section shadows a global section with the same name;
        duplicates within one layer and non-finite orders raise.
        Registration and disposal emit ``system-prompt/change``.

        :param section: the section to register.
        :return: the exact Cordis effect disposer.
        """
        _check_finite_order('section', section.name, section.order)
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.sections.insert(section.name, section),
            'systemPrompt.section()',
        )

    def context(self, context: PromptContext) -> EffectDisposer:
        """Register ordered dynamic context in the calling context's scope.

        Scoped entries shadow global entries with the same name.

        :param context: the context contribution to register.
        :return: the exact Cordis effect disposer.
        """
        _check_finite_order('context', context.name, context.order)
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.contexts.insert(context.name, context),
            'systemPrompt.context()',
        )

    def suppress_runtime_context(self) -> EffectDisposer:
        """Suppress every dynamic runtime-context contribution in the calling
        context's scope without changing the services that own or enforce
        those facts. Multiple suppressors remain independently disposable.

        :return: the exact Cordis effect disposer.
        """
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.runtime_context_suppressors.append(True),
            'systemPrompt.suppress_runtime_context()',
        )

    def tools(self, provider: ToolProvider) -> EffectDisposer:
        """Register a tool-schema provider in the calling context's scope.

        Global and matching scoped providers both contribute; returning the
        reserved ``TOOL_ORDER_REST`` name makes assembly fail.

        :param provider: evaluated for each assembly with its context.
        :return: the exact Cordis effect disposer.
        """
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.tool_providers.append(provider),
            'systemPrompt.tools()',
        )

    def variable(self, name: str, provider: VariableProvider) -> EffectDisposer:
        """Register a prompt variable in the calling context's scope.

        Scoped values shadow globals; invalid or duplicate names raise. A
        provider may return ``None``, but rendering a section that
        references that value then fails.

        :param name: the ``[a-z][a-z0-9_]*`` reference name.
        :param provider: evaluated for each assembly.
        :return: the exact Cordis effect disposer.
        """
        if not _VARIABLE_NAME.match(name):
            raise ValueError(
                f'invalid prompt variable name "{name}" '
                f'(must match /{_VARIABLE_NAME.pattern}/)'
            )
        return self._layers.effect(
            self.ctx,
            lambda layer: layer.variables.insert(name, provider),
            'systemPrompt.variable()',
        )

    async def assemble(
        self,
        context: AssembleContext | None = None,
    ) -> PromptAssembly:
        """Assemble providers, order canonically, then run the waterfall.

        Scoped sections and variables shadow globals. The returned waterfall
        value is authoritative except that an effective complete section is
        restored afterwards as the sole prompt section, and suppressed
        runtime context stays empty.

        :param context: the optional scope and plugin-defined assembly fields.
        :return: the post-waterfall assembly with any complete prompt enforced.
        """
        if context is None:
            context = AssembleContext()
        scope = context.scope
        scope_layers = self._layers.chain_layers(scope)
        runtime_context_suppressed = (
            not self._layers.global_.runtime_context_suppressors.is_empty()
            or any(
                not layer.runtime_context_suppressors.is_empty()
                for layer in scope_layers
            )
        )
        # Scoped variables shadow globals.
        variables: dict[str, str | None] = {}
        for name, provider in self._layers.global_.variables.entries():
            variables[name] = provider(context)
        # Scope-chain variables, farthest first: the nearest scope wins a name.
        for layer in scope_layers:
            for name, provider in layer.variables.entries():
                variables[name] = provider(context)
        # Scoped sections shadow globals before the stable order sort.
        section_by_name = self._layers.merge(scope, lambda layer: layer.sections)
        context_by_name = self._layers.merge(scope, lambda layer: layer.contexts)
        # Validate order against pre-restriction names while collecting
        # visible schemas.
        providers: list[ToolProvider] = [
            *self._layers.global_.tool_providers.values(),
            *(
                tool_provider
                for layer in scope_layers
                for tool_provider in layer.tool_providers.values()
            ),
        ]
        collected: list[ToolSchema] = []
        known_names: set[str] = set()
        for tool_provider in providers:
            provided = tool_provider(context)
            schemas = [
                ToolSchema(
                    name=tool.name,
                    description=tool.description,
                    parameters=copy.deepcopy(tool.parameters),
                )
                for tool in provided.schemas
            ]
            accepted = (
                provided.known_names
                if provided.known_names is not None
                else [tool.name for tool in schemas]
            )
            collected.extend(schemas)
            known_names.update(accepted)
        section_definitions = sorted(
            section_by_name.values(), key=lambda section: section.order
        )
        complete_sections = [
            section for section in section_definitions if section.complete
        ]
        if len(complete_sections) > 1:
            names = ', '.join(
                json.dumps(section.name) for section in complete_sections
            )
            raise ValueError(
                f'multiple complete prompt sections are active: {names}'
            )
        complete_section: AssembledSection | None = None
        sections: list[AssembledSection] = []
        for section in section_definitions:
            assembled = AssembledSection(
                name=section.name,
                text=(
                    section.text(context)
                    if callable(section.text)
                    else section.text
                ),
            )
            if section.complete:
                complete_section = AssembledSection(assembled.name, assembled.text)
            sections.append(assembled)
        assembly = PromptAssembly(
            sections=sections,
            contexts=(
                []
                if runtime_context_suppressed
                else [
                    AssembledContext(
                        name=entry.name,
                        text=(
                            entry.text(context)
                            if callable(entry.text)
                            else entry.text
                        ),
                    )
                    for entry in sorted(
                        context_by_name.values(), key=lambda entry: entry.order
                    )
                ]
            ),
            tools=_order_tools(collected, self._tool_order, known_names),
            variables=variables,
        )
        transformed: PromptAssembly = await maybe_await(self.ctx.waterfall(
            scope_target(self, scope),
            'system-prompt/assemble',
            assembly,
            context,
            lambda *_args: assembly,
        ))
        if complete_section is None and not runtime_context_suppressed:
            return transformed
        result = copy.copy(transformed)
        result.sections = (
            transformed.sections if complete_section is None else [complete_section]
        )
        result.contexts = (
            [] if runtime_context_suppressed else transformed.contexts
        )
        return result
