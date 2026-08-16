"""Request-header reconstruction over full ``request/header`` session events.

Ported from deepseek-harness packages/core/session (MIT),
``src/request-header.ts``.

Anyone holding a session log reconstructs the EpochHeader any request was
built under by taking the latest canonical snapshot; the loop uses the same
equality helper to avoid logging unchanged headers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .json import _deep_equal_json
from .types import EpochHeader, LlmCallConfig, SessionEvent, ToolSchema

__all__ = [
    'call_config_equals',
    'canonical_header',
    'fold_request_header',
    'header_equals',
]


def call_config_equals(a: LlmCallConfig, b: LlmCallConfig) -> bool:
    """Whether two call configurations carry the same JSON values.

    Minimal local stand-in for dsh-llm's ``callConfigEquals`` (deep
    structural equality); the llm port owns the precise field-wise version.
    """
    return _deep_equal_json(a, b)


def canonical_header(header: EpochHeader) -> EpochHeader:
    """Normalize a header to canonical form.

    An empty system prompt and empty tool list become absent fields, and
    ``adapterDefaults`` is kept only when it carries at least one ``True``
    marker, matching how requests are built. Logging, folding, and
    comparison use this one representation.

    :param header: the header to normalize (not mutated).
    :returns: the canonical header.
    """
    canonical = EpochHeader(config=header['config'])
    defaults = header.get('adapterDefaults')
    if defaults is not None and (
        defaults.get('reasoningEffort') is True
        or defaults.get('maxTokens') is True
    ):
        canonical['adapterDefaults'] = defaults
    system = header.get('system')
    if system:
        canonical['system'] = system
    tools = header.get('tools')
    if tools:
        canonical['tools'] = tools
    return canonical


def _same_schema(a: ToolSchema, b: ToolSchema) -> bool:
    """Canonical JSON equality for tool schemas assembled through one path."""
    return json.dumps(a) == json.dumps(b)


def header_equals(a: EpochHeader, b: EpochHeader) -> bool:
    """Field-wise equality over canonical headers.

    Tool schemas compare in order.

    :param a: one canonical header.
    :param b: the other.
    :returns: whether config, adapter defaults, system, and tools all match.
    """
    a_defaults = a.get('adapterDefaults') or {}
    b_defaults = b.get('adapterDefaults') or {}
    if (
        not call_config_equals(a['config'], b['config'])
        or a_defaults.get('reasoningEffort') != b_defaults.get('reasoningEffort')
        or a_defaults.get('maxTokens') != b_defaults.get('maxTokens')
        or a.get('system') != b.get('system')
    ):
        return False
    a_tools = a.get('tools') or []
    b_tools = b.get('tools') or []
    return len(a_tools) == len(b_tools) and all(
        _same_schema(tool, b_tools[index]) for index, tool in enumerate(a_tools)
    )


def fold_request_header(
    events: Sequence[SessionEvent], from_: EpochHeader | None = None
) -> EpochHeader | None:
    """Fold the header events of a log into the EpochHeader in force.

    Non-header events are skipped. This is the pure offline reconstruction
    path; the live session tracks the same fold incrementally.

    :param events: session events in log order.
    :param from_: a previously folded state to continue from.
    :returns: the latest canonical header, or None when none exists yet.
    """
    state = from_
    for event in events:
        if event['type'] == 'request/header':
            state = canonical_header(event['data']['header'])
    return state
