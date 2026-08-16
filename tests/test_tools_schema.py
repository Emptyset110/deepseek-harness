"""Behavior tests for the author-facing schema DSL and ``define_tool``.

Ported-semantics tests for the Python port of deepseek-harness
``packages/core/tools/src/schema.ts`` (MIT): DSL compilation to the raw
subset, author errors, argument validation, and the typed definition helper.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydsh.tools import (
    DefineToolOutput,
    JsonSchemaError,
    ToolArgsError,
    define_tool,
    parameter_schema_spec_to_json_schema,
    validate_args,
    value_schema_spec_to_json_schema,
)


def author_violations(spec: Any, expected: list[str]) -> None:
    with pytest.raises(JsonSchemaError) as exc_info:
        parameter_schema_spec_to_json_schema(spec)
    assert exc_info.value.code == 'UNSUPPORTED_SCHEMA'
    assert exc_info.value.violations == expected


# --- DSL compilation ----------------------------------------------------------


def test_parameter_map_compiles_with_required() -> None:
    schema = parameter_schema_spec_to_json_schema({
        'cmd': {'type': 'string', 'required': True, 'description': 'the command'},
        'verbose': {'type': 'boolean'},
    })
    assert schema == {
        'type': 'object',
        'properties': {
            'cmd': {'type': 'string', 'description': 'the command'},
            'verbose': {'type': 'boolean'},
        },
        'required': ['cmd'],
    }
    # the author-side required annotation is collected, never copied
    assert 'required' not in schema['properties']['cmd']


def test_parameter_map_without_required_omits_key() -> None:
    schema = parameter_schema_spec_to_json_schema({'verbose': {'type': 'boolean'}})
    assert schema == {
        'type': 'object',
        'properties': {'verbose': {'type': 'boolean'}},
    }


def test_nested_object_and_openness() -> None:
    schema = value_schema_spec_to_json_schema({
        'type': 'object',
        'additionalProperties': False,
        'properties': {'port': {'type': 'integer', 'required': True}},
        'title': 'opts',
    })
    assert schema == {
        'type': 'object',
        'title': 'opts',
        'additionalProperties': False,
        'properties': {'port': {'type': 'integer'}},
        'required': ['port'],
    }


def test_json_node_becomes_annotation_only() -> None:
    assert value_schema_spec_to_json_schema({'type': 'json', 'description': 'any'}) == {
        'description': 'any',
    }


def test_one_of_compilation() -> None:
    schema = value_schema_spec_to_json_schema({
        'oneOf': [{'type': 'string'}, {'type': 'null'}],
        'description': 'maybe',
    })
    assert schema == {
        'oneOf': [{'type': 'string'}, {'type': 'null'}],
        'description': 'maybe',
    }


def test_scalar_literals_compile() -> None:
    assert value_schema_spec_to_json_schema({
        'type': 'string', 'enum': ['a', 'b'], 'const': 'a',
    }) == {'type': 'string', 'enum': ['a', 'b'], 'const': 'a'}


# --- author errors -------------------------------------------------------------


def test_unknown_author_key() -> None:
    author_violations(
        {'cmd': {'type': 'string', 'minLength': 2}},
        ['parameters.cmd.minLength is not supported by the value schema DSL'],
    )


def test_required_must_be_true() -> None:
    author_violations(
        {'cmd': {'type': 'string', 'required': False}},
        ['parameters.cmd.required must be true when present'],
    )


def test_non_record_nodes() -> None:
    author_violations([], ['parameters must be an object of value schemas'])
    author_violations(
        {'cmd': 'string'}, ['parameters.cmd must be a value schema object'],
    )


def test_circular_specs_reject() -> None:
    spec: dict = {}
    spec['self'] = spec
    author_violations(spec, ['parameters.self is circular'])


def test_object_openness_is_mandatory() -> None:
    author_violations(
        {'opts': {'type': 'object'}},
        ['parameters.opts.additionalProperties must be explicitly true or false'],
    )


def test_one_of_author_errors() -> None:
    author_violations(
        {'v': {'oneOf': [{'type': 'string'}], 'type': 'string'}},
        ['parameters.v cannot declare both type and oneOf'],
    )
    author_violations(
        {'v': {'oneOf': 'string'}},
        ['parameters.v.oneOf must be an array of at least two value schemas'],
    )
    # length enforcement lives in the raw-subset assertion
    author_violations(
        {'v': {'oneOf': [{'type': 'string'}]}},
        ['schema.properties.v.oneOf must be an array of at least two schemas'],
    )


def test_bad_type_author_error() -> None:
    with pytest.raises(JsonSchemaError) as exc_info:
        value_schema_spec_to_json_schema({'type': 'date'})
    assert exc_info.value.violations == [
        'schema.type must be string/number/integer/boolean/null/array/object/'
        'json, or use oneOf',
    ]


def test_compiled_output_must_stay_in_subset() -> None:
    # annotation values must be lossless JSON even in the DSL
    with pytest.raises(JsonSchemaError) as exc_info:
        value_schema_spec_to_json_schema({'type': 'string', 'default': object()})
    assert exc_info.value.violations == [
        'schema.default annotation must be lossless JSON data',
    ]


# --- validate_args --------------------------------------------------------------


def test_validate_args() -> None:
    spec = {'cmd': {'type': 'string', 'required': True}, 'count': {'type': 'integer'}}
    assert validate_args(spec, {'cmd': 'ls'}) == []
    assert validate_args(spec, {}) == ['missing required property "cmd"']
    assert validate_args(spec, {'cmd': 1, 'count': 'x'}) == [
        '"cmd" must be a string',
        '"count" must be an integer',
    ]


# --- define_tool -----------------------------------------------------------------


def echo_render(args: Any, value: Any) -> list:
    return [{'type': 'text', 'text': str(value)}]


def make_echo_tool(**overrides: Any) -> Any:
    async def execute(args: Any, exec: Any) -> Any:
        return args['cmd']

    options = {
        'name': 'echo',
        'description': 'echo the command',
        'parameters': {'cmd': {'type': 'string', 'required': True}},
        'output': DefineToolOutput(schema={'type': 'string'}, render=echo_render),
        'execute': execute,
    }
    options.update(overrides)
    return define_tool(**options)


async def test_define_tool_validates_args_before_execute() -> None:
    tool = make_echo_tool()
    with pytest.raises(ToolArgsError) as exc_info:
        await tool.execute({'cmd': 1}, None)
    assert exc_info.value.code == 'INVALID_ARGS'
    assert exc_info.value.violations == ['"cmd" must be a string']
    assert str(exc_info.value) == 'invalid arguments: "cmd" must be a string'
    assert await tool.execute({'cmd': 'ls'}, None) == 'ls'


def test_define_tool_compiles_schemas() -> None:
    tool = make_echo_tool()
    assert tool.parameters == {
        'type': 'object',
        'properties': {'cmd': {'type': 'string'}},
        'required': ['cmd'],
    }
    assert tool.output.schema == {'type': 'string'}


def test_define_tool_rejects_bad_timeout() -> None:
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='timeoutMs must be a positive finite'):
            make_echo_tool(timeout_ms=bad)


def test_define_tool_concurrency_classifier_validates_softly() -> None:
    tool = make_echo_tool(is_concurrency_safe=lambda args: True)
    assert tool.is_concurrency_safe({'cmd': 'ls'}) is True
    # invalid arguments classify exclusive instead of raising
    assert tool.is_concurrency_safe({'cmd': 1}) is False


def test_define_tool_without_classifier() -> None:
    tool = make_echo_tool()
    assert tool.is_concurrency_safe is None
    assert tool.finalize_content is None
    assert tool.timeout_ms is None
    assert tool.output.presentation_meta is None
