"""Behavior tests for the enforced JSON Schema subset.

Ported-semantics tests for the Python port of deepseek-harness
``packages/core/tools/src/json-schema.ts`` (MIT): schema assertion
violations, object-root constraint, and value validation diagnostics.
"""

from __future__ import annotations

import pytest

from pydsh.tools import (
    JsonSchemaError,
    assert_object_json_schema,
    assert_supported_json_schema,
    validate_json_schema_value,
)


def assert_schema_violations(schema: object, expected: list[str]) -> None:
    with pytest.raises(JsonSchemaError) as exc_info:
        assert_supported_json_schema(schema)
    assert exc_info.value.code == 'UNSUPPORTED_SCHEMA'
    assert exc_info.value.violations == expected


# --- schema assertion -------------------------------------------------------


def test_valid_schemas_pass() -> None:
    assert_supported_json_schema({})
    assert_supported_json_schema({'description': 'any JSON'})
    assert_supported_json_schema({
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'description': 'n'},
            'tags': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['name'],
        'additionalProperties': False,
    })
    assert_supported_json_schema({
        'oneOf': [{'type': 'string'}, {'type': 'null'}],
    })
    assert_supported_json_schema({'type': 'integer', 'enum': [1, 2.0], 'const': 1})
    assert_supported_json_schema({'type': 'boolean', 'const': False})


def test_non_record_and_circular() -> None:
    assert_schema_violations([], ['schema must be a schema object'])
    node: dict = {'type': 'object', 'properties': {}}
    node['properties']['self'] = node
    assert_schema_violations(node, ['schema.properties.self is circular'])


def test_unknown_and_annotation_keywords() -> None:
    assert_schema_violations(
        {'type': 'string', 'minLength': 2},
        ['schema.minLength is not a supported keyword (subset: '
         'type/oneOf/properties/required/additionalProperties/items/enum/const '
         '+ annotations)'],
    )
    assert_schema_violations(
        {'default': object()},
        ['schema.default annotation must be lossless JSON data'],
    )
    assert_schema_violations(
        {'description': 3, 'title': None},
        ['schema.description must be a string', 'schema.title must be a string'],
    )


def test_type_and_one_of_exclusion() -> None:
    assert_schema_violations(
        {'type': 'string', 'oneOf': [{'type': 'string'}, {'type': 'null'}]},
        ['schema cannot declare both type and oneOf'],
    )
    assert_schema_violations(
        {'properties': {}, 'enum': ['a']},
        [
            'schema.properties requires type or oneOf',
            'schema.enum requires type or oneOf',
        ],
    )


def test_one_of_rules() -> None:
    assert_schema_violations(
        {'oneOf': [{'type': 'string'}]},
        ['schema.oneOf must be an array of at least two schemas'],
    )
    assert_schema_violations(
        {'oneOf': [{'type': 'string'}, {'type': 'null'}], 'const': 'a'},
        ['schema.const is not supported beside oneOf'],
    )
    assert_schema_violations(
        {'oneOf': [{'type': 'string'}, {'type': 'nope'}]},
        ['schema.oneOf[1].type must be one of '
         'object/array/string/number/integer/boolean/null'],
    )


def test_type_keyword_rules() -> None:
    assert_schema_violations(
        {'type': ['string', 'null']},
        ['schema.type must be a single type string '
         '(type arrays are not supported)'],
    )
    assert_schema_violations(
        {'type': 'date'},
        ['schema.type must be one of '
         'object/array/string/number/integer/boolean/null'],
    )
    assert_schema_violations(
        {'type': 'string', 'items': {'type': 'string'}, 'required': []},
        [
            'schema.required is not supported on type "string"',
            'schema.items is not supported on type "string"',
        ],
    )


def test_object_keyword_rules() -> None:
    assert_schema_violations(
        {'type': 'object', 'properties': []},
        ['schema.properties must be an object of schemas'],
    )
    assert_schema_violations(
        {'type': 'object', 'required': 'name'},
        ['schema.required must be an array of strings'],
    )
    assert_schema_violations(
        {'type': 'object', 'properties': {'a': {'type': 'string'}},
         'required': ['a', 'x']},
        ['schema.required names "x" which is not in properties'],
    )
    assert_schema_violations(
        {'type': 'object', 'additionalProperties': 0},
        ['schema.additionalProperties must be a boolean'],
    )
    assert_schema_violations(
        {'type': 'object', 'properties': {'a': {'type': 'nope'}}},
        ['schema.properties.a.type must be one of '
         'object/array/string/number/integer/boolean/null'],
    )


def test_scalar_enum_const_rules() -> None:
    assert_schema_violations(
        {'type': 'string', 'enum': []},
        ['schema.enum must be a non-empty array of string values'],
    )
    assert_schema_violations(
        {'type': 'integer', 'enum': [1, 'a']},
        ['schema.enum must be a non-empty array of integer values'],
    )
    assert_schema_violations(
        {'type': 'boolean', 'const': 1},
        ['schema.const must be a boolean value'],
    )
    assert_schema_violations(
        {'type': 'string', 'enum': ['a', 'b'], 'const': 'c'},
        ['schema.const must be one of schema.enum when both are declared'],
    )
    # JS numeric identity: const 1 matches enum [1.0]
    assert_supported_json_schema({'type': 'number', 'enum': [1.0], 'const': 1})


def test_object_root_constraint() -> None:
    assert_object_json_schema({'type': 'object'})
    for schema in ({}, {'type': 'string'}):
        with pytest.raises(JsonSchemaError) as exc_info:
            assert_object_json_schema(schema)
        assert exc_info.value.violations == [
            'schema.type must be "object" (structured output is object-rooted)',
        ]


# --- value validation --------------------------------------------------------


def test_unconstrained_node_accepts_lossless_json() -> None:
    assert validate_json_schema_value({}, {'a': [1, 'x', None]}) == []
    assert validate_json_schema_value({}, {'a': object()}) == [
        '"value" must be a lossless JSON value',
    ]


def test_scalar_type_mismatches() -> None:
    assert validate_json_schema_value({'type': 'string'}, 1) == [
        '"value" must be a string',
    ]
    assert validate_json_schema_value({'type': 'number'}, '1') == [
        '"value" must be a number',
    ]
    assert validate_json_schema_value({'type': 'number'}, True) == [
        '"value" must be a number',
    ]
    assert validate_json_schema_value({'type': 'number'}, float('nan')) == [
        '"value" must be a finite JSON number',
    ]
    assert validate_json_schema_value({'type': 'integer'}, 1.5) == [
        '"value" must be an integer',
    ]
    assert validate_json_schema_value({'type': 'integer'}, 2.0) == []
    assert validate_json_schema_value({'type': 'boolean'}, 0) == [
        '"value" must be a boolean',
    ]
    assert validate_json_schema_value({'type': 'null'}, 0) == [
        '"value" must be null',
    ]


def test_scalar_enum_and_const() -> None:
    schema = {'type': 'string', 'enum': ['a', 'b']}
    assert validate_json_schema_value(schema, 'a') == []
    assert validate_json_schema_value(schema, 'c') == [
        '"value" must be one of ["a","b"]',
    ]
    assert validate_json_schema_value({'type': 'null', 'const': None}, None) == []
    assert validate_json_schema_value({'type': 'integer', 'const': 3}, 4) == [
        '"value" must be 3',
    ]
    # JS numeric identity: integer 3 matches const 3.0
    assert validate_json_schema_value({'type': 'integer', 'const': 3.0}, 3) == []


def test_object_validation_and_path_labels() -> None:
    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'config': {
                'type': 'object',
                'properties': {'port': {'type': 'integer'}},
                'additionalProperties': False,
            },
        },
        'required': ['name'],
    }
    assert validate_json_schema_value(schema, []) == ['"value" must be an object']
    assert validate_json_schema_value(schema, {}) == [
        'missing required property "value.name"',
    ]
    assert validate_json_schema_value(
        schema, {'name': 'x', 'config': {'port': 'no', 'extra': 1}},
    ) == [
        '"value.config.port" must be an integer',
        '"value.config.extra" is not a declared property '
        '(additionalProperties: false)',
    ]
    assert validate_json_schema_value(
        schema, {'name': 'x', 'unknown': 1},
    ) == []  # open by default
    # the empty root path renders as "arguments" (the parameter validator)
    assert validate_json_schema_value(schema, {}, '') == [
        'missing required property "name"',
    ]
    assert validate_json_schema_value({'type': 'string'}, 1, '') == [
        '"arguments" must be a string',
    ]


def test_object_lossless_tail_check() -> None:
    schema = {'type': 'object', 'properties': {'a': {}}}
    # a child violation wins over the container's own lossless check
    assert validate_json_schema_value(schema, {'a': object()}) == [
        '"value.a" must be a lossless JSON value',
    ]
    # undeclared keys are open, but the whole object must stay lossless
    assert validate_json_schema_value(schema, {'a': None, 'b': object()}) == [
        '"value" must be a lossless JSON object',
    ]


def test_array_validation() -> None:
    schema = {'type': 'array', 'items': {'type': 'integer'}}
    assert validate_json_schema_value(schema, 'x') == ['"value" must be an array']
    assert validate_json_schema_value(schema, [1, 'a', 3]) == [
        '"value[1]" must be an integer',
    ]
    assert validate_json_schema_value({'type': 'array'}, [object()]) == [
        '"value" must be a dense lossless JSON array',
    ]


def test_one_of_exactly_one() -> None:
    schema = {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}
    assert validate_json_schema_value(schema, 'a') == []
    assert validate_json_schema_value(schema, 1) == []
    assert validate_json_schema_value(schema, True) == [
        '"value" must match exactly one oneOf branch (matched 0)',
    ]
    ambiguous = {'oneOf': [{'type': 'integer'}, {'type': 'number'}]}
    assert validate_json_schema_value(ambiguous, 1) == [
        '"value" must match exactly one oneOf branch (matched 2)',
    ]
