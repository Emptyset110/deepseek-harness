"""Enforced JSON Schema subset shared by tool parameters and outputs.

Ported from deepseek-harness packages/core/tools (MIT), ``src/json-schema.ts``.

The subset accepts any JSON root, an annotation-only schema for
unconstrained JSON, one scalar ``type``, object
``properties``/``required``/boolean ``additionalProperties``, array
``items``, type-correct scalar ``enum``/``const``, and exact-one ``oneOf``.
Unsupported or misplaced keywords reject rather than being accepted without
enforcement. Consumers that require an object root apply
:func:`assert_object_json_schema` before accepting input.

Python mappings:

- A "plain JSON record from any JavaScript realm" is an exact ``dict``; a
  "dense undecorated array" is an exact ``list``. Realm, prototype, getter,
  and sparse-array checks have no Python counterpart.
- JavaScript ``number`` is ``int`` or ``float`` excluding ``bool``,
  non-finite floats, and negative zero (the same boundary
  :mod:`pydsh.session.json` enforces); ``1`` and ``1.0`` compare equal,
  matching JS numeric identity.
- Traversal is recursive rather than iterative: nesting depth is bounded by
  the interpreter recursion limit (same trade-off as
  :mod:`pydsh.session.json`).
- Hostile-value containment (the ``catches`` frame flag) is dropped:
  validation here only touches exact ``dict``/``list``/scalar types, which
  cannot raise.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal, TypeAlias, TypedDict, TypeGuard, cast

from pydsh.llm import HarnessError
from pydsh.session.json import is_json_value

__all__ = [
    'JsonSchemaError',
    'JsonSchemaNode',
    'JsonSchemaScalar',
    'JsonSchemaType',
    'ObjectJsonSchema',
    'assert_object_json_schema',
    'assert_supported_json_schema',
    'is_json_schema_record',
    'is_plain_json_array',
    'validate_json_schema_value',
]

#: Scalar JSON values supported by ``enum`` and ``const``.
JsonSchemaScalar: TypeAlias = 'str | int | float | bool | None'

#: Single-type keywords accepted by the enforced subset.
JsonSchemaType = Literal[
    'object', 'array', 'string', 'number', 'integer', 'boolean', 'null',
]


class JsonSchemaNode(TypedDict, total=False):
    """One raw JSON Schema node in the enforced subset.

    The optional fields express the external wire schema;
    :func:`assert_supported_json_schema` rejects invalid combinations before
    a caller treats the node as trusted. Key presence is significant: an
    absent ``additionalProperties`` follows JSON Schema's open default; omit
    ``type`` with no constraints for any JSON value, or use ``oneOf``.
    """

    #: One scalar/container type keyword.
    type: JsonSchemaType
    #: Exactly one branch must validate; at least two branches are required.
    oneOf: list[JsonSchemaNode]
    #: Nested property schemas (``type: 'object'`` only).
    properties: dict[str, JsonSchemaNode]
    #: Required property names; each must appear in ``properties``.
    required: list[str]
    #: ``False`` rejects undeclared keys; absent/``True`` is the open default.
    additionalProperties: bool
    #: Item schema (``type: 'array'`` only); absent accepts any JSON item.
    items: JsonSchemaNode
    #: Allowed values for a scalar node.
    enum: list[JsonSchemaScalar]
    #: The single allowed value for a scalar node.
    const: JsonSchemaScalar
    #: Annotation, ignored for validation.
    description: str
    #: Annotation, ignored for validation.
    title: str
    #: Annotation, ignored for validation but required to be lossless JSON.
    default: Any
    #: Annotation, ignored for validation but required to be lossless JSON.
    examples: Any


class ObjectJsonSchema(JsonSchemaNode):
    """A consumer-constrained object-rooted schema (``type: 'object'``).

    Typed identically to :class:`JsonSchemaNode` — Python's TypedDict cannot
    redeclare an inherited optional key as required, so the narrower root is
    enforced by :func:`assert_object_json_schema` instead.
    """


class JsonSchemaError(HarnessError):
    """Thrown when a raw schema falls outside the enforced subset.

    ``violations`` lists every offending path instead of stopping at the
    first author error.
    """

    def __init__(self, violations: list[str]) -> None:
        super().__init__(
            f'unsupported JSON schema: {"; ".join(violations)}',
            'UNSUPPORTED_SCHEMA',
        )
        #: Individual schema violations in walk order.
        self.violations = violations


_CONSTRAINT_KEYWORDS = {
    'type', 'oneOf', 'properties', 'required', 'additionalProperties',
    'items', 'enum', 'const',
}
_ANNOTATION_KEYWORDS = {'description', 'title', 'default', 'examples'}
_SCHEMA_TYPES = ('object', 'array', 'string', 'number', 'integer', 'boolean', 'null')

#: Keywords that are invalid beside ``oneOf`` (or without ``type``/``oneOf``).
_ONE_OF_SIBLING_KEYWORDS = (
    'properties', 'required', 'additionalProperties', 'items', 'enum', 'const',
)

#: Keyword -> the schema types it is supported on.
_ALLOWED_FOR = {
    'properties': ('object',),
    'required': ('object',),
    'additionalProperties': ('object',),
    'items': ('array',),
    'enum': ('string', 'number', 'integer', 'boolean', 'null'),
    'const': ('string', 'number', 'integer', 'boolean', 'null'),
}


def is_json_schema_record(value: Any) -> TypeGuard[dict[str, Any]]:
    """Test for an ordinary schema record (an exact ``dict``)."""
    return type(value) is dict


def is_plain_json_array(value: Any) -> TypeGuard[list[Any]]:
    """Test for a dense ordinary array (an exact ``list``)."""
    return type(value) is list


def _is_negative_zero(value: float) -> bool:
    return value == 0 and math.copysign(1.0, value) < 0


def _is_json_number(value: Any) -> bool:
    """Lossless finite JSON number, excluding booleans and negative zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value) and not _is_negative_zero(value)


def _is_integer_number(value: Any) -> bool:
    """JSON number with an integer value (JS ``Number.isInteger``)."""
    if isinstance(value, int):
        return True
    return bool(value.is_integer())


def _scalar_matches(type_: str, value: Any) -> bool:
    """Whether a scalar is valid for one declared schema type."""
    if type_ == 'string':
        return isinstance(value, str)
    if type_ == 'number':
        return _is_json_number(value)
    if type_ == 'integer':
        return _is_json_number(value) and _is_integer_number(value)
    if type_ == 'boolean':
        return isinstance(value, bool)
    return value is None


def _json_scalar_equal(a: Any, b: Any) -> bool:
    """JS strict equality over JSON scalars (``1 === 1.0``, ``1 !== true``)."""
    a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_num or b_num:
        return a_num and b_num and a == b
    return type(a) is type(b) and a == b


def _json_stringify(value: Any) -> str:
    """Compact JSON text for diagnostics (JS ``JSON.stringify``)."""
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def _check_schema_node(
    node: Any,
    path: str,
    violations: list[str],
    seen: list[Any],
) -> None:
    """Collect every violation for one raw schema subtree."""
    if not is_json_schema_record(node):
        violations.append(f'{path} must be a schema object')
        return
    if any(known is node for known in seen):
        violations.append(f'{path} is circular')
        return
    seen.append(node)
    try:
        for key in node:
            if key in _CONSTRAINT_KEYWORDS:
                continue
            if key in _ANNOTATION_KEYWORDS:
                if not is_json_value(node[key]):
                    violations.append(
                        f'{path}.{key} annotation must be lossless JSON data'
                    )
                continue
            violations.append(
                f'{path}.{key} is not a supported keyword (subset: '
                'type/oneOf/properties/required/additionalProperties/items/'
                'enum/const + annotations)'
            )
        if 'description' in node and not isinstance(node['description'], str):
            violations.append(f'{path}.description must be a string')
        if 'title' in node and not isinstance(node['title'], str):
            violations.append(f'{path}.title must be a string')

        has_type = 'type' in node
        has_one_of = 'oneOf' in node
        if has_type and has_one_of:
            violations.append(f'{path} cannot declare both type and oneOf')
            return
        if not has_type and not has_one_of:
            for key in _ONE_OF_SIBLING_KEYWORDS:
                if key in node:
                    violations.append(f'{path}.{key} requires type or oneOf')
            return

        if has_one_of:
            one_of = node['oneOf']
            if not is_plain_json_array(one_of) or len(one_of) < 2:
                violations.append(
                    f'{path}.oneOf must be an array of at least two schemas'
                )
            else:
                for index, branch in enumerate(one_of):
                    _check_schema_node(
                        branch, f'{path}.oneOf[{index}]', violations, seen,
                    )
            for key in _ONE_OF_SIBLING_KEYWORDS:
                if key in node:
                    violations.append(
                        f'{path}.{key} is not supported beside oneOf'
                    )
            return

        type_ = node['type']
        if not isinstance(type_, str) or type_ not in _SCHEMA_TYPES:
            violations.append(
                f'{path}.type must be a single type string '
                '(type arrays are not supported)'
                if isinstance(type_, list)
                else f'{path}.type must be one of {"/".join(_SCHEMA_TYPES)}'
            )
            return
        for key, types in _ALLOWED_FOR.items():
            if key in node and type_ not in types:
                violations.append(
                    f'{path}.{key} is not supported on type "{type_}"'
                )

        if type_ == 'object':
            properties = node.get('properties')
            if 'properties' in node:
                if not is_json_schema_record(properties):
                    violations.append(
                        f'{path}.properties must be an object of schemas'
                    )
                else:
                    for key, child in properties.items():
                        _check_schema_node(
                            child, f'{path}.properties.{key}', violations, seen,
                        )
            if 'required' in node:
                required = node['required']
                if not is_plain_json_array(required) or any(
                    not isinstance(entry, str) for entry in required
                ):
                    violations.append(f'{path}.required must be an array of strings')
                else:
                    declared = (
                        properties if is_json_schema_record(properties) else {}
                    )
                    for key in required:
                        if key not in declared:
                            violations.append(
                                f'{path}.required names "{key}" which is not '
                                'in properties'
                            )
            if 'additionalProperties' in node and not isinstance(
                node['additionalProperties'], bool
            ):
                violations.append(
                    f'{path}.additionalProperties must be a boolean'
                )
        elif type_ == 'array':
            if 'items' in node:
                _check_schema_node(
                    node['items'], f'{path}.items', violations, seen,
                )
        else:
            has_enum = 'enum' in node
            allowed = node.get('enum')
            enum_entries: list[Any] = (
                allowed if is_plain_json_array(allowed) else []
            )
            enum_valid = (
                is_plain_json_array(allowed)
                and len(allowed) > 0
                and all(_scalar_matches(type_, entry) for entry in allowed)
            )
            if has_enum and not enum_valid:
                violations.append(
                    f'{path}.enum must be a non-empty array of {type_} values'
                )
            if 'const' in node:
                declared_const = node['const']
                if not _scalar_matches(type_, declared_const):
                    violations.append(f'{path}.const must be a {type_} value')
                elif enum_valid and not any(
                    _json_scalar_equal(entry, declared_const)
                    for entry in enum_entries
                ):
                    violations.append(
                        f'{path}.const must be one of {path}.enum when both '
                        'are declared'
                    )
    finally:
        seen.pop()


def assert_supported_json_schema(schema: Any) -> None:
    """Assert that an arbitrary raw schema uses only the enforced subset.

    Annotation-only schemas are accepted as the standard unconstrained-JSON
    form; callers that require an object root use
    :func:`assert_object_json_schema`.

    :param schema: untrusted raw JSON Schema.
    :raises JsonSchemaError: listing every offending path.
    """
    violations: list[str] = []
    _check_schema_node(schema, 'schema', violations, [])
    if violations:
        raise JsonSchemaError(violations)


def assert_object_json_schema(schema: Any) -> None:
    """Assert the enforced subset plus the object-root constraint.

    :param schema: untrusted caller-supplied schema.
    :raises JsonSchemaError: listing every offending path.
    """
    violations: list[str] = []
    _check_schema_node(schema, 'schema', violations, [])
    if not violations and (
        not is_json_schema_record(schema)
        or 'type' not in schema
        or schema['type'] != 'object'
    ):
        violations.append(
            'schema.type must be "object" (structured output is object-rooted)'
        )
    if violations:
        raise JsonSchemaError(violations)


def _diagnostic_path(path: str) -> str:
    """Root-aware diagnostic path for the parameter validator's sentinel."""
    return 'arguments' if path == '' else path


def _property_path(path: str, key: str) -> str:
    """Append one object property without a leading dot at an implicit root."""
    return key if path == '' else f'{path}.{key}'


def _check_scalar_value(node: dict[str, Any], value: Any, path: str) -> list[str]:
    """Validate one scalar node after its primitive type check."""
    if 'enum' in node:
        allowed = node['enum']
        if not any(_json_scalar_equal(entry, value) for entry in allowed):
            return [
                f'"{_diagnostic_path(path)}" must be one of '
                f'{_json_stringify(allowed)}'
            ]
    if 'const' in node and not _json_scalar_equal(node['const'], value):
        return [
            f'"{_diagnostic_path(path)}" must be '
            f'{_json_stringify(node["const"])}'
        ]
    return []


def _check_value(node: dict[str, Any], value: Any, path: str) -> list[str]:
    """Validate one trusted schema/value pair, returning all violations."""
    if 'oneOf' in node:
        matches = sum(
            1 for branch in node['oneOf'] if not _check_value(branch, value, path)
        )
        if matches == 1:
            return []
        return [
            f'"{_diagnostic_path(path)}" must match exactly one oneOf branch '
            f'(matched {matches})'
        ]

    node_type = node.get('type')
    if node_type is None:
        if is_json_value(value):
            return []
        return [f'"{_diagnostic_path(path)}" must be a lossless JSON value']

    if node_type == 'object':
        if type(value) is not dict:
            return [f'"{_diagnostic_path(path)}" must be an object']
        violations: list[str] = []
        for key in node.get('required') or []:
            if key not in value:
                violations.append(
                    f'missing required property "{_property_path(path, key)}"'
                )
        properties = node.get('properties') or {}
        for key, child in properties.items():
            if key not in value:
                continue
            violations.extend(
                _check_value(child, value[key], _property_path(path, key))
            )
        if node.get('additionalProperties') is False:
            for key in value:
                if key not in properties:
                    violations.append(
                        f'"{_property_path(path, key)}" is not a declared '
                        'property (additionalProperties: false)'
                    )
        if violations:
            return violations
        if is_json_value(value):
            return []
        return [f'"{_diagnostic_path(path)}" must be a lossless JSON object']

    if node_type == 'array':
        if type(value) is not list:
            return [f'"{_diagnostic_path(path)}" must be an array']
        violations = []
        items = node.get('items')
        if items is not None:
            for index, entry in enumerate(value):
                violations.extend(_check_value(items, entry, f'{path}[{index}]'))
        if violations:
            return violations
        if is_json_value(value):
            return []
        return [
            f'"{_diagnostic_path(path)}" must be a dense lossless JSON array'
        ]

    if node_type == 'string':
        if not isinstance(value, str):
            return [f'"{_diagnostic_path(path)}" must be a string']
        return _check_scalar_value(node, value, path)
    if node_type == 'number':
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f'"{_diagnostic_path(path)}" must be a number']
        if not _is_json_number(value):
            return [f'"{_diagnostic_path(path)}" must be a finite JSON number']
        return _check_scalar_value(node, value, path)
    if node_type == 'integer':
        if not (_is_json_number(value) and _is_integer_number(value)):
            return [f'"{_diagnostic_path(path)}" must be an integer']
        return _check_scalar_value(node, value, path)
    if node_type == 'boolean':
        if not isinstance(value, bool):
            return [f'"{_diagnostic_path(path)}" must be a boolean']
        return _check_scalar_value(node, value, path)
    if node_type == 'null':
        if value is not None:
            return [f'"{_diagnostic_path(path)}" must be null']
        return _check_scalar_value(node, value, path)
    raise AssertionError(f'unreachable JsonSchemaType: {node_type!r}')


def validate_json_schema_value(
    schema: JsonSchemaNode,
    value: Any,
    path: str = 'value',
) -> list[str]:
    """Validate a candidate value against an asserted raw schema.

    The function is total for arbitrary values and returns path-qualified
    violations.

    :param schema: a schema accepted by :func:`assert_supported_json_schema`.
    :param value: the candidate JSON value.
    :param path: root label used in diagnostics.
    :return: all violations in walk order; empty means valid.
    """
    return _check_value(cast(dict[str, Any], schema), value, path)
