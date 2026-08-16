"""Author-facing value-schema DSL, compilation, and the ``define_tool`` helper.

Ported from deepseek-harness packages/core/tools (MIT), ``src/schema.ts``.

The DSL compiles implicit parameter maps and explicit value specs into the
enforced raw JSON Schema subset of :mod:`pydsh.tools.json_schema`. The
TypeScript compile-time inference types (``InferValue``/``InferArgs``) and
the presentation callbacks (``presentCall``/``presentResult``) are excluded
— the former has no Python equivalent, the latter belongs to the unported
presentation layer. Runtime compilation is recursive rather than an
explicit task stack: nesting depth is bounded by the interpreter recursion
limit (same trade-off as :mod:`pydsh.session.json`).
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, TypedDict, cast

from pydsh.llm import ContentBlock, HarnessError
from pydsh.session.json import JsonValue

from .json_schema import (
    JsonSchemaError,
    JsonSchemaNode,
    ObjectJsonSchema,
    assert_supported_json_schema,
    is_json_schema_record,
    is_plain_json_array,
    validate_json_schema_value,
)
from .runtime import (
    ToolDefinition,
    ToolExecution,
    ToolExecutionResult,
    ToolOutputDefinition,
)

__all__ = [
    'ArrayValueSchemaSpec',
    'BooleanValueSchemaSpec',
    'DefineToolOutput',
    'IntegerValueSchemaSpec',
    'JsonValueSchemaSpec',
    'NullValueSchemaSpec',
    'NumberValueSchemaSpec',
    'ObjectValueSchemaSpec',
    'OneOfValueSchemaSpec',
    'ParameterJsonSchema',
    'ParameterPropertySpec',
    'ParameterSchemaSpec',
    'StringValueSchemaSpec',
    'ToolArgsError',
    'ValueSchemaAnnotations',
    'ValueSchemaSpec',
    'define_tool',
    'parameter_schema_spec_to_json_schema',
    'validate_args',
    'value_schema_spec_to_json_schema',
]


class ValueSchemaAnnotations(TypedDict, total=False):
    """Annotation keys shared by every author-facing schema node."""

    #: Human-readable description projected into JSON Schema.
    description: str
    #: Human-readable title projected into JSON Schema.
    title: str
    #: Non-validating default annotation; it must be lossless JSON data.
    default: JsonValue
    #: Non-validating examples annotation; it must be lossless JSON data.
    examples: JsonValue


class StringValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """String value schema with type-correct literal constraints."""

    type: Literal['string']
    enum: list[str]
    const: str


class NumberValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Finite JSON-number schema with type-correct literal constraints."""

    type: Literal['number']
    enum: list[int | float]
    const: int | float


class IntegerValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Integer schema with type-correct literal constraints."""

    type: Literal['integer']
    enum: list[int]
    const: int


class BooleanValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Boolean value schema with type-correct literal constraints."""

    type: Literal['boolean']
    enum: list[bool]
    const: bool


class NullValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Null value schema."""

    type: Literal['null']
    const: None


class ArrayValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Array value schema; omitted ``items`` accepts any lossless JSON item."""

    type: Literal['array']
    items: ValueSchemaSpec


class ObjectValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Explicit object value schema.

    Openness is mandatory so a nested or output object never acquires an
    accidental JSON Schema default.
    """

    type: Literal['object']
    properties: ParameterSchemaSpec
    additionalProperties: bool


class JsonValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Author-only unconstrained lossless JSON node."""

    type: Literal['json']


class OneOfValueSchemaSpec(ValueSchemaAnnotations, total=False):
    """Exact-one union schema; at least two branches are required."""

    oneOf: list[ValueSchemaSpec]


#: One author-facing schema for any lossless JSON value root.
ValueSchemaSpec = (
    StringValueSchemaSpec
    | NumberValueSchemaSpec
    | IntegerValueSchemaSpec
    | BooleanValueSchemaSpec
    | NullValueSchemaSpec
    | ArrayValueSchemaSpec
    | ObjectValueSchemaSpec
    | JsonValueSchemaSpec
    | OneOfValueSchemaSpec
)


class ParameterPropertySpec(ValueSchemaAnnotations, total=False):
    """One implicit parameter-root property, optionally required.

    The value keys are any :data:`ValueSchemaSpec` member; ``required`` is
    the implicit-root annotation. The runtime compiler, not the type, is the
    authority on which combinations are legal.
    """

    required: Literal[True]


#: Tool parameter schema: an implicit open object root; requiredness stays a
#: per-property ``required: True`` annotation.
ParameterSchemaSpec = dict[str, ParameterPropertySpec]


class ParameterJsonSchema(ObjectJsonSchema):
    """Raw JSON Schema projection of the implicit parameter object.

    Always carries ``properties`` (and ``required`` when any property opted
    in); TypedDict cannot mark inherited optional keys required, so
    construction through :func:`parameter_schema_spec_to_json_schema` is the
    guarantee.
    """


_ANNOTATION_KEYS = ('description', 'title', 'default', 'examples')


def _author_error(message: str) -> NoReturn:
    """Throw one author-schema violation through the shared schema error."""
    raise JsonSchemaError([message])


def _copy_annotations(source: dict[str, Any], target: dict[str, Any]) -> None:
    """Copy own annotation fields for validation by the raw-schema boundary."""
    for key in _ANNOTATION_KEYS:
        if key in source:
            target[key] = source[key]


def _assert_author_keys(
    source: dict[str, Any],
    path: str,
    allowed: tuple[str, ...],
) -> None:
    """Reject author-only keys outside one node's declared vocabulary."""
    for key in source:
        if key not in allowed:
            _author_error(f'{path}.{key} is not supported by the value schema DSL')


def _compile_property_map(
    input_: Any,
    path: str,
    seen: list[Any],
) -> tuple[dict[str, Any], list[str]]:
    """Compile one implicit property map, collecting per-property requiredness."""
    if not is_json_schema_record(input_):
        _author_error(f'{path} must be an object of value schemas')
    if any(known is input_ for known in seen):
        _author_error(f'{path} is circular')
    seen.append(input_)
    try:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key, prop in input_.items():
            prop_path = f'{path}.{key}'
            if not is_json_schema_record(prop):
                _author_error(f'{prop_path} must be a value schema object')
            if 'required' in prop and prop['required'] is not True:
                _author_error(f'{prop_path}.required must be true when present')
            if prop.get('required') is True:
                required.append(key)
            properties[key] = _compile_value(prop, prop_path, True, seen)
        return properties, required
    finally:
        seen.pop()


def _compile_value(
    input_: Any,
    path: str,
    allow_required: bool,
    seen: list[Any],
) -> dict[str, Any]:
    """Compile one author node without applying any consumer root restriction."""
    if not is_json_schema_record(input_):
        _author_error(f'{path} must be a value schema object')
    if any(known is input_ for known in seen):
        _author_error(f'{path} is circular')
    seen.append(input_)
    try:
        author_keys: tuple[str, ...] = (
            *_ANNOTATION_KEYS,
            *(('required',) if allow_required else ()),
        )
        node: dict[str, Any] = {}

        if 'oneOf' in input_:
            _assert_author_keys(input_, path, (*author_keys, 'oneOf', 'type'))
            if 'type' in input_:
                _author_error(f'{path} cannot declare both type and oneOf')
            one_of = input_['oneOf']
            if not is_plain_json_array(one_of):
                _author_error(
                    f'{path}.oneOf must be an array of at least two value schemas'
                )
            node['oneOf'] = [
                _compile_value(branch, f'{path}.oneOf[{index}]', False, seen)
                for index, branch in enumerate(one_of)
            ]
            _copy_annotations(input_, node)
            return node

        input_type = input_.get('type')
        if input_type == 'json':
            _assert_author_keys(input_, path, (*author_keys, 'type'))
            _copy_annotations(input_, node)
            return node
        if input_type == 'object':
            _assert_author_keys(
                input_, path,
                (*author_keys, 'type', 'properties', 'additionalProperties'),
            )
            if 'additionalProperties' not in input_ or not isinstance(
                input_['additionalProperties'], bool
            ):
                _author_error(
                    f'{path}.additionalProperties must be explicitly true or false'
                )
            node['type'] = 'object'
            _copy_annotations(input_, node)
            node['additionalProperties'] = input_['additionalProperties']
            if 'properties' in input_:
                properties, required = _compile_property_map(
                    input_['properties'], f'{path}.properties', seen,
                )
                node['properties'] = properties
                if required:
                    node['required'] = required
            return node
        if input_type == 'array':
            _assert_author_keys(input_, path, (*author_keys, 'type', 'items'))
            node['type'] = 'array'
            _copy_annotations(input_, node)
            if 'items' in input_:
                node['items'] = _compile_value(
                    input_['items'], f'{path}.items', False, seen,
                )
            return node
        if input_type in ('string', 'number', 'integer', 'boolean', 'null'):
            _assert_author_keys(
                input_, path, (*author_keys, 'type', 'enum', 'const'),
            )
            node['type'] = input_type
            _copy_annotations(input_, node)
            if 'enum' in input_:
                if not is_plain_json_array(input_['enum']):
                    _author_error(
                        f'{path}.enum must be a non-empty array of scalar values'
                    )
                node['enum'] = list(input_['enum'])
            if 'const' in input_:
                node['const'] = input_['const']
            return node
        _author_error(
            f'{path}.type must be '
            'string/number/integer/boolean/null/array/object/json, or use oneOf'
        )
    finally:
        seen.pop()


def value_schema_spec_to_json_schema(spec: ValueSchemaSpec) -> JsonSchemaNode:
    """Compile one author-facing value schema to the enforced raw subset.

    The author-only ``json`` node becomes an annotation-only schema.

    :param spec: schema for any JSON-value root.
    :return: the asserted raw schema projection.
    """
    schema = _compile_value(spec, 'schema', False, [])
    assert_supported_json_schema(schema)
    return cast(JsonSchemaNode, schema)


def parameter_schema_spec_to_json_schema(
    spec: ParameterSchemaSpec,
) -> ParameterJsonSchema:
    """Compile the implicit open parameter object into raw JSON Schema.

    :param spec: per-property parameter definitions.
    :return: an object-rooted raw schema with no implicit-root openness
        override.
    """
    properties, required = _compile_property_map(spec, 'parameters', [])
    schema: dict[str, Any] = {'type': 'object', 'properties': properties}
    if required:
        schema['required'] = required
    assert_supported_json_schema(schema)
    return cast(ParameterJsonSchema, schema)


class ToolArgsError(HarnessError):
    """Invalid model-generated arguments for a typed tool."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__(
            f'invalid arguments: {"; ".join(violations)}',
            'INVALID_ARGS',
        )
        #: Individual violations in schema-walk order.
        self.violations = violations


def validate_args(spec: ParameterSchemaSpec, args: Any) -> list[str]:
    """Validate model-generated arguments against an implicit parameter schema.

    :param spec: declared parameter schema.
    :param args: candidate arguments, however malformed.
    :return: path-qualified violations; empty means valid.
    """
    return validate_json_schema_value(
        parameter_schema_spec_to_json_schema(spec), args, '',
    )


@dataclass(frozen=True)
class DefineToolOutput:
    """Canonical output declaration accepted by :func:`define_tool`.

    ``schema`` is an author-facing :data:`ValueSchemaSpec`, compiled to the
    enforced raw subset at definition time.
    """

    #: Schema enforced against every successful body or policy-replaced value.
    schema: ValueSchemaSpec
    #: Pure Native/model rendering of one validated canonical value.
    render: Callable[[Any, JsonValue], list[ContentBlock]]
    #: Pure replayable presentation metadata for direct top-level calls.
    presentation_meta: Callable[[Any, JsonValue], JsonValue] | None = None


def define_tool(
    *,
    name: str,
    description: str,
    parameters: ParameterSchemaSpec,
    output: DefineToolOutput,
    execute: Callable[[Any, ToolExecution], Awaitable[Any]],
    timeout_ms: float | None = None,
    is_concurrency_safe: Callable[[Any], bool] | None = None,
    finalize_content: Callable[
        [ToolExecution, ToolExecutionResult], list[ContentBlock] | None
    ] | None = None,
) -> ToolDefinition:
    """Define a first-party tool with strict execution-time validation.

    The compiled parameter schema validates arguments before the body runs
    (raising :class:`ToolArgsError`), and the compiled output schema is
    enforced by the registry against every successful value. Unlike
    ``execute``, a ``finalize_content`` callback receives ``arguments`` as
    ``Any`` because invalid-input failures also reach it; it must be total
    and must not throw.

    :param name: tool name (must be unique).
    :param description: human-readable description sent to the model.
    :param parameters: per-property parameter schema compiled to an implicit
        open object root.
    :param output: canonical output declaration.
    :param execute: the tool body, run after argument validation.
    :param timeout_ms: optional positive cooperative timeout budget in
        milliseconds (never sent to the model).
    :param is_concurrency_safe: pure classifier for sibling overlap; invalid
        arguments classify as exclusive.
    :param finalize_content: optional last-mile content transform for every
        normalized outcome.
    :return: a registry-ready definition.
    """
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or not math.isfinite(timeout_ms)
        or timeout_ms <= 0
    ):
        raise ValueError(
            f'defineTool({name}): timeoutMs must be a positive finite number'
        )
    compiled_parameters = parameter_schema_spec_to_json_schema(parameters)
    output_schema = value_schema_spec_to_json_schema(output.schema)

    def validate(args: Any) -> list[str]:
        return validate_json_schema_value(compiled_parameters, args, '')

    async def validated_execute(args: Any, exec: ToolExecution) -> JsonValue:
        violations = validate(args)
        if violations:
            raise ToolArgsError(violations)
        return cast(JsonValue, await execute(args, exec))

    def concurrency_classifier(args: Any) -> bool:
        if validate(args):
            return False
        assert is_concurrency_safe is not None
        return is_concurrency_safe(args)

    return ToolDefinition(
        name=name,
        description=description,
        parameters=dict(compiled_parameters),
        output=ToolOutputDefinition(
            schema=output_schema,
            render=output.render,
            presentation_meta=output.presentation_meta,
        ),
        execute=validated_execute,
        finalize_content=finalize_content,
        timeout_ms=timeout_ms,
        is_concurrency_safe=(
            concurrency_classifier if is_concurrency_safe is not None else None
        ),
    )
