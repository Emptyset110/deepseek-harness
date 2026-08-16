"""Lossless-JSON validation and detached snapshots for durable session data.

Ported from deepseek-harness packages/core/session (MIT), ``src/json.ts``.

Python mapping: "a plain object or array from any JavaScript realm" maps to
an exact ``dict``/``list`` instance; getters, prototypes, sparse arrays, and
``toJSON`` have no Python counterpart. Traversal is recursive rather than
iterative, so nesting depth is bounded by the interpreter recursion limit
instead of available memory.
"""

from __future__ import annotations

import math
from typing import Any, Final, TypeAlias, TypeVar, cast

__all__ = [
    'SNAPSHOT_FAILED',
    'JsonValue',
    'SnapshotFailed',
    'is_json_value',
    'snapshot_json_value',
]

T = TypeVar('T')

JsonValue: TypeAlias = (
    'bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None'
)
"""A value that round-trips losslessly through JSON.

``int`` also covers arbitrary-precision integers (lossless through Python's
JSON codec); non-finite floats and negative zero are excluded and enforced
at runtime by :func:`is_json_value`/:func:`snapshot_json_value`, mirroring
the TypeScript ``number`` constraints.
"""


class SnapshotFailed:
    """Sentinel type returned by :func:`snapshot_json_value` for invalid input.

    A dedicated sentinel (not ``None``) because JSON ``null`` is a valid,
    snapshot-able value.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return 'SNAPSHOT_FAILED'


SNAPSHOT_FAILED: Final = SnapshotFailed()


def _is_negative_zero(value: float) -> bool:
    return value == 0 and math.copysign(1.0, value) < 0


def _walk(value: Any, detach: bool, ancestors: set[int]) -> tuple[bool, Any]:
    """Validate one JSON tree; return ``(ok, snapshot)``.

    With ``detach`` the snapshot is a fresh copy; otherwise it is the
    original value and only validation runs. ``ancestors`` holds the identity
    of containers on the current path to reject cycles.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return True, value
    if isinstance(value, float):
        if not math.isfinite(value) or _is_negative_zero(value):
            return False, None
        return True, value
    if type(value) is list or type(value) is dict:
        marker = id(value)
        if marker in ancestors:
            return False, None
        ancestors.add(marker)
        try:
            if isinstance(value, list):
                snapshot_list: list[Any] = []
                for item in value:
                    ok, child = _walk(item, detach, ancestors)
                    if not ok:
                        return False, None
                    if detach:
                        snapshot_list.append(child)
                return True, snapshot_list if detach else value
            snapshot_dict: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    return False, None
                ok, child = _walk(item, detach, ancestors)
                if not ok:
                    return False, None
                if detach:
                    snapshot_dict[key] = child
            return True, snapshot_dict if detach else value
        finally:
            ancestors.discard(marker)
    return False, None


def snapshot_json_value(value: T) -> T | SnapshotFailed:
    """Validate and detach lossless JSON in one pass.

    Accepts exact ``list``/``dict`` containers with string keys and JSON
    scalars; rejects cyclic, exotic (tuples, sets, class instances,
    subclasses), negative-zero, and non-finite values.

    :param value: the candidate value to validate and detach.
    :returns: the detached snapshot, or ``SNAPSHOT_FAILED`` when the value is
        not losslessly JSON-serializable.
    """
    ok, snapshot = _walk(value, True, set())
    if not ok:
        return SNAPSHOT_FAILED
    return cast(T, snapshot)


def is_json_value(value: object) -> bool:
    """Test the same lossless-JSON boundary as :func:`snapshot_json_value`.

    :param value: the candidate event data to test.
    :returns: whether ``value`` survives a JSON round-trip losslessly.
    """
    ok, _ = _walk(value, False, set())
    return ok


def _deep_equal_json(a: Any, b: Any) -> bool:
    """Deep structural equality over the JSON value domain.

    ``bool`` never equals a number (unlike Python's ``==``); ``int`` and
    ``float`` compare by numeric value, matching JavaScript's single
    ``number`` type. Intra-package helper shared by the surface fold and the
    request-header equality check.
    """
    if type(a) is not type(b):
        numeric = (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        )
        return numeric and bool(a == b)
    if isinstance(a, list):
        return len(a) == len(b) and all(
            _deep_equal_json(item, other) for item, other in zip(a, b, strict=True)
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(
            _deep_equal_json(a[key], b[key]) for key in a
        )
    return bool(a == b)
