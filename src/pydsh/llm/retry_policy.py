"""Provider-owned request-retry policy configuration and resolution.

Ported from deepseek-harness ``packages/llm/llm/src/retry-policy.ts`` (MIT).

Adapters expose one resolved policy per registered provider route; a retry
plugin (out of scope for this port) would execute it on the agent's
failed-step extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .errors import EMPTY_RESPONSE_CODE

#: Largest delay accepted from configuration (2**31 - 1 ms, the timer ceiling
#: shared with the TS ``dsh-timeout`` package).
MAX_TIMER_DELAY_MS = 2_147_483_647

DEFAULT_MAX_RETRIES = 2
DEFAULT_INITIAL_DELAY_MS = 500.0
DEFAULT_MAX_DELAY_MS = 10_000.0
DEFAULT_JITTER_RATIO = 0.1
DEFAULT_RETRYABLE_CODES = (
    EMPTY_RESPONSE_CODE,
    "RATE_LIMIT",
    "SERVER",
    "TIMEOUT",
    "TRANSPORT",
)


@dataclass(frozen=True)
class ResolvedNormalRetryPolicy:
    """Fully resolved bounded transient retry policy."""

    mode: Literal["normal"]
    max_retries: int
    retryable_codes: tuple[str, ...]
    initial_delay_ms: float
    max_delay_ms: float
    jitter_ratio: float


@dataclass(frozen=True)
class ResolvedAlwaysRetryPolicy:
    """Fully resolved unbounded retry policy."""

    mode: Literal["always"]
    initial_delay_ms: float
    max_delay_ms: float
    jitter_ratio: float


#: Immutable provider policy captured when its adapter route is registered.
ResolvedRetryPolicy = ResolvedNormalRetryPolicy | ResolvedAlwaysRetryPolicy

_NORMAL_POLICY_KEYS = frozenset({"mode", "maxRetries", "retryableCodes", "backoff"})
_ALWAYS_POLICY_KEYS = frozenset({"mode", "backoff"})
_BACKOFF_KEYS = frozenset({"initialDelayMs", "maxDelayMs", "jitterRatio"})


def _validate_keys(value: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    for key in value:
        if key not in allowed:
            raise ValueError(f'{path}: unknown key "{key}"')


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_backoff(
    config: dict[str, Any] | None,
    path: str,
) -> tuple[float, float, float]:
    if config is not None:
        _validate_keys(config, _BACKOFF_KEYS, path)
    initial = (config or {}).get("initialDelayMs", DEFAULT_INITIAL_DELAY_MS)
    maximum = (config or {}).get("maxDelayMs", DEFAULT_MAX_DELAY_MS)
    jitter = (config or {}).get("jitterRatio", DEFAULT_JITTER_RATIO)

    if not _is_number(initial) or initial <= 0 or initial > MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"{path}.initialDelayMs must be a positive finite number no greater than"
            f" {MAX_TIMER_DELAY_MS}"
        )
    if not _is_number(maximum) or maximum <= 0 or maximum > MAX_TIMER_DELAY_MS:
        raise ValueError(
            f"{path}.maxDelayMs must be a positive finite number no greater than"
            f" {MAX_TIMER_DELAY_MS}"
        )
    if initial > maximum:
        raise ValueError(
            f"{path}.initialDelayMs must be less than or equal to maxDelayMs"
        )
    if not _is_number(jitter) or jitter < 0 or jitter > 1:
        raise ValueError(f"{path}.jitterRatio must be between 0 and 1")
    return float(initial), float(maximum), float(jitter)


def resolve_retry_policy(
    config: dict[str, Any] | None, path: str
) -> ResolvedRetryPolicy:
    """Validate, default, and detach one provider-owned retry policy.

    ``config`` omission selects normal defaults. Raises ``ValueError`` on any
    bound violation; the result is an immutable value safe to capture in
    provider registration state.
    """
    if config is None:
        initial, maximum, jitter = _resolve_backoff(None, f"{path}.backoff")
        return ResolvedNormalRetryPolicy(
            mode="normal",
            max_retries=DEFAULT_MAX_RETRIES,
            retryable_codes=DEFAULT_RETRYABLE_CODES,
            initial_delay_ms=initial,
            max_delay_ms=maximum,
            jitter_ratio=jitter,
        )

    mode = config.get("mode")
    if mode == "normal":
        _validate_keys(config, _NORMAL_POLICY_KEYS, path)
        max_retries = config.get("maxRetries", DEFAULT_MAX_RETRIES)
        retryable_codes = config.get("retryableCodes", list(DEFAULT_RETRYABLE_CODES))
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise ValueError(f"{path}.maxRetries must be a non-negative safe integer")
        if len(retryable_codes) == 0:
            raise ValueError(f"{path}.retryableCodes must not be empty")
        if any(not isinstance(code, str) or len(code) == 0 for code in retryable_codes):
            raise ValueError(
                f"{path}.retryableCodes must contain only non-empty strings"
            )
        if len(set(retryable_codes)) != len(retryable_codes):
            raise ValueError(f"{path}.retryableCodes must not contain duplicates")
        initial, maximum, jitter = _resolve_backoff(
            config.get("backoff"), f"{path}.backoff"
        )
        return ResolvedNormalRetryPolicy(
            mode="normal",
            max_retries=max_retries,
            retryable_codes=tuple(retryable_codes),
            initial_delay_ms=initial,
            max_delay_ms=maximum,
            jitter_ratio=jitter,
        )
    if mode == "always":
        _validate_keys(config, _ALWAYS_POLICY_KEYS, path)
        initial, maximum, jitter = _resolve_backoff(
            config.get("backoff"), f"{path}.backoff"
        )
        return ResolvedAlwaysRetryPolicy(
            mode="always",
            initial_delay_ms=initial,
            max_delay_ms=maximum,
            jitter_ratio=jitter,
        )
    raise ValueError(f'{path}.mode must be "normal" or "always"')
