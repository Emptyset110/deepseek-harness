"""Harness error base, LLM error type, failure normalization, and API-key checks.

Ported from deepseek-harness ``packages/llm/llm/src/error.ts``,
``adapter-failure.ts``, and ``api-key.ts`` (MIT).

Python mapping: ``ErrorOptions.cause`` becomes the standard ``__cause__``
(``raise ... from ...``); the frozen ``failure`` payload of TS becomes a
plain ``LlmFailure`` dict (JSON-serializable by construction, treated as
read-only by convention).
"""

from __future__ import annotations

import math
import re
from typing import Any, cast

from .types import LlmFailure, ProviderRequestId


class HarnessError(Exception):
    """Base class for all harness errors.

    Carries a ``code`` (stable, programmatic — e.g. ``NO_ADAPTER``,
    ``INVALID_ARGS``, ``INVARIANT``) distinct from the human-readable
    message. Route on ``code``, never by parsing ``message``.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


#: Canonical provider-neutral code for a model request rejected because its
#: context window was exceeded.
CONTEXT_WINDOW_EXCEEDED_CODE = "CONTEXT_WINDOW_EXCEEDED"

#: Canonical provider-neutral code for an exhausted account quota or balance.
QUOTA_EXCEEDED_CODE = "QUOTA"

#: Canonical provider-neutral code for a response that completed normally but
#: carried no content blocks at all.
EMPTY_RESPONSE_CODE = "EMPTY_RESPONSE"

#: Canonical provider-neutral code for a credential that was supplied but
#: cannot be used — malformed rather than absent.
INVALID_CREDENTIAL_CODE = "INVALID_CREDENTIAL"

#: Structured codes and plain phrases that explicitly name a context bound
#: being exceeded.
_STRUCTURED_CONTEXT_OVERFLOW = re.compile(
    r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-]"
    r"(?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

#: Request-size wording that ties "too large" directly to model context capacity.
_TOO_LARGE_FOR_CONTEXT = re.compile(
    r"\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?"
    r"too\s+(?:large|long)\s+for\s+(?:(?:this|the)\s+)?"
    r"(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
    re.IGNORECASE,
)

#: "Exceeds" wording is safe only when its object is explicitly the model context.
_EXCEEDS_MODEL_CONTEXT = re.compile(
    r"\b(?:input|prompt|request|messages?)\b.{0,40}"
    r"\b(?:exceed(?:s|ed)?|overflows?|is\s+larger\s+than)\b.{0,40}"
    r"\b(?:the\s+)?(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b",
    re.IGNORECASE,
)


def is_context_window_exceeded_error(detail: str) -> bool:
    """Recognize the context-overflow wording of OpenAI-compatible providers.

    Adapters pass all available provider code, type, and message text so both
    thrown and in-band delivery styles share one classifier.
    """
    return bool(
        _STRUCTURED_CONTEXT_OVERFLOW.search(detail)
        or re.search(
            r"\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b",
            detail,
            re.IGNORECASE,
        )
        or _TOO_LARGE_FOR_CONTEXT.search(detail)
        or re.search(
            r"\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+for\s+(?:this|the)\s+model\b",
            detail,
            re.IGNORECASE,
        )
        or _EXCEEDS_MODEL_CONTEXT.search(detail)
    )


def is_quota_exceeded_error(detail: str) -> bool:
    """Recognize provider wording for an exhausted account quota.

    True only for terminal quota, balance, credit, budget, or usage-limit
    wording — never for a transient request-rate limit.
    """
    return bool(
        re.search(
            r"\binsufficient[\s_-]+(?:quota|balance|credits?)\b", detail, re.IGNORECASE
        )
        or re.search(
            r"\b(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)\b",
            detail,
            re.IGNORECASE,
        )
        or re.search(
            r"\bexceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?(?:current[\s_-]+)?quota\b",
            detail,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:balance|credits?)[\s_-]+(?:exhausted|depleted)\b",
            detail,
            re.IGNORECASE,
        )
        or re.search(
            r"\bout[\s_-]+of[\s_-]+(?:credits?|budget)\b", detail, re.IGNORECASE
        )
    )


def validate_failure(failure: LlmFailure) -> LlmFailure:
    """Check one failure payload's invariants, raising ``ValueError``."""
    if not isinstance(failure["message"], str) or len(failure["message"]) == 0:
        raise ValueError("failure message must be a non-empty string")
    if not isinstance(failure["code"], str) or len(failure["code"]) == 0:
        raise ValueError("failure code must be a non-empty string")
    status = failure.get("status")
    if status is not None and (
        not isinstance(status, int) or status < 100 or status > 599
    ):
        raise ValueError("failure status must be an integer from 100 through 599")
    retry_after = failure.get("providerRetryAfterMs")
    if retry_after is not None and (
        not isinstance(retry_after, (int, float))
        or isinstance(retry_after, bool)
        or not math.isfinite(retry_after)
        or retry_after <= 0
    ):
        raise ValueError(
            "failure providerRetryAfterMs must be a positive finite number"
        )
    request_id = failure.get("requestId")
    if request_id is not None and (
        not isinstance(request_id, str) or len(request_id) == 0
    ):
        raise ValueError("failure requestId must be a non-empty string")
    return failure


class LlmError(HarnessError):
    """Typed error for LLM-related failures.

    The ``code`` string (e.g. ``AUTH``, ``RATE_LIMIT``, ``NO_ADAPTER``) is
    shared taxonomy. Serializable provider facts are retained beside the live
    exception as ``failure``.
    """

    def __init__(
        self,
        message: str,
        code: str,
        *,
        status: int | None = None,
        provider_retry_after_ms: float | None = None,
        request_id: ProviderRequestId | None = None,
    ) -> None:
        if not isinstance(message, str) or len(message) == 0:
            raise ValueError("LlmError message must be a non-empty string")
        if not isinstance(code, str) or len(code) == 0:
            raise ValueError("LlmError code must be a non-empty string")
        failure: LlmFailure = {"message": message, "code": code}
        if status is not None:
            failure["status"] = status
        if provider_retry_after_ms is not None:
            failure["providerRetryAfterMs"] = provider_retry_after_ms
        if request_id is not None:
            failure["requestId"] = request_id
        validate_failure(failure)
        super().__init__(message, code)
        #: Serializable facts retained beside this live exception (read-only).
        self.failure: LlmFailure = failure


# --- adapter failure normalization (adapter-failure.ts) ---


def normalize_llm_failure(value: Any) -> LlmFailure:
    """Detach serializable provider facts from a value thrown by an adapter.

    Harness-owned codes are trusted; anything else normalizes to ``UNKNOWN``.
    The result is suitable for a terminal finish chunk.
    """
    if isinstance(value, LlmError):
        return cast(LlmFailure, dict(value.failure))
    if isinstance(value, HarnessError):
        return {"message": str(value) or "LLM adapter failed", "code": value.code}
    if isinstance(value, BaseException):
        return {"message": str(value) or "LLM adapter failed", "code": "UNKNOWN"}
    message = str(value)
    return {
        "message": message if len(message) > 0 else "LLM adapter failed",
        "code": "UNKNOWN",
    }


# --- API key checks (api-key.ts) ---

#: Characters an HTTP header value carries verbatim and every known provider
#: key uses: printable ASCII, space excluded.
_LEGAL_API_KEY = re.compile(r"^[\x21-\x7E]+$")


def normalize_api_key(raw: str) -> tuple[bool, str]:
    """Judge one *supplied* API key, trimming surrounding whitespace first.

    Trimming is silent because a padded key has one unambiguous reading.
    Returns ``(True, trimmed)`` on success and ``(False, reason)`` where
    reason is ``'empty'`` or ``'illegalCharacters'`` otherwise.
    """
    value = raw.strip()
    if len(value) == 0:
        return False, "empty"
    if not _LEGAL_API_KEY.match(value):
        return False, "illegalCharacters"
    return True, value


def assert_usable_api_key(raw: str, pkg: str, ref: str) -> str:
    """Accept one supplied credential, or refuse it as unusable.

    The key never enters the error message: ``ref`` names where to fix it,
    and echoing any part of a secret into a log is the failure this
    diagnosis avoids.
    """
    ok, result = normalize_api_key(raw)
    if ok:
        return result
    if result == "empty":
        raise LlmError(
            f"{pkg}: the API key resolved from {ref} is blank; set {ref} to the raw key"
            " or export it in the launching environment",
            INVALID_CREDENTIAL_CODE,
        )
    raise LlmError(
        f"{pkg}: the API key resolved from {ref} contains characters no HTTP"
        f" header can carry; set {ref} to the raw key alone",
        INVALID_CREDENTIAL_CODE,
    )
