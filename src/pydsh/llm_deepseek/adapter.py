"""``DeepSeekAdapter``: HTTP + SSE against a DeepSeek (OpenAI-compatible)
chat-completions endpoint, emitting harness StreamChunks.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/adapter.ts``
(MIT). The adapter is transport-only: connection facts arrive through a
thunk resolved once per stream call and the bearer token through a
per-request resolver, so the registering plugin owns validation, layering,
and credential policy.

Python mapping: TS ``fetch`` becomes ``httpx.AsyncClient`` streaming; the TS
``AbortSignal`` becomes an ``asyncio.Event`` on ``options['signal']`` (set
means aborted); the TS ``idleWatchdog`` becomes an ``asyncio.wait_for``
around each outstanding stream read, mapping expiry to ``TIMEOUT``.
"""

from __future__ import annotations

import email.utils
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import httpx

from pydsh.llm import (
    CONTEXT_WINDOW_EXCEEDED_CODE,
    QUOTA_EXCEEDED_CODE,
    GenerateOptions,
    LlmAdapter,
    LlmError,
    LlmModelInfo,
    LlmProviderInfo,
    LlmReasoningEffortInfo,
    LlmResolvedModelInfo,
    ProviderRequestId,
    ReasoningEffortId,
    ResolvedRetryPolicy,
    StreamChunk,
    attribution_headers,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
)

from .serialize import RequestDefaults, serialize_request
from .sse import parse_sse
from .translate import translate
from .types import WireError, WireErrorBody

#: Default maximum idle interval while an adapter stream read is outstanding.
DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
#: Default combined request/response context capacity.
DEFAULT_CONTEXT_WINDOW = 1_000_000
#: Default per-request output-token cap.
DEFAULT_MAX_TOKENS = 256_000

OFF_REASONING_EFFORT = ReasoningEffortId("off")
HIGH_REASONING_EFFORT = ReasoningEffortId("high")
MAX_REASONING_EFFORT = ReasoningEffortId("max")
REASONING_EFFORTS: tuple[LlmReasoningEffortInfo, ...] = (
    {"id": OFF_REASONING_EFFORT, "name": "Off"},
    {"id": HIGH_REASONING_EFFORT, "name": "High"},
    {"id": MAX_REASONING_EFFORT, "name": "Max"},
)
OFF_ONLY_REASONING_EFFORTS: tuple[LlmReasoningEffortInfo, ...] = (
    {"id": OFF_REASONING_EFFORT, "name": "Off"},
)


@dataclass(frozen=True)
class DeepSeekCatalogModel:
    """One optional model entry advertised by the direct-fetch adapter."""

    #: Wire model id accepted by the configured endpoint.
    id: str
    #: Selector label; defaults to ``id``.
    name: str | None = None
    #: Optional selector detail for deployments with similar model variants.
    description: str | None = None
    #: Known combined request/response context capacity.
    context_window: int | None = None
    #: Per-request output cap for this model; omission falls back to the
    #: profile's ``max_tokens``.
    max_tokens: int | None = None


@dataclass(frozen=True)
class DeepSeekConnectionOptions:
    """Validated connection facts for one operation.

    The plugin's ``resolve_adapter_options`` is the one explicit resolve step
    producing this shape; the adapter trusts it and re-reads it per
    operation, which is what makes a configuration change reach the next
    request without re-registration.
    """

    #: Endpoint base; ``/chat/completions`` is appended.
    base_url: str
    #: Credential reference (environment-variable name) resolved per request.
    #: Travelling with the endpoint is the point: a request can never pair
    #: one generation's URL with another generation's secret.
    api_key_env: str
    #: Request defaults applied to every call (thinking mode, effort).
    defaults: RequestDefaults
    #: Default per-request output cap; explicit request values win.
    max_tokens: int
    #: Positive context capacity used when the selected model has no exact value.
    default_context_window: int
    #: Advisory models exposed to discovery consumers; requests remain unrestricted.
    models: tuple[DeepSeekCatalogModel, ...]
    #: Maximum provider idle time while one stream read is outstanding.
    stream_idle_timeout_ms: int
    #: Provider-owned model-request retry policy, already resolved.
    retry_policy: ResolvedRetryPolicy


def _model_info(provider: str, model: DeepSeekCatalogModel) -> LlmModelInfo:
    info: LlmModelInfo = {
        "provider": provider,
        "id": model.id,
        "name": model.name or model.id,
        "inputModalities": ["text"],
    }
    if model.description is not None:
        info["description"] = model.description
    return info


def provider_retry_after_ms(value: str | None) -> float | None:
    """Parse a ``retry-after`` header into a positive millisecond delay."""
    if value is None:
        return None
    if value.isdigit():
        seconds = int(value) * 1000
        return float(seconds) if seconds > 0 else None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    delay = parsed.timestamp() * 1000 - time.time() * 1000
    return delay if delay > 0 else None


def _request_id(headers: httpx.Headers) -> ProviderRequestId | None:
    value = headers.get("x-request-id") or headers.get("x-deepseek-request-id")
    if not value:
        return None
    return ProviderRequestId(value)


def http_error_code(status: int, error: WireErrorBody | None = None) -> str:
    """Map an HTTP status (plus a parsed provider error body, when available)
    to a stable LlmError code."""
    if status in (401, 403):
        return "AUTH"
    detail = ""
    if error is not None:
        detail = " ".join(
            part
            for part in (error.get("code"), error.get("type"), error.get("message"))
            if part
        )
    if is_quota_exceeded_error(detail):
        return QUOTA_EXCEEDED_CODE
    if status == 429:
        return "RATE_LIMIT"
    if status == 400:
        if is_context_window_exceeded_error(detail):
            return CONTEXT_WINDOW_EXCEEDED_CODE
        return "INVALID_REQUEST"
    if status >= 500:
        return "SERVER"
    return f"HTTP_{status}"


class DeepSeekAdapter(LlmAdapter):
    """One instance serves every model name it was registered under (the
    harness model name IS the wire model name).

    Caller aborts (a set ``options['signal']`` event) map to ``ABORTED``; the
    configured per-read idle watchdog maps to ``TIMEOUT``.
    """

    def __init__(
        self,
        *,
        options: Callable[[], DeepSeekConnectionOptions],
        resolve_api_key: Callable[[DeepSeekConnectionOptions], Awaitable[str]],
        resolve_user_id: Callable[[], str],
        client: httpx.AsyncClient | None = None,
    ) -> None:
        #: Current validated connection facts; called once per operation.
        self._options = options
        #: Resolve the bearer token for one request's connection snapshot.
        self._resolve_api_key = resolve_api_key
        #: Resolve the harness anonymous id shared with telemetry and feedback.
        self._resolve_user_id = resolve_user_id
        self._client = client if client is not None else httpx.AsyncClient()

    def provider_info(self, provider: str) -> LlmProviderInfo:
        return {"id": provider, "name": "DeepSeek"}

    def provider_retry_policy(self, provider: str) -> ResolvedRetryPolicy:
        return self._options().retry_policy

    async def list_models(self, provider: str) -> list[LlmModelInfo]:
        return [_model_info(provider, model) for model in self._options().models]

    def resolve_model(self, provider: str, model: str) -> LlmResolvedModelInfo:
        connection = self._options()
        configured = next(
            (entry for entry in connection.models if entry.id == model), None
        )
        context_window = (
            configured.context_window
            if configured is not None and configured.context_window is not None
            else connection.default_context_window
        )
        # The chat-completions wire route is text-only regardless of catalog
        # membership, so the uncatalogued fallback declares the same negative
        # capability — "unknown" here would let the host accept and persist
        # images the serializer must then reject.
        info: LlmResolvedModelInfo = (
            cast(LlmResolvedModelInfo, {**_model_info(provider, configured)})
            if configured is not None
            else {
                "provider": provider,
                "id": model,
                "name": model,
                "inputModalities": ["text"],
            }
        )
        info["context"] = {"contextWindow": context_window}
        info["defaultMaxTokens"] = (
            configured.max_tokens
            if configured is not None and configured.max_tokens is not None
            else connection.max_tokens
        )
        if connection.defaults.thinking == "disabled":
            off_only: list[LlmReasoningEffortInfo] = [
                {"id": effort["id"], "name": effort["name"]}
                for effort in OFF_ONLY_REASONING_EFFORTS
            ]
            info["reasoning"] = {
                "efforts": off_only,
                "defaultEffort": OFF_REASONING_EFFORT,
            }
        else:
            default = connection.defaults.reasoning_effort
            efforts: list[LlmReasoningEffortInfo] = [
                {"id": effort["id"], "name": effort["name"]}
                for effort in REASONING_EFFORTS
            ]
            info["reasoning"] = {
                "efforts": efforts,
                "defaultEffort": (
                    OFF_REASONING_EFFORT
                    if default == "off"
                    else MAX_REASONING_EFFORT
                    if default == "max"
                    else HIGH_REASONING_EFFORT
                ),
            }
        return info

    async def stream(self, options: GenerateOptions) -> AsyncGenerator[StreamChunk]:
        # One resolution per stream call: connection facts and the credential
        # freeze here and hold for this whole request, so an in-flight stream
        # never observes a configuration change and the next call re-resolves.
        connection = self._options()
        api_key = await self._resolve_api_key(connection)
        user_id = self._resolve_user_id()
        signal = options.get("signal")
        try:
            async for chunk in self._request(options, connection, api_key, user_id):
                if signal is not None and signal.is_set():
                    raise LlmError("DeepSeek request aborted by caller", "ABORTED")
                yield chunk
        except TimeoutError as error:
            raise LlmError(
                "DeepSeek stream idle timeout after"
                f" {connection.stream_idle_timeout_ms}ms",
                "TIMEOUT",
            ) from error
        except LlmError:
            raise
        except httpx.HTTPError as error:
            if signal is not None and signal.is_set():
                raise LlmError(
                    "DeepSeek request aborted by caller", "ABORTED"
                ) from error
            raise LlmError(
                f"DeepSeek API stream from {connection.base_url} failed", "TRANSPORT"
            ) from error

    async def _request(
        self,
        options: GenerateOptions,
        connection: DeepSeekConnectionOptions,
        api_key: str,
        user_id: str,
    ) -> AsyncGenerator[StreamChunk]:
        body = serialize_request(options, connection.defaults)
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
            **attribution_headers(),
            "x-deepseek-harness-user-id": user_id,
        }
        if options.get("sessionId") is not None:
            headers["x-deepseek-harness-session-id"] = str(options["sessionId"])
        if options.get("purpose") == "compaction":
            headers["x-deepseek-harness-compact"] = "1"

        async with self._client.stream(
            "POST",
            f"{connection.base_url}/chat/completions",
            headers=headers,
            json=body,
        ) as response:
            if response.status_code >= 400:
                message = f"DeepSeek API error (HTTP {response.status_code})"
                provider_error: WireErrorBody | None = None
                try:
                    await response.aread()
                    parsed: WireError = response.json()
                    provider_error = parsed.get("error")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Only swallow error-body parsing: the HTTP status still
                    # identifies the failure, so malformed gateway JSON must
                    # not mask it.
                    provider_error = None
                if provider_error is not None and provider_error.get("message"):
                    message = provider_error["message"]
                raise LlmError(
                    message,
                    http_error_code(response.status_code, provider_error),
                    status=response.status_code,
                    provider_retry_after_ms=provider_retry_after_ms(
                        response.headers.get("retry-after")
                    ),
                    request_id=_request_id(response.headers),
                )
            idle_timeout = connection.stream_idle_timeout_ms / 1000
            async for chunk in translate(
                parse_sse(response.aiter_bytes(), idle_timeout=idle_timeout)
            ):
                yield chunk
