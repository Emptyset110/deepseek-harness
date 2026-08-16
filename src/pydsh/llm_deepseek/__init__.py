"""DeepSeek provider plugin: registers a :class:`DeepSeekAdapter` for the
``deepseek-official`` provider route on ``ctx.llm``.

Ported from deepseek-harness ``packages/llm/llm-deepseek/src/index.ts``
(MIT). Function-plugin shape: module-level ``name`` / ``inject`` /
``Config`` / ``apply``; pass this module itself to ``ctx.plugin()``.

Simplifications vs the TS plugin (documented divergence, reinstated when the
Python port grows the corresponding seams):

- No user-settings layering: connection facts are resolved once at load
  (fail loud) instead of per request from a settings snapshot.
- No credentials service: the API key resolves per request from the
  environment variable named by ``apiKeyEnv`` (default
  ``DEEPSEEK_API_KEY``); ``DEEPSEEK_BASE_URL`` supplies the endpoint when
  ``baseURL`` is not configured.
- No configurable-provider directory (``registerConfigurableProviders``) and
  no persisted anonymous user id: a random id is generated per plugin load.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from pycordis import Context
from pydsh.llm import (
    MAX_TIMER_DELAY_MS,
    LlmError,
    assert_usable_api_key,
    resolve_retry_policy,
)

from .adapter import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STREAM_IDLE_TIMEOUT_MS,
    DeepSeekAdapter,
    DeepSeekCatalogModel,
    DeepSeekConnectionOptions,
    http_error_code,
)
from .serialize import RequestDefaults

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_STREAM_IDLE_TIMEOUT_MS",
    "PUBLIC_BASE_URL",
    "Config",
    "DeepSeekAdapter",
    "DeepSeekCatalogModel",
    "DeepSeekConnectionOptions",
    "RequestDefaults",
    "apply",
    "http_error_code",
    "inject",
    "name",
    "resolve_adapter_options",
]

#: Cordis plugin metadata.
name = "llm-deepseek"
inject = ["llm"]

DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
#: Environment variable naming this provider's endpoint.
BASE_URL_ENV = "DEEPSEEK_BASE_URL"
#: The single provider route this plugin owns.
PROVIDER = "deepseek-official"
#: Public API default; an internal endpoint comes from $DEEPSEEK_BASE_URL.
PUBLIC_BASE_URL = "https://api.deepseek.com"

DEFAULT_MODELS = (
    DeepSeekCatalogModel(
        id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash",
        context_window=DEFAULT_CONTEXT_WINDOW,
    ),
    DeepSeekCatalogModel(
        id="deepseek-v4-pro",
        name="DeepSeek-V4-Pro",
        context_window=DEFAULT_CONTEXT_WINDOW,
    ),
)

_CONFIG_KEYS = frozenset(
    {
        "apiKeyEnv",
        "baseURL",
        "thinking",
        "reasoningEffort",
        "maxTokens",
        "defaultContextWindow",
        "models",
        "streamIdleTimeoutMs",
        "retryPolicy",
    }
)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _resolve_models(models: Any) -> tuple[DeepSeekCatalogModel, ...]:
    """Validate, default, and detach the advisory model catalog."""
    if models is None:
        return DEFAULT_MODELS
    if not isinstance(models, list):
        raise ValueError("llm-deepseek: models must be a list")
    seen: set[str] = set()
    resolved = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise ValueError(
                "llm-deepseek: catalog models must be objects with a string id"
            )
        if len(model["id"]) == 0:
            raise ValueError("llm-deepseek: catalog model ids must be non-empty")
        if model.get("name") is not None and len(model["name"]) == 0:
            raise ValueError(
                f'llm-deepseek: catalog model "{model["id"]}" has an empty name'
            )
        context_window = model.get("contextWindow")
        if context_window is not None and not _is_positive_int(context_window):
            raise ValueError(
                f'llm-deepseek: catalog model "{model["id"]}" contextWindow must be a'
                " positive integer"
            )
        max_tokens = model.get("maxTokens")
        if max_tokens is not None and not _is_positive_int(max_tokens):
            raise ValueError(
                f'llm-deepseek: catalog model "{model["id"]}" maxTokens must be a'
                " positive integer"
            )
        if model["id"] in seen:
            raise ValueError(f'llm-deepseek: duplicate catalog model "{model["id"]}"')
        seen.add(model["id"])
        resolved.append(
            DeepSeekCatalogModel(
                id=model["id"],
                name=model.get("name"),
                description=model.get("description"),
                context_window=context_window,
                max_tokens=max_tokens,
            )
        )
    return tuple(resolved)


def Config(config: Any) -> dict[str, Any]:
    """Cordis ``Config`` callable: validate the plugin config dict and apply
    schema defaults.

    Every field is optional in yml: a missing API key resolves through
    ``apiKeyEnv`` at each request (a request without any key fails with
    ``MISSING_CREDENTIAL``, not at plugin load), omitted thinking mode uses
    the provider default, and omitted reasoning effort resolves to ``high``.
    Raises ``ValueError`` on any violation, failing the fiber at load.
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("llm-deepseek: config must be a mapping")
    for key in config:
        if key not in _CONFIG_KEYS:
            raise ValueError(f'llm-deepseek: unknown config key "{key}"')

    validated = dict(config)
    api_key_env = validated.get("apiKeyEnv", DEFAULT_API_KEY_ENV)
    if not isinstance(api_key_env, str) or len(api_key_env) == 0:
        raise ValueError("llm-deepseek: apiKeyEnv must be a non-empty string")
    validated["apiKeyEnv"] = api_key_env

    if validated.get("baseURL") is not None and not isinstance(
        validated["baseURL"], str
    ):
        raise ValueError("llm-deepseek: baseURL must be a string")
    if validated.get("thinking") is not None and validated["thinking"] not in (
        "enabled",
        "disabled",
    ):
        raise ValueError('llm-deepseek: thinking must be "enabled" or "disabled"')
    if validated.get("reasoningEffort") is not None and validated[
        "reasoningEffort"
    ] not in (
        "off",
        "high",
        "max",
    ):
        raise ValueError(
            'llm-deepseek: reasoningEffort must be "off", "high", or "max"'
        )

    max_tokens = validated.get("maxTokens", DEFAULT_MAX_TOKENS)
    if not _is_positive_int(max_tokens):
        raise ValueError("llm-deepseek: maxTokens must be a positive safe integer")
    validated["maxTokens"] = max_tokens

    context_window = validated.get("defaultContextWindow", DEFAULT_CONTEXT_WINDOW)
    if not _is_positive_int(context_window):
        raise ValueError(
            "llm-deepseek: defaultContextWindow must be a positive integer"
        )
    validated["defaultContextWindow"] = context_window

    idle = validated.get("streamIdleTimeoutMs", DEFAULT_STREAM_IDLE_TIMEOUT_MS)
    if (
        not isinstance(idle, (int, float))
        or isinstance(idle, bool)
        or idle <= 0
        or idle > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            "llm-deepseek: streamIdleTimeoutMs must be a positive finite number no"
            f" greater than {MAX_TIMER_DELAY_MS}"
        )
    validated["streamIdleTimeoutMs"] = idle

    validated["models"] = _resolve_models(validated.get("models"))

    retry_policy = validated.get("retryPolicy")
    if retry_policy is not None:
        # Validate eagerly; the resolved value is captured at registration.
        resolve_retry_policy(retry_policy, "llm-deepseek: retryPolicy")

    # Cross-field invariant, re-judged here so misconfiguration fails loud at
    # load rather than at the first request.
    if (
        validated.get("thinking") == "disabled"
        and validated.get("reasoningEffort") is not None
        and validated["reasoningEffort"] != "off"
    ):
        raise ValueError(
            'llm-deepseek: only reasoningEffort "off" can be configured when thinking'
            " is disabled"
        )
    return validated


def resolve_adapter_options(config: dict[str, Any]) -> DeepSeekConnectionOptions:
    """The one explicit resolve step from validated config to connection
    facts. ``Config`` output is trusted here; programmatic callers should run
    their config through :func:`Config` first."""
    return DeepSeekConnectionOptions(
        base_url=config.get("baseURL")
        or os.environ.get(BASE_URL_ENV)
        or PUBLIC_BASE_URL,
        api_key_env=config["apiKeyEnv"],
        defaults=RequestDefaults(
            thinking=config.get("thinking"),
            reasoning_effort=config.get("reasoningEffort"),
        ),
        max_tokens=config["maxTokens"],
        default_context_window=config["defaultContextWindow"],
        models=config["models"],
        stream_idle_timeout_ms=config["streamIdleTimeoutMs"],
        retry_policy=resolve_retry_policy(
            config.get("retryPolicy"), "llm-deepseek: retryPolicy"
        ),
    )


def apply(ctx: Context, config: dict[str, Any]) -> None:
    """Register the DeepSeek adapter on ``ctx.llm``. Connection facts are
    resolved once here; the API key is resolved from the environment per
    request, so a changed key reaches the very next request."""
    options = resolve_adapter_options(config)
    client = httpx.AsyncClient()
    user_id: list[str] = []

    def resolve_user_id() -> str:
        if not user_id:
            user_id.append(uuid.uuid4().hex)
        return user_id[0]

    async def resolve_api_key(connection: DeepSeekConnectionOptions) -> str:
        raw = os.environ.get(connection.api_key_env)
        if raw is None or len(raw) == 0:
            raise LlmError(
                f'llm-deepseek: no API key for provider route "{PROVIDER}";'
                f" export {connection.api_key_env} in the launching environment",
                "MISSING_CREDENTIAL",
            )
        return assert_usable_api_key(raw, "llm-deepseek", connection.api_key_env)

    adapter = DeepSeekAdapter(
        options=lambda: options,
        resolve_api_key=resolve_api_key,
        resolve_user_id=resolve_user_id,
        client=client,
    )
    ctx.llm.register_adapter([PROVIDER], adapter)

    def close_client() -> Any:
        return client.aclose()

    ctx.effect(lambda: close_client, "llm-deepseek httpx client")
