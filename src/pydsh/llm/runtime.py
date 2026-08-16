"""LLM service: adapter registry with a waterfall-interceptable streaming call API.

Ported from deepseek-harness ``packages/llm/llm/src/index.ts`` and
``call-config.ts`` (MIT).

The :class:`LlmRuntime` cordis service (name ``llm``) is Service Definition
and runtime in one: construction registers it on the context. The
``llm/stream`` waterfall wraps every streaming model call; listeners have the
signature ``(runtime, options, next)`` where calling ``next()`` reaches the
resolved adapter's stream and returning without it short-circuits the chain
with the listener's own chunk stream.

Read-only invariant (TS enforces it with ``deepFreeze``; Python documents and
defends it by convention plus defensive copies): the waterfall payload is
read-only. A loop-built request carries the :func:`is_agent_loop_request`
identity and its content is a pure function of the session log — the
"model-visible ⟺ logged" invariant — so listeners read it, never rewrite it.
Nothing in this module mutates a request; materialized adapter defaults
produce a new options dict.

Scope of this port: adapter registration (all-or-nothing, ``replace``),
provider routing, model-info resolution with default materialization, the
streaming dispatch, and the waterfall. The TS configurable-provider
directory, model discovery, and ``prepareCall`` are not ported (no settings
seam exists on the Python side yet).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterator,
)
from dataclasses import dataclass
from typing import Any, cast

from pycordis import Context, Service
from pycordis.utils import schedule_logged

from .errors import LlmError, normalize_llm_failure
from .retry_policy import ResolvedRetryPolicy, resolve_retry_policy
from .types import (
    AbortedFinish,
    ErrorFinish,
    FinishChunk,
    FinishReason,
    GenerateOptions,
    LlmModelContext,
    LlmModelInfo,
    LlmProviderInfo,
    LlmReasoningEffortInfo,
    LlmResolvedModelInfo,
    ReasoningEffortId,
    StreamChunk,
)


class LlmAdapter:
    """Provider-wire adapter for the harness message and stream vocabulary.

    Register implementations with ``ctx.llm.register_adapter(providers,
    adapter)``. Every provider HTTP request must include
    ``attribution_headers()``; prove the headers are added in the wire request.
    """

    def provider_info(self, provider: str) -> LlmProviderInfo:
        """Describe one provider route owned by this adapter."""
        return {"id": provider, "name": provider}

    def provider_retry_policy(self, provider: str) -> ResolvedRetryPolicy | None:
        """Return the provider-owned retry policy captured with this route,
        or ``None`` to use the normal defaults."""
        return None

    async def list_models(self, provider: str) -> list[LlmModelInfo]:
        """List models this adapter can currently advertise for one owned
        provider. The result is advisory: an adapter may accept unlisted model
        ids, and consumers must not turn absence into request rejection."""
        return []

    def resolve_model(self, provider: str, model: str) -> LlmResolvedModelInfo:
        """Resolve all metadata available for one exact model. Independent of
        the advisory catalog; does not validate request routing."""
        return {"provider": provider, "id": model, "name": model}

    def stream(self, options: GenerateOptions) -> AsyncIterable[StreamChunk]:
        """Stream one model call as raw chunks. The only required method.

        Implementations must honor ``options['signal']`` (an
        ``asyncio.Event``; set means aborted).
        """
        raise NotImplementedError


@dataclass
class _AdapterRegistration:
    adapter: LlmAdapter
    provider: LlmProviderInfo
    retry_policy: ResolvedRetryPolicy


class AdapterRegistrationHandle:
    """What :meth:`LlmRuntime.register_adapter` returns: the disposer, plus an
    atomic route replacement for the same adapter instance."""

    def __init__(
        self,
        dispose: Callable[[], None],
        replace: Callable[[list[str]], None],
    ) -> None:
        self._dispose = dispose
        self._replace = replace

    def __call__(self) -> None:
        """Release every route this registration currently holds."""
        self._dispose()

    def replace(self, providers: list[str]) -> None:
        """Replace this registration's routes with ``providers``, keeping the
        same adapter instance.

        The candidate set is validated in full first — a conflict with
        another adapter or an invalid name throws and leaves the current
        routes untouched — and the swap itself is one synchronous section, so
        no request can observe a gap. An empty list is legal here (a live
        registration holding zero routes), unlike an empty initial
        registration. Throws ``LlmError`` with code ``REGISTRATION_DISPOSED``
        once the registration has been released.
        """
        self._replace(providers)


def call_config_equals(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Field-wise equality over the call-config fields of a request:
    ``provider``, ``model``, ``reasoningEffort``, ``temperature``,
    ``maxTokens``, and the ``stop`` list element-wise."""
    for key in ("provider", "model", "reasoningEffort", "temperature", "maxTokens"):
        if a.get(key) != b.get(key):
            return False
    stop_a, stop_b = a.get("stop"), b.get("stop")
    if stop_a is None or stop_b is None:
        return stop_a is stop_b
    return bool(stop_a == stop_b)


def _adapter_failure_chunk(
    error: BaseException, signal: asyncio.Event | None
) -> FinishChunk:
    """Convert one adapter raise into the stream protocol's terminal outcome."""
    failure = normalize_llm_failure(error)
    aborted = (signal is not None and signal.is_set()) or failure["code"] == "ABORTED"
    reason: FinishReason = (
        AbortedFinish(kind="aborted", failure=failure)
        if aborted
        else ErrorFinish(kind="error", failure=failure)
    )
    return {"type": "finish", "reason": reason}


class LlmRuntime(Service):
    """The abstract ``llm`` service: an adapter registry plus a streaming
    model-call API, interceptable via the ``llm/stream`` waterfall.

    Construct through ``ctx.plugin(LlmRuntime)``; construction provides the
    ``llm`` service, and disposal of the owning fiber withdraws it (including
    every adapter registered through it).
    """

    def __init__(self, ctx: Context, config: Any = None) -> None:
        self._adapters: dict[str, _AdapterRegistration] = {}
        super().__init__(ctx, "llm")

    # --- notifications ---

    def _emit_adapters_updated(self) -> None:
        """Notify topology observers without letting one broken listener veto
        the commit. Listener failures are contained and logged."""
        args: list[Any] = ["llm/adapters-updated"]
        for listener in self.ctx.events.dispatch("emit", args):
            try:
                returned = listener()
                if inspect.isawaitable(returned):
                    schedule_logged(returned, self.ctx.logger.error)
            except Exception as error:
                self.ctx.logger.warn("llm: an llm/adapters-updated listener failed")
                self.ctx.logger.warn(error)

    # --- registration ---

    def register_adapter(
        self,
        providers: list[str],
        adapter: LlmAdapter,
    ) -> AdapterRegistrationHandle:
        """Register an adapter for the given provider routes.

        Throws ``LlmError`` with code ``DUPLICATE_ADAPTER`` if any provider
        already has an adapter (all-or-nothing: a rejected candidate set
        changes nothing). The registration is disposed with the calling
        fiber; calling the returned handle releases its routes early.
        """
        owned: set[str] = set()
        # The disposer has run: `owned` being empty cannot say so on its own,
        # because `replace([])` legally leaves a live registration holding none.
        released = False

        def setup() -> Iterator[Callable[[], None]]:
            if len(providers) == 0:
                raise LlmError(
                    "an adapter must register at least one provider", "INVALID_ADAPTER"
                )
            self._commit_routes(owned, self._prepare_routes(providers, adapter, owned))

            def release() -> None:
                nonlocal released
                released = True
                for provider in owned:
                    del self._adapters[provider]
                owned.clear()
                self._emit_adapters_updated()

            yield release

        effect = self.ctx.effect(setup, "llm.register_adapter()")

        def dispose() -> None:
            # The effect's disposer may return a task; this disposer API is
            # synchronous fire-and-forget, so discard it.
            effect()

        def replace(next_providers: list[str]) -> None:
            # Registering here would leak: the effect's disposer already ran,
            # so nothing remains to release whatever this call would add.
            if released:
                raise LlmError(
                    "a disposed adapter registration cannot replace its routes",
                    "REGISTRATION_DISPOSED",
                )
            self._commit_routes(
                owned, self._prepare_routes(next_providers, adapter, owned)
            )

        return AdapterRegistrationHandle(dispose, replace)

    def _prepare_routes(
        self,
        providers: list[str],
        adapter: LlmAdapter,
        owned: set[str],
    ) -> list[_AdapterRegistration]:
        """Validate one candidate route set for ``adapter``, treating routes
        this registration already holds as available. Nothing is mutated: a
        rejected candidate leaves the registry exactly as it was."""
        unique: set[str] = set()
        registrations: list[_AdapterRegistration] = []
        for provider in providers:
            if len(provider) == 0:
                raise LlmError(
                    "adapter provider names must be non-empty", "INVALID_ADAPTER"
                )
            if provider in unique or (
                provider in self._adapters and provider not in owned
            ):
                raise LlmError(
                    f'an adapter for provider "{provider}" is already registered',
                    "DUPLICATE_ADAPTER",
                )
            info = adapter.provider_info(provider)
            if info.get("id") != provider or not info.get("name"):
                raise LlmError(
                    f'adapter metadata for provider "{provider}" must preserve its id'
                    " and have a non-empty name",
                    "INVALID_ADAPTER",
                )
            unique.add(provider)
            retry_policy = adapter.provider_retry_policy(
                provider
            ) or resolve_retry_policy(None, f'llm: provider "{provider}" retryPolicy')
            registrations.append(
                _AdapterRegistration(
                    adapter=adapter,
                    provider={"id": info["id"], "name": info["name"]},
                    retry_policy=retry_policy,
                )
            )
        return registrations

    def _commit_routes(
        self,
        owned: set[str],
        registrations: list[_AdapterRegistration],
    ) -> None:
        """Swap this registration's routes for the prepared ones in one
        synchronous section, so no observer can see the registry between the
        release and the re-registration. This is the route set's one mutation
        point, so ``llm/adapters-updated`` is published here."""
        for provider in owned:
            del self._adapters[provider]
        owned.clear()
        for registration in registrations:
            self._adapters[registration.provider["id"]] = registration
            owned.add(registration.provider["id"])
        self._emit_adapters_updated()

    def list_providers(self) -> list[LlmProviderInfo]:
        """Describe provider routes with a registered adapter, in registration order."""
        return [
            {"id": registration.provider["id"], "name": registration.provider["name"]}
            for registration in self._adapters.values()
        ]

    def _registration(self, provider: str) -> _AdapterRegistration:
        registration = self._adapters.get(provider)
        if registration is None:
            raise LlmError(
                f'no adapter registered for provider "{provider}"', "NO_ADAPTER"
            )
        return registration

    def provider_retry_policy(self, provider: str) -> ResolvedRetryPolicy:
        """Resolve the retry policy captured when one provider route was registered."""
        return self._registration(provider).retry_policy

    async def list_models(self, provider: str) -> list[LlmModelInfo]:
        """Discover models advertised by one registered provider. Catalog
        membership is advisory and never changes routing or request validation."""
        adapter = self._registration(provider).adapter
        models = await adapter.list_models(provider)
        seen: set[str] = set()
        detached: list[LlmModelInfo] = []
        for model in models:
            if (
                model.get("provider") != provider
                or not model.get("id")
                or not model.get("name")
                or model["id"] in seen
            ):
                raise LlmError(
                    f"adapter returned invalid or duplicate model metadata for provider"
                    f' "{provider}"',
                    "INVALID_CATALOG",
                )
            seen.add(model["id"])
            entry: LlmModelInfo = {
                "provider": model["provider"],
                "id": model["id"],
                "name": model["name"],
            }
            if model.get("description") is not None:
                entry["description"] = model["description"]
            if model.get("inputModalities") is not None:
                entry["inputModalities"] = list(model["inputModalities"])
            detached.append(entry)
        return detached

    # --- model info resolution and call-config materialization ---

    def resolve_model_info(self, provider: str, model: str) -> LlmResolvedModelInfo:
        """Resolve and validate all metadata from the adapter that owns one
        exact route. The result is detached from adapter-owned objects."""
        return self._resolve_model_info_for(self._registration(provider), model)

    def _resolve_model_info_for(
        self,
        registration: _AdapterRegistration,
        model: str,
    ) -> LlmResolvedModelInfo:
        provider = registration.provider["id"]
        resolved = registration.adapter.resolve_model(provider, model)
        if (
            resolved.get("provider") != provider
            or resolved.get("id") != model
            or not resolved.get("name")
        ):
            raise LlmError(
                "adapter returned invalid exact model metadata for provider"
                f' "{provider}" model "{model}"',
                "INVALID_MODEL_INFO",
            )
        context = resolved.get("context")
        if context is not None and (
            not isinstance(context.get("contextWindow"), int)
            or context["contextWindow"] <= 0
        ):
            raise LlmError(
                f'adapter returned invalid context metadata for provider "{provider}"'
                f' model "{model}"',
                "INVALID_MODEL_CONTEXT",
            )
        default_max_tokens = resolved.get("defaultMaxTokens")
        if default_max_tokens is not None and (
            not isinstance(default_max_tokens, int) or default_max_tokens <= 0
        ):
            raise LlmError(
                f'adapter returned invalid default maxTokens for provider "{provider}"'
                f' model "{model}"',
                "INVALID_MODEL_MAX_TOKENS",
            )
        info: LlmResolvedModelInfo = {
            "provider": provider,
            "id": model,
            "name": resolved["name"],
        }
        if resolved.get("description") is not None:
            info["description"] = resolved["description"]
        if resolved.get("inputModalities") is not None:
            info["inputModalities"] = list(resolved["inputModalities"])
        if context is not None:
            info["context"] = {"contextWindow": context["contextWindow"]}
        if default_max_tokens is not None:
            info["defaultMaxTokens"] = default_max_tokens
        reasoning = resolved.get("reasoning")
        if reasoning is None:
            return info
        efforts = reasoning.get("efforts") or []
        if len(efforts) == 0:
            raise LlmError(
                f'adapter returned invalid reasoning metadata for provider "{provider}"'
                f' model "{model}"',
                "INVALID_MODEL_REASONING",
            )
        seen: set[str] = set()
        detached_efforts: list[LlmReasoningEffortInfo] = []
        for effort in efforts:
            if not effort.get("id") or not effort.get("name") or effort["id"] in seen:
                raise LlmError(
                    "adapter returned invalid or duplicate reasoning effort"
                    f' metadata for provider "{provider}" model "{model}"',
                    "INVALID_MODEL_REASONING",
                )
            seen.add(effort["id"])
            detached: LlmReasoningEffortInfo = {
                "id": effort["id"],
                "name": effort["name"],
            }
            if effort.get("description") is not None:
                detached["description"] = effort["description"]
            detached_efforts.append(detached)
        default_effort = reasoning.get("defaultEffort")
        if default_effort is not None and default_effort not in seen:
            raise LlmError(
                f"adapter returned an unknown default reasoning effort for provider"
                f' "{provider}" model "{model}"',
                "INVALID_MODEL_REASONING",
            )
        info["reasoning"] = {"efforts": detached_efforts}
        if default_effort is not None:
            info["reasoning"]["defaultEffort"] = ReasoningEffortId(str(default_effort))
        return info

    def resolve_call_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate a conversation call config against its exact model
        capability and materialize adapter-configured defaults. Unsupported
        explicit efforts reject before provider I/O; no clamping or aliasing
        is performed."""
        registration = self._registration(config["provider"])
        return self._resolve_call_for(registration, config)[0]

    def _resolve_call_for(
        self,
        registration: _AdapterRegistration,
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], LlmModelContext | None]:
        info = self._resolve_model_info_for(registration, config["model"])
        defaulted = dict(config)
        if config.get("maxTokens") is None and info.get("defaultMaxTokens") is not None:
            defaulted["maxTokens"] = info["defaultMaxTokens"]
        provider, model = config["provider"], config["model"]
        reasoning = info.get("reasoning")
        requested = config.get("reasoningEffort")
        if reasoning is None:
            if requested is not None:
                raise LlmError(
                    f'provider "{provider}" model "{model}" does not support'
                    f' reasoning effort "{requested}"',
                    "UNSUPPORTED_REASONING_EFFORT",
                )
        else:
            effective = (
                requested if requested is not None else reasoning.get("defaultEffort")
            )
            if effective is not None:
                if not any(
                    effort["id"] == effective for effort in reasoning["efforts"]
                ):
                    raise LlmError(
                        f'provider "{provider}" model "{model}" does not support'
                        f' reasoning effort "{effective}"',
                        "UNSUPPORTED_REASONING_EFFORT",
                    )
                if requested != effective:
                    defaulted["reasoningEffort"] = effective
        return defaulted, info.get("context")

    # --- streaming dispatch ---

    def _for_adapter(
        self, options: GenerateOptions, adapter: LlmAdapter
    ) -> GenerateOptions:
        """Remove replay state whose historical route is owned by another adapter."""
        messages: list[Any] = []
        changed = False
        for message in options["messages"]:
            source = message.get("source", {})
            if (
                message.get("role") != "assistant"
                or source.get("kind") != "model"
                or source.get("replayState") is None
            ):
                messages.append(message)
                continue
            owner = self._adapters.get(source.get("provider", ""))
            if owner is not None and owner.adapter is adapter:
                messages.append(message)
                continue
            messages.append(
                {
                    **message,
                    "source": {
                        "kind": "model",
                        "provider": source["provider"],
                        "model": source["model"],
                    },
                }
            )
            changed = True
        if not changed:
            return options
        return GenerateOptions({**options, "messages": messages})

    async def _adapter_stream(
        self, options: GenerateOptions
    ) -> AsyncGenerator[StreamChunk]:
        """Final adapter boundary. Adapter selection, dispatch, iterator
        construction, and iteration failures become one terminal failure
        chunk. Middleware and downstream consumer failures remain thrown
        plugin or consumer errors."""
        signal = options.get("signal")
        iterator: AsyncIterator[StreamChunk] | None = None
        try:
            registration = self._registration(options["provider"])
            resolved_config, _context = self._resolve_call_for(registration, options)
            resolved_options = (
                options
                if call_config_equals(options, resolved_config)
                else GenerateOptions({**options, **resolved_config})
            )
            adapter = registration.adapter
            stream = adapter.stream(self._for_adapter(resolved_options, adapter))
            iterator = stream.__aiter__()
        except Exception as error:
            yield _adapter_failure_chunk(error, signal)
            return

        completed = False
        try:
            while True:
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    completed = True
                    return
                except Exception as error:
                    completed = True
                    yield _adapter_failure_chunk(error, signal)
                    return
                # End the adapter-owned try before yielding: consumer/middleware
                # failures resumed into this generator must remain thrown.
                yield item
        finally:
            if not completed:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()

    def stream(self, options: GenerateOptions) -> AsyncIterable[StreamChunk]:
        """Stream one model call as raw chunks (token-level deltas).

        Replay state is retained only when the same adapter instance owns its
        historical provider and the target provider. Adapter selection,
        dispatch, and iteration failures become terminal ``error`` or
        ``aborted`` finish chunks; middleware, cleanup, and consumer failures
        remain thrown. The stream passes through the ``llm/stream``
        waterfall: listeners receive ``(runtime, options, next)`` and either
        delegate with ``next()`` or short-circuit with their own chunk
        stream. The payload is read-only (see module docstring).
        """
        return cast(
            AsyncIterable[StreamChunk],
            self.ctx.waterfall(
                self,
                "llm/stream",
                options,
                lambda *_args: self._adapter_stream(options),
            ),
        )
