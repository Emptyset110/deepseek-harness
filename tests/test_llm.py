"""Behavior tests for the pydsh.llm runtime: adapter registry, routing,
streaming dispatch, and the ``llm/stream`` waterfall.

Ported-semantics tests for the deepseek-harness llm capability seam
(packages/llm/llm, MIT).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

import pytest

from pycordis import Context, FiberState
from pydsh.llm import (
    GenerateOptions,
    LlmAdapter,
    LlmError,
    LlmResolvedModelInfo,
    LlmRuntime,
    StreamChunk,
    assert_usable_api_key,
    is_agent_loop_request,
    is_context_window_exceeded_error,
    is_quota_exceeded_error,
    mark_agent_loop_request,
    normalize_api_key,
)


class MockAdapter(LlmAdapter):
    """Canned-chunk adapter; records the options it was called with."""

    def __init__(
        self,
        chunks: list[StreamChunk] | None = None,
        info: LlmResolvedModelInfo | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.chunks = chunks if chunks is not None else [
            {'type': 'block-start', 'index': 0, 'blockType': 'text'},
            {'type': 'text-delta', 'index': 0, 'text': 'hi'},
            {'type': 'block-end', 'index': 0, 'block': {'type': 'text', 'text': 'hi'}},
            {'type': 'finish', 'reason': {'kind': 'stop'}},
        ]
        self.info = info
        self.error = error
        self.calls: list[GenerateOptions] = []

    def resolve_model(self, provider: str, model: str) -> LlmResolvedModelInfo:
        if self.info is not None:
            return self.info
        return super().resolve_model(provider, model)

    def stream(self, options: GenerateOptions) -> AsyncIterable[StreamChunk]:
        self.calls.append(options)
        chunks, error = self.chunks, self.error

        async def gen() -> AsyncGenerator[StreamChunk]:
            for chunk in chunks:
                yield chunk
            if error is not None:
                raise error

        return gen()


def make_options(**overrides: Any) -> GenerateOptions:
    options = GenerateOptions(
        provider='p1',
        model='m1',
        messages=[
            {
                'id': 'msg-1',
                'role': 'user',
                'content': [{'type': 'text', 'text': 'hello'}],
                'source': {'kind': 'user'},
            }
        ],
    )
    options.update(overrides)
    return options


async def collect(stream: AsyncIterable[StreamChunk]) -> list[StreamChunk]:
    return [chunk async for chunk in stream]


@pytest.fixture
async def ctx() -> AsyncGenerator[Context]:
    context = Context()
    await context.plugin(LlmRuntime)
    yield context


async def test_register_and_route(ctx: Context) -> None:
    llm = ctx.get('llm')
    adapter_a, adapter_b = MockAdapter(), MockAdapter()
    llm.register_adapter(['a'], adapter_a)
    llm.register_adapter(['b'], adapter_b)
    assert llm.list_providers() == [{'id': 'a', 'name': 'a'}, {'id': 'b', 'name': 'b'}]

    chunks = await collect(llm.stream(make_options(provider='b')))
    types = [c['type'] for c in chunks]
    assert types == ['block-start', 'text-delta', 'block-end', 'finish']
    assert len(adapter_a.calls) == 0
    assert len(adapter_b.calls) == 1
    assert adapter_b.calls[0]['model'] == 'm1'


async def test_duplicate_adapter_all_or_nothing(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter())
    with pytest.raises(LlmError) as exc_info:
        llm.register_adapter(['b', 'a'], MockAdapter())
    assert exc_info.value.code == 'DUPLICATE_ADAPTER'
    # all-or-nothing: the rejected candidate set registered nothing
    assert [p['id'] for p in llm.list_providers()] == ['a']


async def test_registration_validation(ctx: Context) -> None:
    llm = ctx.get('llm')
    with pytest.raises(LlmError) as exc_info:
        llm.register_adapter([], MockAdapter())
    assert exc_info.value.code == 'INVALID_ADAPTER'
    with pytest.raises(LlmError) as exc_info:
        llm.register_adapter([''], MockAdapter())
    assert exc_info.value.code == 'INVALID_ADAPTER'


async def test_dispose_revokes_routes(ctx: Context) -> None:
    llm = ctx.get('llm')
    handle = llm.register_adapter(['a'], MockAdapter())
    assert [p['id'] for p in llm.list_providers()] == ['a']
    handle()
    assert llm.list_providers() == []
    chunks = await collect(llm.stream(make_options(provider='a')))
    assert chunks == [
        {
            'type': 'finish',
            'reason': {
                'kind': 'error',
                'failure': {
                    'message': 'no adapter registered for provider "a"',
                    'code': 'NO_ADAPTER',
                },
            },
        }
    ]


async def test_fiber_dispose_revokes_routes(ctx: Context) -> None:
    def provider_plugin(c: Context, config: object = None) -> None:
        c.llm.register_adapter(['owned'], MockAdapter())

    provider_plugin.inject = ['llm']
    fiber = ctx.plugin(provider_plugin)
    await fiber
    llm = ctx.get('llm')
    assert [p['id'] for p in llm.list_providers()] == ['owned']
    await fiber.dispose()
    assert llm.list_providers() == []


async def test_replace_routes(ctx: Context) -> None:
    llm = ctx.get('llm')
    handle = llm.register_adapter(['a'], MockAdapter())
    handle.replace(['b'])
    assert [p['id'] for p in llm.list_providers()] == ['b']
    # a conflicting candidate leaves the current routes untouched
    llm.register_adapter(['c'], MockAdapter())
    with pytest.raises(LlmError) as exc_info:
        handle.replace(['c'])
    assert exc_info.value.code == 'DUPLICATE_ADAPTER'
    assert [p['id'] for p in llm.list_providers()] == ['b', 'c']
    # an empty replacement is legal for a live registration
    handle.replace([])
    assert [p['id'] for p in llm.list_providers()] == ['c']
    # disposal wins over replace
    handle()
    with pytest.raises(LlmError) as exc_info:
        handle.replace(['d'])
    assert exc_info.value.code == 'REGISTRATION_DISPOSED'


async def test_adapters_updated_notification(ctx: Context) -> None:
    llm = ctx.get('llm')
    seen: list[str] = []
    ctx.on('llm/adapters-updated', lambda: seen.append('updated'))
    # a broken listener cannot veto the registry mutation
    ctx.on('llm/adapters-updated', lambda: 1 / 0)
    handle = llm.register_adapter(['a'], MockAdapter())
    handle()
    assert seen == ['updated', 'updated']


async def test_stream_materializes_adapter_defaults(ctx: Context) -> None:
    llm = ctx.get('llm')
    adapter = MockAdapter(
        info={
            'provider': 'a',
            'id': 'm1',
            'name': 'M1',
            'defaultMaxTokens': 4096,
            'reasoning': {
                'efforts': [{'id': 'high', 'name': 'High'}],
                'defaultEffort': 'high',
            },
        }
    )
    llm.register_adapter(['a'], adapter)
    await collect(llm.stream(make_options(provider='a')))
    called = adapter.calls[0]
    assert called['maxTokens'] == 4096
    assert called['reasoningEffort'] == 'high'
    # explicit request values win over adapter defaults
    await collect(llm.stream(make_options(provider='a', maxTokens=100)))
    assert adapter.calls[1]['maxTokens'] == 100


async def test_unsupported_reasoning_effort(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter())
    options = make_options(provider='a', reasoningEffort='high')
    chunks = await collect(llm.stream(options))
    finish = chunks[-1]
    assert finish['type'] == 'finish'
    assert finish['reason']['kind'] == 'error'
    assert finish['reason']['failure']['code'] == 'UNSUPPORTED_REASONING_EFFORT'


async def test_no_adapter_finish_chunk(ctx: Context) -> None:
    llm = ctx.get('llm')
    chunks = await collect(llm.stream(make_options(provider='missing')))
    assert len(chunks) == 1
    assert chunks[0]['reason']['kind'] == 'error'  # type: ignore[typeddict-item]
    assert chunks[0]['reason']['failure']['code'] == 'NO_ADAPTER'  # type: ignore[typeddict-item]


async def test_adapter_failure_normalized(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(
        ['a'], MockAdapter(error=LlmError('bad credentials', 'AUTH', status=401))
    )
    llm.register_adapter(['b'], MockAdapter(error=ValueError('boom')))
    chunks_a = await collect(llm.stream(make_options(provider='a')))
    reason = chunks_a[-1]['reason']
    assert reason == {
        'kind': 'error',
        'failure': {'message': 'bad credentials', 'code': 'AUTH', 'status': 401},
    }
    chunks_b = await collect(llm.stream(make_options(provider='b')))
    assert chunks_b[-1]['reason']['failure']['code'] == 'UNKNOWN'  # type: ignore[typeddict-item]


async def test_caller_abort_maps_to_aborted(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter(error=LlmError('stop', 'ABORTED')))
    chunks = await collect(llm.stream(make_options(provider='a')))
    assert chunks[-1]['reason']['kind'] == 'aborted'  # type: ignore[typeddict-item]

    signal = asyncio.Event()
    signal.set()
    llm.register_adapter(['b'], MockAdapter(error=ValueError('boom')))
    chunks = await collect(llm.stream(make_options(provider='b', signal=signal)))
    assert chunks[-1]['reason']['kind'] == 'aborted'  # type: ignore[typeddict-item]


async def test_replay_state_stripped_for_other_adapter(ctx: Context) -> None:
    llm = ctx.get('llm')
    adapter_a = MockAdapter()
    adapter_b = MockAdapter()
    llm.register_adapter(['a'], adapter_a)
    llm.register_adapter(['b'], adapter_b)
    message = {
        'id': 'msg-2',
        'role': 'assistant',
        'content': [{'type': 'text', 'text': 'old'}],
        'source': {
            'kind': 'model',
            'provider': 'a',
            'model': 'm1',
            'replayState': {'x': 1},
        },
    }
    # same adapter owns the historical route and the target: state retained
    await collect(llm.stream(make_options(provider='a', messages=[message])))
    assert adapter_a.calls[0]['messages'][0]['source'].get('replayState') == {'x': 1}
    # a different adapter owns the target: state stripped
    await collect(llm.stream(make_options(provider='b', messages=[message])))
    assert 'replayState' not in adapter_b.calls[0]['messages'][0]['source']


async def test_waterfall_wrap(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter())
    observed: list[GenerateOptions] = []

    def listener(runtime: LlmRuntime, options: GenerateOptions, next_: Any) -> Any:
        observed.append(options)

        async def wrapped() -> AsyncGenerator[StreamChunk]:
            async for chunk in next_():
                yield chunk
            yield {'type': 'usage', 'usage': {'inputTokens': 1, 'outputTokens': 1}}

        return wrapped()

    ctx.on('llm/stream', listener)
    chunks = await collect(llm.stream(make_options(provider='a')))
    assert [c['type'] for c in chunks] == [
        'block-start',
        'text-delta',
        'block-end',
        'finish',
        'usage',
    ]
    assert observed[0]['provider'] == 'a'


async def test_waterfall_short_circuit(ctx: Context) -> None:
    llm = ctx.get('llm')
    adapter = MockAdapter()
    llm.register_adapter(['a'], adapter)

    async def replayed() -> AsyncGenerator[StreamChunk]:
        yield {'type': 'finish', 'reason': {'kind': 'stop'}}

    # not calling next() vetoes the adapter dispatch entirely
    ctx.on('llm/stream', lambda runtime, options, next_: replayed())
    chunks = await collect(llm.stream(make_options(provider='a')))
    assert [c['type'] for c in chunks] == ['finish']
    assert adapter.calls == []


async def test_agent_loop_request_marker() -> None:
    options = make_options()
    assert not is_agent_loop_request(options)
    assert mark_agent_loop_request(options) is options
    assert is_agent_loop_request(options)
    assert not is_agent_loop_request(make_options())


async def test_provider_retry_policy(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter())
    policy = llm.provider_retry_policy('a')
    assert policy.mode == 'normal'
    assert policy.max_retries == 2  # type: ignore[attr-defined]
    with pytest.raises(LlmError) as exc_info:
        llm.provider_retry_policy('missing')
    assert exc_info.value.code == 'NO_ADAPTER'


async def test_list_models_validation(ctx: Context) -> None:
    llm = ctx.get('llm')
    llm.register_adapter(['a'], MockAdapter())
    assert await llm.list_models('a') == []

    class BadCatalogAdapter(MockAdapter):
        async def list_models(self, provider: str) -> list[dict[str, Any]]:
            return [{'provider': provider, 'id': 'x', 'name': 'X'}] * 2

    llm.register_adapter(['b'], BadCatalogAdapter())
    with pytest.raises(LlmError) as exc_info:
        await llm.list_models('b')
    assert exc_info.value.code == 'INVALID_CATALOG'


async def test_llm_error_fact_validation() -> None:
    with pytest.raises(ValueError, match='status'):
        LlmError('m', 'C', status=99)
    with pytest.raises(ValueError, match='providerRetryAfterMs'):
        LlmError('m', 'C', provider_retry_after_ms=0)
    with pytest.raises(ValueError, match='requestId'):
        LlmError('m', 'C', request_id='')  # type: ignore[arg-type]
    error = LlmError('m', 'C', status=500, provider_retry_after_ms=1000)
    assert error.failure == {
        'message': 'm',
        'code': 'C',
        'status': 500,
        'providerRetryAfterMs': 1000,
    }


async def test_api_key_checks() -> None:
    assert normalize_api_key('  sk-abc  ') == (True, 'sk-abc')
    assert normalize_api_key('   ') == (False, 'empty')
    assert normalize_api_key('sk ab\tc') == (False, 'illegalCharacters')
    assert assert_usable_api_key(' sk-abc ', 'pkg', 'REF') == 'sk-abc'
    with pytest.raises(LlmError) as exc_info:
        assert_usable_api_key('', 'pkg', 'REF')
    assert exc_info.value.code == 'INVALID_CREDENTIAL'
    assert 'sk' not in str(exc_info.value)


async def test_failure_wording_classifiers() -> None:
    assert is_context_window_exceeded_error('maximum context length exceeded')
    assert is_context_window_exceeded_error(
        'This model context window is too small x context_length_exceeded'
    )
    assert not is_context_window_exceeded_error('rate limit reached')
    assert is_quota_exceeded_error('insufficient balance')
    assert is_quota_exceeded_error('quota exceeded')
    assert not is_quota_exceeded_error('too many requests')


async def test_call_config_equals() -> None:
    from pydsh.llm import call_config_equals

    a = make_options(stop=['x'])
    assert call_config_equals(a, dict(a))
    assert not call_config_equals(a, {**a, 'stop': ['y']})
    assert not call_config_equals(a, {**a, 'stop': None})
    assert not call_config_equals(a, {**a, 'model': 'other'})


async def test_runtime_unloads_with_fiber() -> None:
    context = Context()
    fiber = context.plugin(LlmRuntime)
    await fiber

    def consumer(c: Context, config: object = None) -> None:
        pass

    consumer.inject = ['llm']
    consumer_fiber = context.plugin(consumer)
    await consumer_fiber
    assert consumer_fiber.state is FiberState.ACTIVE
    await fiber.dispose()
    assert consumer_fiber.state is FiberState.PENDING
