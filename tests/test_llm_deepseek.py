"""Behavior tests for the pydsh.llm_deepseek provider plugin: SSE parsing,
wire translation, request serialization, the HTTP adapter (mocked
transport, prerecorded SSE bytes — no real network), and plugin lifecycle.

Ported-semantics tests for deepseek-harness packages/llm/llm-deepseek (MIT).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

import httpx
import pytest

from pycordis import Context, FiberState
from pydsh.llm import (
    GenerateOptions,
    LlmError,
    LlmRuntime,
    resolve_retry_policy,
)
from pydsh.llm_deepseek import (
    DEFAULT_MAX_TOKENS,
    PUBLIC_BASE_URL,
    Config,
    DeepSeekAdapter,
    DeepSeekCatalogModel,
    DeepSeekConnectionOptions,
    RequestDefaults,
    http_error_code,
)
from pydsh.llm_deepseek.adapter import provider_retry_after_ms
from pydsh.llm_deepseek.serialize import serialize_messages, serialize_request
from pydsh.llm_deepseek.sse import parse_sse
from pydsh.llm_deepseek.translate import map_finish_reason, map_usage, translate


async def byte_stream(*chunks: bytes) -> AsyncGenerator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(stream: AsyncIterable[Any]) -> list[Any]:
    return [item async for item in stream]


# --- SSE framing (sse.ts) ---


async def test_sse_basic_multi_frame() -> None:
    stream = parse_sse(byte_stream(b'data: {"a": 1}\n\ndata: [DONE]\n\n'))
    assert await collect(stream) == ['{"a": 1}', '[DONE]']


async def test_sse_comments_and_non_data_fields() -> None:
    comments: list[str] = []
    stream = parse_sse(
        byte_stream(
            b': keep-alive\n'
            b'event: message\n'
            b'id: 42\n'
            b'data: one\n'
            b'data: two\n'
            b'\n'
            b'data: [DONE]\n\n'
        ),
        on_comment=comments.append,
    )
    # multi-`data:` lines join with newlines; comments never enter the stream
    assert await collect(stream) == ['one\ntwo', '[DONE]']
    assert comments == ['keep-alive']


async def test_sse_crlf_and_lone_cr() -> None:
    stream = parse_sse(byte_stream(b'data: a\r\n\r\ndata: [DONE]\r\n\r\n'))
    assert await collect(stream) == ['a', '[DONE]']
    stream = parse_sse(byte_stream(b'data: a\r\rdata: [DONE]\r\r'))
    assert await collect(stream) == ['a', '[DONE]']


async def test_sse_bom_and_split_multibyte() -> None:
    # BOM stripped; reads may split mid-UTF-8 sequence
    text = 'data: héllo\n\ndata: [DONE]\n\n'.encode()
    stream = parse_sse(byte_stream(b'\xef\xbb\xbf' + text[:8], text[8:]))
    assert await collect(stream) == ['héllo', '[DONE]']


async def test_sse_eof_without_done_raises() -> None:
    stream = parse_sse(byte_stream(b'data: {"a": 1}\n\n'))
    with pytest.raises(LlmError) as exc_info:
        await collect(stream)
    assert exc_info.value.code == 'STREAM_CLOSED'


async def test_sse_unterminated_tail_is_truncation() -> None:
    # an event without its blank-line terminator at EOF is not a payload
    stream = parse_sse(byte_stream(b'data: [DONE]'))
    with pytest.raises(LlmError) as exc_info:
        await collect(stream)
    assert exc_info.value.code == 'STREAM_CLOSED'


async def test_sse_idle_timeout() -> None:
    async def hanging() -> AsyncGenerator[bytes]:
        yield b'data: {"a": 1}\n\n'
        await asyncio.sleep(3600)

    stream = parse_sse(hanging(), idle_timeout=0.05)
    with pytest.raises(TimeoutError):
        await collect(stream)


# --- wire translation (translate.ts) ---


def payloads(*payload: str) -> AsyncIterable[str]:
    async def gen() -> AsyncGenerator[str]:
        for item in payload:
            yield item

    return gen()


def chunk(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


DONE = '[DONE]'


async def test_translate_text_flow_defers_close_to_done() -> None:
    stream = translate(
        payloads(
            chunk({'choices': [{'delta': {'content': 'Hel'}}]}),
            chunk({'choices': [{'delta': {'content': 'lo'}, 'finish_reason': 'stop'}]}),
            DONE,
        )
    )
    result = await collect(stream)
    assert result == [
        {'type': 'block-start', 'index': 0, 'blockType': 'text'},
        {'type': 'text-delta', 'index': 0, 'text': 'Hel'},
        {'type': 'text-delta', 'index': 0, 'text': 'lo'},
        {'type': 'block-end', 'index': 0, 'block': {'type': 'text', 'text': 'Hello'}},
        {'type': 'finish', 'reason': {'kind': 'stop'}},
    ]


async def test_translate_reasoning_empty_first_delta_opens_nothing() -> None:
    stream = translate(
        payloads(
            chunk({'choices': [{'delta': {'reasoning_content': ''}}]}),
            chunk({'choices': [{'delta': {'reasoning_content': 'think'}}]}),
            chunk({
                'choices': [{'delta': {'content': 'out'}, 'finish_reason': 'stop'}]
            }),
            chunk({'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}),
            DONE,
        )
    )
    result = await collect(stream)
    assert result == [
        {'type': 'block-start', 'index': 0, 'blockType': 'reasoning'},
        {'type': 'reasoning-delta', 'index': 0, 'text': 'think'},
        {'type': 'block-start', 'index': 1, 'blockType': 'text'},
        {'type': 'text-delta', 'index': 1, 'text': 'out'},
        {
            'type': 'block-end',
            'index': 0,
            'block': {'type': 'reasoning', 'text': 'think'},
        },
        {'type': 'block-end', 'index': 1, 'block': {'type': 'text', 'text': 'out'}},
        {'type': 'usage', 'usage': {'inputTokens': 10, 'outputTokens': 3}},
        {'type': 'finish', 'reason': {'kind': 'stop'}},
    ]


async def test_translate_tool_calls() -> None:
    first = {
        'choices': [
            {
                'delta': {
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': 'call-1',
                            'function': {'name': 'bash', 'arguments': '{'},
                        }
                    ]
                }
            }
        ]
    }
    second = {
        'choices': [
            {
                'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]},
                'finish_reason': 'tool_calls',
            }
        ]
    }
    stream = translate(payloads(chunk(first), chunk(second), DONE))
    result = await collect(stream)
    assert result == [
        {'type': 'block-start', 'index': 0, 'blockType': 'tool-call'},
        {
            'type': 'tool-call-delta',
            'index': 0,
            'id': 'call-1',
            'name': 'bash',
            'argumentsDelta': '{',
        },
        {
            'type': 'tool-call-delta',
            'index': 0,
            'id': 'call-1',
            'name': 'bash',
            'argumentsDelta': '}',
        },
        {
            'type': 'block-end',
            'index': 0,
            'block': {
                'type': 'tool-call',
                'id': 'call-1',
                'name': 'bash',
                'arguments': '{}',
            },
        },
        {'type': 'finish', 'reason': {'kind': 'tool-calls'}},
    ]


async def test_translate_empty_response_is_an_error_finish() -> None:
    stream = translate(payloads(chunk({'choices': []}), DONE))
    result = await collect(stream)
    assert result == [
        {
            'type': 'finish',
            'reason': {
                'kind': 'error',
                'failure': {
                    'message': 'model returned a completed response with no content',
                    'code': 'EMPTY_RESPONSE',
                },
            },
        }
    ]


async def test_translate_malformed_payload_aborts() -> None:
    stream = translate(payloads('not json'))
    with pytest.raises(LlmError) as exc_info:
        await collect(stream)
    assert exc_info.value.code == 'MALFORMED_RESPONSE'


async def test_translate_unknown_finish_reason_maps_to_error() -> None:
    finish_chunk = {
        'choices': [{'delta': {'content': 'x'}, 'finish_reason': 'content_filter'}]
    }
    stream = translate(payloads(chunk(finish_chunk), DONE))
    result = await collect(stream)
    assert result[-1] == {
        'type': 'finish',
        'reason': {
            'kind': 'error',
            'failure': {
                'message': 'model stopped: content_filter',
                'code': 'CONTENT_FILTER',
            },
        },
    }


def test_map_usage_disjoint_cache_counts() -> None:
    # DeepSeek's prompt_tokens INCLUDES cache hits; harness counts are disjoint
    usage = map_usage({
        'prompt_tokens': 100,
        'completion_tokens': 20,
        'prompt_cache_hit_tokens': 30,
        'completion_tokens_details': {'reasoning_tokens': 5},
    })
    assert usage == {
        'inputTokens': 70,
        'outputTokens': 20,
        'cacheReadTokens': 30,
        'reasoningTokens': 5,
    }
    # the OpenAI-compat spelling of the hit count wins
    usage = map_usage({
        'prompt_tokens': 100,
        'completion_tokens': 20,
        'prompt_cache_hit_tokens': 30,
        'prompt_tokens_details': {'cached_tokens': 25},
    })
    assert usage['cacheReadTokens'] == 25
    assert usage['inputTokens'] == 75


def test_map_finish_reason() -> None:
    assert map_finish_reason('stop') == {'kind': 'stop'}
    assert map_finish_reason('tool_calls') == {'kind': 'tool-calls'}
    assert map_finish_reason('length') == {'kind': 'max-tokens'}


# --- request serialization (serialize.ts) ---


def user_message(text: str) -> dict[str, Any]:
    return {
        'id': 'm-u',
        'role': 'user',
        'content': [{'type': 'text', 'text': text}],
        'source': {'kind': 'user'},
    }


def options(**overrides: Any) -> GenerateOptions:
    base = GenerateOptions(
        provider='deepseek-official', model='deepseek-v4-flash', messages=[]
    )
    base.update(overrides)
    return base


def test_serialize_request_basics() -> None:
    tool_schema = {
        'name': 'bash',
        'description': 'run',
        'parameters': {'type': 'object'},
    }
    body = serialize_request(
        options(
            messages=[user_message('hi')],
            system='you are helpful',
            tools=[tool_schema],
            temperature=0.5,
            maxTokens=100,
            stop=['END'],
        )
    )
    assert body == {
        'model': 'deepseek-v4-flash',
        'messages': [
            {'role': 'system', 'content': 'you are helpful'},
            {'role': 'user', 'content': 'hi'},
        ],
        'stream': True,
        'stream_options': {'include_usage': True},
        'tools': [{'type': 'function', 'function': tool_schema}],
        'temperature': 0.5,
        'max_tokens': 100,
        'stop': ['END'],
    }


def test_serialize_request_thinking_resolution() -> None:
    # no defaults, no request effort: nothing on the wire
    body = serialize_request(options(messages=[]))
    assert 'thinking' not in body and 'reasoning_effort' not in body
    # adapter default effort
    defaults = RequestDefaults(reasoning_effort='high')
    body = serialize_request(options(messages=[]), defaults)
    assert body['thinking'] == {'type': 'enabled'}
    assert body['reasoning_effort'] == 'high'
    # off disables thinking and is never a wire effort
    body = serialize_request(options(reasoningEffort='off', messages=[]))
    assert body['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in body
    # a deployment with thinking disabled rejects a non-off effort
    with pytest.raises(LlmError) as exc_info:
        serialize_request(
            options(reasoningEffort='high', messages=[]),
            RequestDefaults(thinking='disabled'),
        )
    assert exc_info.value.code == 'UNSUPPORTED_REASONING_EFFORT'
    # unknown effort values reject before the wire
    with pytest.raises(LlmError) as exc_info:
        serialize_request(options(reasoningEffort='low', messages=[]))
    assert exc_info.value.code == 'UNSUPPORTED_REASONING_EFFORT'
    # session-title calls always disable thinking
    body = serialize_request(
        options(purpose='session-title', messages=[]),
        RequestDefaults(reasoning_effort='high'),
    )
    assert body['thinking'] == {'type': 'disabled'}
    assert 'reasoning_effort' not in body


def test_serialize_messages_roles() -> None:
    tool_call_block = {
        'type': 'tool-call',
        'id': 'call-1',
        'name': 'bash',
        'arguments': '{}',
    }
    messages = [
        {
            'id': 'm-s',
            'role': 'system',
            'content': [{'type': 'text', 'text': 'rules'}],
            'source': {'kind': 'plugin', 'plugin': 'test'},
        },
        user_message('hello'),
        {
            'id': 'm-a',
            'role': 'assistant',
            'content': [
                {'type': 'reasoning', 'text': 'because'},
                tool_call_block,
            ],
            'source': {'kind': 'model', 'provider': 'p', 'model': 'm'},
        },
        {
            'id': 'm-t',
            'role': 'user',
            'content': [
                {
                    'type': 'tool-result',
                    'toolCallId': 'call-1',
                    'content': [],
                    'isError': False,
                }
            ],
            'source': {'kind': 'tool', 'callId': 'call-1'},
        },
        {
            'id': 'm-a2',
            'role': 'assistant',
            'content': [
                {'type': 'reasoning', 'text': 'hidden'},
                {'type': 'text', 'text': 'done'},
            ],
            'source': {'kind': 'model', 'provider': 'p', 'model': 'm'},
        },
    ]
    wire = serialize_messages(messages)  # type: ignore[arg-type]
    assert wire == [
        {'role': 'system', 'content': 'rules'},
        {'role': 'user', 'content': 'hello'},
        # reasoning passes back on tool-call turns; content is '' never null
        {
            'role': 'assistant',
            'content': '',
            'reasoning_content': 'because',
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'bash', 'arguments': '{}'},
                }
            ],
        },
        # empty tool output still needs SOME content on the wire
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': '(no output)'},
        # reasoning is dropped on tool-call-free turns to save tokens
        {'role': 'assistant', 'content': 'done'},
    ]


def test_serialize_messages_rejects_images() -> None:
    message = user_message('look')
    message['content'].append({'type': 'image', 'attachment': {'ref': 'x'}})
    with pytest.raises(LlmError) as exc_info:
        serialize_messages([message])  # type: ignore[arg-type]
    assert exc_info.value.code == 'UNSUPPORTED_CONTENT'


# --- the HTTP adapter (adapter.ts) over a mock transport ---

SUCCESS_SSE = (
    b'data: {"choices": [{"delta": {"reasoning_content": ""}}]}\n\n'
    b'data: {"choices": [{"delta": {"reasoning_content": "thinking"}}]}\n\n'
    b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}], '
    b'"usage": {"prompt_tokens": 12, "completion_tokens": 2, '
    b'"prompt_cache_hit_tokens": 4}}\n\n'
    b'data: [DONE]\n\n'
)


def connection_options(**overrides: Any) -> DeepSeekConnectionOptions:
    base: dict[str, Any] = {
        'base_url': 'https://api.test',
        'api_key_env': 'TEST_DEEPSEEK_KEY',
        'defaults': RequestDefaults(),
        'max_tokens': DEFAULT_MAX_TOKENS,
        'default_context_window': 1_000_000,
        'models': (),
        'stream_idle_timeout_ms': 300_000,
        'retry_policy': resolve_retry_policy(None, 'test'),
    }
    base.update(overrides)
    return DeepSeekConnectionOptions(**base)


def make_adapter(
    handler: Any,
    connection: DeepSeekConnectionOptions | None = None,
    api_key: str | LlmError = 'sk-test',
) -> DeepSeekAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def resolve_api_key(conn: DeepSeekConnectionOptions) -> str:
        if isinstance(api_key, LlmError):
            raise api_key
        return api_key

    return DeepSeekAdapter(
        options=lambda: connection or connection_options(),
        resolve_api_key=resolve_api_key,
        resolve_user_id=lambda: 'user-1',
        client=client,
    )


async def test_adapter_stream_success_and_request_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['headers'] = request.headers
        captured['body'] = json.loads(request.content)
        return httpx.Response(200, content=SUCCESS_SSE)

    adapter = make_adapter(
        handler, connection_options(defaults=RequestDefaults(reasoning_effort='high'))
    )
    stream = adapter.stream(
        options(messages=[user_message('hi')], sessionId='sess-1', purpose='compaction')
    )
    chunks = await collect(stream)
    assert [c['type'] for c in chunks] == [
        'block-start',
        'reasoning-delta',
        'block-start',
        'text-delta',
        'text-delta',
        'block-end',
        'block-end',
        'usage',
        'finish',
    ]
    usage = chunks[-2]['usage']
    assert usage == {'inputTokens': 8, 'outputTokens': 2, 'cacheReadTokens': 4}
    assert chunks[-1]['reason'] == {'kind': 'stop'}

    headers, body = captured['headers'], captured['body']
    assert headers['authorization'] == 'Bearer sk-test'
    assert headers['accept'] == 'text/event-stream'
    assert headers['user-agent'].startswith('deepseek-harness/')
    assert headers['x-deepseek-harness-user-id'] == 'user-1'
    assert headers['x-deepseek-harness-session-id'] == 'sess-1'
    assert headers['x-deepseek-harness-compact'] == '1'
    assert body['model'] == 'deepseek-v4-flash'
    assert body['stream'] is True
    assert body['thinking'] == {'type': 'enabled'}
    assert body['reasoning_effort'] == 'high'


async def test_adapter_http_error_mapping() -> None:
    cases = [
        (401, {}, 'AUTH'),
        (403, {}, 'AUTH'),
        (429, {}, 'RATE_LIMIT'),
        (
            400,
            {'error': {'message': 'maximum context length exceeded'}},
            'CONTEXT_WINDOW_EXCEEDED',
        ),
        (400, {'error': {'message': 'bad field'}}, 'INVALID_REQUEST'),
        (429, {'error': {'message': 'insufficient balance'}}, 'QUOTA'),
        (500, {}, 'SERVER'),
        (418, {}, 'HTTP_418'),
    ]
    for status, body, code in cases:
        def handler(
            request: httpx.Request, s: int = status, b: Any = body
        ) -> httpx.Response:
            return httpx.Response(s, json=b)

        adapter = make_adapter(handler)
        with pytest.raises(LlmError) as exc_info:
            await collect(adapter.stream(options(messages=[user_message('hi')])))
        assert exc_info.value.code == code, status
        assert exc_info.value.failure['status'] == status


async def test_adapter_error_facts_retry_after_and_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={'error': {'message': 'slow down'}},
            headers={'retry-after': '2', 'x-request-id': 'req-1'},
        )

    adapter = make_adapter(handler)
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(options(messages=[user_message('hi')])))
    error = exc_info.value
    assert str(error) == 'slow down'
    assert error.failure['providerRetryAfterMs'] == 2000
    assert error.failure['requestId'] == 'req-1'


async def test_adapter_malformed_error_body_keeps_status() -> None:
    adapter = make_adapter(lambda request: httpx.Response(502, content=b'not json'))
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(options(messages=[user_message('hi')])))
    assert exc_info.value.code == 'SERVER'
    assert str(exc_info.value) == 'DeepSeek API error (HTTP 502)'


class HangingByteStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncGenerator[bytes]:
        yield b'data: {"choices": []}\n\n'
        await asyncio.sleep(3600)

    async def aclose(self) -> None:
        pass


async def test_adapter_idle_timeout_maps_to_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=HangingByteStream())

    adapter = make_adapter(handler, connection_options(stream_idle_timeout_ms=50))
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(options(messages=[user_message('hi')])))
    assert exc_info.value.code == 'TIMEOUT'


async def test_adapter_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('dns failed')

    adapter = make_adapter(handler)
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(options(messages=[user_message('hi')])))
    assert exc_info.value.code == 'TRANSPORT'
    assert 'https://api.test' in str(exc_info.value)


async def test_adapter_missing_credential() -> None:
    adapter = make_adapter(
        lambda request: httpx.Response(200),
        api_key=LlmError('no key', 'MISSING_CREDENTIAL'),
    )
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(options(messages=[user_message('hi')])))
    assert exc_info.value.code == 'MISSING_CREDENTIAL'


async def test_adapter_caller_abort_maps_to_aborted() -> None:
    adapter = make_adapter(lambda request: httpx.Response(200, content=SUCCESS_SSE))
    signal = asyncio.Event()
    signal.set()
    aborted_options = options(messages=[user_message('hi')], signal=signal)
    with pytest.raises(LlmError) as exc_info:
        await collect(adapter.stream(aborted_options))
    assert exc_info.value.code == 'ABORTED'


def test_retry_after_parsing() -> None:
    assert provider_retry_after_ms(None) is None
    assert provider_retry_after_ms('3') == 3000
    assert provider_retry_after_ms('0') is None
    assert provider_retry_after_ms('not a date') is None


def test_http_error_code_without_body() -> None:
    assert http_error_code(503) == 'SERVER'
    assert http_error_code(400) == 'INVALID_REQUEST'


# --- the plugin (index.ts) ---


def test_config_defaults() -> None:
    validated = Config(None)
    assert validated['apiKeyEnv'] == 'DEEPSEEK_API_KEY'
    assert validated['maxTokens'] == DEFAULT_MAX_TOKENS
    assert validated['defaultContextWindow'] == 1_000_000
    assert validated['streamIdleTimeoutMs'] == 300_000
    model_ids = [m.id for m in validated['models']]
    assert model_ids == ['deepseek-v4-flash', 'deepseek-v4-pro']


def test_config_validation_failures() -> None:
    bad_configs = [
        {'unknown': 1},
        {'thinking': 'maybe'},
        {'reasoningEffort': 'low'},
        {'maxTokens': 0},
        {'defaultContextWindow': -1},
        {'streamIdleTimeoutMs': 0},
        {'streamIdleTimeoutMs': 2_147_483_648},
        {'thinking': 'disabled', 'reasoningEffort': 'high'},
        {'models': [{'id': 'a'}, {'id': 'a'}]},
        {'models': [{'id': ''}]},
        {'retryPolicy': {'mode': 'sometimes'}},
    ]
    for config in bad_configs:
        with pytest.raises(ValueError, match='llm-deepseek'):
            Config(config)


async def test_inject_gating_and_registration() -> None:
    import pydsh.llm_deepseek as deepseek_plugin

    ctx = Context()
    fiber = ctx.plugin(deepseek_plugin)
    # the declared inject=['llm'] keeps the plugin PENDING until the runtime loads
    assert fiber.state is FiberState.PENDING
    runtime_fiber = ctx.plugin(LlmRuntime)
    await runtime_fiber
    await fiber
    assert fiber.state is FiberState.ACTIVE
    llm = ctx.get('llm')
    assert llm.list_providers() == [{'id': 'deepseek-official', 'name': 'DeepSeek'}]
    policy = llm.provider_retry_policy('deepseek-official')
    assert policy.mode == 'normal'

    await fiber.dispose()
    assert llm.list_providers() == []
    # losing the llm service parks the plugin again
    fiber2 = ctx.plugin(deepseek_plugin)
    await fiber2
    await runtime_fiber.dispose()
    assert fiber2.state is FiberState.PENDING


async def test_config_failure_fails_the_fiber() -> None:
    import pydsh.llm_deepseek as deepseek_plugin

    ctx = Context()
    await ctx.plugin(LlmRuntime)
    fiber = ctx.plugin(deepseek_plugin, {'streamIdleTimeoutMs': -1})
    with pytest.raises(ValueError, match='llm-deepseek'):
        await fiber
    assert fiber.state is FiberState.FAILED


async def test_plugin_stream_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pydsh.llm_deepseek as deepseek_plugin

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['headers'] = request.headers
        captured['body'] = json.loads(request.content)
        return httpx.Response(200, content=SUCCESS_SSE)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        deepseek_plugin.httpx,
        'AsyncClient',
        lambda *args, **kwargs: real_client(transport=transport),
    )
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-plugin')

    ctx = Context()
    await ctx.plugin(LlmRuntime)
    fiber = ctx.plugin(deepseek_plugin)
    await fiber

    llm = ctx.get('llm')
    chunks = await collect(llm.stream(options(messages=[user_message('hi')])))
    assert chunks[-1] == {'type': 'finish', 'reason': {'kind': 'stop'}}
    assert captured['headers']['authorization'] == 'Bearer sk-plugin'
    # runtime-materialized adapter defaults reach the wire
    assert captured['body']['reasoning_effort'] == 'high'
    assert captured['body']['max_tokens'] == DEFAULT_MAX_TOKENS


async def test_plugin_missing_key_surfaces_as_finish_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pydsh.llm_deepseek as deepseek_plugin

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError('no request may leave without a credential')

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        deepseek_plugin.httpx,
        'AsyncClient',
        lambda *args, **kwargs: real_client(transport=transport),
    )
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.delenv('DEEPSEEK_BASE_URL', raising=False)

    ctx = Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(deepseek_plugin)
    stream = ctx.get('llm').stream(options(messages=[user_message('hi')]))
    chunks = await collect(stream)
    reason = chunks[-1]['reason']
    assert reason['kind'] == 'error'
    assert reason['failure']['code'] == 'MISSING_CREDENTIAL'


def test_base_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydsh.llm_deepseek import resolve_adapter_options

    config = Config(None)
    monkeypatch.delenv('DEEPSEEK_BASE_URL', raising=False)
    assert resolve_adapter_options(config).base_url == PUBLIC_BASE_URL
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'https://internal.test')
    assert resolve_adapter_options(config).base_url == 'https://internal.test'
    config = Config({'baseURL': 'https://explicit.test'})
    assert resolve_adapter_options(config).base_url == 'https://explicit.test'


async def test_adapter_model_metadata() -> None:
    adapter = make_adapter(
        lambda request: httpx.Response(200),
        connection_options(
            models=(
                DeepSeekCatalogModel(id='m-known', name='Known', context_window=64_000),
            )
        ),
    )
    info = adapter.resolve_model('deepseek-official', 'm-known')
    assert info['name'] == 'Known'
    assert info['context'] == {'contextWindow': 64_000}
    assert info['defaultMaxTokens'] == DEFAULT_MAX_TOKENS
    assert info['reasoning']['defaultEffort'] == 'high'
    assert [e['id'] for e in info['reasoning']['efforts']] == ['off', 'high', 'max']
    # uncatalogued models fall back to deployment defaults, text-only
    fallback = adapter.resolve_model('deepseek-official', 'm-other')
    assert fallback['name'] == 'm-other'
    assert fallback['context'] == {'contextWindow': 1_000_000}
    assert fallback['inputModalities'] == ['text']
    # thinking-disabled deployments offer only "off"
    disabled = make_adapter(
        lambda request: httpx.Response(200),
        connection_options(defaults=RequestDefaults(thinking='disabled')),
    )
    info = disabled.resolve_model('deepseek-official', 'm')
    assert [e['id'] for e in info['reasoning']['efforts']] == ['off']
    assert info['reasoning']['defaultEffort'] == 'off'
