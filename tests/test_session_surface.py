"""Behavior tests for the session surface fold, JSON boundary, and headers.

Ported-semantics tests for the pydsh.session Python port (ported from
deepseek-harness packages/core/session, MIT).
"""

from __future__ import annotations

import pytest

from pydsh.session import (
    SNAPSHOT_FAILED,
    Session,
    SessionId,
    canonical_header,
    derive_event_message,
    fold_request_header,
    fold_surface,
    header_equals,
    is_append_surface_event,
    is_json_value,
    is_replacement_surface_event,
    is_surface_eligible_type,
    is_surface_event,
    snapshot_json_value,
)


def user_message(text: str, message_id: str = 'm-user') -> dict:
    return {
        'id': message_id,
        'role': 'user',
        'source': {'kind': 'user'},
        'content': [{'type': 'text', 'text': text}],
    }


def assistant_message(text: str, message_id: str = 'm-assistant') -> dict:
    return {
        'id': message_id,
        'role': 'assistant',
        'source': {'kind': 'model', 'provider': 'mock', 'model': 'mock'},
        'content': [{'type': 'text', 'text': text}],
    }


def tool_result_message(call_id: str, text: str = 'ok') -> dict:
    return {
        'id': f'm-tool-{call_id}',
        'role': 'user',
        'source': {'kind': 'tool', 'callId': call_id},
        'content': [
            {
                'type': 'tool-result',
                'toolCallId': call_id,
                'content': [{'type': 'text', 'text': text}],
            }
        ],
    }


def tool_result_data(call_id: str, text: str = 'ok', **extra: object) -> dict:
    return {
        'turn': 1,
        'step': 1,
        'message': tool_result_message(call_id, text),
        **extra,
    }


# --- lossless-JSON boundary ---


async def test_is_json_value_and_snapshot() -> None:
    tree = {'a': [1, 'x', True, None, 1.5], 'b': {'c': []}}
    assert is_json_value(tree)
    snapshot = snapshot_json_value(tree)
    assert snapshot == tree
    assert snapshot is not tree
    tree['a'].append('later')
    assert snapshot['a'] == [1, 'x', True, None, 1.5]
    # JSON null is valid and snapshot-able, distinct from the sentinel
    assert snapshot_json_value(None) is None
    assert SNAPSHOT_FAILED is not None


@pytest.mark.parametrize(
    'value',
    [
        float('nan'),
        float('inf'),
        -0.0,
        (1, 2),
        {1, 2},
        object(),
    ],
)
async def test_json_rejects_non_lossless_scalars(value: object) -> None:
    assert not is_json_value(value)
    assert snapshot_json_value(value) is SNAPSHOT_FAILED


async def test_json_rejects_exotic_containers() -> None:
    class DictSubclass(dict):
        pass

    assert not is_json_value(DictSubclass())
    assert not is_json_value({'k': (1,)})
    assert not is_json_value({1: 'x'})
    cyclic: list = []
    cyclic.append(cyclic)
    assert not is_json_value(cyclic)
    assert snapshot_json_value(cyclic) is SNAPSHOT_FAILED


# --- surface fold ---


def _ev(type_: str, seq: int, data: dict, time: int = 1, **extra: object) -> dict:
    return {'type': type_, 'seq': seq, 'time': time, 'data': data, **extra}


async def test_fold_surface_appends_and_skips_log_only() -> None:
    events = [
        _ev('turn/start', 0, {'turn': 1}),
        _ev('user/message', 1, user_message('a'), surfaceOp='append'),
        _ev('assistant/chunk', 2, {'turn': 1, 'step': 1, 'chunk': {}}),
        _ev(
            'assistant/message',
            3,
            {'turn': 1, 'step': 1, 'message': assistant_message('b')},
            surfaceOp='append',
        ),
    ]
    result = fold_surface(events)
    assert result['nodes'] == [1, 3]
    assert result['replacements'] == []


async def test_surface_replace_shadows_range_and_rebuilds_history() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('one', 'm1'), {'surfaceOp': 'append'})
    session.append('user/message', user_message('two', 'm2'), {'surfaceOp': 'append'})
    session.append('user/message', user_message('three', 'm3'), {'surfaceOp': 'append'})

    before = session.derive_messages()
    assert [m['id'] for m in before] == ['m1', 'm2', 'm3']

    session.append(
        'assistant/message',
        {'turn': 1, 'step': 1, 'message': assistant_message('summary', 'm4')},
        {
            'surfaceOp': {'op': 'replace', 'start': 1, 'end': 2},
            'sourceEventSeqs': [1, 2],
        },
    )
    assert session.surface.nodes == [0, 3]
    assert session.surface.replace_generation == 1
    # the replace generation invalidates the derived-message cache
    after = session.derive_messages()
    assert [m['id'] for m in after] == ['m1', 'm4']
    assert session.surface is session.surface


async def test_surface_replace_single_node() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('one', 'm1'), {'surfaceOp': 'append'})
    session.append(
        'user/message',
        user_message('one-edited', 'm1b'),
        {
            'surfaceOp': {'op': 'replace', 'start': 0, 'end': 0},
            'sourceEventSeqs': [0],
        },
    )
    assert session.surface.nodes == [1]
    assert [m['id'] for m in session.derive_messages()] == ['m1b']


async def test_surface_rejects_non_contiguous_seq() -> None:
    events = [_ev('user/message', 1, user_message('a'), surfaceOp='append')]
    with pytest.raises(ValueError, match='not contiguous; expected 0'):
        fold_surface(events)


async def test_surface_marker_validation() -> None:
    session = Session.create(SessionId('surf'))
    with pytest.raises(ValueError, match='requires a surfaceOp marker'):
        session.append('user/message', user_message('a'))
    with pytest.raises(ValueError, match='carries an invalid surfaceOp'):
        session.append(
            'user/message', user_message('a'), {'surfaceOp': 'bogus'}
        )
    with pytest.raises(ValueError, match='carries an invalid replace surfaceOp'):
        session.append(
            'user/message', user_message('a'), {'surfaceOp': {'op': 'replace'}}
        )
    with pytest.raises(ValueError, match='not surface-eligible and cannot carry'):
        session.append('turn/start', {'turn': 1}, {'sourceEventSeqs': [0]})


async def test_surface_replace_range_validation() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('one', 'm1'), {'surfaceOp': 'append'})
    session.append('user/message', user_message('two', 'm2'), {'surfaceOp': 'append'})

    with pytest.raises(ValueError, match='start seq 9 not found in surface'):
        session.append(
            'user/message',
            user_message('x'),
            {'surfaceOp': {'op': 'replace', 'start': 9, 'end': 9}},
        )
    with pytest.raises(
        ValueError, match='start seq 1 \\(index 1\\) is after end seq 0'
    ):
        session.append(
            'user/message',
            user_message('x'),
            {
                'surfaceOp': {'op': 'replace', 'start': 1, 'end': 0},
                'sourceEventSeqs': [0, 1],
            },
        )


async def test_surface_provenance_validation() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('one', 'm1'), {'surfaceOp': 'append'})

    with pytest.raises(ValueError, match='must reference earlier events: 1'):
        session.append(
            'user/message',
            user_message('x'),
            {'surfaceOp': 'append', 'sourceEventSeqs': [1]},
        )
    with pytest.raises(ValueError, match='must not contain duplicates'):
        session.append(
            'user/message',
            user_message('x'),
            {'surfaceOp': 'append', 'sourceEventSeqs': [0, 0]},
        )
    with pytest.raises(
        ValueError, match='must not be empty except on assistant/message'
    ):
        session.append(
            'user/message',
            user_message('x'),
            {'surfaceOp': 'append', 'sourceEventSeqs': []},
        )

    # an empty source list is allowed on assistant/message (known empty stream)
    session.append(
        'assistant/message',
        {'turn': 1, 'step': 1, 'message': assistant_message('b')},
        {'surfaceOp': 'append', 'sourceEventSeqs': []},
    )
    assert session.surface.nodes == [0, 1]


async def test_surface_replace_requires_shadowed_coverage() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('one', 'm1'), {'surfaceOp': 'append'})
    session.append('user/message', user_message('two', 'm2'), {'surfaceOp': 'append'})
    session.append('user/message', user_message('three', 'm3'), {'surfaceOp': 'append'})
    with pytest.raises(
        ValueError, match='must include every shadowed surface node; missing 2'
    ):
        session.append(
            'user/message',
            user_message('x'),
            {
                'surfaceOp': {'op': 'replace', 'start': 1, 'end': 2},
                'sourceEventSeqs': [1],
            },
        )


async def test_tool_result_rewrite_restrictions() -> None:
    session = Session.create(SessionId('surf'))
    session.append('user/message', user_message('q', 'm0'), {'surfaceOp': 'append'})
    session.append(
        'tool/result',
        tool_result_data('c1', 'old'),
        {'surfaceOp': 'append'},
    )
    # content-only rewrite is allowed
    session.append(
        'tool/result',
        tool_result_data('c1', 'new'),
        {
            'surfaceOp': {'op': 'replace', 'start': 1, 'end': 1},
            'sourceEventSeqs': [1],
        },
    )
    assert session.surface.nodes == [0, 2]
    assert session.derive_messages()[1]['content'][0]['content'][0]['text'] == 'new'

    # changing anything beyond content is rejected
    with pytest.raises(ValueError, match='may change only content'):
        session.append(
            'tool/result',
            tool_result_data('c1', 'x', error={'name': 'E', 'code': 'C'}),
            {
                'surfaceOp': {'op': 'replace', 'start': 2, 'end': 2},
                'sourceEventSeqs': [2],
            },
        )
    # the replacement must target a current tool/result
    with pytest.raises(
        ValueError, match='must target a current tool/result'
    ):
        session.append(
            'tool/result',
            tool_result_data('c1', 'x'),
            {
                'surfaceOp': {'op': 'replace', 'start': 0, 'end': 0},
                'sourceEventSeqs': [0],
            },
        )
    # and it must shadow exactly one node
    with pytest.raises(
        ValueError, match='must rewrite exactly one current node'
    ):
        session.append(
            'tool/result',
            tool_result_data('c1', 'x'),
            {
                'surfaceOp': {'op': 'replace', 'start': 0, 'end': 2},
                'sourceEventSeqs': [0, 2],
            },
        )


async def test_surface_guards_and_projection() -> None:
    appended = _ev('user/message', 0, user_message('a'), surfaceOp='append')
    replaced = _ev(
        'user/message',
        1,
        user_message('b'),
        surfaceOp={'op': 'replace', 'start': 0, 'end': 0},
        sourceEventSeqs=[0],
    )
    chunk = _ev('assistant/chunk', 2, {'turn': 1, 'step': 1, 'chunk': {}})
    markerless = _ev('user/message', 3, user_message('c'))

    assert is_surface_eligible_type('user/message')
    assert not is_surface_eligible_type('turn/start')
    assert is_surface_event(appended)
    assert is_append_surface_event(appended)
    assert not is_replacement_surface_event(appended)
    assert is_replacement_surface_event(replaced)
    assert not is_surface_event(chunk)
    assert not is_surface_event(markerless)

    assert derive_event_message(appended) == appended['data']
    assert derive_event_message(chunk) is None
    empty_assistant = _ev(
        'assistant/message',
        4,
        {'turn': 1, 'step': 1, 'message': assistant_message('')},
        surfaceOp='append',
    )
    empty_assistant['data']['message']['content'] = []
    assert derive_event_message(empty_assistant) is None
    assert Session.derive_event_message(appended) == appended['data']


# --- request-header reconstruction ---


def _header(**overrides: object) -> dict:
    header: dict = {'config': {'provider': 'deepseek', 'model': 'chat'}}
    header.update(overrides)
    return header


async def test_canonical_header_drops_empty_optionals() -> None:
    assert canonical_header(_header(system='', tools=[])) == _header()
    kept = _header(
        system='sys',
        tools=[{'name': 'echo'}],
        adapterDefaults={'reasoningEffort': True},
    )
    assert canonical_header(kept) == kept
    # adapter defaults without a True marker are dropped
    assert canonical_header(_header(adapterDefaults={})) == _header()


async def test_header_equals() -> None:
    base = canonical_header(_header(system='s', tools=[{'name': 'a'}]))
    same = canonical_header(_header(system='s', tools=[{'name': 'a'}]))
    other_system = canonical_header(_header(system='t', tools=[{'name': 'a'}]))
    assert header_equals(base, same)
    assert not header_equals(base, other_system)
    assert not header_equals(
        base, canonical_header(_header(system='s', tools=[{'name': 'b'}]))
    )
    assert not header_equals(base, canonical_header(_header()))
    with_defaults = canonical_header(
        _header(system='s', tools=[{'name': 'a'}], adapterDefaults={'maxTokens': True})
    )
    assert not header_equals(base, with_defaults)


async def test_fold_request_header() -> None:
    events = [
        _ev('turn/start', 0, {'turn': 1}),
        _ev(
            'request/header',
            1,
            {'header': _header(system='one'), 'reason': 'initial'},
        ),
        _ev(
            'request/header',
            2,
            {'header': _header(system='two'), 'reason': 'change'},
        ),
    ]
    assert fold_request_header([]) is None
    assert fold_request_header(events) == _header(system='two')
    # continue from a previously folded state
    assert fold_request_header(events[2:], _header(system='zero')) == _header(
        system='two'
    )
    assert fold_request_header(events[:1], _header(system='zero')) == _header(
        system='zero'
    )
