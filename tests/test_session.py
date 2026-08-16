"""Behavior tests for the in-memory session store and event log.

Ported-semantics tests for the pydsh.session Python port (ported from
deepseek-harness packages/core/session, MIT).
"""

from __future__ import annotations

import asyncio

import pytest

from pycordis import Context
from pydsh.session import (
    SESSION_FORMAT_VERSION,
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    CreateSessionMeta,
    CreateSessionOptions,
    Session,
    SessionForkError,
    SessionId,
    SessionPreparation,
    SessionStore,
    adopt_session_event,
    interrupted_turn_closers,
    snapshot_session_event,
)


def user_message(
    text: str, message_id: str = 'm-user', source: dict | None = None
) -> dict:
    return {
        'id': message_id,
        'role': 'user',
        'source': source or {'kind': 'user'},
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


def make_store() -> tuple[Context, SessionStore]:
    ctx = Context()
    return ctx, SessionStore(ctx)


async def test_create_mints_ids_and_registers_service() -> None:
    ctx, store = make_store()
    assert ctx.sessions.list() == []

    first = store.create()
    second = store.create()
    assert first.id == 'session-1'
    assert second.id == 'session-2'
    assert store.list() == [first, second]
    assert store.get(SessionId('session-1')) is first
    assert store.get(SessionId('missing')) is None
    # a manually claimed id is skipped by the mint
    _, store2 = make_store()
    claimed = store2.create(SessionId('session-1'))
    assert claimed.id == 'session-1'
    assert store2.create().id == 'session-2'


async def test_create_duplicate_id_rejected() -> None:
    _, store = make_store()
    store.create(SessionId('s'))
    with pytest.raises(ValueError, match='session "s" already exists'):
        store.create(SessionId('s'))
    # prepare() checks eagerly, and enter() re-checks after arbitrary
    # interleaved work (here: a create racing the prepared session)
    with pytest.raises(ValueError, match='session "s" already exists'):
        store.prepare(SessionId('s'))
    prepared = store.prepare(SessionId('raced'))
    store.create(SessionId('raced'))
    with pytest.raises(ValueError, match='session "raced" already exists'):
        store.enter(prepared)


async def test_created_and_disposed_events() -> None:
    ctx, store = make_store()
    created: list[Session] = []
    disposed: list[Session] = []
    ctx.on('session/created', lambda s: created.append(s))
    ctx.on('session/disposed', lambda s: disposed.append(s))

    session = store.create()
    assert created == [session]
    assert disposed == []

    # manual lifecycle to control detach explicitly
    manual = store.prepare()
    detach = store.enter(manual)
    store.announce(manual)
    assert created == [session, manual]
    detach()
    assert disposed == [manual]
    assert store.get(manual.id) is None
    # detach is single-shot
    detach()
    assert disposed == [manual]


async def test_created_veto_rolls_back_with_paired_disposal() -> None:
    ctx, store = make_store()
    disposed: list[Session] = []
    ctx.on('session/disposed', lambda s: disposed.append(s))

    def veto(session: Session) -> None:
        raise RuntimeError('veto')

    ctx.on('session/created', veto)
    with pytest.raises(RuntimeError, match='veto'):
        store.create(SessionId('vetoed'))
    # the attach was rolled back, and the partial creation paired a disposal
    assert store.get(SessionId('vetoed')) is None
    assert [s.id for s in disposed] == ['vetoed']


async def test_announce_twice_and_unentered_rejected() -> None:
    _, store = make_store()
    session = store.prepare()
    with pytest.raises(ValueError, match='not live in this store'):
        store.announce(session)
    store.enter(session)
    store.announce(session)
    with pytest.raises(ValueError, match='already announced'):
        store.announce(session)


async def test_detach_during_announce_is_deferred() -> None:
    ctx, store = make_store()
    disposed: list[Session] = []
    ctx.on('session/disposed', lambda s: disposed.append(s))
    session = store.prepare()
    detach = store.enter(session)

    def on_created(s: Session) -> None:
        detach()
        # removal waits until the creation dispatch unwinds
        assert store.get(s.id) is s

    ctx.on('session/created', on_created)
    store.announce(session)
    assert store.get(session.id) is None
    assert disposed == [session]


async def test_append_publishes_after_commit() -> None:
    ctx, store = make_store()
    seen: list[tuple[Session, dict]] = []

    def listener(s: Session, event: dict) -> None:
        seen.append((s, event))
        # the append is committed before listeners observe it
        assert s.seq == event['seq'] + 1

    ctx.on('session/event', listener)
    session = store.create()
    event = session.append('turn/start', {'turn': 1})
    assert event['seq'] == 0
    assert event['type'] == 'turn/start'
    assert isinstance(event['time'], int)
    assert event['data'] == {'turn': 1}
    assert seen == [(session, event)]
    assert session.events == (event,)
    # a previously returned snapshot does not grow later
    snapshot = session.events
    session.append('turn/end', {'turn': 1, 'reason': {'kind': 'completed'}})
    assert len(snapshot) == 1
    assert len(session.events) == 2


async def test_append_detaches_caller_data() -> None:
    _, store = make_store()
    session = store.create()
    data = {'todos': [{'content': 'x', 'status': 'pending'}]}
    event = session.append('todo/write', data)
    data['todos'].append({'content': 'y', 'status': 'completed'})
    assert event['data']['todos'] == [{'content': 'x', 'status': 'pending'}]


@pytest.mark.parametrize(
    'payload',
    [
        {'v': float('nan')},
        {'v': float('inf')},
        {'v': -0.0},
        {'v': (1, 2)},
        {'v': object()},
        {1: 'x'},
    ],
)
async def test_append_rejects_non_json_data(payload: dict) -> None:
    _, store = make_store()
    session = store.create()
    with pytest.raises(ValueError, match='non-JSON-serializable data'):
        session.append('todo/write', payload)
    assert session.seq == 0


async def test_append_rejects_cyclic_data() -> None:
    _, store = make_store()
    session = store.create()
    cyclic: dict = {'todos': []}
    cyclic['self'] = cyclic
    with pytest.raises(ValueError, match='non-JSON-serializable data'):
        session.append('todo/write', cyclic)


async def test_append_surface_metadata_rules() -> None:
    _, store = make_store()
    session = store.create()
    with pytest.raises(ValueError, match='requires a surfaceOp marker'):
        session.append('user/message', user_message('hi'))
    with pytest.raises(ValueError, match='not surface-eligible and cannot carry'):
        session.append('turn/start', {'turn': 1}, {'surfaceOp': 'append'})
    assert session.seq == 0


async def test_append_reentrancy_rejected() -> None:
    ctx, store = make_store()
    errors: list[str] = []

    def listener(s: Session, event: dict) -> None:
        try:
            s.append('todo/write', {'todos': []})
        except RuntimeError as error:
            errors.append(str(error))

    ctx.on('session/event', listener)
    session = store.create()
    session.append('turn/start', {'turn': 1})
    assert errors == [
        'session append cannot reenter while another append is being published'
    ]
    # only the outer append committed
    assert session.seq == 1


async def test_event_listener_failure_is_contained() -> None:
    ctx, store = make_store()

    def bad(s: Session, event: dict) -> None:
        raise RuntimeError('boom')

    async def bad_async(s: Session, event: dict) -> None:
        raise RuntimeError('async boom')

    ctx.on('session/event', bad)
    ctx.on('session/event', bad_async)
    session = store.create()
    event = session.append('turn/start', {'turn': 1})
    assert session.events == (event,)
    await asyncio.sleep(0)


async def test_seed_replay_appends_end_seed_marker_once() -> None:
    original = Session.create(SessionId('s3'))
    original.append('turn/start', {'turn': 1})
    original.append('user/message', user_message('q'), {'surfaceOp': 'append'})
    original.append('turn/end', {'turn': 1, 'reason': {'kind': 'completed'}})

    replayed = Session.create(SessionId('s3-replay'), list(original.events))
    assert list(replayed.events[: original.seq]) == list(original.events)
    assert replayed.seq == original.seq + 1
    assert replayed.first_live_seq == original.seq
    assert replayed.events[-1]['type'] == 'session/end-seed'
    assert replayed.events[-1]['data'] == {}
    assert replayed.derive_messages() == original.derive_messages()

    # reopening an untouched session does not grow its log per pickup
    reopened = Session.create(SessionId('s3-reopen'), list(replayed.events))
    assert list(reopened.events) == list(replayed.events)
    assert reopened.first_live_seq == replayed.seq


async def test_empty_seed_marked_but_fresh_session_not() -> None:
    fresh = Session.create(SessionId('fresh'))
    assert fresh.events == ()

    resumed = Session.create(SessionId('resumed'), [])
    assert resumed.first_live_seq == 0
    assert resumed.events[0]['type'] == 'session/end-seed'
    assert resumed.events[0]['data'] == {}


async def test_seed_validation() -> None:
    gap = {'type': 'turn/start', 'seq': 1, 'time': 1, 'data': {'turn': 1}}
    with pytest.raises(ValueError, match='must be contiguous from 0'):
        Session.create(SessionId('bad'), [gap])

    extra = {
        'type': 'turn/start',
        'seq': 0,
        'time': 1,
        'data': {'turn': 1},
        'bogus': 1,
    }
    with pytest.raises(ValueError, match='invalid event envelope'):
        Session.create(SessionId('bad'), [extra])

    missing_data = {'type': 'turn/start', 'seq': 0, 'time': 1}
    with pytest.raises(ValueError, match='invalid event envelope'):
        Session.create(SessionId('bad'), [missing_data])

    bad_message = {
        'type': 'user/message',
        'seq': 0,
        'time': 1,
        'data': {'role': 'user'},
        'surfaceOp': 'append',
    }
    with pytest.raises(ValueError, match='lacks an identified message'):
        Session.create(SessionId('bad'), [bad_message])

    legacy = {
        'type': 'request/header-delta',
        'seq': 0,
        'time': 1,
        'data': {'delta': {}},
    }
    with pytest.raises(ValueError, match='legacy request/header-delta'):
        Session.create(SessionId('bad'), [legacy])

    old_header = {
        'type': 'request/header',
        'seq': 0,
        'time': 1,
        'data': {'header': {'config': {'model': 'old'}}, 'reason': 'initial'},
    }
    with pytest.raises(ValueError, match='lacks provider/model'):
        Session.create(SessionId('bad'), [old_header])

    non_json = {'type': 'turn/start', 'seq': 0, 'time': 1, 'data': {'v': float('nan')}}
    with pytest.raises(ValueError, match='not losslessly JSON-serializable'):
        Session.create(SessionId('bad'), [non_json])


async def test_append_rejects_legacy_request_header_reason() -> None:
    _, store = make_store()
    session = store.create()
    header = {'config': {'provider': 'p', 'model': 'm'}}
    with pytest.raises(ValueError, match='legacy request/header reason'):
        session.append('request/header', {'header': header, 'reason': 'fallback'})


async def test_header_meta_and_cwd_validation() -> None:
    _, store = make_store()
    session = store.create(
        options=CreateSessionOptions(
            meta=CreateSessionMeta(cwd='/repo', agentPreset='default')
        )
    )
    assert session.header['version'] == SESSION_FORMAT_VERSION
    assert session.header['id'] == session.id
    assert session.header['cwd'] == '/repo'
    assert session.header['agentPreset'] == 'default'
    assert isinstance(session.header['createdAt'], int)

    with pytest.raises(ValueError, match='cwd must be an absolute path'):
        store.create(options=CreateSessionOptions(meta={'cwd': 'relative'}))
    with pytest.raises(ValueError, match='cwd must be a string'):
        store.create(options=CreateSessionOptions(meta={'cwd': 1}))


async def test_derive_messages_projects_surface_only() -> None:
    session = Session.create(SessionId('s1'))
    session.append('turn/start', {'turn': 1})
    session.append('user/message', user_message('hello'), {'surfaceOp': 'append'})
    session.append(
        'assistant/chunk',
        {
            'turn': 1,
            'step': 1,
            'chunk': {'type': 'text-delta', 'index': 0, 'text': 'h'},
        },
    )
    message = assistant_message('let me check')
    message['content'].append(
        {'type': 'tool-call', 'id': 'c1', 'name': 'echo', 'arguments': '{}'}
    )
    session.append(
        'assistant/message',
        {'turn': 1, 'step': 1, 'message': message},
        {'surfaceOp': 'append'},
    )
    session.append(
        'tool/result',
        {'turn': 1, 'step': 1, 'message': tool_result_message('c1')},
        {'surfaceOp': 'append'},
    )
    session.append('turn/end', {'turn': 1, 'reason': {'kind': 'completed'}})

    messages = session.derive_messages()
    assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
    # raw chunks do not appear in derived history
    assert len(messages[1]['content']) == 2
    assert messages[2]['content'][0]['type'] == 'tool-result'


async def test_empty_assistant_message_hosts_usage_only() -> None:
    session = Session.create(SessionId('s1'))
    message = assistant_message('')
    message['content'] = []
    session.append(
        'assistant/message',
        {'turn': 1, 'step': 1, 'message': message, 'usage': {'totalTokens': 3}},
        {'surfaceOp': 'append'},
    )
    assert session.surface.nodes == [0]
    assert session.derive_messages() == []


async def test_request_header_and_context_folds() -> None:
    session = Session.create(SessionId('s1'))
    assert session.request_header() is None
    assert session.request_context() is None

    header = {
        'config': {'provider': 'deepseek', 'model': 'chat'},
        'system': '',
        'tools': [],
    }
    session.append('request/header', {'header': header, 'reason': 'initial'})
    # canonical form: empty system/tools become absent
    assert session.request_header() == {
        'config': {'provider': 'deepseek', 'model': 'chat'}
    }
    session.append(
        'request/context',
        {'provider': 'deepseek', 'model': 'chat', 'contextWindow': 64000},
    )
    assert session.request_context() == {
        'provider': 'deepseek',
        'model': 'chat',
        'contextWindow': 64000,
    }


async def test_fork_happy_path() -> None:
    _, store = make_store()
    source = store.create(options=CreateSessionOptions(meta={'cwd': '/repo'}))
    source.append('turn/start', {'turn': 1})
    source.append('user/message', user_message('q'), {'surfaceOp': 'append'})
    source.append('turn/end', {'turn': 1, 'reason': {'kind': 'completed'}})
    source.append('turn/start', {'turn': 2})
    source.append('turn/end', {'turn': 2, 'reason': {'kind': 'completed'}})

    child = store.fork(source)
    assert child.header['parentSession'] == source.id
    assert child.header['seedLength'] == 5
    assert child.header['cwd'] == '/repo'
    assert child.first_live_seq == 5
    assert list(child.events[:5]) == list(source.events)
    assert child.events[-1]['type'] == 'session/end-seed'
    assert child.derive_messages() == source.derive_messages()

    # fork by id through an explicit inclusive boundary between turns
    bounded = store.fork(source.id, 2, SessionId('bounded'))
    assert bounded.id == 'bounded'
    assert bounded.header['seedLength'] == 3
    assert [e['type'] for e in bounded.events] == [
        'turn/start',
        'user/message',
        'turn/end',
        'session/end-seed',
    ]

    # forking an empty source forks an empty child (marker only)
    empty = store.create()
    child_of_empty = store.fork(empty)
    assert child_of_empty.header['seedLength'] == 0
    assert child_of_empty.events[0]['type'] == 'session/end-seed'


async def test_fork_errors() -> None:
    _, store = make_store()
    source = store.create()
    source.append('turn/start', {'turn': 1})
    source.append('turn/end', {'turn': 1, 'reason': {'kind': 'completed'}})

    with pytest.raises(SessionForkError, match='not found') as not_found:
        store.fork(SessionId('nope'))
    assert not_found.value.code == 'SESSION_NOT_FOUND'

    detached = Session.create(source.id)
    with pytest.raises(SessionForkError, match='not the live store instance') as stale:
        store.fork(detached)
    assert stale.value.code == 'SESSION_NOT_LIVE'

    with pytest.raises(SessionForkError, match='already exists') as duplicate:
        store.fork(source, child_session_id=source.id)
    assert duplicate.value.code == 'SESSION_ALREADY_EXISTS'

    with pytest.raises(SessionForkError, match='non-negative safe integer') as neg:
        store.fork(source, -1)
    assert neg.value.code == 'INVALID_BOUNDARY'

    with pytest.raises(SessionForkError, match='does not exist') as beyond:
        store.fork(source, 99)
    assert beyond.value.code == 'INVALID_BOUNDARY'

    open_turn_source = store.create()
    open_turn_source.append('turn/start', {'turn': 1})
    with pytest.raises(SessionForkError, match='ends inside open turn 1') as open_:
        store.fork(open_turn_source)
    assert open_.value.code == 'OPEN_TURN'


async def test_flush_checkpoint() -> None:
    ctx, store = make_store()
    session = store.create()
    # no durability listener participates
    assert await store.flush(session) is False

    calls: list[str] = []

    async def first(s: Session) -> None:
        await asyncio.sleep(0)
        calls.append('first')

    def second(s: Session) -> None:
        calls.append('second')

    ctx.on('session/flush', first)
    ctx.on('session/flush', second)
    assert await store.flush(session) is True
    assert sorted(calls) == ['first', 'second']


async def test_flush_raises_first_failure_after_all_settle() -> None:
    ctx, store = make_store()
    session = store.create()
    ran: list[str] = []

    async def bad(s: Session) -> None:
        await asyncio.sleep(0)
        raise RuntimeError('flush boom')

    async def good(s: Session) -> None:
        await asyncio.sleep(0)
        ran.append('good')

    ctx.on('session/flush', bad)
    ctx.on('session/flush', good)
    with pytest.raises(RuntimeError, match='flush boom'):
        await store.flush(session)
    # every listener settled before the failure propagated
    assert ran == ['good']


async def test_flush_requires_live_session() -> None:
    _, store = make_store()
    detached = Session.create(SessionId('detached'))
    with pytest.raises(ValueError, match='not live in this store'):
        await store.flush(detached)


async def test_preparation_release() -> None:
    released: list[bool] = []
    session = Session.create(SessionId('p'))
    preparation = SessionPreparation.create(
        session, release=lambda: released.append(True)
    )
    assert preparation.session is session
    preparation.close()
    preparation.close()
    assert released == [True]

    with SessionPreparation.create(session, release=lambda: released.append(True)):
        pass
    assert released == [True, True]


async def test_event_snapshot_and_adopt() -> None:
    session = Session.create(SessionId('s'))
    session.append('user/message', user_message('hi'), {'surfaceOp': 'append'})
    event = session.events[0]
    snapshot = snapshot_session_event(event)
    assert snapshot == event
    assert snapshot is not event
    assert snapshot['data'] is not event['data']

    bad = {'type': 'user/message', 'seq': 0, 'time': 1, 'data': {'role': 'user'}}
    with pytest.raises(ValueError, match='lacks an identified message'):
        adopt_session_event(bad)


def _ev(type_: str, seq: int, data: dict, time: int = 1000, **extra: object) -> dict:
    return {'type': type_, 'seq': seq, 'time': time, 'data': data, **extra}


async def test_interrupted_turn_closers_balanced_log() -> None:
    assert interrupted_turn_closers([]) == []
    balanced = [
        _ev('turn/start', 0, {'turn': 1}),
        _ev('turn/end', 1, {'turn': 1, 'reason': {'kind': 'completed'}}),
    ]
    assert interrupted_turn_closers(balanced) == []


async def test_interrupted_turn_closers_open_turn_without_step() -> None:
    log = [_ev('turn/start', 0, {'turn': 1}, time=42)]
    closers = interrupted_turn_closers(log)
    assert len(closers) == 1
    assert closers[0]['type'] == 'turn/end'
    assert closers[0]['seq'] == 1
    assert closers[0]['time'] == 42
    assert closers[0]['data'] == {'turn': 1, 'reason': {'kind': 'interrupted'}}


async def test_interrupted_turn_closers_started_call() -> None:
    message = assistant_message('working')
    message['content'] = [
        {'type': 'tool-call', 'id': 'c1', 'name': 'echo', 'arguments': '{}'}
    ]
    log = [
        _ev('turn/start', 0, {'turn': 1}),
        _ev('step/start', 1, {'turn': 1, 'step': 1}),
        _ev(
            'assistant/message',
            2,
            {'turn': 1, 'step': 1, 'message': message},
            surfaceOp='append',
        ),
        _ev('tool/call', 3, {'turn': 1, 'step': 1, 'callId': 'c1',
                             'name': 'echo', 'arguments': '{}'}),
    ]
    closers = interrupted_turn_closers(log)
    assert [c['type'] for c in closers] == ['tool/result', 'step/end', 'turn/end']
    result = closers[0]
    assert result['seq'] == 4
    assert result['time'] == 1000
    assert result['surfaceOp'] == 'append'
    assert result['sourceEventSeqs'] == [3]
    assert result['data']['error'] == {
        'name': 'ToolOutcomeUnknownError',
        'code': TOOL_OUTCOME_UNKNOWN,
    }
    assert result['data']['message']['source'] == {'kind': 'tool', 'callId': 'c1'}
    assert result['data']['message']['content'][0]['isError'] is True
    assert closers[1]['data'] == {'turn': 1, 'step': 1}
    assert closers[2]['data']['reason'] == {'kind': 'interrupted'}


async def test_interrupted_turn_closers_not_started_call() -> None:
    message = assistant_message('working')
    message['content'] = [
        {'type': 'tool-call', 'id': 'c2', 'name': 'echo', 'arguments': '{}'}
    ]
    log = [
        _ev('turn/start', 0, {'turn': 1}),
        _ev('step/start', 1, {'turn': 1, 'step': 1}),
        _ev(
            'assistant/message',
            2,
            {'turn': 1, 'step': 1, 'message': message},
            surfaceOp='append',
        ),
    ]
    closers = interrupted_turn_closers(log)
    result = closers[0]
    assert result['data']['error'] == {
        'name': 'ToolNotStartedError',
        'code': TOOL_NOT_STARTED,
    }
    # no recorded tool/call to cite
    assert 'sourceEventSeqs' not in result
