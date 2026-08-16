"""Event-sourced session service: append-only log, in-memory store, history.

Ported from deepseek-harness packages/core/session (MIT), ``src/index.ts``.

Persistence is intentionally not implemented here — persistence plugins
subscribe to ``session/event`` and flush on ``session/flush`` / dispose.

Event contract (dispatched through the Cordis event bus; listener argument
lists are the payload):

- ``session/created(session)`` — creation announcement during publication.
  A synchronous raise vetoes and rolls back with a paired disposal; detach
  requested during dispatch is deferred. A returned-awaitable rejection is
  logged but cannot retroactively veto this synchronous boundary.
- ``session/disposed(session)`` — emitted once when an announced session
  leaves the store, including publication rollback, but never for an entry
  whose creation announcement did not begin. Failures are logged/contained.
- ``session/event(session, event)`` — post-commit, fire-and-forget append
  feed. The listener snapshot resolves before the log push, but callbacks
  run after it; observer failures are logged and contained without making
  the committed append fail.
- ``session/flush(session)`` — awaited parallel durability checkpoint:
  every listener runs and the caller awaits all of them, with no veto.

Simplifications from the TypeScript original:

- No ``dsh-scope`` scoped dispatch: events carry no ``thisArg`` carrier, so
  listeners receive exactly the payload arguments listed above.
- No ``typert`` lookup registration (the RPC layer is not ported).
- JavaScript ``Object.freeze``/``structuredClone`` deep immutability has no
  Python equivalent for JSON dicts: events, headers, and folded views are
  detached snapshots and immutable BY CONVENTION. ``Session.events`` returns
  a tuple so the container itself cannot grow under a holder.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import os
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Final, Literal, TypeAlias, TypeGuard, TypeVar, cast, overload

from pycordis import Context, Service
from pycordis.utils import maybe_await, schedule_logged

from .json import SNAPSHOT_FAILED, SnapshotFailed, snapshot_json_value
from .request_header import fold_request_header
from .surface import SurfaceManager, derive_event_message
from .types import (
    SESSION_FORMAT_VERSION,
    AssistantChunkData,
    AssistantMessageData,
    CreateSessionMeta,
    CreateSessionOptions,
    EpochHeader,
    Message,
    PrepareSessionOptions,
    RequestContext,
    RequestHeaderData,
    RestoredSessionOptions,
    SessionEvent,
    SessionHeader,
    SessionId,
    StepEndData,
    StepStartData,
    SurfaceIntent,
    TodoWriteData,
    ToolCallData,
    ToolResultData,
    TurnEndData,
    TurnStartData,
    UserMessage,
)

__all__ = [
    'Session',
    'SessionForkError',
    'SessionForkErrorCode',
    'SessionForkSource',
    'SessionStore',
    'adopt_session_event',
    'snapshot_session_event',
]

T = TypeVar('T')

_MAX_SAFE_INTEGER: Final = 2**53 - 1

_ENVELOPE_KEYS: Final = frozenset(
    {'type', 'seq', 'time', 'data', 'surfaceOp', 'sourceEventSeqs', 'ignorable'}
)

_ALLOWED_ADAPTER_KEYS: Final = frozenset({'reasoningEffort', 'maxTokens'})

MessageEventType: Final = frozenset(
    {'user/message', 'assistant/message', 'tool/result'}
)


def _now_ms() -> int:
    """Current Unix epoch milliseconds (JavaScript ``Date.now()``)."""
    return int(time.time() * 1000)


def _is_safe_integer(value: Any) -> TypeGuard[int]:
    """Whether a runtime value is a safe integer (JS Number.isSafeInteger)."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) <= _MAX_SAFE_INTEGER
    )


def _validate_session_header(id: SessionId, input: Any) -> SessionHeader:
    """Validate one detached creation header."""
    if not isinstance(input, dict):
        raise ValueError('session header is not a plain JSON record')
    if input.get('version') != SESSION_FORMAT_VERSION:
        raise ValueError(
            f'session header version must be {SESSION_FORMAT_VERSION},'
            f' got {input.get("version")}'
        )
    if input.get('id') != id:
        raise ValueError(
            f'session header id "{input.get("id")}" does not match session id "{id}"'
        )
    created_at = input.get('createdAt')
    if not _is_safe_integer(created_at) or created_at < 0:
        raise ValueError('session header createdAt must be a non-negative safe integer')
    if 'cwd' in input:
        cwd = input['cwd']
        if not isinstance(cwd, str):
            raise ValueError('session header cwd must be a string')
        if not os.path.isabs(cwd):
            raise ValueError(
                f'session header cwd must be an absolute path, got "{cwd}"'
            )
    if 'parentSession' in input and not isinstance(input['parentSession'], str):
        raise ValueError('session header parentSession must be a string')
    if 'seedLength' in input and (
        not _is_safe_integer(input['seedLength']) or input['seedLength'] < 0
    ):
        raise ValueError(
            'session header seedLength must be a non-negative safe integer'
        )
    if 'origin' in input and input['origin'] != 'subagent':
        raise ValueError('session header origin must be "subagent"')
    if 'delegationDepth' in input and (
        not _is_safe_integer(input['delegationDepth']) or input['delegationDepth'] < 0
    ):
        raise ValueError(
            'session header delegationDepth must be a non-negative safe integer'
        )
    if 'agentPreset' in input and not isinstance(input['agentPreset'], str):
        raise ValueError('session header agentPreset must be a string')
    return cast(SessionHeader, input)


def _validate_restored_session_header(id: SessionId, input: Any) -> SessionHeader:
    """Validate one exclusively owned persistence header in place."""
    # A restored header crosses the durable boundary: require an exact plain
    # record, not a dict subclass (the TS plain-prototype check).
    if type(input) is not dict:
        raise ValueError('session header is not a plain JSON record')
    return _validate_session_header(id, input)


def _snapshot_session_header(
    id: SessionId, source: SessionHeader | None
) -> SessionHeader:
    """Detach, validate, and freeze the creation metadata published by a session."""
    input: Any = (
        {'version': SESSION_FORMAT_VERSION, 'id': id, 'createdAt': _now_ms()}
        if source is None
        else source
    )
    snapshot = snapshot_json_value(input)
    if snapshot is SNAPSHOT_FAILED:
        raise ValueError('session header is not losslessly JSON-serializable')
    return _validate_session_header(id, snapshot)


def _has_provider_model(value: Any) -> bool:
    """Whether an unknown value carries the current provider/model pair."""
    if not isinstance(value, dict):
        return False
    provider = value.get('provider')
    model = value.get('model')
    return (
        isinstance(provider, str)
        and len(provider) > 0
        and isinstance(model, str)
        and len(model) > 0
    )


def _assert_adapter_defaults(
    header_record: dict[str, Any], config: dict[str, Any], index: int
) -> None:
    """Validate adapter-default markers imported from a durable request header."""
    if 'adapterDefaults' not in header_record:
        return
    defaults = header_record['adapterDefaults']
    if not isinstance(defaults, dict):
        raise ValueError(
            f'seed request/header at index {index} has invalid adapterDefaults'
        )
    if (
        any(key not in _ALLOWED_ADAPTER_KEYS for key in defaults)
        or any(marker is not True for marker in defaults.values())
        or (
            defaults.get('reasoningEffort') is True
            and 'reasoningEffort' not in config
        )
        or (defaults.get('maxTokens') is True and 'maxTokens' not in config)
    ):
        raise ValueError(
            f'seed request/header at index {index} has invalid adapterDefaults'
        )


def _assert_message_event_shape(event: dict[str, Any], subject: str) -> None:
    """Validate only the event-specific invariants needed to replay a message."""
    type_ = event.get('type')
    if type_ not in MessageEventType:
        return
    data = event.get('data')
    record = data if isinstance(data, dict) else None
    message = record if type_ == 'user/message' else (record or {}).get('message')
    if (
        not isinstance(message, dict)
        or not isinstance(message.get('id'), str)
        or message.get('id') == ''
    ):
        raise ValueError(f'{subject} lacks an identified message')
    expected_role = 'assistant' if type_ == 'assistant/message' else 'user'
    if message.get('role') != expected_role:
        raise ValueError(f'{subject} message must have role "{expected_role}"')
    source = message.get('source')
    if (
        not isinstance(source, dict)
        or not isinstance(source.get('kind'), str)
        or source.get('kind') == ''
    ):
        raise ValueError(f'{subject} message has invalid source')
    if not isinstance(message.get('content'), list):
        raise ValueError(f'{subject} message has invalid content')
    if type_ == 'assistant/message':
        if source.get('kind') != 'model' or not _has_provider_model(source):
            raise ValueError(f'{subject} message must have model source')
        return
    if type_ != 'tool/result':
        return
    call_id = source.get('callId')
    if source.get('kind') != 'tool' or not isinstance(call_id, str) or call_id == '':
        raise ValueError(f'{subject} message must have tool source')
    content = message['content']
    block = content[0] if len(content) == 1 else None
    if (
        not isinstance(block, dict)
        or block.get('type') != 'tool-result'
        or not isinstance(block.get('content'), list)
    ):
        raise ValueError(f'{subject} message must contain one tool-result block')
    if block.get('toolCallId') != call_id:
        raise ValueError(f'{subject} message has mismatched tool call ids')


def _assert_current_llm_shape(event: dict[str, Any], index: int) -> None:
    """Reject obsolete request headers and malformed messages at the seed boundary."""
    type_ = event.get('type')
    data = event.get('data')
    record = data if isinstance(data, dict) else None
    if type_ == 'request/header':
        header = (record or {}).get('header')
        header_record = header if isinstance(header, dict) else {}
        config = header_record.get('config')
        if not _has_provider_model(config):
            raise ValueError(
                f'seed request/header at index {index} lacks provider/model'
            )
        config_record = cast(dict[str, Any], config)
        if 'reasoningEffort' in config_record and (
            not isinstance(config_record['reasoningEffort'], str)
            or len(config_record['reasoningEffort']) == 0
        ):
            raise ValueError(
                f'seed request/header at index {index} has an invalid reasoningEffort'
            )
        _assert_adapter_defaults(header_record, config_record, index)
    if type_ not in MessageEventType:
        return
    _assert_message_event_shape(event, f'seed {type_} at index {index}')


def _assert_session_event_envelope(value: dict[str, Any], index: int) -> None:
    """Validate the fixed event envelope after one-pass JSON materialization."""
    if value.get('type') == 'request/header-delta':
        raise ValueError(
            f'seed event at index {index} uses unsupported legacy'
            ' request/header-delta format'
        )
    for key in value:
        if key not in _ENVELOPE_KEYS:
            raise ValueError(
                f'seed event at index {index} has an invalid event envelope'
            )
    type_ = value.get('type')
    seq = value.get('seq')
    time_ = value.get('time')
    if (
        not isinstance(type_, str)
        or not _is_safe_integer(seq)
        or seq < 0
        or not _is_safe_integer(time_)
        or 'data' not in value
        or ('ignorable' in value and value['ignorable'] is not True)
    ):
        raise ValueError(
            f'seed event at index {index} has an invalid event envelope'
        )
    if type_ in ('request/header', 'user/message', 'assistant/message', 'tool/result'):
        _assert_current_llm_shape(value, index)


def _assert_supported_request_header(type: str, data: Any, location: str) -> None:
    """Reject request-header vocabulary removed with the legacy delta codec."""
    if type == 'request/header-delta':
        raise ValueError(
            f'{location} uses unsupported legacy request/header-delta format'
        )
    if (
        type == 'request/header'
        and isinstance(data, dict)
        and data.get('reason') == 'fallback'
    ):
        raise ValueError(
            f'{location} uses unsupported legacy request/header reason "fallback"'
        )


def adopt_session_event(event: T) -> T:
    """Validate one exclusively owned event imported across a trusted boundary.

    The caller transfers an object graph that no producer retains and that
    shares no mutable children with another event; the event is not copied.
    The TS original also deep-freezes the identified message — a no-op under
    Python's by-convention immutability (see module docstring). Use
    :func:`snapshot_session_event` when exclusive ownership is not guaranteed.

    :param event: exclusively owned event imported across a trusted boundary.
    :returns: the same event object, validated.
    """
    _assert_message_event_shape(
        cast(dict[str, Any], event), f'session event at seq {cast(Any, event)["seq"]}'
    )
    return event


def snapshot_session_event(event: T) -> T:
    """Detach one event imported across a query or persistence boundary.

    :param event: event to detach.
    :returns: a detached event snapshot with a validated message.
    """
    return adopt_session_event(copy.deepcopy(event))


_SessionCallback = Callable[..., Any]


def _collect_session_callbacks(
    ctx: Context, args: list[Any]
) -> list[_SessionCallback]:
    """Resolve one listener snapshot, including Cordis's dispatch checks."""
    return list(ctx.events.dispatch('emit', args))


def _invoke_contained_session_observers(
    ctx: Context,
    name: Literal['session/event', 'session/disposed'],
    id: SessionId,
    args: list[Any],
    callbacks: list[_SessionCallback],
) -> None:
    """Invoke one resolved observe-only listener snapshot with containment."""
    for callback in callbacks:
        try:
            returned = callback(*args)
            if inspect.isawaitable(returned):
                schedule_logged(
                    returned,
                    lambda error: ctx.logger.warn(
                        f'session "{id}": {name} listener rejected: {error}'
                    ),
                )
        except Exception as error:
            ctx.logger.warn(f'session "{id}": {name} listener threw: {error}')


class Session:
    """An event-sourced session: an append-only log of SessionEvents.

    Plain class (not a Service) — create live instances via
    ``ctx.sessions.create()`` and detached instances via :meth:`create`.
    Seeding with an existing event log replays/forks a session.
    """

    def __init__(
        self,
        id: SessionId,
        seed: Sequence[SessionEvent] | None = None,
        header: SessionHeader | None = None,
        mode: Literal['snapshot', 'restore'] = 'snapshot',
    ) -> None:
        """Construct a detached session; prefer :meth:`create`/:meth:`from_restore`.

        A seed is validated to the SAME invariants ``append`` enforces
        (lossless-JSON data, fixed envelope, contiguous seq from 0, canonical
        surface transitions), so a replay/fork cannot construct a live log
        that no persistence backend could store. With ``mode='restore'`` the
        seed and header graphs are adopted in place (ownership transfer);
        otherwise they are snapshotted.

        :param id: session identity.
        :param seed: optional replay or fork events.
        :param header: optional storage metadata.
        :param mode: ``snapshot`` detaches borrowed values; ``restore``
            adopts fresh persistence-owned values in place.
        """
        self._log: list[SessionEvent] = []
        #: Single incremental owner of surface acceptance and projection state.
        self._surface_manager = SurfaceManager(self._log)
        self._events_snapshot: tuple[SessionEvent, ...] | None = None
        self._header_fold: EpochHeader | None = None
        self._header_fold_seq = 0
        self._context_fold: RequestContext | None = None
        self._context_fold_seq = 0
        self._derived: list[Message] = []
        self._derived_nodes = 0
        self._derived_generation = 0

        restored_header = (
            _validate_restored_session_header(id, header) if mode == 'restore' else None
        )
        if seed is not None:
            for index, source in enumerate(seed):
                # The seed is a persistence/replay boundary: validate and
                # detach the complete event in one lossless-JSON pass.
                snapshot: Any = (
                    source if mode == 'restore' else snapshot_json_value(source)
                )
                if snapshot is SNAPSHOT_FAILED:
                    raise ValueError(
                        f'seed event at index {index} is not losslessly'
                        ' JSON-serializable'
                    )
                snapshot = cast(dict[str, Any], snapshot)
                _assert_session_event_envelope(snapshot, index)
                _assert_supported_request_header(
                    cast(str, snapshot['type']),
                    snapshot['data'],
                    f'seed event at index {index}',
                )
                if snapshot['seq'] != index:
                    raise ValueError(
                        f'seed event at index {index} has seq {snapshot["seq"]}'
                        f' (expected {index}); seed must be contiguous from 0'
                    )
                # A seed is accepted incrementally through the same transition
                # as a live append; the candidate is planned before it enters
                # the log, so a failure cannot partially mutate the surface.
                try:
                    self._surface_manager.validate_next(cast(SessionEvent, snapshot))
                except ValueError as error:
                    raise ValueError(
                        f'invalid seed event at index {index}: {error}'
                    ) from error
                self._log.append(cast(SessionEvent, snapshot))
        self.first_live_seq = len(self._log)
        self.header: SessionHeader = (
            restored_header
            if restored_header is not None
            else _snapshot_session_header(id, header)
        )
        # Appended here so the marker is already in events when a backend
        # captures the creation seed: no load-time write. Re-marking is
        # skipped because a cold session is resumed on first touch, so
        # repeatedly opening one must not grow its log per open.
        if seed is not None and (
            not self._log or self._log[-1]['type'] != 'session/end-seed'
        ):
            self.append('session/end-seed', {})

    @classmethod
    def create(
        cls,
        id: SessionId,
        seed: Sequence[SessionEvent] | None = None,
        header: SessionHeader | None = None,
    ) -> Session:
        """Create a detached session by validating and snapshotting borrowed values.

        :param id: session identity.
        :param seed: optional borrowed replay or fork events.
        :param header: optional borrowed storage metadata.
        :returns: a detached session.
        """
        return cls(id, seed, header)

    @classmethod
    def from_restore(
        cls, id: SessionId, seed: Sequence[SessionEvent], header: SessionHeader
    ) -> Session:
        """Restore a detached session by taking ownership of fresh storage values.

        The storage format, event envelopes, sequence continuity, surface
        transitions, and header fields are validated before the restored
        objects are adopted.

        :param id: restored session identity.
        :param seed: fresh detached events whose ownership is transferred.
        :param header: fresh detached metadata whose ownership is transferred.
        :returns: a restored detached session.
        """
        return cls(id, seed, header, 'restore')

    @property
    def surface(self) -> SurfaceManager:
        """The ordered surface over this session's event log."""
        return self._surface_manager

    @property
    def id(self) -> SessionId:
        """The session identity, derived from its durable header's single copy."""
        return self.header['id']

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """An immutable snapshot of the append-only event log.

        The snapshot is reused until the next append; a previously returned
        tuple does not grow later. Event payloads are detached snapshots,
        immutable by convention (see module docstring).
        """
        if self._events_snapshot is None:
            self._events_snapshot = tuple(self._log)
        return self._events_snapshot

    @property
    def seq(self) -> int:
        """The next event's sequence number — always the log length."""
        return len(self._log)

    # The overloads pair each core event type with its payload, mirroring the
    # TS conditional-rest signature: surface-eligible types REQUIRE the
    # SurfaceIntent, log-only types FORBID it. The final fallback covers
    # plugin-merged extension events.
    @overload
    def append(
        self, type: Literal['turn/start'], data: TurnStartData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['turn/end'], data: TurnEndData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['step/start'], data: StepStartData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['step/end'], data: StepEndData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['user/message'], data: UserMessage, opts: SurfaceIntent
    ) -> SessionEvent: ...
    @overload
    def append(
        self,
        type: Literal['assistant/message'],
        data: AssistantMessageData,
        opts: SurfaceIntent,
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['assistant/chunk'], data: AssistantChunkData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['tool/call'], data: ToolCallData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['tool/result'], data: ToolResultData, opts: SurfaceIntent
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['todo/write'], data: TodoWriteData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['request/header'], data: RequestHeaderData
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['request/context'], data: RequestContext
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: Literal['session/end-seed'], data: dict[str, Any]
    ) -> SessionEvent: ...
    @overload
    def append(
        self, type: str, data: Any, opts: SurfaceIntent | None = None
    ) -> SessionEvent: ...

    def append(
        self, type: str, data: Any, opts: SurfaceIntent | None = None
    ) -> SessionEvent:
        """Append one typed event to the log and synchronously notify observers.

        The hot path never blocks on I/O — persistence plugins buffer
        asynchronously. Once the event enters the log, the append is
        committed: observer failures are logged and contained per listener,
        so they do not change the return value or prevent later listeners
        from observing the same accepted event.

        :param type: the event type (a SessionEventMap key).
        :param data: the event payload; must be JSON-serializable.
        :param opts: surface metadata — REQUIRED for message-producing events
            and forbidden on log-only events.
        :returns: the logged event — its assigned ``seq``/``time`` plus the
            SNAPSHOT of ``data`` that entered the log, so reading
            ``event['data']`` back sees the logged value, never the caller's
            still-mutable input.
        :raises ValueError: if ``data`` or the surface metadata is not
            losslessly JSON-serializable, or when the candidate violates the
            canonical surface contract (marker shape and eligibility, unique
            earlier source-event references, positional replacement validity,
            complete shadowed-node coverage).
        :raises RuntimeError: when append is reentered while this
            acceptance/publication boundary is open.
        """
        surface_metadata: dict[str, Any] = {}
        if opts is not None:
            if 'sourceEventSeqs' in opts:
                surface_metadata['sourceEventSeqs'] = opts['sourceEventSeqs']
            if 'surfaceOp' in opts:
                surface_metadata['surfaceOp'] = opts['surfaceOp']
        data_snapshot = snapshot_json_value(data)
        if data_snapshot is SNAPSHOT_FAILED:
            raise ValueError(
                f'session event "{type}" carries non-JSON-serializable data'
            )
        _assert_supported_request_header(type, data_snapshot, f'session event "{type}"')
        metadata_snapshot = snapshot_json_value(surface_metadata)
        if isinstance(metadata_snapshot, SnapshotFailed):
            raise ValueError(
                f'session event "{type}" carries non-JSON-serializable surface metadata'
            )
        entry = _attachments.get(self)
        if entry is not None and entry.appending:
            raise RuntimeError(
                'session append cannot reenter while another append is being published'
            )
        event = cast(
            SessionEvent,
            {
                'type': type,
                'seq': len(self._log),
                'time': _now_ms(),
                'data': data_snapshot,
                **metadata_snapshot,
            },
        )
        self._surface_manager.validate_next(event)

        if entry is not None:
            entry.appending = True
        try:
            callbacks: list[_SessionCallback] | None = None
            callback_args: list[Any] = [self, event]
            if entry is not None:
                callbacks = _collect_session_callbacks(
                    entry.emit_ctx, ['session/event', *callback_args]
                )
            self._log.append(event)
            self._events_snapshot = None
            if callbacks is not None and entry is not None:
                _invoke_contained_session_observers(
                    entry.emit_ctx, 'session/event', entry.id, callback_args, callbacks
                )
            return event
        finally:
            if entry is not None:
                entry.appending = False
                if entry.detach_requested and not entry.announcing:
                    entry.detach()

    def request_header(self) -> EpochHeader | None:
        """The EpochHeader in force after the log's last header event.

        This is the header the NEXT request will be compared against, or None
        before the first ``request/header`` snapshot. The live,
        incrementally-maintained form of ``fold_request_header(session.events)``:
        each header event is folded once, when first seen, so a per-step read
        costs O(new events).

        :returns: the folded header, or None when no header event exists yet.
        """
        if self._header_fold_seq < len(self._log):
            self._header_fold = fold_request_header(
                self._log[self._header_fold_seq :], self._header_fold
            )
            self._header_fold_seq = len(self._log)
        return self._header_fold

    def request_context(self) -> RequestContext | None:
        """The latest resolved route metadata.

        :returns: the latest route metadata, or None before the first
            ``request/context`` event. Each event is folded once.
        """
        if self._context_fold_seq < len(self._log):
            for event in self._log[self._context_fold_seq :]:
                if event['type'] == 'request/context':
                    self._context_fold = cast(RequestContext, dict(event['data']))
            self._context_fold_seq = len(self._log)
        return self._context_fold

    def derive_messages(self) -> list[Message]:
        """Derive the LLM message history by walking the ordered surface.

        The surface is the single source of derived history: every
        message-producing append records its ``surfaceOp``, so a raw event
        with no marker (a chunk, a turn boundary) is correctly absent, and a
        compaction ``replace`` deletes the shadowed nodes from the
        derivation. The projection rules are :func:`derive_event_message`,
        folded per node.

        CACHED: each surface node is projected exactly once, when first seen
        — a call costs O(new nodes), and a surface rewrite (a ``replace``)
        rebuilds. The returned list is a fresh snapshot per call; the
        Message dicts in it are SHARED with the durable event data.

        :returns: a fresh list of the shared derived history.
        """
        nodes = self._surface_manager.nodes
        generation = self._surface_manager.replace_generation
        if generation != self._derived_generation:
            self._derived = []
            self._derived_nodes = 0
            self._derived_generation = generation
        for seq in nodes[self._derived_nodes :]:
            message = derive_event_message(self._log[seq])
            # A surface node is one of the message-producing types, but an
            # empty-content assistant/message (a max-tokens step that hosts
            # only usage) derives to None and must not enter the transcript.
            if message is not None:
                self._derived.append(message)
        self._derived_nodes = len(nodes)
        return list(self._derived)

    @staticmethod
    def derive_event_message(event: SessionEvent) -> Message | None:
        """Instance face of the pure per-node projection from ``surface.py``.

        :param event: the event to project.
        :returns: the derived message, or None when the event produces none.
        """
        return derive_event_message(event)


class _SessionEntry:
    """All mutable lifecycle state for one exact store entry."""

    __slots__ = (
        'announced',
        'announcing',
        'appending',
        'detach_requested',
        'emit_ctx',
        'id',
        'session',
        'store',
    )

    def __init__(self, store: SessionStore, id: SessionId, session: Session) -> None:
        self.store = store
        self.id = id
        self.session = session
        self.emit_ctx = store.ctx
        self.announced = False
        self.announcing = False
        self.appending = False
        self.detach_requested = False

    def detach(self) -> None:
        """Remove this exact entry and publish its paired disposal edge."""
        self.store._detach_entered(self)


#: Store attachment for the append path; module-private to keep Session
#: store-agnostic publicly.
_attachments: weakref.WeakKeyDictionary[Session, _SessionEntry] = (
    weakref.WeakKeyDictionary()
)

SessionForkErrorCode = Literal[
    'SESSION_NOT_FOUND',
    'SESSION_NOT_LIVE',
    'SESSION_ALREADY_EXISTS',
    'INVALID_BOUNDARY',
    'OPEN_TURN',
]
"""Rejection codes for session forking."""

SessionForkSource: TypeAlias = 'Session | SessionId'
"""A fork source: either the live session object or its live store id."""


class SessionForkError(Exception):
    """Typed error for session fork rejections."""

    def __init__(self, message: str, code: SessionForkErrorCode) -> None:
        super().__init__(message)
        self.code = code


class SessionStore(Service):
    """In-memory session store (``ctx.sessions``).

    Persistence is intentionally not implemented here — persistence plugins
    subscribe to ``session/event`` and flush on ``session/flush`` / dispose.
    """

    def __init__(self, ctx: Context) -> None:
        self._store: dict[SessionId, _SessionEntry] = {}
        self._counter = 0
        super().__init__(ctx, 'sessions')

    def create(
        self, id: SessionId | None = None, options: PrepareSessionOptions | None = None
    ) -> Session:
        """Create a session owned by the calling fiber.

        Disposing that fiber stops event notification and removes the
        session from the store. ``options.seed`` populates the session with a
        copy of those events (replay/fork); ``options.meta`` attaches
        creation metadata as the immutable SessionHeader (the store fills
        ``version``/``id``/``createdAt``).

        For an agent whose session must be torn down IN ORDER with its loop,
        do NOT use this — fold the session lifecycle into the agent's own
        effect via :meth:`prepare` + :meth:`enter` + :meth:`announce`.

        :param id: the session id; omitted, the store mints ``session-<n>``.
        :param options: seed events and/or creation metadata for the header.
        :returns: the live session, already entered and announced.
        :raises ValueError: if a session with ``id`` already exists, metadata
            is not a plain lossless-JSON record with valid scalar fields, or
            ``meta.cwd`` is a non-absolute path.
        """
        session = self.prepare(id, options)

        # Single effect owned by the calling fiber. Yield the detach BEFORE
        # announcing so a throwing session/created listener rolls the attach
        # back (the generator effect disposes already-yielded disposers on a
        # raise) instead of leaking the store entry and its publication hooks.
        def effect() -> Iterator[Callable[[], None]]:
            yield self.enter(session)
            self.announce(session)

        self.ctx.effect(effect, 'sessions.create()')
        return session

    def prepare(
        self, id: SessionId | None = None, options: PrepareSessionOptions | None = None
    ) -> Session:
        """Build a session WITHOUT entering it into the store.

        Validates the id/cwd and constructs the Session (with its immutable
        SessionHeader). Pairs with :meth:`enter` + :meth:`announce`: a caller
        that owns a composite ``ctx.effect`` folds the session lifecycle into
        that ONE effect so a fiber unload tears the session + agent down as a
        single ORDERED chain.

        :param id: the session id; omitted, the store mints ``session-<n>``.
        :param options: seed events and/or creation metadata. With
            RestoredSessionOptions, metadata and events must be fresh
            detached graphs whose ownership transfers to this call.
        :returns: the constructed session, NOT yet in the store.
        :raises ValueError: if a session with ``id`` already exists, metadata
            is not a plain lossless-JSON record with valid scalar fields, or
            ``meta.cwd`` is a non-absolute path.
        """
        if id is None:
            session_id = self._mint_id()
        else:
            session_id = SessionId(id)
        if session_id in self._store:
            raise ValueError(f'session "{session_id}" already exists')
        if isinstance(options, RestoredSessionOptions):
            return Session.from_restore(session_id, options.seed, options.meta)
        seed = options.seed if options is not None else None
        meta: CreateSessionMeta | None = options.meta if options is not None else None
        if meta is not None and 'createdAt' in meta:
            created_at: Any = meta['createdAt']
        else:
            created_at = _now_ms()
        header = SessionHeader(
            version=SESSION_FORMAT_VERSION,
            id=session_id,
            createdAt=created_at,
        )
        if meta is not None:
            if 'cwd' in meta:
                header['cwd'] = meta['cwd']
            if 'parentSession' in meta:
                header['parentSession'] = meta['parentSession']
            if 'seedLength' in meta:
                header['seedLength'] = meta['seedLength']
            if 'origin' in meta:
                header['origin'] = meta['origin']
            if 'delegationDepth' in meta:
                header['delegationDepth'] = meta['delegationDepth']
            if 'agentPreset' in meta:
                header['agentPreset'] = meta['agentPreset']
        return Session.create(session_id, seed, header)

    def enter(self, session: Session) -> Callable[[], None]:
        """Enter a prepared session into the store.

        Installs the module-private append publication hooks and adds the
        session to the store. Does NOT emit ``session/created`` — the caller
        yields the returned disposer inside its effect and THEN calls
        :meth:`announce`, so a throwing ``session/created`` listener rolls
        the attach back instead of leaking it.

        Re-checks the id for a duplicate: ``prepare`` and ``enter`` are
        public primitives and a caller may interleave arbitrary work between
        them, so a stale prepared session must NOT overwrite a live store
        entry of the same id.

        :param session: a prepared session not yet in the store.
        :returns: the detach disposer (publication hooks + store removal).
            When called from a synchronous ``session/created`` listener,
            removal and disposal wait until that creation dispatch unwinds.
        :raises ValueError: if a session with this id is already in the store.
        """
        id = session.id
        # This is the authoritative collision boundary after arbitrary
        # unpublished preparation. Only one exact same-id transaction can
        # publish.
        if id in self._store:
            raise ValueError(f'session "{id}" already exists')
        if session in _attachments:
            raise ValueError(f'session "{id}" is already attached to a store')
        entry = _SessionEntry(self, id, session)
        self._store[id] = entry
        _attachments[session] = entry
        entered = True

        def detach() -> None:
            nonlocal entered
            if not entered:
                return
            entered = False
            # A lifecycle listener may own the advanced detach capability.
            # Keep the entry and its publication hooks live until synchronous
            # creation or append publication unwinds, then publish the paired
            # disposal edge.
            if entry.announcing or entry.appending:
                entry.detach_requested = True
                return
            entry.detach()

        return detach

    def announce(self, session: Session) -> None:
        """Emit ``session/created`` exactly once for an entered session.

        Separate from :meth:`enter` so the caller can yield the detach
        disposer first (rollback safety — see :meth:`enter`).

        :param session: the entered session to announce to listeners.
        :raises ValueError: if the session is not live or its announcement
            already began, including a reentrant call from a creation listener.
        """
        entry = self._live_entry_for(session)
        if entry.announced or entry.announcing:
            raise ValueError(f'session "{entry.id}" was already announced')
        # Mark before dispatch: emit may deliver to earlier listeners and
        # then raise. Rollback must still pair that partial creation with
        # disposal, and a listener cannot recursively create a second
        # lifecycle edge.
        entry.announced = True
        entry.announcing = True
        try:
            callbacks = _collect_session_callbacks(
                self.ctx, ['session/created', session]
            )
            for callback in callbacks:
                # Synchronous raises intentionally propagate and veto
                # publication; the yielded detach then emits the paired
                # disposal edge. An async listener's returned awaitable is
                # observed instead: rejection is too late to roll back and
                # must be logged rather than abandoned.
                returned = callback(session)
                if inspect.isawaitable(returned):
                    schedule_logged(
                        returned,
                        lambda error: self.ctx.logger.warn(
                            f'session "{entry.id}": session/created listener'
                            f' rejected: {error}'
                        ),
                    )
        finally:
            entry.announcing = False
            if entry.detach_requested and not entry.appending:
                entry.detach()

    async def flush(self, session: Session) -> bool:
        """Dispatch the awaited ``session/flush`` durability checkpoint.

        THE flush entry point: the store owns publication, so callers (the
        checkpoint policy's per-request barrier, teardown drains, and
        consumers that flush themselves before reading storage) must come
        through here rather than dispatch a raw parallel event.

        :param session: the session whose buffered events must reach durable
            storage.
        :returns: whether at least one durability listener participated,
            after every listener has settled successfully.
        :raises Exception: the first registered listener failure after every
            listener settles.
        """
        # The lookup itself is the liveness gate for the checkpoint.
        self._live_entry_for(session)
        callbacks = _collect_session_callbacks(self.ctx, ['session/flush', session])

        async def invoke(callback: _SessionCallback) -> Any:
            return await maybe_await(callback(session))

        results = await asyncio.gather(
            *(invoke(callback) for callback in callbacks),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return len(callbacks) > 0

    def get(self, id: SessionId) -> Session | None:
        """Look up a live session.

        :param id: the session id to look up.
        :returns: the session, or None when no live session has that id.
        """
        entry = self._store.get(id)
        return entry.session if entry is not None else None

    def list(self) -> list[Session]:
        """All live sessions, in creation order.

        :returns: a fresh list; mutating it does not affect the store.
        """
        return [entry.session for entry in self._store.values()]

    def fork(
        self,
        source: Session | SessionId,
        boundary: int | None = None,
        child_session_id: SessionId | None = None,
    ) -> Session:
        """Create a live child session from a stable prefix of a live source.

        ``boundary`` is an inclusive source event seq; omitted means the
        source's current last event. The selected slice may end with a
        between-turn event but must not end inside an open turn.

        :param source: live source session object or id.
        :param boundary: inclusive source event seq to fork through; omitted
            means the source's current last event, and omitted on an empty
            source forks an empty child.
        :param child_session_id: optional child session id; omitted delegates
            to the store's id policy.
        :returns: the created live child session.
        :raises SessionForkError: with code SESSION_NOT_FOUND,
            SESSION_NOT_LIVE, SESSION_ALREADY_EXISTS, INVALID_BOUNDARY, or
            OPEN_TURN.
        """
        if child_session_id is not None and self.get(child_session_id) is not None:
            raise SessionForkError(
                f'session "{child_session_id}" already exists', 'SESSION_ALREADY_EXISTS'
            )
        live_source = self._resolve_fork_source(source)
        seed = self._fork_seed(live_source, boundary)
        meta = CreateSessionMeta(
            parentSession=live_source.id,
            seedLength=len(seed),
        )
        if 'cwd' in live_source.header:
            meta['cwd'] = live_source.header['cwd']
        return self.create(child_session_id, CreateSessionOptions(seed=seed, meta=meta))

    def _fork_seed(
        self, session: Session, requested_boundary: int | None
    ) -> Sequence[SessionEvent]:
        events = session.events
        last_event = events[-1] if events else None
        if requested_boundary is not None:
            boundary = requested_boundary
        else:
            if last_event is None:
                return []
            boundary = last_event['seq']
        if not _is_safe_integer(boundary) or boundary < 0:
            raise SessionForkError(
                f'fork boundary for session "{session.id}" must be a non-negative'
                f' safe integer, got {boundary}',
                'INVALID_BOUNDARY',
            )
        if boundary >= len(events):
            last_seq: Any = events[-1]['seq'] if events else 'none'
            raise SessionForkError(
                f'fork boundary {boundary} does not exist in session'
                f' "{session.id}" (last seq: {last_seq})',
                'INVALID_BOUNDARY',
            )
        boundary_event = events[boundary]
        if boundary_event['seq'] != boundary:
            raise SessionForkError(
                f'fork boundary {boundary} does not match a contiguous event seq'
                f' in session "{session.id}"',
                'INVALID_BOUNDARY',
            )
        last_turn_boundary: SessionEvent | None = None
        for event in events[: boundary + 1]:
            if event['type'] in ('turn/start', 'turn/end'):
                last_turn_boundary = event
        if (
            last_turn_boundary is not None
            and last_turn_boundary['type'] == 'turn/start'
        ):
            raise SessionForkError(
                f'fork boundary {boundary} in session "{session.id}" ends inside'
                f' open turn {last_turn_boundary["data"]["turn"]}',
                'OPEN_TURN',
            )
        return list(events[: boundary + 1])

    def _resolve_fork_source(self, source: Session | SessionId) -> Session:
        if not isinstance(source, Session):
            session = self.get(source)
            if session is None:
                raise SessionForkError(
                    f'session "{source}" not found', 'SESSION_NOT_FOUND'
                )
            return session
        live = self.get(source.id)
        if live is None:
            raise SessionForkError(
                f'session "{source.id}" not found', 'SESSION_NOT_FOUND'
            )
        if live is not source:
            raise SessionForkError(
                f'session "{source.id}" is not the live store instance',
                'SESSION_NOT_LIVE',
            )
        return source

    def _mint_id(self) -> SessionId:
        while True:
            self._counter += 1
            session_id = SessionId(f'session-{self._counter}')
            if session_id not in self._store:
                return session_id

    def _detach_entered(self, entry: _SessionEntry) -> None:
        """Remove one exact entered session and emit its paired disposal."""
        entry.detach_requested = False
        # A stale capability cannot remove observers or storage belonging to
        # a later same-id lifecycle.
        if self._store.get(entry.id) is not entry:
            return
        del self._store[entry.id]
        _attachments.pop(entry.session, None)
        if entry.announced:
            self._emit_disposed(entry)

    def _emit_disposed(self, entry: _SessionEntry) -> None:
        """Emit the paired teardown notification with per-listener containment."""
        try:
            callbacks = _collect_session_callbacks(
                self.ctx, ['session/disposed', entry.session]
            )
        except Exception as error:
            self.ctx.logger.warn(
                f'session "{entry.id}": session/disposed dispatch threw: {error}'
            )
            return
        _invoke_contained_session_observers(
            self.ctx, 'session/disposed', entry.id, [entry.session], callbacks
        )

    def _live_entry_for(self, session: Session) -> _SessionEntry:
        """Return the exact live entry; detached/prepared objects reject."""
        entry = _attachments.get(session)
        if entry is None or self._store.get(entry.id) is not entry:
            raise ValueError(f'session "{session.id}" is not live in this store')
        return entry
