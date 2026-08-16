"""Incremental projection of durable agent inbox events.

Ported from deepseek-harness ``packages/core/agent/src/inbox.ts`` (MIT).

Messages are canonical :class:`pydsh.session.Message` dicts: every durable
``agent/inbox/spliced`` event carries them into the session log, so they must
be lossless-JSON records. The splice coordinates mirror JavaScript's
``Array.prototype.splice`` normalization, clamped to safe integers.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydsh.session import Message, Session

from .runtime_types import InboxTarget

__all__ = ['Inbox', 'InboxNotifications']


class InboxNotifications(Protocol):
    """Live notifications committed by inbox mutations."""

    def inserted(self, message: Message) -> None:
        """Publish one inserted message."""
        ...

    def discarded(self, message: Message) -> None:
        """Publish one discarded message."""
        ...

    def claimed(self, message: Message, turn: int) -> None:
        """Publish one claimed message inside its owning turn."""
        ...


class Inbox:
    """A replay-once projection that incrementally consumes later inbox splices."""

    def __init__(self, session: Session, notifications: InboxNotifications) -> None:
        self._session = session
        self._notifications = notifications
        self._state: dict[InboxTarget, list[Message]] = {
            'next-turn': [],
            'next-step': [],
        }
        seed_length = session.header.get('seedLength') or 0
        for event in session.events[seed_length:]:
            if event['type'] != 'agent/inbox/spliced':
                continue
            try:
                self._apply(event['data'])
            except Exception as error:
                raise ValueError(
                    'invalid persisted inbox splice at session seq'
                    f' {event["seq"]}'
                ) from error

    @property
    def next_turn(self) -> list[Message]:
        """Prompts awaiting individual turns."""
        return self._state['next-turn']

    @property
    def next_step(self) -> list[Message]:
        """Input awaiting the next step boundary."""
        return self._state['next-step']

    @property
    def has_pending(self) -> bool:
        """Whether either pending-message list contains work."""
        return len(self.next_turn) > 0 or len(self.next_step) > 0

    def clear(self) -> None:
        """Durably cancel all pending input, clearing next-step before next-turn."""
        self.splice('next-step', 0, len(self.next_step), [])
        self.splice('next-turn', 0, len(self.next_turn), [])

    def claim(self, target: InboxTarget, turn: int) -> list[Message]:
        """Remove and return the complete batch proposed for one step.

        The durable splices are pure deletions.

        :param target: whether this boundary also consumes one queued turn.
        :param turn: turn that will own the claimed batch.
        :returns: next-step input followed by the queued turn, when requested.
        """
        claimed = self._mutate('next-step', 0, len(self.next_step), [], False)
        if target == 'next-turn':
            claimed.extend(self._mutate('next-turn', 0, 1, [], False))
        for message in claimed:
            self._notifications.claimed(message, turn)
        return claimed

    def append(self, target: InboxTarget, message: Message) -> None:
        """Append one message to a pending list and durably record the insertion."""
        self.splice(target, len(self._state[target]), 0, [message])

    def prepend(self, target: InboxTarget, message: Message) -> None:
        """Prepend one message to a pending list and durably record the insertion."""
        self.splice(target, 0, 0, [message])

    def replace(self, message_id: str, new_message: Message) -> bool:
        """Replace one pending message in place, possibly changing its identity.

        :returns: whether the message was still pending.
        """
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [new_message])
        return True

    def remove(self, message_id: str) -> bool:
        """Remove one pending message and durably record its cancellation.

        :returns: whether the message was still pending.
        """
        location = self._locate(message_id)
        if location is None:
            return False
        target, index = location
        self.splice(target, index, 1, [])
        return True

    def splice(
        self,
        target: InboxTarget,
        start: int,
        delete_count: int,
        inserted: list[Message],
    ) -> list[Message]:
        """Apply standard splice semantics and durably record the normalized result.

        The durable event commits before the live projection mutates, so
        synchronous ``session/event`` observers see the pre-splice lists and
        can reconstruct the removed messages from the normalized coordinates.

        :returns: messages removed by the splice.
        """
        return self._mutate(target, start, delete_count, inserted, True)

    def _locate(self, message_id: str) -> tuple[InboxTarget, int] | None:
        """Locate one pending identity across both owned lists."""
        for target in ('next-turn', 'next-step'):
            inbox = self._state[target]
            for index, message in enumerate(inbox):
                if message['id'] == message_id:
                    return target, index
        return None

    def _mutate(
        self,
        target: InboxTarget,
        start: int,
        delete_count: int,
        inserted: list[Message],
        discard_removed: bool,
    ) -> list[Message]:
        """Commit one normalized mutation and publish its live notifications."""
        inbox = self._state[target]
        offset = int(start)
        actual_start = (
            max(len(inbox) + offset, 0) if offset < 0 else min(offset, len(inbox))
        )
        actual_delete_count = min(max(int(delete_count), 0), len(inbox) - actual_start)
        if actual_delete_count == 0 and len(inserted) == 0:
            return []
        outcome = 'canceled' if discard_removed and actual_delete_count > 0 else None
        splice: dict[str, Any] = {'target': target, 'start': actual_start}
        if actual_delete_count > 0:
            splice['removedCount'] = actual_delete_count
        splice['inserted'] = inserted
        if outcome is not None:
            splice['outcome'] = outcome
        self._validate(splice)
        event = self._session.append('agent/inbox/spliced', splice)
        removed = inbox[actual_start : actual_start + actual_delete_count]
        inbox[actual_start : actual_start + actual_delete_count] = event['data'][
            'inserted'
        ]
        if discard_removed:
            for message in removed:
                self._notifications.discarded(message)
        for message in event['data']['inserted']:
            self._notifications.inserted(message)
        return removed

    def _apply(self, splice: dict[str, Any]) -> list[Message]:
        """Apply one normalized durable splice to the projection."""
        self._validate(splice)
        inbox = self._state[splice['target']]
        start: int = splice['start']
        removed_count: int = splice.get('removedCount', 0)
        removed = inbox[start : start + removed_count]
        inbox[start : start + removed_count] = splice['inserted']
        return removed

    def _validate(self, splice: dict[str, Any]) -> None:
        """Validate one normalized splice against the current projection."""
        inbox: list[Message] = self._state[splice['target']]
        start = splice['start']
        removed_count = splice.get('removedCount', 0)
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start < 0
            or start > len(inbox)
            or not isinstance(removed_count, int)
            or isinstance(removed_count, bool)
            or removed_count < 0
            or start + removed_count > len(inbox)
        ):
            raise ValueError('invalid inbox splice')
        candidate = [
            *inbox[:start],
            *splice['inserted'],
            *inbox[start + removed_count :],
        ]
        ids: set[str] = set()
        others = self.next_step if splice['target'] == 'next-turn' else self.next_turn
        for message in [*candidate, *others]:
            if message['id'] in ids:
                raise ValueError(f'message "{message["id"]}" is already pending')
            ids.add(message['id'])
