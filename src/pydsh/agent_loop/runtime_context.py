"""Durable projection state for dynamic runtime context.

Ported from deepseek-harness
``packages/core/agent-loop/src/runtime-context.ts`` (MIT).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pycordis import Context
from pydsh.session import Message, Session, is_replacement_surface_event
from pydsh.system_prompt import ContextSnapshotSection

from .llm_helpers import create_user_message

_SOURCE = '@deepseek-ai/dsh-system-prompt'
_CLEARED = (
    'Current runtime context: none. Earlier runtime-context snapshots no'
    ' longer apply.'
)

#: ``retained`` state before any snapshot ever existed (TS ``undefined``).
_NEVER: Any = object()


def _is_owned(message: Message) -> bool:
    source = message.get('source')
    return (
        isinstance(source, dict)
        and source.get('kind') == 'plugin'
        and source.get('plugin') == _SOURCE
    )


def _text_of(message: Message) -> str | None:
    content = message.get('content')
    if not isinstance(content, list) or len(content) != 1:
        return None
    block = content[0]
    if isinstance(block, dict) and block.get('type') == 'text':
        text = block.get('text')
        return text if isinstance(text, str) else None
    return None


class RuntimeContextProjection:
    """Tracks the last retained runtime-context snapshot without owning its commit.

    ``retained`` is ``_NEVER`` when no snapshot ever existed, ``None`` when
    none is retained, and otherwise the retained event's seq and text.
    """

    def __init__(self, ctx: Context, session: Session) -> None:
        self._retained: Any = _NEVER
        surface = set(session.surface.nodes)
        for event in reversed(session.events):
            if event['type'] != 'user/message' or not _is_owned(event['data']):
                continue
            if self._retained is _NEVER:
                self._retained = None
            if event['seq'] in surface:
                self._retained = {'seq': event['seq'], 'text': _text_of(event['data'])}
                break

        def on_session_event(subject: Session, event: Any) -> None:
            if subject is not session:
                return
            if event['type'] == 'user/message' and _is_owned(event['data']):
                self._retained = {
                    'seq': event['seq'],
                    'text': _text_of(event['data']),
                }
            elif (
                isinstance(self._retained, dict)
                and is_replacement_surface_event(event)
                and self._retained['seq'] in (event.get('sourceEventSeqs') or [])
            ):
                self._retained = None

        ctx.on('session/event', on_session_event)

    def project(
        self, current: str, sections: Sequence[ContextSnapshotSection]
    ) -> Message | None:
        """Create an uncommitted snapshot only when the retained value differs.

        :param current: fully rendered dynamic context.
        :param sections: named contributions that formed the current snapshot.
        :returns: a candidate user message, or ``None`` when no update is needed.
        """
        if self._retained is _NEVER and len(current) == 0:
            return None
        snapshot = _CLEARED if len(current) == 0 else current
        if isinstance(self._retained, dict) and self._retained['text'] == snapshot:
            return None
        source: dict[str, Any] = (
            # The cleared marker has no contributions left to attribute.
            {'kind': 'plugin', 'plugin': _SOURCE}
            if len(sections) == 0
            else {
                'kind': 'plugin',
                'plugin': _SOURCE,
                'form': 'snapshot',
                'sections': [
                    {'name': section.name, 'text': section.text}
                    for section in sections
                ],
            }
        )
        return create_user_message([{'type': 'text', 'text': snapshot}], source)
