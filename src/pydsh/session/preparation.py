"""Ownership of one unpublished Session before registry publication.

Ported from deepseek-harness packages/core/session (MIT),
``src/preparation.ts``. The TypeScript ``Disposable`` protocol maps onto
``close()`` plus the context-manager protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import Session

__all__ = ['SessionPreparation']


class SessionPreparation:
    """One exact unpublished Session and the provider state that keeps it usable.

    Disposal is synchronous and idempotent. Providers decide whether release
    returns the Session to a cache or discards it; publication may consume
    that state before disposal, making the callback a no-op.
    """

    def __init__(
        self, session: Session, release: Callable[[], None] | None = None
    ) -> None:
        """Wrap an unpublished Session in one preparation lifetime.

        Prefer :meth:`create`; direct construction is equivalent.

        :param session: exact unpublished Session.
        :param release: releases provider-owned state when the Session was
            not published.
        """
        self.session = session
        self._release = release
        self._released = False

    @classmethod
    def create(
        cls, session: Session, release: Callable[[], None] | None = None
    ) -> SessionPreparation:
        """Wrap an unpublished Session in one preparation lifetime.

        :param session: exact unpublished Session.
        :param release: optional provider release behavior.
        :returns: a preparation disposed after publication or rollback.
        """
        return cls(session, release)

    def close(self) -> None:
        """Release provider state once when this preparation leaves its caller."""
        if self._released:
            return
        self._released = True
        if self._release is not None:
            self._release()

    def __enter__(self) -> SessionPreparation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
