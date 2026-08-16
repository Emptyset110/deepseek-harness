"""Filesystem Service Definition for one execution world (``ctx.fs``).

Ported from deepseek-harness ``packages/fs/fs/src/index.ts`` (MIT).

Backends own stable target identity, process paths and file URIs,
containment, text reads, decoding, binary rejection, and atomic mutations.
Read windows and observed-state policy stay in consumer and policy plugins;
``edit_text`` remains here so version check, literal match, and rewrite share
one critical section.

This package also owns the ``fs/*`` policy event vocabulary (declared via
declaration merging in TS; cordis-py events are dynamic, so these are
documentation-only contracts):

- ``fs/write-intent(target, actor, next)`` — waterfall. Single-slot decision
  for the next :meth:`FileSystem.write_text`; calling ``next()`` yields the
  bare provider's unconditional write, the first listener that returns an
  :data:`FsWriteIntent` owns the decision.
- ``fs/edit-intent(target, actor, next)`` — waterfall. Single-slot decision
  for the next :meth:`FileSystem.edit_text`; the first returned
  ``FsVersion`` guard wins.
- ``fs/observed(target, observation, actor)`` — emit. Records an
  authoritative positive or negative :data:`FsObservation`; listeners must be
  synchronous recorders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pycordis import Service

from .types import (
    AbortSignal,
    FsCreateIfAbsent,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsErrorCode,
    FsInfo,
    FsObservation,
    FsObservationAbsent,
    FsObservationPresent,
    FsPathInfo,
    FsReplaceIfVersion,
    FsTarget,
    FsTargetKey,
    FsVersion,
    FsWriteIntent,
    FsWriteOutcome,
)

if TYPE_CHECKING:
    from pycordis import Context

__all__ = [
    'AbortSignal',
    'FileSystem',
    'FsCreateIfAbsent',
    'FsDirEntry',
    'FsEditOutcome',
    'FsEditRequest',
    'FsError',
    'FsErrorCode',
    'FsInfo',
    'FsObservation',
    'FsObservationAbsent',
    'FsObservationPresent',
    'FsPathInfo',
    'FsReplaceIfVersion',
    'FsTarget',
    'FsTargetKey',
    'FsVersion',
    'FsWriteIntent',
    'FsWriteOutcome',
]


class FileSystem(Service, ABC):
    """Abstract filesystem provider registered as ``ctx.fs``.

    Targets must preserve identity across aliases; reads expose regular UTF-8
    text or typed errors, listings are stable and content-free, and mutations
    are atomic. Optional guards add stale protection without changing the
    unguarded provider contract.
    """

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx, 'fs')

    @property
    def sandbox_mode(self) -> str | None:
        """The sandbox mode this backend enforces on mutations by default.

        ``None`` when it does not confine at all — the capability fact the
        tool layer reads to advertise escalation fields honestly. The base
        class and the bare local backend report ``None``; a sandboxing
        backend overrides this with the deployment default (``str`` mode
        name). The sandbox packages themselves are out of scope for this
        port.
        """
        return None

    @abstractmethod
    async def resolve(
        self,
        path: str,
        *,
        cwd: str | None = None,
        signal: AbortSignal | None = None,
    ) -> FsTarget:
        """Resolve a model/plugin-supplied path into a stable ``FsTarget``.

        May perform I/O (a remote/sandboxed backend may need a round-trip),
        hence async even though the local backend only normalizes +
        realpaths. Relative paths resolve against ``cwd`` (omitted ⇒ the
        backend default). The same file yields the same ``target_key``.
        """

    @abstractmethod
    def process_path(self, target: FsTarget) -> str:
        """Return the canonical absolute path a subprocess in this
        filesystem's execution world can open.

        Deliberately separate from ``FsTarget.target_key``: consumers may
        pass this value to another OS capability, but must continue treating
        the target key as opaque.
        """

    @abstractmethod
    def file_url(self, target: FsTarget) -> str:
        """Return the canonical ``file:`` URI for a target.

        Backends own URI encoding because the host platform may differ from
        the execution platform.
        """

    @abstractmethod
    def contains(self, parent: FsTarget, child: FsTarget) -> bool:
        """Test canonical containment without exposing or parsing target keys.

        Both targets must come from this provider. Returns true when ``child``
        is ``parent`` or a descendant of it.
        """

    @abstractmethod
    async def stat(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> FsInfo | None:
        """Return target metadata, or ``None`` when the target does not exist.

        Metadata only, never content.
        """

    @abstractmethod
    async def lstat(
        self,
        path: str,
        *,
        cwd: str | None = None,
        signal: AbortSignal | None = None,
    ) -> FsPathInfo | None:
        """Return path metadata without following a final symlink component.

        Intentionally path-shaped, not target-shaped: :meth:`resolve` follows
        symlinks to produce the stable identity used by normal reads/writes,
        while ``lstat`` lets a consumer reject the path itself before that
        follow happens. ``cwd`` follows :meth:`resolve`'s cwd rules. ``None``
        means the path is absent.
        """

    @abstractmethod
    async def read_text(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> str:
        """Read the whole regular text file as a single decoded string."""

    @abstractmethod
    def stream_text(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> AsyncIterator[str]:
        """Stream the whole regular text file as decoded text chunks.

        Same text semantics as :meth:`read_text`, for large files. The
        backend owns cross-chunk UTF-8 decoding and binary rejection so the
        policy layer never touches raw bytes.
        """

    @abstractmethod
    async def read_bytes(
        self,
        target: FsTarget,
        *,
        max_bytes: int,
        signal: AbortSignal | None = None,
    ) -> bytes:
        """Read the whole regular file as raw bytes with no decoding.

        The bound lives at this seam so a backend can never buffer an
        unbounded file: a target known or discovered to exceed ``max_bytes``
        fails with ``FS_TOO_LARGE`` instead of returning a truncated result.
        """

    @abstractmethod
    async def list_dir(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> list[FsDirEntry]:
        """List direct children of a directory in stable name order.

        Returns resolved child targets plus cheap metadata only; never reads
        file contents.
        """

    @abstractmethod
    async def write_text(
        self,
        target: FsTarget,
        content: str,
        *,
        expected: FsWriteIntent | None = None,
        signal: AbortSignal | None = None,
    ) -> FsWriteOutcome:
        """Atomically create or replace UTF-8 text.

        ``expected`` guards intent and staleness; omission allows
        unconditional overwrite. Aborts (``FS_ABORTED``) take effect before
        atomic publication.
        """

    @abstractmethod
    async def edit_text(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        *,
        expected: FsVersion | None = None,
        signal: AbortSignal | None = None,
    ) -> FsEditOutcome:
        """Atomically edit literal text.

        When supplied, the ``expected`` version guard is checked before
        matching so stale content reports ``FS_STALE_VERSION``; omission edits
        the current content without a freshness precondition. (TS passes
        ``{ version }``; the version alone is the guard here.)
        """
