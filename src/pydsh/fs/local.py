"""Host-filesystem implementation of ``ctx.fs``.

Ported from deepseek-harness ``packages/fs/fs-local/src/index.ts`` (MIT).

Realpath-derived target identity makes aliases share stale guards, and
writes through a symlink update its target without replacing the link.
Loading :class:`LocalFileSystem` as a plugin populates ``ctx.fs``. Raw I/O
lives in :mod:`pydsh.fs.fsio` (Cordis-free); this module is the thin service
wiring.

Omitted from the TS surface: the ``sandboxPolicy`` per-call parameter and the
``internals`` test hooks (sandbox packages and native-fault injection are out
of scope for this port).
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from . import (
    AbortSignal,
    FileSystem,
    FsCreateIfAbsent,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsInfo,
    FsPathInfo,
    FsReplaceIfVersion,
    FsTarget,
    FsVersion,
    FsWriteIntent,
    FsWriteOutcome,
)
from .fsio import (
    LocalTarget,
    PathInfo,
    _absolute,
    apply_literal_edit,
    list_directory,
    normalize_line_endings,
    probe,
    probe_no_follow,
    read_for_edit,
    read_text_for_diff,
    read_whole_bytes,
    read_whole_text,
    resolve_local_target,
    restore_line_endings,
    stream_whole_text,
    write_file_atomic,
)

if TYPE_CHECKING:
    from pycordis import Context

T = TypeVar('T')

DEFAULT_DIFF_BASIS_MAX_BYTES = 10 * 1024 * 1024
# Python analogue of Node's MAX_LENGTH/MAX_STRING_LENGTH cap.
_MAX_DIFF_BASIS_BYTES = sys.maxsize


@dataclass(frozen=True)
class LocalFsConfig:
    """Validated configuration for the local filesystem backend."""

    #: Base directory for relative paths (a resolution default, NOT a
    #: containment boundary). Defaults to ``os.getcwd()``.
    cwd: str
    #: Exclusive UTF-8 byte limit on each overwrite-diff side. Defaults to
    #: 10 MiB.
    diff_basis_max_bytes: int


def _resolve_config(config: dict[str, Any] | None) -> LocalFsConfig:
    """Validate plugin config, applying defaults (the schemastery role)."""
    values = dict(config or {})
    cwd = values.get('cwd', os.getcwd())
    if not isinstance(cwd, str):
        raise ValueError('fs-local: cwd must be a string')
    limit = values.get('diff_basis_max_bytes', DEFAULT_DIFF_BASIS_MAX_BYTES)
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
        or limit > _MAX_DIFF_BASIS_BYTES
    ):
        raise ValueError(
            'fs-local: diff_basis_max_bytes must be a positive safe integer'
            f' no greater than {_MAX_DIFF_BASIS_BYTES}'
        )
    return LocalFsConfig(cwd=cwd, diff_basis_max_bytes=limit)


class LocalFileSystem(FileSystem):
    """The host-filesystem backend.

    Reads resolve relative paths from ``config.cwd`` (a resolution default,
    NOT a containment boundary); enforce containment with a stricter backend
    or a permission plugin. Load with ``ctx.plugin(LocalFileSystem, {...})``.
    """

    #: Cordis plugin config validator: ``dict | None -> LocalFsConfig``.
    Config = staticmethod(_resolve_config)

    def __init__(
        self,
        ctx: Context,
        config: LocalFsConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(ctx)
        self.config = (
            config if isinstance(config, LocalFsConfig) else _resolve_config(config)
        )
        # Per-target-key locks serialize mutating ops so the read→guard→write
        # window can't interleave, making concurrent writes/edits
        # deterministically ordered (one wins, the rest see the new version
        # and reject as stale).
        self._locks: dict[str, asyncio.Lock] = {}

    async def _with_lock(self, target_key: str, op: Coroutine[Any, Any, T]) -> T:
        """Run ``op`` with exclusive access to ``target_key``."""
        lock = self._locks.get(target_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[target_key] = lock
        async with lock:
            result = await op
        # No await between the release above and this check, so a popped
        # lock can never have a waiter.
        if not lock.locked():
            self._locks.pop(target_key, None)
        return result

    def _local(self, target: FsTarget) -> LocalTarget:
        return LocalTarget(
            display_path=target.display_path, target_key=target.target_key
        )

    async def resolve(
        self,
        path: str,
        *,
        cwd: str | None = None,
        signal: AbortSignal | None = None,
    ) -> FsTarget:
        if signal is not None and signal.is_set():
            raise FsError('resolve aborted', 'FS_ABORTED')
        base = cwd if cwd is not None else self.config.cwd
        local = await resolve_local_target(base, path)
        if signal is not None and signal.is_set():
            raise FsError('resolve aborted', 'FS_ABORTED')
        return FsTarget(target_key=local.target_key, display_path=local.display_path)

    def process_path(self, target: FsTarget) -> str:
        return str(target.target_key)

    def file_url(self, target: FsTarget) -> str:
        return Path(self.process_path(target)).as_uri()

    def contains(self, parent: FsTarget, child: FsTarget) -> bool:
        path = os.path.relpath(self.process_path(child), self.process_path(parent))
        return path == '.' or (
            path != '..'
            and not path.startswith(f'..{os.sep}')
            and not os.path.isabs(path)
        )

    async def stat(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> FsInfo | None:
        if signal is not None and signal.is_set():
            raise FsError('stat aborted', 'FS_ABORTED')
        info = await probe(target.target_key)
        if signal is not None and signal.is_set():
            raise FsError('stat aborted', 'FS_ABORTED')
        if info is None:
            return None
        return FsInfo(version=info.version, type=info.type, size=info.size)

    async def lstat(
        self,
        path: str,
        *,
        cwd: str | None = None,
        signal: AbortSignal | None = None,
    ) -> FsPathInfo | None:
        if signal is not None and signal.is_set():
            raise FsError('lstat aborted', 'FS_ABORTED')
        if not path.strip():
            raise FsError('file_path must be a non-empty string', 'FS_NOT_FOUND')
        base = cwd if cwd is not None else self.config.cwd
        info = await probe_no_follow(_absolute(base, path))
        if signal is not None and signal.is_set():
            raise FsError('lstat aborted', 'FS_ABORTED')
        if info is None:
            return None
        return FsPathInfo(version=info.version, type=info.type, size=info.size)

    async def read_text(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> str:
        return await read_whole_text(self._local(target), signal)

    def stream_text(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> AsyncIterator[str]:
        return stream_whole_text(self._local(target), signal)

    async def read_bytes(
        self,
        target: FsTarget,
        *,
        max_bytes: int,
        signal: AbortSignal | None = None,
    ) -> bytes:
        return await read_whole_bytes(self._local(target), signal, max_bytes)

    async def list_dir(
        self,
        target: FsTarget,
        *,
        signal: AbortSignal | None = None,
    ) -> list[FsDirEntry]:
        entries = await list_directory(self._local(target), signal)
        return [
            FsDirEntry(
                name=entry.name,
                type=entry.type,
                target=FsTarget(
                    target_key=entry.target.target_key,
                    display_path=entry.target.display_path,
                ),
                version=entry.version,
                size=entry.size,
            )
            for entry in entries
        ]

    async def write_text(
        self,
        target: FsTarget,
        content: str,
        *,
        expected: FsWriteIntent | None = None,
        signal: AbortSignal | None = None,
    ) -> FsWriteOutcome:
        async def op() -> FsWriteOutcome:
            existing = await probe(target.target_key)
            if existing is not None and existing.type != 'file':
                raise FsError(
                    f'cannot write "{target.display_path}": not a regular file',
                    'FS_NOT_REGULAR_FILE',
                )

            if isinstance(expected, FsReplaceIfVersion):
                # Stale guard: the file must still exist at the version the
                # owner observed.
                if existing is None:
                    raise FsError(
                        f'cannot write "{target.display_path}": file no longer exists',
                        'FS_STALE_VERSION',
                    )
                if existing.version != expected.version:
                    raise FsError(
                        f'cannot write "{target.display_path}": file changed since'
                        ' it was read',
                        'FS_STALE_VERSION',
                    )
            elif isinstance(expected, FsCreateIfAbsent) and existing is not None:
                # createIfAbsent onto an existing file: a blind overwrite —
                # require a read first.
                raise FsError(
                    f'cannot overwrite existing "{target.display_path}" without'
                    ' reading it first',
                    'FS_NOT_OBSERVED',
                )
            # No expectation means an unconditional but still atomic write.

            # Capture an optional contextual-diff basis before the write. The
            # bounded reader checks the opened file itself, so an external
            # replacement after ``probe()`` cannot turn this best-effort
            # presentation read into an unbounded allocation. Either side
            # at/above the configured limit yields ``before: None``.
            diffable = (
                existing is not None
                and len(content.encode('utf-8')) < self.config.diff_basis_max_bytes
            )
            before = (
                await read_text_for_diff(
                    str(target.target_key), self.config.diff_basis_max_bytes, signal
                )
                if diffable
                else None
            )
            await write_file_atomic(
                str(target.target_key),
                content,
                existing.mode if existing is not None else None,
                signal,
                target.display_path if isinstance(expected, FsCreateIfAbsent) else None,
            )
            after = await probe(target.target_key)
            return FsWriteOutcome(
                operation='update' if existing is not None else 'create',
                version=self._version_after_write(after, target),
                before=before,
                # LF-normalized to share the diff basis with ``before``: a
                # CRLF overwrite must not read as every line changed.
                after=normalize_line_endings(content),
            )

        return await self._with_lock(str(target.target_key), op())

    async def edit_text(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        *,
        expected: FsVersion | None = None,
        signal: AbortSignal | None = None,
    ) -> FsEditOutcome:
        async def op() -> FsEditOutcome:
            existing = await probe(target.target_key)
            # Stale guard before literal matching: an edit based on an old
            # read reports FS_STALE_VERSION, never FS_EDIT_NOT_FOUND /
            # FS_AMBIGUOUS_EDIT against newer content. Missing targets use
            # the same stale code on guarded and unconditional edit paths.
            if existing is None:
                raise FsError(
                    f'cannot edit "{target.display_path}": file changed since it'
                    ' was read',
                    'FS_STALE_VERSION',
                )
            if existing.type != 'file':
                raise FsError(
                    f'cannot edit "{target.display_path}": not a regular file',
                    'FS_NOT_REGULAR_FILE',
                )
            # expected is None: unconditional edit of the current content —
            # no version guard. Still inside the per-target lock, so the
            # read→match→write window is serialized and atomic.
            if expected is not None and existing.version != expected:
                raise FsError(
                    f'cannot edit "{target.display_path}": file changed since it'
                    ' was read',
                    'FS_STALE_VERSION',
                )

            original = await read_for_edit(
                str(target.target_key), target.display_path, signal
            )
            edited, _replacements = apply_literal_edit(
                original.content,
                edit.old_string,
                edit.new_string,
                edit.replace_all,
                target.display_path,
            )
            content = restore_line_endings(edited, original.line_endings)
            await write_file_atomic(
                str(target.target_key), content, existing.mode, signal
            )

            after = await probe(target.target_key)
            return FsEditOutcome(
                version=self._version_after_write(after, target),
                # The LF-normalized before/after text (the applied-hunk diff
                # basis); line-ending restoration is a storage detail the
                # diff ignores.
                before=original.content,
                after=edited,
            )

        return await self._with_lock(str(target.target_key), op())

    def _version_after_write(
        self, after: PathInfo | None, target: FsTarget
    ) -> FsVersion:
        # The post-write probe finding the file absent requires a concurrent
        # unlink between publication and stat; fall back to a sentinel.
        if after is not None:
            return after.version
        return FsVersion(f'missing:{target.target_key}')
