"""Cordis-free local filesystem mechanics.

Ported from deepseek-harness ``packages/fs/fs-local/src/fsio.ts`` (MIT).

This provider layer returns validated UTF-8 text, streams large files, and
rejects binary data; line windows belong to the tool layer. Writes stage an
exclusive owner-only file in a private sibling directory and atomically
publish it. All blocking calls run through :func:`asyncio.to_thread`; the TS
``AbortSignal`` maps to a settable :class:`asyncio.Event` checked between
steps. Windows-specific DACL preservation is not ported (POSIX semantics
only).
"""

from __future__ import annotations

import asyncio
import codecs
import os
import shutil
import stat as stat_module
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .types import AbortSignal, FsError, FsTargetKey, FsVersion

_BINARY_SAMPLE_BYTES = 8192
_READ_CHUNK_BYTES = 64 * 1024

#: Line ending style detected before LF normalization.
LineEndings: TypeAlias = Literal['LF', 'CRLF']


def _throw_if_aborted(signal: AbortSignal | None, verb: str) -> None:
    if signal is not None and signal.is_set():
        raise FsError(f'{verb} aborted', 'FS_ABORTED')


def _absolute(cwd: str, path: str) -> str:
    """Join and normalize a path against a base directory (pure string op)."""
    return os.path.abspath(os.path.join(cwd, path))


def _version_of(info: os.stat_result) -> FsVersion:
    """Opaque version token from identity and freshness metadata."""
    return FsVersion(
        f'{info.st_dev}:{info.st_ino}:{info.st_size}'
        f':{info.st_mtime_ns}:{info.st_ctime_ns}'
    )


@dataclass(frozen=True)
class LocalTarget:
    """A resolved local path: the display path and its realpath identity."""

    #: Absolute path (symlinks not resolved) — used for display.
    display_path: str
    #: Realpath identity — used as the stable target key and the I/O path.
    target_key: FsTargetKey


@dataclass(frozen=True)
class PathInfo:
    """Result of probing a path: ``None`` from ``probe`` means absent."""

    version: FsVersion
    mode: int
    type: Literal['file', 'directory', 'other']
    size: int


@dataclass(frozen=True)
class PathLinkInfo:
    """Result of probing a path without following the final symlink."""

    version: FsVersion
    mode: int
    type: Literal['file', 'directory', 'symlink', 'other']
    size: int


@dataclass(frozen=True)
class LocalDirEntry:
    """One local directory child with a resolved target and cheap metadata."""

    name: str
    type: Literal['file', 'directory', 'other']
    target: LocalTarget
    version: FsVersion | None = None
    size: int | None = None


async def resolve_local_target(cwd: str, path: str) -> LocalTarget:
    """Resolve a path to its absolute display path and realpath identity.

    For a missing target, realpath the nearest existing ancestor and append
    the missing suffix, preserving identity across symlinked ancestors before
    and after creation. Empty/whitespace-only ``path`` throws
    ``FS_NOT_FOUND``; a parent segment that is a regular file also throws
    ``FS_NOT_FOUND`` (the target can never exist).
    """
    if not path.strip():
        raise FsError('file_path must be a non-empty string', 'FS_NOT_FOUND')
    display_path = _absolute(cwd, path)
    try:
        # Prefer the file's own realpath (resolves a symlinked file to its
        # target). strict=True restores Node's ENOENT/ENOTDIR behavior, which
        # Python's non-strict realpath never raises.
        real = await asyncio.to_thread(os.path.realpath, display_path, strict=True)
        return LocalTarget(display_path=display_path, target_key=FsTargetKey(real))
    except NotADirectoryError:
        # A path component is a file, not a directory (e.g. "afile/child.txt"
        # where "afile" is a regular file): the target can neither exist nor
        # be created.
        raise FsError(
            f'cannot resolve "{display_path}": a parent path segment is not'
            ' a directory',
            'FS_NOT_FOUND',
        ) from None
    except FileNotFoundError:
        pass
    # File absent: realpath the nearest existing ancestor and re-append the
    # missing suffix (the file basename plus any not-yet-created intermediate
    # dirs), so the key is stable across creation of those dirs.
    missing = [os.path.basename(display_path)]
    ancestor = os.path.dirname(display_path)
    while True:
        try:
            real_ancestor = await asyncio.to_thread(
                os.path.realpath, ancestor, strict=True
            )
            return LocalTarget(
                display_path=display_path,
                target_key=FsTargetKey(os.path.join(real_ancestor, *missing)),
            )
        except FileNotFoundError:
            parent = os.path.dirname(ancestor)
            # The filesystem root always realpaths, so the walk terminates
            # before parent == ancestor.
            if parent == ancestor:
                return LocalTarget(
                    display_path=display_path, target_key=FsTargetKey(display_path)
                )
            missing.insert(0, os.path.basename(ancestor))
            ancestor = parent


def _path_type(info: os.stat_result) -> Literal['file', 'directory', 'other']:
    if stat_module.S_ISREG(info.st_mode):
        return 'file'
    if stat_module.S_ISDIR(info.st_mode):
        return 'directory'
    return 'other'


def _path_link_type(
    info: os.stat_result,
) -> Literal['file', 'directory', 'symlink', 'other']:
    if stat_module.S_ISLNK(info.st_mode):
        return 'symlink'
    return _path_type(info)


async def probe(absolute_path: str | FsTargetKey) -> PathInfo | None:
    """Probe a path for its version, mode, type, and size.

    ``None`` when the path — or a parent segment — does not exist. Any other
    metadata failure (permission/IO fault) propagates.
    """
    try:
        info = await asyncio.to_thread(os.stat, absolute_path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return PathInfo(
        version=_version_of(info),
        mode=stat_module.S_IMODE(info.st_mode),
        type=_path_type(info),
        size=info.st_size,
    )


async def probe_no_follow(absolute_path: str) -> PathLinkInfo | None:
    """Probe a path without following the final symlink component.

    ``None`` when the path entry is absent.
    """
    try:
        info = await asyncio.to_thread(os.lstat, absolute_path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return PathLinkInfo(
        version=_version_of(info),
        mode=stat_module.S_IMODE(info.st_mode),
        type=_path_link_type(info),
        size=info.st_size,
    )


# --- Directory listing ---


def _listing_io_error(display_path: str, error: BaseException) -> FsError:
    # Pass-through for races where a child resolver has already produced a
    # structured FsError.
    if isinstance(error, FsError):
        return error
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return FsError(
            f'cannot list "{display_path}": not found', 'FS_NOT_FOUND', cause=error
        )
    if isinstance(error, PermissionError):
        return FsError(
            f'cannot list "{display_path}": permission denied',
            'FS_PERMISSION_DENIED',
            cause=error,
        )
    return FsError(f'cannot list "{display_path}": {error}', 'FS_IO_ERROR', cause=error)


async def list_directory(
    target: LocalTarget, signal: AbortSignal | None = None
) -> list[LocalDirEntry]:
    """List direct children of a directory in stable name order.

    Each child includes a resolved target plus stat metadata when still
    available; file contents are never read. A missing or non-directory
    target throws; ``signal`` aborts the listing, checked between children
    (``FS_ABORTED``).
    """
    _throw_if_aborted(signal, 'list')
    try:
        info = await probe(target.target_key)
    except OSError as error:
        raise _listing_io_error(target.display_path, error) from error
    if info is None:
        raise FsError(f'cannot list "{target.display_path}": not found', 'FS_NOT_FOUND')
    if info.type != 'directory':
        raise FsError(
            f'cannot list "{target.display_path}": not a directory', 'FS_NOT_DIRECTORY'
        )

    try:
        names = await asyncio.to_thread(os.listdir, target.target_key)
    except OSError as error:
        raise _listing_io_error(target.display_path, error) from error
    _throw_if_aborted(signal, 'list')

    result: list[LocalDirEntry] = []
    for name in sorted(names):
        _throw_if_aborted(signal, 'list')
        try:
            identity = await resolve_local_target(str(target.target_key), name)
            child_target = LocalTarget(
                display_path=os.path.join(target.display_path, name),
                target_key=identity.target_key,
            )
            child_info = await probe(child_target.target_key)
        except (OSError, FsError) as error:
            raise _listing_io_error(
                os.path.join(target.display_path, name), error
            ) from error
        result.append(
            LocalDirEntry(
                name=name,
                type=child_info.type if child_info is not None else 'other',
                target=child_target,
                version=child_info.version if child_info is not None else None,
                size=(
                    child_info.size
                    if child_info is not None and child_info.type == 'file'
                    else None
                ),
            )
        )
        _throw_if_aborted(signal, 'list')
    return result


# --- Reading ---


def _not_text_error(verb: Literal['read', 'edit'], display_path: str) -> FsError:
    return FsError(f'cannot {verb} "{display_path}": invalid UTF-8 text', 'FS_NOT_TEXT')


def _decode_utf8(
    buffer: bytes, verb: Literal['read', 'edit'], display_path: str
) -> str:
    try:
        return buffer.decode('utf-8')
    except UnicodeDecodeError:
        raise _not_text_error(verb, display_path) from None


async def _stat_regular_file(
    target: LocalTarget,
    verb: Literal['read'],
    signal: AbortSignal | None,
) -> os.stat_result:
    _throw_if_aborted(signal, verb)
    try:
        info = await asyncio.to_thread(os.stat, target.target_key)
    except FileNotFoundError:
        raise FsError(
            f'cannot {verb} "{target.display_path}": not found', 'FS_NOT_FOUND'
        ) from None
    if not stat_module.S_ISREG(info.st_mode):
        raise FsError(
            f'cannot {verb} "{target.display_path}": not a regular file',
            'FS_NOT_REGULAR_FILE',
        )
    return info


def _read_file_bytes(path: str) -> bytes:
    with open(path, 'rb') as handle:
        return handle.read()


async def read_whole_text(
    target: LocalTarget, signal: AbortSignal | None = None
) -> str:
    """Read a whole regular UTF-8 text file into a single decoded string.

    Rejects non-regular files, invalid UTF-8, and NUL-byte binary samples.
    Returns the full decoded text byte-for-byte (no normalization).
    """
    await _stat_regular_file(target, 'read', signal)
    raw = await asyncio.to_thread(_read_file_bytes, str(target.target_key))
    _throw_if_aborted(signal, 'read')
    if b'\x00' in raw[:_BINARY_SAMPLE_BYTES]:
        raise FsError(
            f'cannot read "{target.display_path}": binary file', 'FS_NOT_TEXT'
        )
    return _decode_utf8(raw, 'read', target.display_path)


def _read_up_to(path: str, limit: int) -> bytes:
    """Read at most ``limit`` bytes (one past the cap detects growth)."""
    with open(path, 'rb') as handle:
        return handle.read(limit)


async def read_whole_bytes(
    target: LocalTarget,
    signal: AbortSignal | None,
    max_bytes: int,
) -> bytes:
    """Read a whole regular file as raw bytes with no decoding or rejection.

    ``max_bytes`` bounds the complete content: the stat size short-circuits
    an oversized file before any content I/O, and the read takes at most one
    byte beyond the cap so a file growing after stat cannot cause unbounded
    buffering (``FS_TOO_LARGE`` either way).
    """
    info = await _stat_regular_file(target, 'read', signal)
    if info.st_size > max_bytes:
        raise FsError(
            f'cannot read "{target.display_path}": {info.st_size} bytes exceeds'
            f' the {max_bytes}-byte limit',
            'FS_TOO_LARGE',
        )
    data = await asyncio.to_thread(_read_up_to, str(target.target_key), max_bytes + 1)
    if len(data) > max_bytes:
        raise FsError(
            f'cannot read "{target.display_path}": content exceeds'
            f' the {max_bytes}-byte limit',
            'FS_TOO_LARGE',
        )
    return data


async def stream_whole_text(
    target: LocalTarget, signal: AbortSignal | None = None
) -> AsyncIterator[str]:
    """Stream a whole regular UTF-8 text file as decoded text chunks.

    Same text semantics as :func:`read_whole_text` (regular-file check,
    binary/NUL rejection, cross-chunk UTF-8 decoding), but never holds the
    whole file in memory. Chunk boundaries carry no meaning.
    """
    await _stat_regular_file(target, 'read', signal)
    handle = await asyncio.to_thread(open, str(target.target_key), 'rb')
    decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')
    sampled = 0
    try:
        while True:
            _throw_if_aborted(signal, 'read')
            chunk = await asyncio.to_thread(handle.read, _READ_CHUNK_BYTES)
            if not chunk:
                break
            if sampled < _BINARY_SAMPLE_BYTES:
                sample = chunk[: _BINARY_SAMPLE_BYTES - sampled]
                if b'\x00' in sample:
                    raise FsError(
                        f'cannot read "{target.display_path}": binary file',
                        'FS_NOT_TEXT',
                    )
                sampled += len(sample)
            try:
                text = decoder.decode(chunk)
            except UnicodeDecodeError:
                raise _not_text_error('read', target.display_path) from None
            if text:
                yield text
        try:
            tail = decoder.decode(b'', True)
        except UnicodeDecodeError:
            raise _not_text_error('read', target.display_path) from None
        if tail:
            yield tail
    finally:
        await asyncio.to_thread(handle.close)


# --- Writing ---


def _make_staging_dir(staging_dir: str) -> None:
    os.mkdir(staging_dir, 0o700)
    os.chmod(staging_dir, 0o700)


def _write_temp_file(temp_path: str, data: bytes, mode: int | None) -> None:
    # Exclusive owner-only create; fsync before publication.
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        if mode is not None:
            os.fchmod(fd, mode)
    finally:
        os.close(fd)


async def _publish_no_replace(
    temp_path: str, absolute_path: str, display_path: str
) -> None:
    """Publish by hard-linking the staged file into place (no replace).

    A concurrent creator's file is preserved and this write is rejected with
    ``FS_NOT_OBSERVED``; a non-regular path entry is preserved and rejected
    with ``FS_NOT_REGULAR_FILE``.
    """
    try:
        await asyncio.to_thread(os.link, temp_path, absolute_path)
        return
    except OSError as error:
        # Link errno values vary by platform and filesystem. Inspect the
        # target entry after failure so a collision is not confused with
        # missing hard-link support.
        existing: os.stat_result | None
        try:
            existing = await asyncio.to_thread(os.lstat, absolute_path)
        except (FileNotFoundError, NotADirectoryError):
            existing = None
        except OSError as metadata_error:
            raise FsError(
                f'cannot write "{display_path}": {metadata_error}',
                'FS_IO_ERROR',
                cause=metadata_error,
            ) from metadata_error

        if existing is not None:
            if not stat_module.S_ISREG(existing.st_mode):
                raise FsError(
                    f'cannot write "{display_path}": not a regular file',
                    'FS_NOT_REGULAR_FILE',
                    cause=error,
                ) from error
            raise FsError(
                f'cannot overwrite existing "{display_path}" without reading it first',
                'FS_NOT_OBSERVED',
                cause=error,
            ) from error
        if isinstance(error, FileExistsError):
            raise FsError(
                f'cannot overwrite existing "{display_path}" without reading it first',
                'FS_NOT_OBSERVED',
                cause=error,
            ) from error
        raise FsError(
            f'cannot write "{display_path}": {error}', 'FS_IO_ERROR', cause=error
        ) from error


async def write_file_atomic(
    absolute_path: str,
    content: str,
    mode: int | None,
    signal: AbortSignal | None,
    create_if_absent_display: str | None = None,
) -> None:
    """Atomically replace a file through a private, synced staging file.

    The staging directory (``0o700``) and temp file (``0o600``) live next to
    the target; missing parent directories are created. ``mode`` preserves an
    existing destination's POSIX mode, or is ``None`` for a new file.
    ``signal`` is checked before final publication. When
    ``create_if_absent_display`` is given, publication is a hard-link
    no-replace primitive guarded under that display path. Post-commit staging
    cleanup is best effort; pre-commit failure cleans up and rethrows.
    """
    _throw_if_aborted(signal, 'write')
    directory = os.path.dirname(absolute_path)
    await asyncio.to_thread(os.makedirs, directory, exist_ok=True)

    _throw_if_aborted(signal, 'write')
    basename = os.path.basename(absolute_path)
    staging_dir = os.path.join(
        directory, f'.{basename}.{os.getpid()}.{uuid.uuid4().hex}.tmpdir'
    )
    temp_path = os.path.join(staging_dir, f'{basename}.tmp')
    try:
        await asyncio.to_thread(_make_staging_dir, staging_dir)
        await asyncio.to_thread(
            _write_temp_file, temp_path, content.encode('utf-8'), mode
        )
        _throw_if_aborted(signal, 'write')
        if create_if_absent_display is not None:
            await _publish_no_replace(
                temp_path, absolute_path, create_if_absent_display
            )
        else:
            await asyncio.to_thread(os.replace, temp_path, absolute_path)
    except BaseException:
        # Pre-commit failure: the target was never published; drop staging.
        await asyncio.to_thread(shutil.rmtree, staging_dir, ignore_errors=True)
        raise
    # The target is committed; owner-only staging residue cannot turn that
    # write into a failure.
    await asyncio.to_thread(shutil.rmtree, staging_dir, ignore_errors=True)


# --- Editing ---


def normalize_line_endings(content: str) -> str:
    """Collapse CRLF to LF — the canonical in-memory edit/diff basis.

    Lone ``\\r`` characters (not followed by ``\\n``) are left untouched.
    """
    return content.replace('\r\n', '\n')


def _detect_line_endings(raw: str) -> LineEndings:
    sample = raw[:4096]
    crlf_count = sample.count('\r\n')
    lf_count = sample.count('\n') - crlf_count
    return 'CRLF' if crlf_count > lf_count else 'LF'


def restore_line_endings(content: str, line_endings: LineEndings) -> str:
    """Convert LF-normalized content back to the detected style.

    ``LF`` returns the content unchanged; ``CRLF`` re-normalizes first so an
    already-CRLF sequence is never doubled to ``\\r\\r\\n``.
    """
    if line_endings == 'LF':
        return content
    return normalize_line_endings(content).replace('\n', '\r\n')


@dataclass(frozen=True)
class EditReadResult:
    """LF-normalized content plus the original style for write-back."""

    content: str
    line_endings: LineEndings


async def read_for_edit(
    absolute_path: str,
    display_path: str,
    signal: AbortSignal | None = None,
) -> EditReadResult:
    """Read and decode a file for editing.

    Rejects binaries (NUL anywhere), returns LF-normalized content plus the
    original line-ending style for write-back.
    """
    _throw_if_aborted(signal, 'edit')
    buffer = await asyncio.to_thread(_read_file_bytes, absolute_path)
    _throw_if_aborted(signal, 'edit')
    if b'\x00' in buffer:
        raise FsError(f'cannot edit "{display_path}": binary file', 'FS_NOT_TEXT')
    raw = _decode_utf8(buffer, 'edit', display_path)
    return EditReadResult(
        content=normalize_line_endings(raw), line_endings=_detect_line_endings(raw)
    )


def _read_diff_basis(path: str, max_bytes: int) -> str | None:
    with open(path, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat_module.S_ISREG(info.st_mode):
            return None
        if info.st_size >= max_bytes:
            return None
        opened_size = info.st_size
        # One extra byte detects growth after stat without retaining per-read
        # backing buffers.
        data = handle.read(opened_size + 1)
    if len(data) != opened_size:
        return None
    if b'\x00' in data:
        return None
    try:
        return normalize_line_endings(data.decode('utf-8'))
    except UnicodeDecodeError:
        return None


async def read_text_for_diff(
    absolute_path: str,
    max_bytes: int,
    signal: AbortSignal | None = None,
) -> str | None:
    """Best-effort overwrite diff basis.

    Binary, invalid UTF-8, a file at/above the byte limit, or a file
    deleted/made unreadable after the caller's preflight returns ``None`` so
    the write still succeeds and presentation falls back to a whole-file
    diff. The bound is enforced on the opened descriptor rather than a prior
    path stat. Cancellation (``FS_ABORTED``) propagates, unlike I/O failure.
    """
    _throw_if_aborted(signal, 'read')
    try:
        return await asyncio.to_thread(_read_diff_basis, absolute_path, max_bytes)
    except OSError:
        # A descriptor-phase errno — deleted or made unreadable after the
        # caller's preflight — costs only the optional basis: a committed
        # write must not fail for a presentation-only pre-read.
        return None


def apply_literal_edit(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    display_path: str,
) -> tuple[str, int]:
    """Apply a literal replacement to LF-normalized content.

    Empty or missing search text throws ``FS_EDIT_NOT_FOUND``; multiple
    matches throw ``FS_AMBIGUOUS_EDIT`` unless ``replace_all`` is true.
    Returns the edited LF-normalized content plus the replacement count.
    """
    old_norm = normalize_line_endings(old_string)
    if not old_norm:
        raise FsError('old_string must be a non-empty string', 'FS_EDIT_NOT_FOUND')
    new_norm = normalize_line_endings(new_string)
    replacements = content.count(old_norm)
    if replacements == 0:
        raise FsError(
            f'old_string was not found in "{display_path}"', 'FS_EDIT_NOT_FOUND'
        )
    if not replace_all and replacements > 1:
        raise FsError(
            f'old_string matched {replacements} times in "{display_path}";'
            ' provide a more specific old_string or set replace_all to true',
            'FS_AMBIGUOUS_EDIT',
        )
    return content.replace(old_norm, new_norm), replacements
