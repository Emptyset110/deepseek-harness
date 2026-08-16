"""Tests for the fs capability seam: the ``ctx.fs`` Service Definition and
its local backend.

Ported-semantics tests for deepseek-harness ``packages/fs`` (MIT):
registration/inject gating/disposal of the abstract service, and the local
backend's resolve/stat/lstat/read/stream/list/write/edit behavior including
guards, error taxonomy, symlink identity, and per-target serialization.
"""

from __future__ import annotations

import asyncio
import os
import stat as stat_module
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pycordis import Context, FiberState
from pydsh.fs import (
    FileSystem,
    FsCreateIfAbsent,
    FsDirEntry,
    FsEditOutcome,
    FsEditRequest,
    FsError,
    FsInfo,
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
from pydsh.fs.local import DEFAULT_DIFF_BASIS_MAX_BYTES, LocalFileSystem, LocalFsConfig


class FakeFileSystem(FileSystem):
    """A minimal in-memory backend implementing the provider primitives."""

    marker = 'fake'

    def __init__(self, ctx: Context, config: object = None) -> None:
        super().__init__(ctx)
        self.files: dict[str, str] = {}

    async def resolve(self, path: str, **kwargs: object) -> FsTarget:
        return FsTarget(target_key=FsTargetKey(path), display_path=path)

    def process_path(self, target: FsTarget) -> str:
        return str(target.target_key)

    def file_url(self, target: FsTarget) -> str:
        return f'file:///{target.target_key}'

    def contains(self, parent: FsTarget, child: FsTarget) -> bool:
        return child.target_key == parent.target_key or str(
            child.target_key
        ).startswith(f'{parent.target_key}/')

    async def stat(self, target: FsTarget, **kwargs: object) -> FsInfo | None:
        content = self.files.get(str(target.target_key))
        if content is None:
            return None
        return FsInfo(version=FsVersion('v1'), type='file', size=len(content))

    async def lstat(self, path: str, **kwargs: object) -> FsPathInfo | None:
        content = self.files.get(path)
        if content is None:
            return None
        return FsPathInfo(version=FsVersion('v1'), type='file', size=len(content))

    async def read_text(self, target: FsTarget, **kwargs: object) -> str:
        content = self.files.get(str(target.target_key))
        if content is None:
            raise FsError(f'not found: {target.display_path}', 'FS_NOT_FOUND')
        return content

    def stream_text(self, target: FsTarget, **kwargs: object) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield await self.read_text(target)

        return gen()

    async def read_bytes(
        self, target: FsTarget, *, max_bytes: int, **kwargs: object
    ) -> bytes:
        data = (await self.read_text(target)).encode()
        if len(data) > max_bytes:
            raise FsError(f'too large: {target.display_path}', 'FS_TOO_LARGE')
        return data

    async def list_dir(self, target: FsTarget, **kwargs: object) -> list[FsDirEntry]:
        raise FsError(f'not a directory: {target.display_path}', 'FS_NOT_DIRECTORY')

    async def write_text(
        self,
        target: FsTarget,
        content: str,
        *,
        expected: FsWriteIntent | None = None,
        **kwargs: object,
    ) -> FsWriteOutcome:
        before = self.files.get(str(target.target_key))
        self.files[str(target.target_key)] = content
        return FsWriteOutcome(
            operation='update' if before is not None else 'create',
            version=FsVersion('v2'),
            before=before,
            after=content,
        )

    async def edit_text(
        self,
        target: FsTarget,
        edit: FsEditRequest,
        *,
        expected: FsVersion | None = None,
        **kwargs: object,
    ) -> FsEditOutcome:
        content = self.files.get(str(target.target_key), '')
        after = content.replace(edit.old_string, edit.new_string)
        self.files[str(target.target_key)] = after
        return FsEditOutcome(version=FsVersion('v2'), before=content, after=after)


# --- Service Definition contract ---


async def test_fs_unreachable_without_backend() -> None:
    ctx = Context()
    # No provider: the strict store read yields nothing.
    assert ctx.get('fs') is None
    # Root-context attribute access resolves to None as well.
    assert ctx.fs is None

    def peek(c: Context, config: object = None) -> None:
        _ = c.fs  # not declared in inject

    failed = ctx.plugin(peek)
    with pytest.raises(AttributeError, match='without inject'):
        await failed
    assert failed.state is FiberState.FAILED


async def test_inject_gates_on_fs_service() -> None:
    ctx = Context()
    seen: list[FileSystem] = []

    consumer = ctx.inject(['fs'], lambda c, config=None: seen.append(c.fs))
    assert consumer.state is FiberState.PENDING

    provider = ctx.plugin(FakeFileSystem)
    await provider
    await consumer
    assert consumer.state is FiberState.ACTIVE
    assert len(seen) == 1
    # attribute reads resolve through the per-access service view
    assert seen[0].marker == 'fake'


async def test_provider_dispose_revokes_fs_and_unloads_dependent() -> None:
    ctx = Context()
    unloads: list[str] = []

    def consumer(c: Context, config: object = None) -> object:
        return lambda: unloads.append('unload')

    consumer_fiber = ctx.inject(['fs'], consumer)
    provider = ctx.plugin(FakeFileSystem)
    await provider
    await consumer_fiber
    assert consumer_fiber.state is FiberState.ACTIVE

    await provider.dispose()
    assert consumer_fiber.state is FiberState.PENDING
    assert unloads == ['unload']
    assert ctx.get('fs') is None


async def test_definition_basics() -> None:
    ctx = Context()
    fiber = ctx.plugin(FakeFileSystem)
    await fiber
    fs = ctx.get('fs')
    # the base class reports no sandbox confinement
    assert fs.sandbox_mode is None
    # branded ids are validation-free string brands
    assert FsTargetKey('k') == 'k'
    assert FsVersion('v') == 'v'
    # observation vocabulary
    assert FsObservationPresent(FsVersion('v')).kind == 'present'
    assert FsObservationAbsent().kind == 'absent'
    # FsError carries the stable code and chains a cause
    cause = OSError('disk')
    error = FsError('boom', 'FS_IO_ERROR', cause=cause)
    assert error.code == 'FS_IO_ERROR'
    assert error.__cause__ is cause
    assert str(error) == 'boom'
    await fiber.dispose()


async def test_abstract_service_cannot_be_instantiated() -> None:
    ctx = Context()
    with pytest.raises(TypeError, match='abstract'):
        FileSystem(ctx)  # type: ignore[abstract]


# --- Local backend ---


@pytest.fixture
async def backend(tmp_path: Path):
    ctx = Context()
    fiber = ctx.plugin(LocalFileSystem, {'cwd': str(tmp_path)})
    await fiber
    fs = ctx.get('fs')
    # ctx.get returns the per-access service view over the backend instance
    assert fs is not None and fs.config.cwd == str(tmp_path)
    yield ctx, fiber, fs
    await fiber.dispose()


async def _version_of(fs: LocalFileSystem, target: FsTarget) -> FsVersion:
    info = await fs.stat(target)
    assert info is not None
    return info.version


def _realpath(path: Path) -> str:
    return os.path.realpath(path)


def _dir_names(path: Path) -> list[str]:
    return [entry.name for entry in path.iterdir()]


async def test_local_registration_defaults() -> None:
    ctx = Context()
    fiber = ctx.plugin(LocalFileSystem)
    await fiber
    fs = ctx.get('fs')
    assert fs.config.cwd == os.getcwd()
    assert fs.config.diff_basis_max_bytes == DEFAULT_DIFF_BASIS_MAX_BYTES
    assert isinstance(fs.config, LocalFsConfig)
    await fiber.dispose()


async def test_local_rejects_invalid_diff_basis_limit() -> None:
    for bad in (0, -1, 1.5, True):
        ctx = Context()
        fiber = ctx.plugin(LocalFileSystem, {'diff_basis_max_bytes': bad})
        with pytest.raises(ValueError, match='diff_basis_max_bytes'):
            await fiber
        assert fiber.state is FiberState.FAILED


async def test_resolve_relative_absolute_and_cwd_override(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'x.txt').write_text('content')
    target = await fs.resolve('x.txt')
    assert target.display_path == str(tmp_path / 'x.txt')
    assert await fs.read_text(target) == 'content'

    # an absolute path ignores both config.cwd and the cwd override
    target = await fs.resolve(str(tmp_path / 'x.txt'), cwd='/nonexistent-base')
    assert await fs.read_text(target) == 'content'

    # a cwd override bases the relative path elsewhere
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'y.txt').write_text('in other')
    target = await fs.resolve('y.txt', cwd=str(other))
    assert await fs.read_text(target) == 'in other'
    with pytest.raises(FsError) as exc_info:
        await fs.read_text(await fs.resolve('y.txt'))
    assert exc_info.value.code == 'FS_NOT_FOUND'


async def test_resolve_rejects_empty_path(
    backend: tuple[Context, object, LocalFileSystem],
) -> None:
    _, _, fs = backend
    for bad in ('', '   '):
        with pytest.raises(FsError) as exc_info:
            await fs.resolve(bad)
        assert exc_info.value.code == 'FS_NOT_FOUND'


async def test_resolve_through_a_regular_file_parent(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'afile').write_text('not a dir')
    with pytest.raises(FsError) as exc_info:
        await fs.resolve('afile/child.txt')
    assert exc_info.value.code == 'FS_NOT_FOUND'
    assert 'not a directory' in str(exc_info.value)


async def test_resolve_missing_target_uses_realpath_ancestor(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    real_dir = tmp_path / 'real'
    real_dir.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real_dir, target_is_directory=True)
    # the not-yet-created file's key already resolves through the symlinked
    # ancestor, so identity is stable across creation
    target = await fs.resolve('link/new/deep.txt')
    assert target.target_key == str(
        Path(_realpath(real_dir)) / 'new' / 'deep.txt'
    )
    await fs.write_text(target, 'created')
    assert (real_dir / 'new' / 'deep.txt').read_text() == 'created'
    again = await fs.resolve('link/new/deep.txt')
    assert again.target_key == target.target_key


async def test_resolve_symlink_aliases_share_identity(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'file.txt').write_text('shared')
    (tmp_path / 'alias.txt').symlink_to(tmp_path / 'file.txt')
    direct = await fs.resolve('file.txt')
    via_link = await fs.resolve('alias.txt')
    assert direct.target_key == via_link.target_key
    # display paths stay un-resolved
    assert via_link.display_path == str(tmp_path / 'alias.txt')
    # writing through the alias updates the target, preserving the link
    await fs.write_text(via_link, 'updated')
    assert (tmp_path / 'file.txt').read_text() == 'updated'
    assert (tmp_path / 'alias.txt').is_symlink()


async def test_process_path_file_url_contains(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'file.txt').write_text('text')
    root = await fs.resolve('.')
    child = await fs.resolve('nested/file.txt')
    assert fs.process_path(child) == str(child.target_key)
    assert fs.file_url(child) == Path(fs.process_path(child)).as_uri()
    assert fs.contains(root, root)
    assert fs.contains(root, child)
    assert not fs.contains(child, root)
    sibling = await fs.resolve('nested')
    assert fs.contains(sibling, child)  # a directory contains its file


async def test_stat_and_lstat(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'f.txt').write_text('hello')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'l.txt').symlink_to(tmp_path / 'f.txt')

    target = await fs.resolve('f.txt')
    info = await fs.stat(target)
    assert info is not None
    assert info.type == 'file'
    assert info.size == 5
    assert info.version

    directory = await fs.stat(await fs.resolve('sub'))
    assert directory is not None and directory.type == 'directory'

    assert await fs.stat(await fs.resolve('missing.txt')) is None

    # lstat does not follow the final symlink
    link_info = await fs.lstat('l.txt')
    assert link_info is not None and link_info.type == 'symlink'
    file_info = await fs.lstat('f.txt')
    assert file_info is not None and file_info.type == 'file'
    assert await fs.lstat('missing.txt') is None
    with pytest.raises(FsError) as exc_info:
        await fs.lstat('  ')
    assert exc_info.value.code == 'FS_NOT_FOUND'


async def test_read_text(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'f.txt').write_text('héllo\nworld\n')
    target = await fs.resolve('f.txt')
    assert await fs.read_text(target) == 'héllo\nworld\n'

    with pytest.raises(FsError) as exc_info:
        await fs.read_text(await fs.resolve('missing.txt'))
    assert exc_info.value.code == 'FS_NOT_FOUND'

    (tmp_path / 'dir').mkdir()
    with pytest.raises(FsError) as exc_info:
        await fs.read_text(await fs.resolve('dir'))
    assert exc_info.value.code == 'FS_NOT_REGULAR_FILE'

    (tmp_path / 'bin').write_bytes(b'abc\x00def')
    with pytest.raises(FsError) as exc_info:
        await fs.read_text(await fs.resolve('bin'))
    assert exc_info.value.code == 'FS_NOT_TEXT'

    (tmp_path / 'bad.txt').write_bytes(b'\xff\xfe invalid')
    with pytest.raises(FsError) as exc_info:
        await fs.read_text(await fs.resolve('bad.txt'))
    assert exc_info.value.code == 'FS_NOT_TEXT'


async def test_stream_text(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    # larger than one 64 KiB chunk so streaming crosses chunk boundaries
    content = 'αβγ line\n' * 20000
    (tmp_path / 'big.txt').write_text(content)
    target = await fs.resolve('big.txt')
    chunks = [chunk async for chunk in fs.stream_text(target)]
    assert len(chunks) > 1
    assert ''.join(chunks) == content

    # the NUL sample covers only the first 8192 bytes — keep the NUL early
    (tmp_path / 'bin').write_bytes(b'ok\x00' + b'ok' * 5000)
    with pytest.raises(FsError) as exc_info:
        async for _ in fs.stream_text(await fs.resolve('bin')):
            pass
    assert exc_info.value.code == 'FS_NOT_TEXT'

    with pytest.raises(FsError) as exc_info:
        async for _ in fs.stream_text(await fs.resolve('missing.txt')):
            pass
    assert exc_info.value.code == 'FS_NOT_FOUND'


async def test_read_bytes(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'bin').write_bytes(b'\x00\x01\x02\xff')
    target = await fs.resolve('bin')
    # no decoding or binary rejection
    assert await fs.read_bytes(target, max_bytes=4) == b'\x00\x01\x02\xff'
    # stat short-circuits a known-oversized file
    with pytest.raises(FsError) as exc_info:
        await fs.read_bytes(target, max_bytes=3)
    assert exc_info.value.code == 'FS_TOO_LARGE'

    with pytest.raises(FsError) as exc_info:
        await fs.read_bytes(await fs.resolve('missing'), max_bytes=10)
    assert exc_info.value.code == 'FS_NOT_FOUND'


async def test_list_dir(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'b.txt').write_text('bb')
    (tmp_path / 'a.txt').write_text('a')
    (tmp_path / 'csub').mkdir()
    target = await fs.resolve('.')
    entries = await fs.list_dir(target)
    # stable name order, metadata and resolved child targets, no contents
    assert [entry.name for entry in entries] == ['a.txt', 'b.txt', 'csub']
    assert [entry.type for entry in entries] == ['file', 'file', 'directory']
    first = entries[0]
    assert first.size == 1
    assert first.version
    assert first.target.display_path == str(tmp_path / 'a.txt')
    assert await fs.read_text(first.target) == 'a'
    assert entries[2].size is None

    with pytest.raises(FsError) as exc_info:
        await fs.list_dir(await fs.resolve('missing'))
    assert exc_info.value.code == 'FS_NOT_FOUND'

    with pytest.raises(FsError) as exc_info:
        await fs.list_dir(await fs.resolve('a.txt'))
    assert exc_info.value.code == 'FS_NOT_DIRECTORY'


async def test_write_text_create_and_update(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    target = await fs.resolve('new.txt')
    outcome = await fs.write_text(target, 'first')
    assert outcome.operation == 'create'
    assert outcome.before is None
    assert outcome.after == 'first'
    assert outcome.version
    assert (tmp_path / 'new.txt').read_text() == 'first'
    # no staging residue survives a committed write
    assert _dir_names(tmp_path) == ['new.txt']

    outcome = await fs.write_text(target, 'second')
    assert outcome.operation == 'update'
    assert outcome.before == 'first'
    assert outcome.after == 'second'
    assert (tmp_path / 'new.txt').read_text() == 'second'


async def test_write_text_normalizes_crlf_in_outcome(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    target = await fs.resolve('crlf.txt')
    outcome = await fs.write_text(target, 'a\r\nb\r\n')
    # storage keeps the caller's bytes; the outcome's `after` is LF-normalized
    assert (tmp_path / 'crlf.txt').read_bytes() == b'a\r\nb\r\n'
    assert outcome.after == 'a\nb\n'


async def test_write_text_preserves_mode(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    path = tmp_path / 'mode.txt'
    path.write_text('old')
    path.chmod(0o640)
    target = await fs.resolve('mode.txt')
    await fs.write_text(target, 'new')
    assert stat_module.S_IMODE(path.stat().st_mode) == 0o640
    # a new file is created owner-only
    created = tmp_path / 'fresh.txt'
    await fs.write_text(await fs.resolve('fresh.txt'), 'x')
    assert stat_module.S_IMODE(created.stat().st_mode) == 0o600


async def test_write_text_create_if_absent(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    target = await fs.resolve('guard.txt')
    outcome = await fs.write_text(target, 'created', expected=FsCreateIfAbsent())
    assert outcome.operation == 'create'

    with pytest.raises(FsError) as exc_info:
        await fs.write_text(target, 'blind', expected=FsCreateIfAbsent())
    assert exc_info.value.code == 'FS_NOT_OBSERVED'
    # the existing file is preserved
    assert (tmp_path / 'guard.txt').read_text() == 'created'


async def test_write_text_replace_if_version(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    target = await fs.resolve('v.txt')
    # a missing target fails the stale guard
    with pytest.raises(FsError) as exc_info:
        await fs.write_text(
            target, 'x', expected=FsReplaceIfVersion(FsVersion('anything'))
        )
    assert exc_info.value.code == 'FS_STALE_VERSION'

    await fs.write_text(target, 'one')
    version = await _version_of(fs, target)
    outcome = await fs.write_text(
        target, 'two', expected=FsReplaceIfVersion(version)
    )
    assert outcome.operation == 'update'
    assert outcome.before == 'one'

    with pytest.raises(FsError) as exc_info:
        await fs.write_text(
            target, 'three', expected=FsReplaceIfVersion(version)
        )
    assert exc_info.value.code == 'FS_STALE_VERSION'
    assert (tmp_path / 'v.txt').read_text() == 'two'


async def test_write_text_rejects_non_regular_target(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'dir').mkdir()
    with pytest.raises(FsError) as exc_info:
        await fs.write_text(await fs.resolve('dir'), 'x')
    assert exc_info.value.code == 'FS_NOT_REGULAR_FILE'


async def test_write_text_creates_missing_parents(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    target = await fs.resolve('deep/nested/file.txt')
    outcome = await fs.write_text(target, 'nested')
    assert outcome.operation == 'create'
    assert (tmp_path / 'deep' / 'nested' / 'file.txt').read_text() == 'nested'


async def test_edit_text_basic(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'e.txt').write_text('hello world')
    target = await fs.resolve('e.txt')
    version = await _version_of(fs, target)
    outcome = await fs.edit_text(
        target, FsEditRequest(old_string='world', new_string='there', replace_all=False)
    )
    assert outcome.before == 'hello world'
    assert outcome.after == 'hello there'
    assert outcome.version != version
    assert (tmp_path / 'e.txt').read_text() == 'hello there'


async def test_edit_text_version_guard(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'e.txt').write_text('aaa bbb')
    target = await fs.resolve('e.txt')
    version = await _version_of(fs, target)
    outcome = await fs.edit_text(
        target,
        FsEditRequest(old_string='aaa', new_string='ccc', replace_all=False),
        expected=version,
    )
    assert outcome.after == 'ccc bbb'

    # stale guard is checked BEFORE literal matching: old_string still matches
    # the current content, yet the stale version reports FS_STALE_VERSION
    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            target,
            FsEditRequest(old_string='ccc', new_string='ddd', replace_all=False),
            expected=version,
        )
    assert exc_info.value.code == 'FS_STALE_VERSION'

    # a missing target reports FS_STALE_VERSION either way
    missing = await fs.resolve('missing.txt')
    for expected in (None, FsVersion('v')):
        with pytest.raises(FsError) as exc_info:
            await fs.edit_text(
                missing,
                FsEditRequest(old_string='x', new_string='y', replace_all=False),
                expected=expected,
            )
        assert exc_info.value.code == 'FS_STALE_VERSION'


async def test_edit_text_match_errors(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'e.txt').write_text('dup dup dup')
    target = await fs.resolve('e.txt')

    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            target, FsEditRequest(old_string='', new_string='x', replace_all=False)
        )
    assert exc_info.value.code == 'FS_EDIT_NOT_FOUND'

    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            target,
            FsEditRequest(old_string='absent', new_string='x', replace_all=False),
        )
    assert exc_info.value.code == 'FS_EDIT_NOT_FOUND'

    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            target, FsEditRequest(old_string='dup', new_string='x', replace_all=False)
        )
    assert exc_info.value.code == 'FS_AMBIGUOUS_EDIT'

    outcome = await fs.edit_text(
        target, FsEditRequest(old_string='dup', new_string='x', replace_all=True)
    )
    assert outcome.after == 'x x x'


async def test_edit_text_preserves_crlf(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'win.txt').write_bytes(b'one\r\ntwo\r\nthree\r\n')
    target = await fs.resolve('win.txt')
    outcome = await fs.edit_text(
        target, FsEditRequest(old_string='two', new_string='TWO', replace_all=False)
    )
    # the diff basis is LF-normalized
    assert outcome.before == 'one\ntwo\nthree\n'
    assert outcome.after == 'one\nTWO\nthree\n'
    # storage restores the dominant CRLF style
    assert (tmp_path / 'win.txt').read_bytes() == b'one\r\nTWO\r\nthree\r\n'


async def test_edit_text_rejects_binary_and_non_regular(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'bin').write_bytes(b'ab\x00cd')
    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            await fs.resolve('bin'),
            FsEditRequest(old_string='ab', new_string='x', replace_all=False),
        )
    assert exc_info.value.code == 'FS_NOT_TEXT'

    (tmp_path / 'dir').mkdir()
    with pytest.raises(FsError) as exc_info:
        await fs.edit_text(
            await fs.resolve('dir'),
            FsEditRequest(old_string='a', new_string='b', replace_all=False),
        )
    assert exc_info.value.code == 'FS_NOT_REGULAR_FILE'


async def test_concurrent_guarded_edits_one_wins(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'race.txt').write_text('start')
    target = await fs.resolve('race.txt')
    version = await _version_of(fs, target)

    async def edit(old: str, new: str) -> FsEditOutcome:
        return await fs.edit_text(
            target,
            FsEditRequest(old_string=old, new_string=new, replace_all=False),
            expected=version,
        )

    results = await asyncio.gather(
        edit('start', 'first'), edit('start', 'second'), return_exceptions=True
    )
    wins = [r for r in results if isinstance(r, FsEditOutcome)]
    stale = [
        r
        for r in results
        if isinstance(r, FsError) and r.code == 'FS_STALE_VERSION'
    ]
    # the per-target lock serializes the read→guard→write window: one wins,
    # the other sees the new version and rejects as stale
    assert len(wins) == 1
    assert len(stale) == 1
    assert (tmp_path / 'race.txt').read_text() in ('first', 'second')
    # locks are cleaned up after the mutation
    assert fs._locks == {}


async def test_abort_signal(
    backend: tuple[Context, object, LocalFileSystem], tmp_path: Path
) -> None:
    _, _, fs = backend
    (tmp_path / 'f.txt').write_text('x')
    aborted = asyncio.Event()
    aborted.set()

    with pytest.raises(FsError) as exc_info:
        await fs.resolve('f.txt', signal=aborted)
    assert exc_info.value.code == 'FS_ABORTED'

    target = await fs.resolve('f.txt')
    with pytest.raises(FsError) as exc_info:
        await fs.stat(target, signal=aborted)
    assert exc_info.value.code == 'FS_ABORTED'

    with pytest.raises(FsError) as exc_info:
        await fs.read_text(target, signal=aborted)
    assert exc_info.value.code == 'FS_ABORTED'

    with pytest.raises(FsError) as exc_info:
        await fs.write_text(target, 'y', signal=aborted)
    assert exc_info.value.code == 'FS_ABORTED'

    # an aborted write never publishes
    assert (tmp_path / 'f.txt').read_text() == 'x'


async def test_local_backend_hmr_roundtrip(tmp_path: Path) -> None:
    ctx = Context()
    fiber = ctx.plugin(LocalFileSystem, {'cwd': str(tmp_path)})
    await fiber
    assert ctx.get('fs') is not None
    await fiber.dispose()
    # unloading the backend revokes the service
    assert ctx.get('fs') is None
    fiber = ctx.plugin(LocalFileSystem, {'cwd': str(tmp_path)})
    await fiber
    assert ctx.get('fs') is not None
    await fiber.dispose()
