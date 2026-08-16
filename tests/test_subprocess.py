"""Definition-level tests for the subprocess seam: environment scrubbing, the
abort-signal port, bounded-output collection, and cordis inject gating.

Behavior contract tests for the deepseek-harness subprocess port (MIT);
ported from deepseek-harness packages/subprocess (subprocess and
subprocess-local spawn.spec.ts collector cases).
"""

from __future__ import annotations

import os
import signal
import stat
from pathlib import Path

import pytest

from pycordis import Context, FiberState
from pydsh.subprocess import (
    AbortController,
    AbortError,
    SubprocessRuntime,
    scrubbed_parent_env,
)
from pydsh.subprocess.local import LocalSubprocessRuntime
from pydsh.subprocess.local.process_inspector import parse_proc_stat
from pydsh.subprocess.local.spawn import OutputCollector, child_env, kill_group


def test_scrubbed_parent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DSH_TEST_API_KEY', 'super-secret')
    monkeypatch.setenv('SUBPROCESS_TEST_PASSWORD', 'password-secret')
    monkeypatch.setenv('dsh_lowercase_fact', 'visible')
    monkeypatch.setenv('ORDINARY_SETTING', 'kept')
    env = scrubbed_parent_env()
    assert 'DSH_TEST_API_KEY' not in env
    assert 'SUBPROCESS_TEST_PASSWORD' not in env
    # the DSH_ scrub matches case-insensitively, like Windows env names
    assert 'dsh_lowercase_fact' not in env
    assert env['ORDINARY_SETTING'] == 'kept'
    assert env.get('PATH')


def test_child_env_merges_after_the_scrub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AMBIENT_TOMBSTONE_PROBE', 'ambient-value')
    env = child_env({'EXTRA_ONE': 'alpha', 'AMBIENT_TOMBSTONE_PROBE': None})
    assert env['EXTRA_ONE'] == 'alpha'
    # a None tombstone removes an ordinary ambient entry
    assert 'AMBIENT_TOMBSTONE_PROBE' not in env


def test_child_env_explicit_entry_survives_credential_scrub() -> None:
    # A credential-shaped name deliberately supplied through the spec is an
    # opt-in: the scrub only drops AMBIENT entries.
    env = child_env({'EXPLICIT_OVERRIDE_PASSWORD': 'explicit-wins'})
    assert env['EXPLICIT_OVERRIDE_PASSWORD'] == 'explicit-wins'


def test_abort_signal_and_controller() -> None:
    controller = AbortController()
    fired: list[None] = []
    controller.signal.add_listener(lambda: fired.append(None))
    assert not controller.signal.aborted
    controller.abort('deadline')
    assert controller.signal.aborted
    assert controller.signal.reason == 'deadline'
    assert len(fired) == 1
    # abort is single-shot; late listeners fire immediately
    controller.abort('again')
    late: list[None] = []
    controller.signal.add_listener(lambda: late.append(None))
    assert len(fired) == 1
    assert len(late) == 1
    with pytest.raises(AbortError, match='deadline'):
        controller.signal.throw_if_aborted()


def test_abort_signal_remove_listener() -> None:
    controller = AbortController()
    fired: list[None] = []

    def listener() -> None:
        fired.append(None)

    controller.signal.add_listener(listener)
    controller.signal.remove_listener(listener)
    controller.signal.remove_listener(listener)  # unknown listeners no-op
    controller.abort()
    assert fired == []


def test_throw_if_aborted_default_reason() -> None:
    controller = AbortController()
    controller.abort()
    with pytest.raises(AbortError, match=r'^aborted$'):
        controller.signal.throw_if_aborted()


def test_parse_proc_stat() -> None:
    with open('/proc/self/stat') as file:
        parsed = parse_proc_stat(file.read())
    assert parsed is not None
    assert parsed.pid == os.getpid()
    assert parse_proc_stat('garbage without parens') is None
    assert parse_proc_stat('1 (comm) X') is None


def test_kill_group_never_raises() -> None:
    kill_group(-1, signal.SIGTERM)
    kill_group(0, signal.SIGTERM)
    # a vanished group reports ESRCH and is contained
    kill_group(2**22, signal.SIGTERM)


class TestOutputCollector:
    def test_keeps_tail_of_single_oversized_chunk(self, tmp_path: Path) -> None:
        collector = OutputCollector(10, 100, 'test', str(tmp_path))
        collector.push(b'0123456789abcdef')
        out = collector.finalize()
        assert out.text == '6789abcdef'
        assert out.truncated
        assert out.spill_path is not None
        with open(out.spill_path) as file:
            assert file.read() == '0123456789abcdef'

    def test_byte_exact_tail_across_uneven_chunks(self, tmp_path: Path) -> None:
        # A diagnostic tail must be exactly the LAST max_bytes regardless of
        # chunking; dropping only whole chunks would under-retain.
        collector = OutputCollector(10, None, 'exact-tail', str(tmp_path))
        collector.push(b'aaaa')
        collector.push(b'bbbbbb')
        collector.push(b'cc')
        out = collector.finalize()
        assert out.text == 'aabbbbbbcc'
        assert len(out.text.encode()) == 10
        assert out.truncated
        assert out.spill_path is None

    def test_read_from_increments_and_lossy(self, tmp_path: Path) -> None:
        collector = OutputCollector(10, 100, 'test', str(tmp_path))
        collector.push(b'aaaaa')
        first = collector.read_from(0)
        assert first.text == 'aaaaa'
        assert not first.lossy
        assert first.next_offset == 5

        collector.push(b'bbbbb')
        second = collector.read_from(first.next_offset)
        assert second.text == 'bbbbb'
        assert not second.lossy

        # Push enough to slide the window past the last offset.
        collector.push(b'c' * 20)
        third = collector.read_from(second.next_offset)
        assert third.lossy
        assert third.text == 'c' * 10
        assert third.spill_path is not None

    def test_spill_cap_discards_incomplete_spill(self, tmp_path: Path) -> None:
        collector = OutputCollector(4, 8, 'bounded', str(tmp_path))
        collector.push(b'aaaa')
        collector.push(b'bbbb')
        spill_path = collector.read_from(0).spill_path
        assert spill_path is not None
        with open(spill_path) as file:
            assert file.read() == 'aaaabbbb'

        collector.push(b'c')
        collector.push(b'dddd')
        out = collector.finalize()
        assert out.text == 'dddd'
        assert out.truncated
        assert out.spill_path is None
        assert not os.path.exists(spill_path)

    def test_no_spill_when_first_overflow_exceeds_cap(self, tmp_path: Path) -> None:
        collector = OutputCollector(4, 4, 'no-spill', str(tmp_path))
        collector.push(b'abcdefgh')
        out = collector.finalize()
        assert out.text == 'efgh'
        assert out.truncated
        assert out.spill_path is None

    def test_spill_file_permissions(self, tmp_path: Path) -> None:
        collector = OutputCollector(4, 100, 'perms', str(tmp_path))
        collector.push(b'aaaa')
        collector.push(b'bbbb')
        spill_path = collector.finalize().spill_path
        assert spill_path is not None
        mode = stat.S_IMODE(os.stat(spill_path).st_mode)
        assert mode == 0o600


async def test_inject_gating_until_backend_loaded() -> None:
    ctx = Context()
    seen: list[SubprocessRuntime] = []
    fiber = ctx.inject(['subprocess'], lambda c, config=None: seen.append(c.subprocess))
    # the unloaded backend keeps the dependent fiber PENDING
    assert fiber.state is FiberState.PENDING
    assert seen == []

    provider = ctx.plugin(LocalSubprocessRuntime)
    await provider
    await fiber
    assert fiber.state is FiberState.ACTIVE
    assert len(seen) == 1

    await fiber.dispose()
    await provider.dispose()


async def test_access_without_inject_fails() -> None:
    ctx = Context()
    provider = ctx.plugin(LocalSubprocessRuntime)
    await provider

    def peek(c: Context, config: object = None) -> None:
        _ = c.subprocess  # not declared in inject

    failed = ctx.plugin(peek)
    with pytest.raises(AttributeError, match='without inject'):
        await failed
    assert failed.state is FiberState.FAILED
    await provider.dispose()


async def test_root_access_without_backend_is_none() -> None:
    ctx = Context()
    assert ctx.get('subprocess') is None


async def test_duplicate_provider_rejected() -> None:
    ctx = Context()
    first = ctx.plugin(LocalSubprocessRuntime)
    await first
    second = ctx.plugin(LocalSubprocessRuntime)
    with pytest.raises(RuntimeError, match='has been registered'):
        await second
    # the first registration survives the failed second provide
    service = ctx.get('subprocess')
    assert service is not None and service.name == 'subprocess'
    await first.dispose()


async def test_service_provides_itself_as_subprocess() -> None:
    ctx = Context()
    fiber = ctx.plugin(LocalSubprocessRuntime)
    await fiber
    # ctx.get returns a ServiceView rebind of the tracked service, so assert
    # through the proxied attributes instead of isinstance.
    service = ctx.get('subprocess')
    assert service is not None
    assert service.name == 'subprocess'
    assert isinstance(object.__getattribute__(service, '_target'), SubprocessRuntime)
    await fiber.dispose()
