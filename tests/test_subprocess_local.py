"""Behavior tests for the local subprocess backend, using real child
processes (/bin/echo, /bin/cat, /bin/sh, sleep, false).

Ported-semantics tests for the deepseek-harness subprocess port (MIT);
source: packages/subprocess/subprocess-local/tests/spawn.spec.ts and
local.spec.ts.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from pycordis import Context
from pydsh.subprocess import (
    MAX_TIMER_DELAY_MS,
    AbortController,
    AbortError,
    SpillLimit,
    StdinData,
    SubprocessCollect,
    SubprocessHandle,
    SubprocessOutcome,
    SubprocessOutputRead,
    SubprocessSpawnSpec,
    SubprocessStdinMode,
    SubprocessStdio,
    SubprocessTerminalSpawnSpec,
)
from pydsh.subprocess.local import LocalSubprocessRuntime

GRACE_MS = 3_000
BIG_SPILL = 64 * 1024 * 1024


def spec(
    argv: list[str],
    *,
    cwd: str | None = None,
    stdin: SubprocessStdinMode = 'ignore',
    stdout_max: int = 64_000,
    stderr_max: int = 64_000,
    spill: int | None = BIG_SPILL,
    grace_ms: float = GRACE_MS,
    signal: Any = None,
    env: dict[str, str | None] | None = None,
) -> SubprocessSpawnSpec:
    """Build a fully-specified collect-mode spawn spec, like the TS test
    helper."""
    return SubprocessSpawnSpec(
        argv=argv,
        cwd=cwd if cwd is not None else os.getcwd(),
        stdio=SubprocessStdio(
            stdin=stdin,
            stdout=SubprocessCollect(
                stdout_max, SpillLimit(spill) if spill is not None else None
            ),
            stderr=SubprocessCollect(
                stderr_max, SpillLimit(spill) if spill is not None else None
            ),
        ),
        grace_ms=grace_ms,
        signal=signal,
        env=env,
    )


def sh(script: str, **kwargs: Any) -> SubprocessSpawnSpec:
    return spec(['/bin/sh', '-c', script], **kwargs)


async def finish(
    handle: SubprocessHandle,
) -> tuple[SubprocessOutcome, SubprocessOutputRead, SubprocessOutputRead]:
    """Await settlement and project both collected streams like a batch
    outcome."""
    outcome = await handle.done
    assert handle.collected.stdout is not None
    assert handle.collected.stderr is not None
    return (
        outcome,
        handle.collected.stdout.read_from(0),
        handle.collected.stderr.read_from(0),
    )


def _proc_state(pid: int) -> str | None:
    """The state letter from ``/proc/<pid>/stat``; None when the entry is
    unreadable or gone."""
    try:
        with open(f'/proc/{pid}/stat') as file:
            text = file.read()
    except OSError:
        return None
    return text[text.rfind(')') + 2:text.rfind(')') + 3]


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None  # the child shell has not written the pid file yet
    return pid if pid > 0 else None


def _read_text(path: str) -> str:
    with open(path) as file:
        return file.read()


async def wait_gone(pid: int, limit: float = 5.0) -> None:
    """Poll until a pid no longer exists, or is only a zombie on Linux."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        state = _proc_state(pid)
        if state is None or state in ('Z', 'X'):
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'pid {pid} still alive after {limit}s')


async def wait_for_stdout(
    handle: SubprocessHandle, expected: str, limit: float = 5.0
) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        assert handle.collected.stdout is not None
        if expected in handle.collected.stdout.read_from(0).text:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'stdout did not include {expected!r} after {limit}s')


async def wait_for_pid_file(path: Path, limit: float = 5.0) -> int:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        pid = _read_pid(path)
        if pid is not None:
            return pid
        await asyncio.sleep(0.02)
    raise AssertionError(f'pid file {path} was not written after {limit}s')


@pytest.fixture
async def svc() -> Any:
    ctx = Context()
    fiber = ctx.plugin(LocalSubprocessRuntime)
    await fiber
    yield ctx.get('subprocess')
    await fiber.dispose()


class TestSpawnBasics:
    async def test_captures_stdout_on_success(self, svc: Any) -> None:
        handle = await svc.spawn(spec(['/bin/echo', 'hello']))
        outcome, stdout, stderr = await finish(handle)
        assert outcome.exit_code == 0
        assert outcome.signal is None
        assert stdout.text == 'hello\n'
        assert not stdout.lossy
        assert stderr.text == ''

    async def test_captures_stderr_separately(self, svc: Any) -> None:
        outcome, stdout, stderr = await finish(await svc.spawn(sh('echo oops >&2')))
        assert outcome.exit_code == 0
        assert stdout.text == ''
        assert stderr.text == 'oops\n'

    async def test_captures_both_streams(self, svc: Any) -> None:
        _, stdout, stderr = await finish(await svc.spawn(sh('echo out; echo err >&2')))
        assert stdout.text == 'out\n'
        assert stderr.text == 'err\n'

    async def test_reports_nonzero_exit_codes(self, svc: Any) -> None:
        # A non-zero exit is NOT an error: it settles done with exit facts.
        outcome, _, _ = await finish(await svc.spawn(spec(['false'])))
        assert outcome.exit_code == 1
        assert outcome.signal is None
        outcome, _, _ = await finish(await svc.spawn(sh('exit 42')))
        assert outcome.exit_code == 42

    async def test_runs_in_requested_cwd(self, svc: Any) -> None:
        _, stdout, _ = await finish(await svc.spawn(sh('pwd', cwd='/tmp')))
        assert stdout.text.strip().endswith('/tmp')

    async def test_spawn_level_failure_raises(self, svc: Any) -> None:
        # Infrastructure failure (bad cwd) is distinct from a non-zero exit:
        # it raises from spawn itself, before any handle exists.
        with pytest.raises(FileNotFoundError):
            await svc.spawn(spec(['/bin/echo', 'hi'], cwd='/nonexistent-dir-dsh-test'))

    @pytest.mark.parametrize(
        'grace_ms', [0, -1, float('nan'), float('inf'), MAX_TIMER_DELAY_MS + 1]
    )
    async def test_rejects_invalid_grace(self, svc: Any, grace_ms: float) -> None:
        with pytest.raises(ValueError, match='grace_ms'):
            await svc.spawn(spec(['true'], grace_ms=grace_ms))

    async def test_rejects_invalid_argv(self, svc: Any) -> None:
        with pytest.raises(ValueError, match='non-empty program name'):
            await svc.spawn(spec([]))
        with pytest.raises(ValueError, match='non-empty program name'):
            await svc.spawn(spec(['']))

    async def test_argv_is_never_shell_interpreted(self, svc: Any) -> None:
        _, stdout, _ = await finish(await svc.spawn(spec(['printf', '%s', '$HOME'])))
        assert stdout.text == '$HOME'

    async def test_reports_signal_of_self_killed_command(self, svc: Any) -> None:
        outcome, _, _ = await finish(await svc.spawn(sh('kill -TERM $$')))
        assert outcome.signal == 'SIGTERM'
        assert outcome.exit_code is None


class TestStdinAndEnv:
    async def test_writes_batch_stdin_and_closes(self, svc: Any) -> None:
        outcome, stdout, _ = await finish(
            await svc.spawn(spec(['/bin/cat'], stdin=StdinData('hello from stdin\n')))
        )
        assert outcome.exit_code == 0
        assert stdout.text == 'hello from stdin\n'

    async def test_ignore_stdin_reads_eof(self, svc: Any) -> None:
        # No stdin → fd 0 is /dev/null, so cat reads EOF and exits 0 with no
        # output (it does NOT block).
        outcome, stdout, _ = await finish(await svc.spawn(spec(['/bin/cat'])))
        assert outcome.exit_code == 0
        assert stdout.text == ''

    async def test_batch_stdin_handle_exposes_no_stdin(self, svc: Any) -> None:
        handle = await svc.spawn(spec(['/bin/cat'], stdin=StdinData('batch\n')))
        assert handle.stdin is None
        await handle.done
        assert handle.collected.stdout is not None
        assert handle.collected.stdout.read_from(0).text == 'batch\n'

    async def test_merges_ordinary_env_entries(self, svc: Any) -> None:
        env = {'EXTRA_ONE': 'alpha', 'EXTRA_TWO': 'beta'}
        _, stdout, _ = await finish(
            await svc.spawn(sh('echo "$EXTRA_ONE/$EXTRA_TWO"', env=env))
        )
        assert stdout.text == 'alpha/beta\n'

    async def test_tombstone_removes_ambient_env(
        self, svc: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('SUBPROCESS_TOMBSTONE_PROBE', 'ambient-value')
        _, stdout, _ = await finish(
            await svc.spawn(
                sh(
                    'echo "${SUBPROCESS_TOMBSTONE_PROBE:-absent}"',
                    env={'SUBPROCESS_TOMBSTONE_PROBE': None},
                )
            )
        )
        assert stdout.text == 'absent\n'

    async def test_explicit_env_overrides_credential_scrub(self, svc: Any) -> None:
        _, stdout, _ = await finish(
            await svc.spawn(
                sh(
                    'echo "$EXPLICIT_OVERRIDE_PASSWORD"',
                    env={'EXPLICIT_OVERRIDE_PASSWORD': 'explicit-wins'},
                )
            )
        )
        assert stdout.text == 'explicit-wins\n'

    async def test_scrubs_ambient_credentials_and_dsh_vars(
        self, svc: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('DSH_TEST_TOKEN', 'also-secret')
        monkeypatch.setenv('SUBPROCESS_TEST_PASSWORD', 'password-secret')
        script = (
            'echo "[${DSH_TEST_TOKEN:-absent}|${SUBPROCESS_TEST_PASSWORD:-absent}]"'
        )
        _, stdout, _ = await finish(await svc.spawn(sh(script)))
        assert stdout.text.strip() == '[absent|absent]'


class TestTermination:
    async def test_timeout_via_abort_signal(self, svc: Any) -> None:
        # The seam owns no deadline: the caller aborts and classifies; the
        # abort starts the SIGTERM escalation.
        controller = AbortController()
        start = time.monotonic()
        handle = await svc.spawn(spec(['sleep', '60'], signal=controller.signal))
        asyncio.get_running_loop().call_later(0.1, controller.abort, 'deadline')
        outcome = await handle.done
        assert time.monotonic() - start < 5
        assert outcome.signal == 'SIGTERM'
        assert outcome.exit_code is None

    async def test_aborted_before_spawn_raises(self, svc: Any) -> None:
        controller = AbortController()
        controller.abort('too late')
        with pytest.raises(AbortError, match='aborted before spawn: too late'):
            await svc.spawn(spec(['/bin/echo', 'hi'], signal=controller.signal))

    async def test_terminate_escalates_when_term_trapped(self, svc: Any) -> None:
        script = "trap '' TERM; echo ready; while :; do sleep 60 & wait $!; done"
        handle = await svc.spawn(sh(script, grace_ms=200))
        await wait_for_stdout(handle, 'ready\n')
        handle.terminate()
        outcome = await handle.done
        assert outcome.signal == 'SIGKILL'

    async def test_terminate_is_idempotent(self, svc: Any) -> None:
        handle = await svc.spawn(spec(['sleep', '60']))
        handle.terminate()
        handle.terminate()
        outcome = await handle.done
        assert outcome.signal == 'SIGTERM'

    async def test_terminate_kills_whole_group(self, svc: Any, tmp_path: Path) -> None:
        # The subshell writes the sleep's pid then waits on it; terminating
        # the group must take the sleep down with the shell.
        pid_file = tmp_path / 'grandchild.pid'
        handle = await svc.spawn(sh(f'sleep 60 & echo $! > {pid_file}; wait'))
        grandchild = await wait_for_pid_file(pid_file)

        handle.terminate()
        outcome = await handle.done
        assert outcome.signal == 'SIGTERM'
        await wait_gone(grandchild)

    async def test_wait_for_exit_covers_whole_tree(
        self, svc: Any, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / 'tree-wait.pid'
        handle = await svc.spawn(sh(f'sleep 60 & echo $! > {pid_file}; wait'))
        grandchild = await wait_for_pid_file(pid_file)
        handle.terminate()
        await handle.done
        assert await handle.wait_for_exit()
        await wait_gone(grandchild, 0.1)

    async def test_bounded_wait_for_exit_reports_false(self, svc: Any) -> None:
        handle = await svc.spawn(spec(['sleep', '60']))
        controller = AbortController()
        controller.abort()
        assert await handle.wait_for_exit(controller.signal) is False
        handle.terminate()
        await handle.done

    async def test_terminate_after_exit_is_noop(self, svc: Any) -> None:
        handle = await svc.spawn(spec(['true']))
        await handle.done
        handle.terminate()
        assert await handle.wait_for_exit()

    async def test_fiber_unload_reclaims_processes(self) -> None:
        ctx = Context()
        fiber = ctx.plugin(LocalSubprocessRuntime)
        await fiber
        service = ctx.get('subprocess')
        handle = await service.spawn(spec(['sleep', '60']))
        pid = handle.pid
        await fiber.dispose()
        # Teardown itself terminated and joined the tree.
        await wait_gone(pid, 0.1)

    async def test_teardown_awaits_tree_survivors(self, tmp_path: Path) -> None:
        ctx = Context()
        fiber = ctx.plugin(LocalSubprocessRuntime)
        await fiber
        service = ctx.get('subprocess')
        pid_file = tmp_path / 'survivor.pid'
        handle = await service.spawn(
            sh(f'sleep 60 & echo $! > {pid_file}; echo leader-done', grace_ms=200)
        )
        grandchild = await wait_for_pid_file(pid_file)
        await handle.done
        await fiber.dispose()
        # Disposal escalated to the surviving group member and awaited it.
        await wait_gone(grandchild, 0.1)


class TestStdioModes:
    async def test_pipe_mode_exposes_raw_streams(self, svc: Any) -> None:
        handle = await svc.spawn(
            SubprocessSpawnSpec(
                argv=['/bin/cat'],
                cwd=os.getcwd(),
                stdio=SubprocessStdio(
                    stdin='pipe',
                    stdout='pipe',
                    stderr=SubprocessCollect(1000),
                ),
                grace_ms=GRACE_MS,
            )
        )
        assert handle.stdin is not None
        assert handle.stdout is not None
        assert handle.stderr is None
        assert handle.collected.stdout is None
        assert handle.collected.stderr is not None

        handle.stdin.write(b'through the pipe\n')
        await handle.stdin.drain()
        echoed = await asyncio.wait_for(handle.stdout.readline(), 5)
        assert echoed == b'through the pipe\n'
        handle.stdin.close()
        outcome = await handle.done
        assert outcome.exit_code == 0

    async def test_inherit_stdout_wires_only_requested_collector(
        self, svc: Any
    ) -> None:
        handle = await svc.spawn(
            SubprocessSpawnSpec(
                argv=['/bin/sh', '-c', 'echo to-parent; echo err >&2'],
                cwd=os.getcwd(),
                stdio=SubprocessStdio(
                    stdin='ignore',
                    stdout='inherit',
                    stderr=SubprocessCollect(1000),
                ),
                grace_ms=GRACE_MS,
            )
        )
        outcome = await handle.done
        assert outcome.exit_code == 0
        assert handle.stdout is None
        assert handle.collected.stdout is None
        assert handle.collected.stderr is not None
        assert handle.collected.stderr.read_from(0).text == 'err\n'

    async def test_collect_without_spill_keeps_memory_tail_only(self, svc: Any) -> None:
        handle = await svc.spawn(
            sh(
                'for i in $(seq 1 200); do printf "line-%04d\\n" $i; done',
                stdout_max=100,
                stderr_max=100,
                spill=None,
            )
        )
        await handle.done
        assert handle.collected.stdout is not None
        read = handle.collected.stdout.read_from(0)
        assert read.lossy
        assert 'line-0200' in read.text
        assert read.spill_path is None

    async def test_inherited_pipe_drain_is_bounded(
        self, svc: Any, tmp_path: Path
    ) -> None:
        # A surviving descendant holding the collected pipe must not hold the
        # outcome open: the same grace that governs kills bounds the drain.
        pid_file = tmp_path / 'pipe-holder.pid'
        start = time.monotonic()
        handle = await svc.spawn(
            sh(f'sleep 60 & echo $! > {pid_file}; echo shell-done', grace_ms=100)
        )
        grandchild = await wait_for_pid_file(pid_file)
        try:
            outcome, stdout, _ = await finish(handle)
            assert time.monotonic() - start < 2
            assert outcome.exit_code == 0
            assert 'shell-done\n' in stdout.text
        finally:
            os.kill(grandchild, 9)
            await wait_gone(grandchild)


class TestTruncationAndSpill:
    async def test_independent_stream_caps(self, svc: Any) -> None:
        _, stdout, stderr = await finish(
            await svc.spawn(
                sh(
                    'printf "%.0sx" $(seq 1 500); printf "%.0se" $(seq 1 500) >&2',
                    stdout_max=500,
                    stderr_max=100,
                )
            )
        )
        assert not stdout.lossy
        assert stdout.text == 'x' * 500
        assert stderr.lossy
        assert len(stderr.text) <= 100

    async def test_keeps_tail_and_spills_full_stream(
        self, svc: Any, tmp_path: Path
    ) -> None:
        svc.spill_dir = str(tmp_path)
        _, stdout, _ = await finish(
            await svc.spawn(
                sh(
                    'for i in $(seq 1 200); do printf "line-%04d\\n" $i; done',
                    stdout_max=500,
                )
            )
        )
        assert stdout.lossy
        assert len(stdout.text) <= 500
        assert 'line-0200' in stdout.text
        assert 'line-0001' not in stdout.text
        assert stdout.spill_path is not None
        full = _read_text(stdout.spill_path)
        assert 'line-0001' in full
        assert 'line-0200' in full

    async def test_exact_cap_is_not_truncated(self, svc: Any) -> None:
        _, stdout, _ = await finish(
            await svc.spawn(sh('printf "%.0sx" $(seq 1 500)', stdout_max=500))
        )
        assert not stdout.lossy
        assert len(stdout.text) == 500
        assert stdout.spill_path is None

    async def test_spill_files_land_in_private_dir(self, svc: Any) -> None:
        script = 'for i in $(seq 1 200); do printf "line-%04d\\n" $i; done'
        _, stdout, _ = await finish(await svc.spawn(sh(script, stdout_max=500)))
        assert stdout.spill_path is not None
        assert 'dsh-subprocess-' in os.path.dirname(stdout.spill_path)


class TestResolveExecutable:
    async def test_absolute_path(self, svc: Any) -> None:
        assert await svc.resolve_executable('/bin/echo') == '/bin/echo'

    async def test_bare_path_name(self, svc: Any) -> None:
        resolved = await svc.resolve_executable('echo')
        assert os.path.isabs(resolved)
        assert os.path.basename(resolved) == 'echo'

    async def test_rejects_relative_path(self, svc: Any) -> None:
        with pytest.raises(ValueError, match='relative path'):
            await svc.resolve_executable('bin/echo')

    async def test_missing_command_raises(self, svc: Any) -> None:
        with pytest.raises(FileNotFoundError, match='not found on PATH'):
            await svc.resolve_executable('dsh-no-such-command')
        with pytest.raises(FileNotFoundError, match='not an executable file'):
            await svc.resolve_executable('/nonexistent-dir-dsh-test/echo')
        with pytest.raises(ValueError, match='non-empty'):
            await svc.resolve_executable('')

    async def test_aborted_signal_raises(self, svc: Any) -> None:
        controller = AbortController()
        controller.abort('cancelled')
        with pytest.raises(AbortError, match='cancelled'):
            await svc.resolve_executable('echo', signal=controller.signal)


class TestTerminal:
    async def test_pipe_terminal_roundtrip(self, svc: Any) -> None:
        handle = await svc.spawn_terminal(
            SubprocessTerminalSpawnSpec(
                argv=['/bin/cat'], cwd=os.getcwd(), rows=24, cols=80, grace_ms=1000
            )
        )
        await handle.write('hello terminal\n')
        line = await asyncio.wait_for(handle.output.readline(), 5)
        assert line == b'hello terminal\n'
        await handle.terminate()
        outcome = await handle.done
        assert outcome.signal == 'SIGTERM'
        with pytest.raises(RuntimeError, match='has exited'):
            await handle.write('too late\n')

    async def test_foreground_ops_need_a_real_pty(self, svc: Any) -> None:
        handle = await svc.spawn_terminal(
            SubprocessTerminalSpawnSpec(
                argv=['/bin/cat'], cwd=os.getcwd(), rows=24, cols=80, grace_ms=1000
            )
        )
        try:
            # Simplified port: pipes have no controlling tty, so there is no
            # resolvable foreground process group.
            assert await handle.inspect_foreground() is None
            with pytest.raises(RuntimeError, match='controlling terminal'):
                await handle.signal_foreground('SIGINT')
        finally:
            await handle.terminate()

    async def test_terminal_rejects_bad_argv(self, svc: Any) -> None:
        with pytest.raises(ValueError, match='argv'):
            await svc.spawn_terminal(
                SubprocessTerminalSpawnSpec(
                    argv=[], cwd=os.getcwd(), rows=24, cols=80, grace_ms=1000
                )
            )
