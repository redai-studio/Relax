# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run fault-injection cases in private Linux child owners, never PID 1."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _subreaper(value: int | None = None) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if value is not None:
        assert libc.prctl(36, value, 0, 0, 0) == 0
    current = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(current), 0, 0, 0) == 0
    return current.value


def _child_pids() -> list[int]:
    return [int(value) for value in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()]


def _clean_private_test_children() -> None:
    # Only used in the dedicated scenario interpreter. It has no unrelated
    # children and is an emergency subreaper even if the implementation fails.
    deadline = time.monotonic() + 5
    while True:
        for pid in _child_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                break
        if time.monotonic() > deadline:
            raise RuntimeError("Test could not clean its own children")
        time.sleep(0.005)


def _fault_command(pidfile: Path, *, exits: bool, exit_code: int = 0) -> list[str]:
    # Both descendants escape the inherited process group. Cleanup must follow
    # ownership/adoption rather than assuming killpg alone is sufficient.
    grandchild = "import time; time.sleep(60)"
    child = (
        "import os,subprocess,sys,time,json; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True); "
        f"Path({str(pidfile)!r}).write_text(json.dumps({{'leader':os.getppid(),'child':os.getpid(),'grandchild':p.pid}})); "
        "time.sleep(60)"
    )
    worker = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True)\n"
        f"deadline=time.monotonic()+3\nwhile not Path({str(pidfile)!r}).exists():\n"
        " if time.monotonic()>deadline: raise RuntimeError('descendants did not start')\n"
        " time.sleep(.005)\n"
        "print('descendants-ready',flush=True)\n"
    )
    worker += f"raise SystemExit({exit_code})\n" if exits else "time.sleep(60)\n"
    return [sys.executable, "-c", worker]


def _wait_pidfile(path: Path) -> dict[str, int]:
    deadline = time.monotonic() + 3
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("worker did not record descendants")
        time.sleep(0.005)
    return json.loads(path.read_text())


def _assert_gone(pids: list[int]) -> None:
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids), pids


def _scenario(name: str, directory: Path) -> dict[str, Any]:
    from scripts.benchmarks import benchmark_session_export as bench
    from scripts.benchmarks.process_supervisor import ManagedProcess

    pidfile = directory / "pids.json"
    # This belongs to the private test caller, outside the helper's worker tree.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    before = _subreaper()
    supervisors = []
    if name in {"leader-exits", "nonzero", "timeout"}:
        command = _fault_command(pidfile, exits=name != "timeout", exit_code=7 if name == "nonzero" else 0)

        def replacement(_command: list[str], **kwargs: Any) -> ManagedProcess:
            process = ManagedProcess(command, **kwargs)
            supervisors.append(process)
            return process

        bench.ManagedProcess = replacement
        if name == "timeout":
            with pytest.raises(TimeoutError):
                bench.isolated_run({}, timeout=0.75)
        else:
            with pytest.raises(RuntimeError, match=f"Worker failed \\({7 if name == 'nonzero' else 0}\\)"):
                bench.isolated_run({}, timeout=3)
        recorded = _wait_pidfile(pidfile)
        assert supervisors[0].pid == recorded["leader"]
        _assert_gone([*recorded.values(), supervisors[0].supervisor_pid])
        assert supervisors[0].reaped_descendants == 2
    elif name in {"interrupt", "repeated-interrupt"}:
        process = ManagedProcess(_fault_command(pidfile, exits=False))
        supervisors.append(process)
        recorded = _wait_pidfile(pidfile)
        if name == "interrupt":
            with pytest.raises(KeyboardInterrupt):
                with process:
                    raise KeyboardInterrupt
        else:
            original_wait = process.wait
            remaining = 2

            def interrupted_wait(timeout: float | None = None) -> int:
                nonlocal remaining
                if remaining:
                    remaining -= 1
                    raise KeyboardInterrupt
                return original_wait(timeout)

            process.wait = interrupted_wait
            with pytest.raises(KeyboardInterrupt):
                process.close()
            process.close()
        _assert_gone([*recorded.values(), process.supervisor_pid])
    elif name == "success":
        with ManagedProcess([sys.executable, "-c", "print('success')"]) as process:
            assert process.pid != process.supervisor_pid
            assert process.stdout.read() == b"success\n"
            assert process.stderr.read() == b""
            assert process.wait() == 0
            assert process.poll() == 0
        _assert_gone([process.pid, process.supervisor_pid])
    elif name == "relative-cwd":
        with ManagedProcess(
            [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=os.path.relpath(directory, Path.cwd())
        ) as process:
            assert process.stdout.read().decode().strip() == str(directory.resolve())
            assert process.wait() == 0
        _assert_gone([process.pid, process.supervisor_pid])
    elif name == "pipe-failure":
        # Exhaustion of the second private pipe must release the first one.
        from scripts.benchmarks import process_supervisor

        initial = len(list(Path("/proc/self/fd").iterdir()))
        original_pipe = process_supervisor.os.pipe
        calls = 0

        def failing_pipe() -> tuple[int, int]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected file descriptor exhaustion")
            return original_pipe()

        process_supervisor.os.pipe = failing_pipe
        try:
            with pytest.raises(OSError, match="descriptor exhaustion"):
                ManagedProcess([sys.executable, "-c", "pass"])
        finally:
            process_supervisor.os.pipe = original_pipe
        assert len(list(Path("/proc/self/fd").iterdir())) == initial
    elif name == "startup-failure":
        with pytest.raises(RuntimeError, match="could not start worker"):
            ManagedProcess([str(directory / "missing-executable")])
        assert _child_pids() == [unrelated.pid]
    elif name == "fd-stability":
        initial = len(list(Path("/proc/self/fd").iterdir()))
        for _ in range(10):
            with ManagedProcess([sys.executable, "-c", "pass"]) as process:
                assert process.wait() == 0
            _assert_gone([process.pid, process.supervisor_pid])
        assert len(list(Path("/proc/self/fd").iterdir())) == initial
    else:
        raise AssertionError(name)
    assert unrelated.poll() is None, "helper killed an unrelated caller child"
    assert _subreaper() == before, "helper changed the caller's subreaper state"
    unrelated.terminate()
    unrelated.wait()
    assert not _child_pids()
    return {"scenario": name, "unrelated_child_preserved": True, "all_owned_processes_reaped": True}


def _scenario_main(name: str, directory: Path) -> None:
    previous = _subreaper()
    _subreaper(1)

    def timed_out(_signal: int, _frame: Any) -> None:
        raise TimeoutError("scenario watchdog expired")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(10)
    try:
        result = _scenario(name, directory)
    finally:
        signal.alarm(0)
        _clean_private_test_children()
        _subreaper(previous)
    print(json.dumps(result))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl and /proc ownership checks")
@pytest.mark.parametrize(
    "scenario",
    [
        "leader-exits",
        "nonzero",
        "timeout",
        "interrupt",
        "repeated-interrupt",
        "success",
        "relative-cwd",
        "pipe-failure",
        "startup-failure",
        "fd-stability",
    ],
)
def test_session_benchmark_process_tree_cleanup(scenario: str, tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--scenario", scenario, str(tmp_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["all_owned_processes_reaped"]
    assert result["unrelated_child_preserved"]


if __name__ == "__main__":
    _scenario_main(sys.argv[2], Path(sys.argv[3]))
