# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Linux subprocess-tree ownership for CPU benchmarks.

A dedicated supervisor is the subreaper, so the caller's prctl state and its
unrelated subprocesses are untouched. Only the supervisor calls waitpid(-1):
its children consist exclusively of the one worker tree it launched.
"""

from __future__ import annotations

import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


class ManagedProcess:
    """Own a worker and all descendants until the context is closed.

    ``pid`` identifies the real worker for CPU/RSS sampling, not the additional
    supervisor process. stdout/stderr are binary pipes. stdin is /dev/null.
    Success, errors, timeouts and KeyboardInterrupt all terminate and reap the
    owned tree before context exit. The supervisor also handles caller death.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        startup_timeout: float = 10.0,
    ) -> None:
        if sys.platform != "linux":
            raise RuntimeError("ManagedProcess requires Linux prctl and /proc")
        self.command = list(command)
        self.returncode: int | None = None
        self.reaped_descendants = 0
        self._closed = False
        self._buffer = b""
        cwd = str(Path(cwd).resolve()) if cwd is not None else None
        opened_fds = []
        try:
            config_read, config_write = os.pipe()
            opened_fds.extend((config_read, config_write))
            control_read, control_write = os.pipe()
            opened_fds.extend((control_read, control_write))
            self._control = control_read
            self._owner = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--supervise",
                    str(config_read),
                    str(control_write),
                    str(os.getpid()),
                ],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(config_read, control_write),
            )
        except BaseException:
            for fd in opened_fds:
                os.close(fd)
            raise
        os.close(config_read)
        os.close(control_write)
        self.stdout = self._owner.stdout
        self.stderr = self._owner.stderr
        self.supervisor_pid = self._owner.pid
        try:
            with os.fdopen(config_write, "wb") as stream:
                stream.write(
                    json.dumps({"command": self.command, "cwd": os.fspath(cwd) if cwd is not None else None}).encode()
                    + b"\n"
                )
            packet = self._receive(startup_timeout)
            if packet.get("kind") != "started":
                raise RuntimeError(f"Supervisor could not start worker: {packet}")
            self.pid = int(packet["pid"])
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _receive(self, timeout: float | None) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not select.select([self._control], [], [], remaining)[0]:
                raise subprocess.TimeoutExpired(self.command, timeout)
            chunk = os.read(self._control, 65536)
            if not chunk:
                raise RuntimeError("Supervisor exited without a complete status message")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def poll(self) -> int | None:
        if self._owner.poll() is None:
            return None
        return self.wait()

    def wait(self, timeout: float | None = None) -> int:
        self._owner.wait(timeout=timeout)
        if self.returncode is None:
            packet = self._receive(None)
            if packet.get("kind") != "exit" or not packet.get("cleanup_complete"):
                raise RuntimeError(f"Supervisor failed: {packet}")
            self.returncode = int(packet["returncode"])
            self.reaped_descendants = int(packet["reaped_descendants"])
        return self.returncode

    def terminate(self) -> None:
        """Request complete tree cleanup, including descendants in new
        sessions."""
        if self._owner.poll() is None:
            self._owner.send_signal(signal.SIGTERM)

    def close(self) -> None:
        if self._closed:
            return
        self.terminate()
        interrupted = False
        try:
            while True:
                try:
                    self.wait()
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    self.terminate()
        finally:
            if self.stdout is not None:
                self.stdout.close()
            if self.stderr is not None:
                self.stderr.close()
            os.close(self._control)
            self._closed = True
        if interrupted:
            raise KeyboardInterrupt

    def __enter__(self) -> "ManagedProcess":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _set_prctl(option: int, value: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _children() -> list[int]:
    """Only direct children of this dedicated, single-threaded supervisor."""
    path = Path(f"/proc/self/task/{os.getpid()}/children")
    return [int(value) for value in path.read_text().split()]


def _terminate_and_reap(worker: subprocess.Popen[bytes]) -> int:
    """Kill parents, adopt orphans, and repeat until the private tree is empty.

    No cached PID/PGID is signalled after reaping, avoiding PID-reuse races.
    Children in other process groups/sessions are adopted just like others. The
    calling process has no unrelated children.
    """
    reaped = 0
    while True:
        children = _children()
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        no_children = False
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                no_children = True
                break
            if pid == 0:
                break
            if pid == worker.pid and worker.returncode is None:
                worker.returncode = os.waitstatus_to_exitcode(status)
            else:
                reaped += 1
        if no_children:
            return reaped
        time.sleep(0.005)


def _supervise(config_fd: int, control_fd: int, expected_parent: int) -> int:
    stopping = False

    def stop(_sig: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    worker = None
    control = os.fdopen(control_fd, "wb", buffering=0)

    def send(packet: dict[str, Any]) -> None:
        control.write(json.dumps(packet).encode() + b"\n")

    try:
        _set_prctl(36, 1)  # PR_SET_CHILD_SUBREAPER, scoped to this dedicated owner.
        _set_prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: clean up if the caller dies.
        if os.getppid() != expected_parent:
            raise RuntimeError("Benchmark caller exited during supervisor startup")
        with os.fdopen(config_fd, "rb") as stream:
            config = json.loads(stream.readline())
        if stopping:
            raise RuntimeError("Supervisor was stopped before worker startup")
        worker = subprocess.Popen(config["command"], cwd=config["cwd"], stdin=subprocess.DEVNULL)
        send({"kind": "started", "pid": worker.pid})
        while not stopping and worker.poll() is None:
            time.sleep(0.01)
        reaped = _terminate_and_reap(worker)
        send({"kind": "exit", "returncode": worker.returncode, "reaped_descendants": reaped, "cleanup_complete": True})
        return 0
    except BaseException as error:
        if worker is not None:
            _terminate_and_reap(worker)
        try:
            send({"kind": "error", "message": f"{type(error).__name__}: {error}"})
        except BrokenPipeError:
            pass
        return 1
    finally:
        control.close()


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--supervise":
        raise SystemExit("This module is an internal subprocess supervisor")
    raise SystemExit(_supervise(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
