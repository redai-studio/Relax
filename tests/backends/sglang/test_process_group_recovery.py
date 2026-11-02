# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""R4b: delayed spawn, unreaped zombies, and late process-group evidence."""

import ast
import ctypes
import multiprocessing
import os
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from types import SimpleNamespace

import psutil
import pytest

from relax.backends.sglang import process_cleanup
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from tests.backends.sglang.test_server_process_group import SOURCE, _entry, _recv


pytest_plugins = ["tests.backends.sglang.test_server_process_group"]


def _unpickle_args(pipe):
    # This barrier is before the production target, like importing torch and
    # SGLang while spawn unpickles ServerArgs. No scheduling sleeps are needed.
    pipe.send("unpickling")
    if pipe.recv() == "exit":
        os._exit(0)
    return SimpleNamespace(test_pipe=pipe)


class _SlowArgs:
    node_rank = 1

    def __init__(self, pipe):
        self.pipe = pipe

    def __reduce__(self):
        return _unpickle_args, (self.pipe,)


@pytest.fixture
def slow_launch(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    processes = []

    def make_process(**kwargs):
        process = ctx.Process(**kwargs)
        processes.append(process)
        return process

    namespace = {
        "multiprocessing": SimpleNamespace(Process=make_process, set_start_method=lambda *a, **kw: None),
        "os": os,
        "uuid": uuid,
        "ServerArgs": _SlowArgs,
        "Envs": Envs,
        "logger": get_logger(__name__),
        "_launch_server_with_patches": _entry,
        "_wait_server_healthy": lambda **kw: pytest.fail("follower waited for HTTP readiness"),
    }
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "launch_server_process")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield SimpleNamespace(
            start=lambda: executor.submit(namespace[node.name], _SlowArgs(remote)),
            pipe=local,
            processes=processes,
        )
    finally:
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(timeout=5)
        executor.shutdown(wait=True)
        local.close()
        remote.close()


@pytest.mark.parametrize("exit_before_target", [False, True])
def test_follower_waits_through_slow_unpickle_before_returning(slow_launch, monkeypatch, exit_before_target):
    first_poll, second_poll, advance, finish = Event(), Event(), Event(), Event()
    getpgid = os.getpgid
    monotonic = time.monotonic
    polls = []

    def observe(pid):
        pgid = getpgid(pid)
        if slow_launch.processes and pid == slow_launch.processes[0].pid:
            polls.append(pgid)
            if len(polls) == 1:
                first_poll.set()
                assert advance.wait(5)
            elif len(polls) == 2:
                second_poll.set()
                assert finish.wait(5)
        return pgid

    monkeypatch.setattr(os, "getpgid", observe)
    # Advance ten seconds while unpickling remains blocked: the former 5s
    # capture budget cannot cover this, but the startup budget must do so.
    monkeypatch.setattr(
        process_cleanup,
        "time",
        SimpleNamespace(monotonic=lambda: monotonic() + (10 if advance.is_set() else 0), sleep=time.sleep),
    )
    future = slow_launch.start()
    try:
        assert _recv(slow_launch.pipe) == "unpickling"
        assert first_poll.wait(2), "follower returned before waiting for its process group"
        assert not future.done()
        advance.set()
        assert second_poll.wait(2), "startup used the short capture budget"
        assert not future.done()
        slow_launch.pipe.send("exit" if exit_before_target else "continue")
        if not exit_before_target:
            assert _recv(slow_launch.pipe) == "ready"
        finish.set()
        process = future.result(timeout=5)
        identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234, timeout=0)
        if exit_before_target:
            assert "pgid" not in identity
        else:
            assert identity["pgid"] == process.pid != os.getpgrp()
        assert os.getpgrp() != process.pid and psutil.Process().is_running()
    finally:
        advance.set()
        finish.set()


def test_follower_group_wait_timeout_preserves_process_for_later_cleanup(slow_launch, monkeypatch, caplog):
    monkeypatch.setenv("RELAX_SGLANG_PROCESS_GROUP_TIMEOUT_S", "0.03")
    future = slow_launch.start()
    assert _recv(slow_launch.pipe) == "unpickling"
    process = future.result(timeout=3)
    assert process.is_alive()
    assert "process group" in caplog.text and "0.03" in caplog.text
    identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234, timeout=0)
    assert "pgid" not in identity
    slow_launch.pipe.send("continue")
    assert _recv(slow_launch.pipe) == "ready"
    process_cleanup.terminate_server(process, lambda pid: pytest.fail("legacy kill"), identity=identity, timeout=2)
    assert process.exitcode == -signal.SIGKILL


def test_cleanup_upgrades_token_identity_and_joins_its_server(delayed_server, monkeypatch):
    process, pipe = delayed_server
    identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234, timeout=0)
    assert "pgid" not in identity
    pipe.send("group")
    assert _recv(pipe) == "group-ready"
    killpg = os.killpg
    signals = []

    def record(pgid, sig):
        signals.append((pgid, sig))
        return killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record)
    process_cleanup.terminate_server(process, lambda pid: pytest.fail("legacy kill"), identity=identity, timeout=2)
    assert (process.pid, signal.SIGKILL) in signals
    assert process.exitcode == -signal.SIGKILL and not process.is_alive()


def _unreaping_parent(pipe):
    local, remote = multiprocessing.get_context("fork").Pipe()
    pid = os.fork()
    if pid == 0:
        os.setpgid(0, 0)
        remote.send("group-ready")
        remote.recv()
        os._exit(0)
    try:
        assert local.recv() == "group-ready"
        pipe.send(pid)
        assert pipe.recv() == "exit-child"
        local.send("exit")
        # Observe termination without reaping: emulate a container PID 1 that
        # keeps zombies indefinitely, without changing pytest's own parenting.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        pipe.send("zombie")
        pipe.recv()
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        local.close()
        remote.close()


def test_group_with_only_unreaped_zombies_confirms_release_after_sigkill(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    holder = ctx.Process(target=_unreaping_parent, args=(remote,))
    holder.start()
    try:
        pid = _recv(local)
        server = SimpleNamespace(pid=pid, _relax_server_incarnation=uuid.uuid4().hex)
        identity = process_cleanup.capture_server_identity(server, node_id="follower", port=1234)
        local.send("exit-child")
        assert _recv(local) == "zombie"
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        assert psutil.Process(pid).ppid() == holder.pid
        events = []
        killpg, process_iter = os.killpg, psutil.process_iter

        def signal_group(pgid, sig):
            result = killpg(pgid, sig)
            if sig == signal.SIGKILL:
                events.append("SIGKILL-delivered")
            return result

        def enumerate_processes():
            if "SIGKILL-delivered" in events:
                events.append("post-kill-enumeration")
            return process_iter()

        monkeypatch.setattr(os, "killpg", signal_group)
        monkeypatch.setattr(psutil, "process_iter", enumerate_processes)
        process_cleanup.cleanup_server_identity(identity, timeout=0.5)
        assert "post-kill-enumeration" in events
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        killpg(pid, 0)  # ESRCH is deliberately still false after success.
        assert holder.is_alive()
    finally:
        local.send("reap-child")
        holder.join(timeout=5)
        if holder.is_alive():
            holder.kill()
            holder.join(timeout=5)
        local.close()
        remote.close()


def test_group_sigkill_ack_without_exit_keeps_fence_until_retry(delayed_server, monkeypatch):
    process, pipe = delayed_server
    pipe.send("group")
    assert _recv(pipe) == "group-ready"
    identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234)
    killpg, kill = os.killpg, psutil.Process.kill
    delivered = []

    def delayed_group_kill(pgid, sig):
        if pgid == process.pid and sig == signal.SIGKILL:
            delivered.append(pgid)
            return  # Fault injection: signal accepted, exit not yet observable.
        return killpg(pgid, sig)

    def delayed_pid_kill(proc):
        if proc.pid != process.pid:
            return kill(proc)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "killpg", delayed_group_kill)
        patcher.setattr(psutil.Process, "kill", delayed_pid_kill)
        with pytest.raises(RuntimeError, match="exit was not confirmed"):
            process_cleanup.cleanup_server_identity(identity, timeout=0.03)
        assert delivered and process.is_alive()
    process_cleanup.terminate_server(process, lambda pid: pytest.fail("legacy kill"), identity=identity, timeout=2)
    assert not process.is_alive()


def test_legacy_identity_keeps_live_server_fenced_without_token_or_group(delayed_server, monkeypatch):
    process, _ = delayed_server
    identity = process_cleanup.capture_server_identity(SimpleNamespace(pid=process.pid), node_id="follower", port=1234)
    assert "pgid" not in identity and "incarnation" not in identity
    kill = psutil.Process.kill

    def delayed_pid_kill(proc):
        if proc.pid != process.pid:
            return kill(proc)

    with monkeypatch.context() as patcher:
        patcher.setattr(psutil.Process, "kill", delayed_pid_kill)
        with pytest.raises(RuntimeError, match="exit was not confirmed"):
            process_cleanup.terminate_server(
                process, lambda pid: pytest.fail("legacy kill"), identity=identity, timeout=0.03
            )
        assert process.is_alive()


def _zombie_thread_leader(pipe):
    os.setpgid(0, 0)
    pipe.send("group-ready")
    assert pipe.recv() == "exit-main-thread"

    def survivor():
        while psutil.Process().status() != psutil.STATUS_ZOMBIE:
            os.sched_yield()
        pipe.send("leader-zombie-thread-alive")
        pipe.recv()

    Thread(target=survivor).start()
    libc = ctypes.CDLL(None)
    libc.pthread_exit.argtypes = [ctypes.c_void_p]
    libc.pthread_exit.restype = None
    libc.pthread_exit(None)


def test_zombie_leader_with_live_thread_keeps_group_fenced(monkeypatch):
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    process = ctx.Process(target=_zombie_thread_leader, args=(remote,))
    process._relax_server_incarnation = uuid.uuid4().hex
    process.start()
    try:
        assert _recv(local) == "group-ready"
        identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234)
        local.send("exit-main-thread")
        assert _recv(local) == "leader-zombie-thread-alive"
        leader = psutil.Process(process.pid)
        assert leader.status() == psutil.STATUS_ZOMBIE and leader.num_threads() == 2
        killpg = os.killpg

        def delayed_group_kill(pgid, sig):
            if pgid == process.pid and sig == signal.SIGKILL:
                return
            return killpg(pgid, sig)

        with monkeypatch.context() as patcher:
            patcher.setattr(os, "killpg", delayed_group_kill)
            patcher.setattr(psutil.Process, "kill", lambda proc: None)
            with pytest.raises(RuntimeError, match="exit was not confirmed"):
                process_cleanup.cleanup_server_identity(identity, timeout=0)
            assert "released_process_group" not in identity
            assert leader.num_threads() == 2
        process_cleanup.terminate_server(process, lambda pid: pytest.fail("legacy kill"), identity=identity, timeout=2)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        local.close()
        remote.close()
