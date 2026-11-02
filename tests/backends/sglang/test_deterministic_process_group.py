# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A managed server owns its PID-named group even before entry or after
exit."""

import ast
import ctypes
import multiprocessing
import os
import signal
import traceback
import uuid
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

from relax.backends.sglang import process_cleanup
from relax.utils.logging_utils import get_logger
from tests.backends.sglang.test_server_process_group import SOURCE, TOKEN, _alive, _recv


pytest_plugins = ["tests.backends.sglang.test_server_process_group"]


def _fork_server(pipe):
    pipe.send("before-group")  # Model spawn/unpickling before the managed entry.
    assert pipe.recv() == "group"
    os.setpgid(0, 0)
    process_cleanup.inherit_server_incarnation()
    pipe.send("group-ready")
    assert pipe.recv() == "fork"
    worker = os.fork()
    if worker == 0:
        pipe.close()
        Event().wait()
        os._exit(0)
    pipe.send({"worker": worker, "token": psutil.Process(worker).environ().get(TOKEN)})
    assert pipe.recv() == "exit"
    os._exit(0)


def _shutdown(engine):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SGLangEngine")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    ns = {"logger": get_logger(__name__), "kill_process_tree": lambda pid: pytest.fail("legacy cleanup")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    ns["shutdown"](engine)


def _late_group_case(result_pipe, mode):
    # Only this isolated test process becomes a subreaper; never change the
    # pytest/Ray caller's parenting. Reap only the two PIDs created here.
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    root = ctx.Process(target=_fork_server, args=(remote,))
    root._relax_server_incarnation = uuid.uuid4().hex
    root.start()
    worker = None
    result = {}

    def fork_worker():
        nonlocal worker
        local.send("group")
        assert _recv(local) == "group-ready"
        local.send("fork")
        details = _recv(local)
        worker = psutil.Process(details["worker"])
        assert details["token"] is None
        assert os.getpgid(worker.pid) == root.pid != os.getpgrp()

    try:
        assert _recv(local) == "before-group"
        saved = None
        if mode in {"fallback_then_root_dies", "birth_between_signals"}:
            saved = process_cleanup.capture_server_identity(root, node_id="owned-test", port=1, timeout=0)
        if mode == "birth_between_signals":
            # Isolate signal ordering with an already-recorded, derived PGID.
            saved["pgid"] = root.pid
            kill = psutil.Process.kill

            def kill_root(process):
                if process.pid == root.pid and worker is None:
                    fork_worker()  # Root advances immediately before its PID kill.
                return kill(process)

            with patch.object(psutil.Process, "kill", kill_root):
                process_cleanup.terminate_server(root, lambda pid: pytest.fail("legacy cleanup"), identity=saved)
            result["success"] = True
        else:
            local.send("group")
            assert _recv(local) == "group-ready"
            if mode == "observed_live":
                saved = process_cleanup.capture_server_identity(root, node_id="owned-test", port=1, timeout=0)
                assert [row["pid"] for row in saved["processes"]] == [root.pid]
            if mode == "waiter_then_capture_zombie":
                # The former launcher waiter observed this same value but did
                # not persist it; capture now happens only after root exits.
                assert os.getpgid(root.pid) == root.pid
            local.send("fork")
            details = _recv(local)
            worker = psutil.Process(details["worker"])
            assert details["token"] is None
            local.send("exit")
            # Raw fork inherits the multiprocessing sentinel FD. waitid gives
            # an exit barrier without waiting for that worker or reaping root.
            os.waitid(os.P_PID, root.pid, os.WEXITED | os.WNOWAIT)
            assert psutil.Process(root.pid).status() == psutil.STATUS_ZOMBIE
            assert _alive(worker)
            if saved is None:
                saved = process_cleanup.capture_server_identity(root, node_id="owned-test", port=1, timeout=0)
            else:
                assert not root.is_alive()  # Reap root before using its saved identity.
                assert not psutil.pid_exists(root.pid)
            engine = SimpleNamespace(
                args=SimpleNamespace(rollout_external=False),
                process=root,
                _release_evidence=saved,
                server_host="owned-test",
                server_port=1,
                unregister_from_router=lambda **kwargs: None,
            )
            _shutdown(engine)
            result["success"] = engine.process is None
        result.update(worker_alive=_alive(worker), pgid=saved.get("pgid"), server_pid=root.pid)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if root.is_alive():
            root.kill()
        if worker is not None:
            if _alive(worker):
                worker.kill()
            try:
                os.waitpid(worker.pid, 0)
            except ChildProcessError:
                pass
        root.join(timeout=5)
        local.close()
        remote.close()
    result_pipe.send(result)
    result_pipe.close()


@pytest.mark.parametrize(
    "mode",
    [
        "capture_zombie",
        "waiter_then_capture_zombie",
        "fallback_then_root_dies",
        "observed_live",
        "birth_between_signals",
    ],
)
def test_derived_group_retires_tokenless_worker_after_root_exit(mode):
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    process = ctx.Process(target=_late_group_case, args=(remote, mode))
    process.start()
    try:
        assert local.poll(30), "isolated cleanup did not finish"
        result = local.recv()
        process.join(timeout=5)
        assert process.exitcode == 0
        assert "error" not in result, result.get("error")
        assert result["success"] and not result["worker_alive"], result
        assert result["pgid"] == result["server_pid"]
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        local.close()
        remote.close()


@pytest.mark.parametrize("killed_before_capture", [False, True])
def test_derived_group_cleanup_before_setpgid_never_signals_actor(delayed_server, monkeypatch, killed_before_capture):
    process, _ = delayed_server
    actor_group = os.getpgrp()
    if killed_before_capture:
        process.kill()
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    identity = process_cleanup.capture_server_identity(process, node_id="owned-test", port=1, timeout=0)
    signals = []
    killpg = os.killpg

    def record(pgid, sig):
        assert pgid == process.pid != actor_group
        signals.append((pgid, sig))
        return killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record)
    process_cleanup.terminate_server(process, lambda pid: pytest.fail("legacy cleanup"), identity=identity, timeout=2)
    assert identity["pgid"] == process.pid
    assert (process.pid, signal.SIGKILL) in signals
    assert not process.is_alive() and process.exitcode == -signal.SIGKILL
    assert os.getpgrp() == actor_group


def test_managed_server_entry_sets_own_group_before_any_other_statement():
    tree = ast.parse(SOURCE.read_text())
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_server_with_patches")
    body = entry.body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # A docstring is not an executable entry operation.
    expected = ast.parse("os.setpgid(0, 0)").body[0]
    assert ast.dump(body[0], include_attributes=False) == ast.dump(expected, include_attributes=False)
