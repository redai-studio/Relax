# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""N1 follow-up: real owned CPU processes and unchanged production
lifecycle."""

import ast
import ctypes
import multiprocessing
import os
import signal
import sys
import uuid
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

from relax.utils.logging_utils import get_logger


TOKEN = "RELAX_SGLANG_SERVER_INCARNATION"
SOURCE = Path(__file__).parents[4] / "relax/backends/sglang/sglang_engine.py"


def _identity():
    import setproctitle

    setproctitle.setproctitle("sglang::redteam-n1-worker")
    return {"pid": os.getpid(), "token": os.environ.get(TOKEN), "proc_token": psutil.Process().environ().get(TOKEN)}


def _install_real_parent_guard():
    from importlib.metadata import distribution

    source = distribution("sglang").locate_file("sglang/srt/utils/common.py")
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kill_itself_when_parent_died")
    namespace = {"sys": sys, "ctypes": ctypes, "signal": signal}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    namespace[node.name]()


def _leaf(pipe, late_guard):
    pipe.send(_identity())
    if late_guard:
        assert pipe.recv() == "install-guard-after-reparent"
        _install_real_parent_guard()
        pipe.send({"guarded_parent": os.getppid(), "alive": True})
    pipe.recv()
    os._exit(0)


def _middle(pipe, leaf_pipe, late_guard):
    pipe.send(_identity())
    while True:
        command = pipe.recv()
        if command == "spawn":
            if late_guard:
                _install_real_parent_guard()
            proc = multiprocessing.get_context("spawn").Process(target=_leaf, args=(leaf_pipe, late_guard))
            proc.start()
            pipe.send("leaf-started")
        elif command == "exit":
            os._exit(0)
        else:
            raise AssertionError(command)


def _cpu_server(args):
    args.test_pipe.send("ready")
    assert args.test_pipe.recv() == "spawn"
    proc = multiprocessing.get_context("spawn").Process(
        target=_middle, args=(args.middle_pipe, args.leaf_pipe, args.late_guard)
    )
    proc.start()
    args.test_pipe.send("middle-started")
    assert args.test_pipe.recv() == "exit"
    os._exit(0)


def _entry(args, parent_pid):
    # Fake the GPU target; execute production entry and the installed SGLang parent guard.
    http = ModuleType("sglang.srt.entrypoints.http_server")
    http.launch_server = _cpu_server
    utils = ModuleType("sglang.srt.utils")
    utils.kill_itself_when_parent_died = _install_real_parent_guard
    base = ModuleType("sglang.srt.multimodal.processors.base_processor")
    base.BaseMultimodalProcessor = type("BaseMultimodalProcessor", (), {})
    for module in (http, utils, base):
        sys.modules[module.__name__] = module
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_server_with_patches")
    ns = {"ServerArgs": SimpleNamespace, "os": os, "Envs": SimpleNamespace(RELAX_OPTIMIZE_ROUTING_REPLAY=False)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    ns[node.name](args, parent_pid)


def _recv(pipe):
    assert pipe.poll(10), "owned process did not reach barrier"
    return pipe.recv()


def _alive(proc):
    try:
        return proc.is_running() and (proc.status() != psutil.STATUS_ZOMBIE or proc.num_threads() > 1)
    except psutil.NoSuchProcess:
        return False


class Follower:
    def __init__(self, late_guard=False):
        ctx = multiprocessing.get_context("spawn")
        self.local, self.remote = ctx.Pipe()
        self.middle_local, self.middle_remote = ctx.Pipe()
        self.leaf_local, self.leaf_remote = ctx.Pipe()
        self.middle = self.leaf = None
        self.events = []
        self.actor_token = os.environ.get(TOKEN)
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SGLangEngine")
        ns = {
            "multiprocessing": multiprocessing,
            "os": os,
            "uuid": uuid,
            "ServerArgs": SimpleNamespace,
            "Envs": SimpleNamespace(
                RELAX_OPTIMIZE_ROUTING_REPLAY=False,
                RELAX_OPD_PER_POS_TOKEN_IDS=False,
            ),
            "logger": get_logger(__name__),
            "_launch_server_with_patches": _entry,
            "_wait_server_healthy": lambda **kw: pytest.fail("follower unexpectedly waited for readiness"),
            "ray": SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "cpu-follower")),
            "kill_process_tree": lambda pid: pytest.fail("saved identity must own cleanup"),
        }
        names = {"launch_server_process", "_init_normal", "get_release_evidence", "shutdown"}
        for node in [*tree.body, *cls.body]:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
        self.shutdown = ns["shutdown"]
        self.engine = SimpleNamespace(
            args=SimpleNamespace(rollout_external=False),
            server_host="fake",
            server_port=18000,
            worker_type="regular",
            _skip_router_registration=True,
            unregister_from_router=lambda **kw: None,
        )
        self.engine.get_release_evidence = MethodType(ns["get_release_evidence"], self.engine)
        ns["_init_normal"](
            self.engine,
            {
                "host": "127.0.0.1",
                "node_rank": 1,
                "dp_size": 1,
                "test_pipe": self.remote,
                "middle_pipe": self.middle_remote,
                "leaf_pipe": self.leaf_remote,
                "late_guard": late_guard,
            },
            apply_policy_load_plan=False,
        )
        self.process = self.engine.process
        assert [r["pid"] for r in self.engine._release_evidence["processes"]] == [self.process.pid]
        assert _recv(self.local) == "ready"
        self.events.append("production_follower_captured_parent_only")
        assert os.environ.get(TOKEN) == self.actor_token

    def start_middle(self):
        self.local.send("spawn")
        assert _recv(self.local) == "middle-started"
        info = _recv(self.middle_local)
        self.middle = psutil.Process(info["pid"])
        assert info["token"] == info["proc_token"] == self.engine._release_evidence["incarnation"]
        self.events.append("spawn_child_renamed_token_visible")

    def start_leaf(self):
        self.middle_local.send("spawn")
        assert _recv(self.middle_local) == "leaf-started"
        info = _recv(self.leaf_local)
        self.leaf = psutil.Process(info["pid"])
        assert info["token"] == info["proc_token"] == self.engine._release_evidence["incarnation"]
        self.events.append("spawn_grandchild_renamed_token_visible")

    def exit_parent(self):
        self.local.send("exit")
        self.process.join(timeout=5)
        assert not self.process.is_alive()
        self.events.append("recorded_parent_exited")

    def exit_middle(self):
        self.middle_local.send("exit")
        psutil.wait_procs([self.middle], timeout=2)
        assert not _alive(self.middle)
        self.events.append("intermediate_exited_grandchild_reparented")

    def teardown(self):
        from ray.core.generated.gcs_pb2 import ActorTableData

        from relax.distributed.ray.engine_pool import EnginePool, EnginePoolSpec

        class Handle:
            def _get_local_state(self):
                return ActorTableData.ALIVE

            def __getattr__(self, name):
                return SimpleNamespace(remote=lambda **kw: (name, kw))

        spec = EnginePoolSpec(
            actor_class=lambda: None,
            resolve_placement=lambda _: None,
            actor_options=lambda *a: {},
            ctor=lambda *a: ((), {}),
            init_kwargs=lambda *a: {},
            teardown_timeout_s=1,
        )
        pool = EnginePool(spec, [Handle()])
        pool.release_evidence[0] = self.engine._release_evidence

        def get(ref, timeout=None):
            name, kwargs = ref
            assert name == "shutdown"
            self.shutdown(self.engine)

        with patch("ray.get", get), patch("ray.kill", lambda *a, **kw: self.events.append("wrapper_retired")):
            pool.teardown([0])
        assert pool.slots == [None] and self.engine.process is None
        return pool

    def close(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=5)
        for proc in (self.leaf, self.middle):
            if proc is not None and _alive(proc):
                proc.kill()
                psutil.wait_procs([proc], timeout=0.5)
        for conn in (
            self.local,
            self.remote,
            self.middle_local,
            self.middle_remote,
            self.leaf_local,
            self.leaf_remote,
        ):
            conn.close()


@pytest.mark.parametrize("grandchild", [False, True])
def test_original_late_worker_and_grandchild_are_cleaned_other_generation_survives(grandchild):
    records = []
    try:
        old = Follower()
        records.append(old)
        other = Follower()
        records.append(other)
        for record in records:
            record.start_middle()
            if grandchild:
                record.start_leaf()
                record.exit_middle()
            record.exit_parent()
        old_worker = old.leaf if grandchild else old.middle
        other_worker = other.leaf if grandchild else other.middle
        old.teardown()
        assert not _alive(old_worker) and _alive(other_worker)
        other.teardown()
        assert not _alive(other_worker)
        assert os.environ.get(TOKEN) == old.actor_token
        assert os.getpgrp() != old.engine._release_evidence.get("pgid")
    finally:
        for record in records:
            record.close()


@pytest.mark.parametrize("birth_in_second_scan,late_guard", [(False, False), (True, False), (True, True)])
def test_birth_during_first_or_last_empty_scan(monkeypatch, birth_in_second_scan, late_guard):
    record = Follower(late_guard=late_guard)
    original = psutil.process_iter
    enumerations = []
    try:

        def enumerate_snapshot():
            snapshot = list(original())
            enumerations.append(len(snapshot))
            if len(enumerations) == 1:
                record.events.append("first_proc_snapshot_taken")
                record.start_middle()
                record.exit_parent()
            elif len(enumerations) == 2 and birth_in_second_scan:
                record.events.append("second_proc_snapshot_taken")
                record.start_leaf()
                record.exit_middle()
                if late_guard:
                    record.leaf_local.send("install-guard-after-reparent")
                    installed = _recv(record.leaf_local)
                    assert installed["guarded_parent"] != record.middle.pid
                    record.events.append("real_SGLang_parent_guard_installed_after_reparent")
            return iter(snapshot)

        monkeypatch.setattr(psutil, "process_iter", enumerate_snapshot)
        try:
            record.teardown()
        except RuntimeError:
            # An unproven group must retain its handle for a later retry.
            assert record.engine.process is record.process
        else:
            survivor = record.leaf if birth_in_second_scan else record.middle
            assert not _alive(survivor), "released while a same-generation descendant is alive"
        assert os.getpid() > 0 and psutil.Process().is_running()
    finally:
        record.close()


def _delayed_group(pipe):
    pipe.send("before-group")
    if pipe.recv() == "exit":
        os._exit(0)
    os.setpgid(0, 0)
    pipe.send("group-ready")
    pipe.recv()
    os._exit(0)


@pytest.fixture
def delayed_server():
    ctx = multiprocessing.get_context("spawn")
    local, remote = ctx.Pipe()
    process = ctx.Process(target=_delayed_group, args=(remote,))
    process._relax_server_incarnation = uuid.uuid4().hex
    process.start()
    assert _recv(local) == "before-group"
    try:
        yield process, local
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        local.close()
        remote.close()


@pytest.mark.parametrize("exit_before_group", [False, True])
def test_capture_derives_group_without_observing_actor_group(delayed_server, monkeypatch, exit_before_group):
    from relax.backends.sglang.process_cleanup import capture_server_identity

    process, pipe = delayed_server
    actor_pgid = os.getpgrp()
    assert os.getpgid(process.pid) == actor_pgid
    if exit_before_group:
        pipe.send("exit")
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    monkeypatch.setattr(os, "getpgid", lambda pid: pytest.fail("capture observed PGID instead of deriving it"))
    identity = capture_server_identity(process, node_id="follower", port=1234)
    assert identity["pgid"] == process.pid != actor_pgid
    assert identity["incarnation"] == process._relax_server_incarnation
    assert identity["processes"][0]["pid"] == process.pid
    assert os.getpgrp() == actor_pgid and psutil.Process().is_running()


def test_capture_no_longer_uses_observation_budget(delayed_server, monkeypatch):
    from relax.backends.sglang import process_cleanup

    process, _ = delayed_server
    monkeypatch.setattr(
        process_cleanup,
        "time",
        SimpleNamespace(monotonic=lambda: pytest.fail("capture started a wait budget")),
    )
    identity = process_cleanup.capture_server_identity(process, node_id="follower", port=1234)
    assert identity["pgid"] == process.pid
    assert identity["incarnation"] == process._relax_server_incarnation
    assert process.is_alive()


def test_terminate_reaps_zombie_server_before_confirming_empty_group(delayed_server):
    from multiprocessing.connection import wait

    from relax.backends.sglang.process_cleanup import capture_server_identity, terminate_server

    process, pipe = delayed_server
    pipe.send("group")
    assert _recv(pipe) == "group-ready"
    identity = capture_server_identity(process, node_id="follower", port=1234)
    assert identity["pgid"] == process.pid
    pipe.send("exit")
    assert wait([process.sentinel], timeout=5)
    assert psutil.Process(process.pid).status() == psutil.STATUS_ZOMBIE
    os.killpg(process.pid, 0)  # A zombie still keeps its group alive.
    terminate_server(process, lambda pid: pytest.fail("legacy kill used"), timeout=2, identity=identity)
    assert process.exitcode == 0 and not process.is_alive()
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    assert psutil.Process().is_running()


@pytest.mark.parametrize("escaped_token", [False, True])
def test_cleanup_treats_reused_group_as_empty_without_signalling_it(delayed_server, monkeypatch, escaped_token):
    import subprocess

    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    foreign, pipe = delayed_server
    pipe.send("group")
    assert _recv(pipe) == "group-ready"
    identity = {
        "server_pid": foreign.pid,
        "pgid": foreign.pid,
        "incarnation": "retired-generation",
        "processes": [{"pid": foreign.pid, "create_time": psutil.Process(foreign.pid).create_time() - 100}],
    }
    escaped = None
    killpg = os.killpg
    signals = []

    def record_signal(pgid, sig):
        signals.append((pgid, sig))
        assert pgid != os.getpgrp()
        assert sig != signal.SIGKILL, "a token outside this group cannot authorize group kill"
        return killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", record_signal)
    try:
        if escaped_token:
            escaped = subprocess.Popen(
                [sys.executable, "-c", 'import sys; print("ready", flush=True); sys.stdin.read()'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env={**os.environ, TOKEN: identity["incarnation"]},
            )
            assert escaped.stdout.readline().strip() == "ready"
        cleanup_server_identity(identity, timeout=0.5)
        assert foreign.is_alive()
        assert signals == []
        if escaped is not None:
            assert escaped.wait(timeout=2) is not None
    finally:
        if escaped is not None:
            if escaped.poll() is None:
                escaped.kill()
            escaped.wait(timeout=2)
            escaped.stdin.close()
            escaped.stdout.close()


def test_cleanup_rejects_actor_process_group_without_any_group_signal(monkeypatch):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    parent = psutil.Process()
    identity = {
        "server_pid": os.getpgrp(),
        "pgid": os.getpgrp(),
        "incarnation": "invalid-server",
        "processes": [{"pid": parent.pid, "create_time": parent.create_time()}],
    }
    monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("signalled the actor group"))
    monkeypatch.setattr(psutil.Process, "kill", lambda *args: pytest.fail("killed the actor"))
    with pytest.raises(RuntimeError, match="group"):
        cleanup_server_identity(identity, timeout=0)
    assert parent.is_running()
