# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Clean up live fork descendants and reparented spawn workers."""

import ast
import concurrent.futures
import multiprocessing
import os
import signal
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import psutil
import pytest


pytest_plugins = ["tests.backends.sglang.test_router_registration"]
TOKEN_ENV = "RELAX_SGLANG_SERVER_INCARNATION"


def _alive(process):
    try:
        return process.is_running() and (process.status() != psutil.STATUS_ZOMBIE or process.num_threads() > 1)
    except psutil.NoSuchProcess:
        return False


def _worker_identity():
    import setproctitle

    # Real SGLang schedulers/DP/tokenizer workers rename themselves. The
    # default setproctitle behavior overwrites /proc's exec environment.
    setproctitle.setproctitle("sglang::test-worker")
    return (
        os.getpid(),
        os.environ.get(TOKEN_ENV),
        psutil.Process().environ().get(TOKEN_ENV),
        multiprocessing.get_start_method(),
    )


def _worker(pipe):
    pipe.send(_worker_identity())
    pipe.close()
    threading.Event().wait(60)


def _cpu_server(server_args):
    pipe = server_args.test_pipe
    pipe.send("ready")
    pipe.recv()  # The owner has already captured the early parent-only tree.
    if server_args.worker_kind == "cpu_pool":
        from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

        executor = BaseMultimodalProcessor()._create_cpu_executor()
        pipe.send(executor.submit(_worker_identity).result(timeout=10))
    else:
        local, child = multiprocessing.get_context("spawn").Pipe()
        worker = multiprocessing.get_context("spawn").Process(target=_worker, args=(child,))
        worker.start()
        assert local.poll(10)
        pipe.send(local.recv())
    pipe.recv()
    os._exit(0)  # Deliberately leave a live, reparented worker for shutdown.


def _server_entry(server_args, parent_pid):
    # Execute the production entry in a real spawn child, replacing only the
    # GPU server and its parent-death signal guard with controlled CPU peers.
    http = ModuleType("sglang.srt.entrypoints.http_server")
    http.launch_server = _cpu_server
    utils = ModuleType("sglang.srt.utils")
    utils.kill_itself_when_parent_died = lambda: None
    base = ModuleType("sglang.srt.multimodal.processors.base_processor")

    class BaseMultimodalProcessor:
        # SGLang 0.5.21 explicitly forks this pool for cpu/cuda_ipc transports.
        def _create_cpu_executor(self):
            return concurrent.futures.ProcessPoolExecutor(
                mp_context=multiprocessing.get_context("fork"), max_workers=1
            )

    base.BaseMultimodalProcessor = BaseMultimodalProcessor
    sys.modules[base.__name__] = base
    sys.modules[http.__name__] = http
    sys.modules[utils.__name__] = utils
    source = Path(__file__).parents[3] / "relax/backends/sglang/sglang_engine.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch_server_with_patches")
    namespace = {
        "ServerArgs": SimpleNamespace,
        "os": os,
        "multiprocessing": multiprocessing,
        "Envs": SimpleNamespace(RELAX_OPTIMIZE_ROUTING_REPLAY=False),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    namespace[node.name](server_args, parent_pid)


@pytest.fixture
def follower(monkeypatch, sglang_engine_module, request):
    module = sglang_engine_module
    monkeypatch.setattr(module, "_launch_server_with_patches", _server_entry)
    monkeypatch.setattr(module, "ServerArgs", SimpleNamespace)
    monkeypatch.setattr(module.Envs, "RELAX_OPTIMIZE_ROUTING_REPLAY", False, raising=False)
    monkeypatch.setattr(module.Envs, "RELAX_OPD_PER_POS_TOKEN_IDS", False, raising=False)
    monkeypatch.setattr(
        module, "_wait_server_healthy", lambda **kwargs: pytest.fail("a follower must still return before readiness")
    )
    monkeypatch.setattr(module.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "follower"))
    monkeypatch.setenv("SGLANG_CPU_WORKERS", "1")
    owned = []

    def launch():
        local, child = multiprocessing.get_context("spawn").Pipe()
        engine = object.__new__(module.SGLangEngine)
        engine.args = SimpleNamespace(rollout_external=False)
        engine.server_host, engine.server_port, engine.worker_type = "localhost", 18000, "regular"
        engine._skip_router_registration = True
        engine.unregister_from_router = lambda **kwargs: None
        engine._init_normal(
            {
                "host": "localhost",
                "node_rank": 1,
                "dp_size": 1,
                "test_pipe": child,
                "worker_kind": getattr(request, "param", "spawn"),
            },
            apply_policy_load_plan=False,
        )
        record = SimpleNamespace(engine=engine, process=engine.process, pipe=local, child=child, worker=None)
        owned.append(record)
        assert [p["pid"] for p in engine._release_evidence["processes"]] == [engine.process.pid]
        assert local.poll(10) and local.recv() == "ready"
        return record

    yield launch
    for record in owned:
        if record.process.is_alive():
            record.process.kill()
        record.process.join(timeout=5)
        if record.worker is not None and record.worker.is_running():
            record.worker.kill()
            psutil.wait_procs([record.worker], timeout=0.1)
        record.engine.process = None
        record.pipe.close()
        record.child.close()


def _start_late_worker(record):
    record.pipe.send("spawn after capture")
    assert record.pipe.poll(10)
    pid, environ_token, proc_token, record.start_method = record.pipe.recv()
    record.worker = psutil.Process(pid)
    return environ_token, proc_token


def _exit_parent(record):
    record.pipe.send("parent exits")
    record.process.join(timeout=5)
    assert not record.process.is_alive()
    assert _alive(record.worker)


@pytest.mark.parametrize("follower", ["spawn"], indirect=True)
def test_follower_shutdown_cleans_late_worker_without_killing_other_incarnation(follower):
    old, other = follower(), follower()
    old_token, old_proc_token = _start_late_worker(old)
    other_token, other_proc_token = _start_late_worker(other)
    _exit_parent(old)
    _exit_parent(other)

    old.engine.shutdown()
    assert not _alive(old.worker)
    assert old.engine.process is None
    assert _alive(other.worker)
    assert old_token == old_proc_token == old.engine._release_evidence["incarnation"]
    assert other_token == other_proc_token == other.engine._release_evidence["incarnation"]
    assert old_token != other_token
    other.engine.shutdown()
    assert not _alive(other.worker)


@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_live_parent_shutdown_cleans_fork_cpu_pool_without_killing_other_incarnation(follower):
    old, other = follower(), follower()
    _start_late_worker(old)
    _start_late_worker(other)
    assert old.start_method == other.start_method == "fork"
    assert old.process.is_alive() and other.process.is_alive()
    # These workers appeared after the saved tree; the live parent lets
    # shutdown find them through descendants without changing the pool mode.
    assert old.worker in psutil.Process(old.process.pid).children(recursive=True)
    old.engine.shutdown()
    assert not _alive(old.worker)
    assert old.engine.process is None
    assert other.process.is_alive()
    assert _alive(other.worker)
    other.engine.shutdown()
    assert not _alive(other.worker)


@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_reaped_server_group_still_owns_fork_workers_without_proc_token(follower):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    old, other = follower(), follower()
    for record in (old, other):
        token, proc_token = _start_late_worker(record)
        assert record.start_method == "fork"
        assert token == record.engine._release_evidence["incarnation"]
        assert proc_token is None
        _exit_parent(record)
        assert not psutil.pid_exists(record.process.pid)
        assert os.getpgid(record.worker.pid) == record.process.pid
    cleanup_server_identity(old.engine._release_evidence, timeout=0.5)
    assert not _alive(old.worker)
    assert _alive(other.worker)
    cleanup_server_identity(other.engine._release_evidence, timeout=0.5)
    assert not _alive(other.worker)


@pytest.mark.parametrize("split_retry", [False, True])
@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_reused_group_stays_retired_when_new_leader_exits_while_cleaning_escapee(follower, monkeypatch, split_retry):
    import subprocess

    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    foreign = follower()
    _start_late_worker(foreign)
    identity = {
        "server_pid": foreign.process.pid,
        "pgid": foreign.process.pid,
        "incarnation": "retired-server-generation",
        "processes": [
            {"pid": foreign.process.pid, "create_time": psutil.Process(foreign.process.pid).create_time() - 100}
        ],
    }
    escaped = subprocess.Popen(
        [sys.executable, "-c", 'import sys; print("ready", flush=True); sys.stdin.read()'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, TOKEN_ENV: identity["incarnation"]},
    )
    try:
        assert escaped.stdout.readline().strip() == "ready"
        kill = psutil.Process.kill
        attempts = []
        retrying = False

        def delay_escapee_exit(proc):
            if proc.pid == escaped.pid:
                attempts.append(proc.pid)
                if len(attempts) == 1:
                    # Group reuse has already been observed, then its new
                    # leader exits. Its foreign fork worker must stay alive.
                    _exit_parent(foreign)
                    return
                if split_retry and not retrying:
                    return
            return kill(proc)

        with monkeypatch.context() as patcher:
            patcher.setattr(os, "killpg", lambda *a: pytest.fail("signalled a retired/reused group"))
            patcher.setattr(psutil.Process, "kill", delay_escapee_exit)
            if split_retry:
                with pytest.raises(RuntimeError, match="exit was not confirmed"):
                    cleanup_server_identity(identity, timeout=0)
                retrying = True
            cleanup_server_identity(identity, timeout=10)
        assert len(attempts) >= 2
        assert escaped.wait(timeout=2) is not None
        assert not psutil.pid_exists(foreign.process.pid)
        assert _alive(foreign.worker)
        assert os.getpgid(foreign.worker.pid) == identity["pgid"]
    finally:
        if escaped.poll() is None:
            escaped.kill()
        escaped.wait(timeout=2)
        escaped.stdin.close()
        escaped.stdout.close()


@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_post_kill_group_scan_ignores_stale_process_iter_identity(follower, monkeypatch):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    record = follower()
    _start_late_worker(record)
    _exit_parent(record)
    stale = psutil.Process(record.worker.pid)
    stale._ident = (stale.pid, stale.create_time() - 100)
    killpg = os.killpg

    def delayed_group_kill(pgid, sig):
        if pgid == record.process.pid and sig == signal.SIGKILL:
            return
        return killpg(pgid, sig)

    with monkeypatch.context() as patcher:
        patcher.setattr(psutil, "process_iter", lambda: iter([stale]))
        patcher.setattr(os, "killpg", delayed_group_kill)
        patcher.setattr(psutil.Process, "kill", lambda proc: None)
        with pytest.raises(RuntimeError, match="exit was not confirmed"):
            cleanup_server_identity(record.engine._release_evidence, timeout=0)
        assert _alive(record.worker)
    cleanup_server_identity(record.engine._release_evidence, timeout=2)
    assert not _alive(record.worker)


@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_derived_group_evidence_survives_exit_pending_timeout(follower, monkeypatch):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    record = follower()
    _start_late_worker(record)
    identity = record.engine._release_evidence
    assert identity["pgid"] == record.process.pid
    killpg = os.killpg

    def delayed_group_kill(pgid, sig):
        if pgid == record.process.pid and sig == signal.SIGKILL:
            _exit_parent(record)
            return
        return killpg(pgid, sig)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "killpg", delayed_group_kill)
        patcher.setattr(psutil.Process, "kill", lambda proc: None)
        with pytest.raises(RuntimeError, match="exit was not confirmed"):
            cleanup_server_identity(identity, timeout=0)
    assert identity["pgid"] == record.process.pid
    assert _alive(record.worker)
    cleanup_server_identity(identity, timeout=2)
    assert not _alive(record.worker)


@pytest.mark.parametrize("follower", ["spawn"], indirect=True)
def test_server_incarnation_reaches_spawn_exec_environment_without_tagging_actor(follower):
    actor_token = os.environ.get(TOKEN_ENV)
    record = follower()
    environ_token, proc_token = _start_late_worker(record)
    assert environ_token and environ_token == proc_token
    assert record.engine._release_evidence["incarnation"] == proc_token
    assert os.environ.get(TOKEN_ENV) == actor_token


def test_shutdown_rescans_workers_spawned_during_the_first_membership_scan(follower, monkeypatch):
    record = follower()
    original = psutil.process_iter
    enumerations = []

    def enumerate_processes():
        found = list(original())
        enumerations.append(found)
        if len(enumerations) == 1:
            # The first snapshot cannot see this worker. Its parent disappears
            # before cleanup signals it, so only a second token scan can help.
            _start_late_worker(record)
            _exit_parent(record)
        return iter(found)

    monkeypatch.setattr(psutil, "process_iter", enumerate_processes)
    record.engine.shutdown()
    assert len(enumerations) >= 2
    assert not _alive(record.worker)


def test_cleanup_keeps_known_live_worker_when_its_environment_becomes_unreadable(monkeypatch):
    from relax.backends.sglang.process_cleanup import cleanup_server_identity

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.running = True

        def create_time(self):
            if not self.running:
                raise psutil.NoSuchProcess(self.pid)
            return float(self.pid)

        def children(self, recursive):
            return [child] if self is parent else []

        def environ(self):
            raise psutil.AccessDenied(self.pid)

        def kill(self):
            if self is parent:
                self.running = False

        def is_running(self):
            return self.running

        def status(self):
            return psutil.STATUS_RUNNING

    parent, child = Process(11), Process(12)
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter([child]))
    monkeypatch.setattr(psutil, "wait_procs", lambda processes, timeout: ([parent], [child]))
    identity = {"processes": [{"pid": 11, "create_time": 11.0}], "incarnation": "test-server"}
    with pytest.raises(RuntimeError, match="not confirmed.*12"):
        cleanup_server_identity(identity, timeout=0)
