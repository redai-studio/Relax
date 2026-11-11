# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression tests for the mini-swe-agent server process-management helpers.

Focus is the orphan reaper vs. subprocess exit-code race (a global
``waitpid(-1)`` could reap a live subprocess and make ``communicate()`` report
returncode 0, silently turning a failed apptainer/setup/reward command into a
false success) and the pid-scoped Apptainer instance naming.
"""

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


# agent_server imports heavy runtime deps (flask, litellm, minisweagent, ...);
# skip on a CPU-only checkout that lacks the mini-swe-agent extras rather than
# failing collection.
pytest.importorskip("flask")
pytest.importorskip("minisweagent")
pytest.importorskip("litellm")

_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "mini_swe_agent"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))
agent_server = importlib.import_module("agent_server")


@pytest.mark.parametrize("resumable", [False, True])
def test_resumable_model_read_wait_uses_runtime_deadline(monkeypatch, resumable):
    monkeypatch.setenv("AGENT_SERVER_RESUMABLE_GENERATION", "1" if resumable else "0")
    captured = {}

    def completion(**kwargs):
        captured.update(kwargs)
        return "response"

    monkeypatch.setattr("litellm.completion", completion)
    model = agent_server.MyLitellmModel(model_name="openai/test", model_kwargs={"timeout": 7200})
    assert model._query([{"role": "user", "content": "task"}]) == "response"
    if resumable:
        assert captured["timeout"].read is None
        assert captured["timeout"].connect == captured["timeout"].write == captured["timeout"].pool == 30.0
    else:
        assert captured["timeout"] == 7200
    assert json.loads(model.config.model_dump_json())["model_kwargs"]["timeout"] == 7200
    assert agent_server.SETUP_AND_REWARD_TIMEOUT_SECONDS == 300


def test_tracked_run_preserves_nonzero_exit() -> None:
    result = agent_server._tracked_run(["bash", "-c", "exit 7"], text=True)
    assert result.returncode == 7


def test_tracked_run_check_raises_on_nonzero() -> None:
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        agent_server._tracked_run(
            ["bash", "-c", "echo boom; exit 3"],
            text=True,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert excinfo.value.returncode == 3


def test_tracked_run_timeout_reports_partial_output() -> None:
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        agent_server._tracked_run(
            ["bash", "-c", "echo partial; sleep 5"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=0.3,
        )
    assert b"partial" in (excinfo.value.output or b"")


def test_tracked_run_exit_codes_not_swallowed_by_reaper() -> None:
    """The core regression: an aggressive orphan reaper running concurrently
    must never reap a pid a ``_tracked_run`` is waiting on, which would make
    the command look like it exited 0."""
    stop = threading.Event()

    def hammer() -> None:
        while not stop.is_set():
            agent_server._reap_orphan_children()
            time.sleep(0.001)

    reaper = threading.Thread(target=hammer, daemon=True)
    reaper.start()

    results: list[tuple[int, int]] = []
    errors: list[Exception] = []

    def worker(code: int) -> None:
        try:
            result = agent_server._tracked_run(["bash", "-c", f"sleep 0.05; exit {code}"], text=True)
            results.append((code, result.returncode))
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 20 + 1,)) for i in range(60)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        stop.set()
        reaper.join(timeout=2)

    assert not errors
    assert len(results) == 60
    swallowed = [(expected, actual) for expected, actual in results if expected != actual]
    assert not swallowed, f"reaper swallowed exit codes: {swallowed}"


def test_instance_owner_pid_roundtrip() -> None:
    name = agent_server._new_instance_name()
    assert name.startswith(agent_server.INSTANCE_PREFIX)
    assert agent_server._instance_owner_pid(name) == os.getpid()


def test_instance_owner_pid_legacy_and_malformed_return_none() -> None:
    # Legacy `mswe-<uuid>` names (no owner segment) and non-mswe names.
    assert agent_server._instance_owner_pid("mswe-deadbeefcafe") is None
    assert agent_server._instance_owner_pid("mswe-") is None
    assert agent_server._instance_owner_pid("other-123-x") is None


def test_session_creation_is_idempotent_and_release_waits_for_future():
    server = object.__new__(agent_server.AgentServer)
    server.lock = threading.Lock()
    server.records, server._session_keys = {}, {}
    server._stopping = False
    leases = []

    def lease(group):
        leases.append(group)
        return 0

    server.modes = {"train": SimpleNamespace(lease_sample=lease)}
    server.executors = {"train": ThreadPoolExecutor(max_workers=1)}
    entered, release = threading.Event(), threading.Event()

    def worker(_sid, _sample):
        entered.set()
        assert release.wait(timeout=5)

    server._run_supervised_session = worker
    payload = {
        "mode": "train",
        "group_id": "g",
        "session_id": "client",
        "base_url": "http://local",
        "api_key": "client",
        "release_nonce": "nonce",
    }
    try:
        created = server.create_session(payload)
        assert entered.wait(timeout=2)
        assert server.create_session(payload)["session_id"] == created["session_id"]
        assert leases == ["g"]
        record = server.records[created["session_id"]]
        record.resources_released = True
        record.finish({"n_calls": 1, "reward": 0.0}, "")
        assert server.get_session(created["session_id"])["resource_release"]["status"] == "unknown"
        queued = server.create_session({**payload, "session_id": "queued"})
        cancelled, finish_cancel, poll_started = threading.Event(), threading.Event(), threading.Event()

        def hold_cancel(_future):
            cancelled.set()
            assert finish_cancel.wait(timeout=5)

        def poll_queued():
            poll_started.set()
            return server.get_session(queued["session_id"])

        server.records[queued["session_id"]].future.add_done_callback(hold_cancel)
        with ThreadPoolExecutor(max_workers=2) as control:
            cancelling = control.submit(server.cancel_session, queued["session_id"])
            try:
                assert cancelled.wait(timeout=2)
                poll = control.submit(poll_queued)
                assert poll_started.wait(timeout=2)
                # Cancellation and its no-worker proof are one public transition.
                with pytest.raises(TimeoutError):
                    poll.result(timeout=0.05)
            finally:
                finish_cancel.set()
            cancelling.result(timeout=2)
            assert poll.result(timeout=2)["resource_release"]["status"] == "released"
        assert server.get_session(queued["session_id"])["resource_release"]["status"] == "released"
        release.set()
        record.future.result(timeout=2)
        assert server.get_session(created["session_id"])["resource_release"] == {
            "schema": 1,
            "session_id": "client",
            "nonce": "nonce",
            "status": "released",
        }
    finally:
        release.set()
        server.executors["train"].shutdown(wait=True)


def test_shutdown_cancels_queued_sessions_and_joins_supervised_workers(monkeypatch):
    server = object.__new__(agent_server.AgentServer)
    server.lock = threading.Lock()
    server.records, server._session_keys = {}, {}
    server._stopping = False
    server.apptainer_args = ("apptainer", [])
    server.modes = {"train": SimpleNamespace(lease_sample=lambda _group: 0)}
    server.executors = {"train": ThreadPoolExecutor(max_workers=1)}
    entered = threading.Event()
    queued_done = threading.Event()
    started = []

    def worker(sid, _sample):
        record = server.records[sid]
        started.append(record.session_id)
        entered.set()
        assert record.cancel_event.wait(timeout=5)
        assert queued_done.wait(timeout=5)
        record.resources_released = True
        record.cancel()

    server._run_supervised_session = worker
    monkeypatch.setattr(agent_server, "_stop_mswe_instances", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agent_server, "_drain_orphans", lambda: None)
    payload = {
        "mode": "train",
        "group_id": "g",
        "session_id": "active",
        "base_url": "http://local",
        "api_key": "client",
        "release_nonce": "nonce",
    }
    try:
        active = server.create_session(payload)["session_id"]
        assert entered.wait(timeout=2)
        queued = server.create_session({**payload, "session_id": "queued"})["session_id"]
        server.records[queued].future.add_done_callback(lambda _future: queued_done.set())
        server.shutdown()
        server.records[active].future.result(timeout=1)
        assert server.records[queued].future.cancelled()
        assert started == ["active"]
        for sid in (active, queued):
            result = server.get_session(sid)
            assert result["status"] == "cancelled"
            assert result["resource_release"]["status"] == "released"
        with pytest.raises(RuntimeError, match="stopping"):
            server.create_session(payload)
    finally:
        queued_done.set()
        for record in server.records.values():
            record.cancel_event.set()
        server.executors["train"].shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def real_popen():
    return subprocess.Popen


@pytest.fixture
def supervised_server(tmp_path, monkeypatch, real_popen):
    server = object.__new__(agent_server.AgentServer)
    server.lock = threading.Lock()
    server.tmp_base = server.work_dir = server.sif_dir = server.wheel_dir = tmp_path
    server.base_config = server.r2e_config = server.sifs = {}
    server.apptainer_args = ("apptainer", [])
    server.modes = {"train": SimpleNamespace(get=lambda _: {"docker_image": "test-image"})}
    server.records = {
        sid: agent_server.SessionRecord(
            session_id=sid,
            group_id="g",
            sample_idx=0,
            mode="train",
            base_url="http://local",
            api_key=sid,
            release_nonce="nonce",
            status="running",
        )
        for sid in ("finishing", "other")
    }

    class CompletedWorker:
        pid = 999999999
        returncode = 0

        def __init__(self, command, **_kwargs):
            spec = json.loads(Path(command[2]).read_text())
            Path(command[3]).write_text(
                json.dumps(
                    {
                        "nonce": spec["release_nonce"],
                        "client_session_id": spec["record"]["session_id"],
                        "resources_released": True,
                        "children_reaped": True,
                        "instance_absent": True,
                        "cleanup_error": "",
                        "info": {"n_calls": 1, "reward": 0.0},
                        "error": "",
                    }
                )
            )

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(agent_server.subprocess, "Popen", CompletedWorker)
    return server


@pytest.mark.parametrize(
    ("report_kind", "exit_code"),
    [
        ("valid", 7),
        ("bad_nonce", 7),
        ("bad_session", 7),
        ("missing", 7),
        ("false", 7),
        ("false", 0),
        ("incomplete", 7),
        ("inconsistent", 7),
        ("invalid_bool", 7),
    ],
)
def test_supervisor_nonzero_exit_preserves_only_complete_release_proof(
    supervised_server, real_popen, monkeypatch, report_kind, exit_code
):
    server = supervised_server
    processes = []
    child = """
import json
import os
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text())
target = Path(sys.argv[2])
kind = sys.argv[3]
report = {
    "nonce": spec["release_nonce"],
    "client_session_id": spec["record"]["session_id"],
    "resources_released": True,
    "children_reaped": True,
    "instance_absent": True,
    "cleanup_error": "",
    "info": {"n_calls": 1, "reward": 0.0},
    "error": "",
}
if kind == "bad_nonce":
    report["nonce"] = "different-nonce"
elif kind == "bad_session":
    report["client_session_id"] = "different-session"
elif kind == "false":
    report.update(resources_released=False, instance_absent=False, cleanup_error="instance still present")
elif kind == "incomplete":
    report.pop("children_reaped")
elif kind == "inconsistent":
    report["children_reaped"] = False
elif kind == "invalid_bool":
    report["resources_released"] = 1
if kind != "missing":
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report))
    temporary.replace(target)
os._exit(int(sys.argv[4]))
"""

    def launch(command, **kwargs):
        # Exercise real process exit/wait and atomic report publication without
        # launching an agent, container or GPU workload.
        process = real_popen(
            [sys.executable, "-c", child, command[2], command[3], report_kind, str(exit_code)], **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(agent_server.subprocess, "Popen", launch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(server._run_supervised_session, "finishing", 0)
        server.records["finishing"].future = future
        future.result(timeout=5)
    assert len(processes) == 1 and processes[0].returncode == exit_code
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)
    result = server.get_session("finishing")
    released = report_kind == "valid"
    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert result["resource_release"]["status"] == ("released" if released else "unknown")
    assert result["resource_release_failed"] is not released
    if released:
        assert "Session supervisor exited with code 7" in result["error"]
    else:
        assert result["error"]


@pytest.mark.parametrize("failure", ["event_storage", "control_cleanup"])
def test_successful_session_ignores_diagnostic_cleanup_failure(supervised_server, monkeypatch, failure):
    server = supervised_server
    open_path = Path.open

    def fail_event_storage(path, *args, **kwargs):
        if path.name in {"session_releases.jsonl", "agent_server_events.jsonl"}:
            raise OSError("event storage unavailable")
        return open_path(path, *args, **kwargs)

    def fail_cleanup(_path):
        raise OSError("control directory storage unavailable")

    if failure == "event_storage":
        monkeypatch.setattr(Path, "open", fail_event_storage)
    else:
        monkeypatch.setattr(agent_server.shutil, "rmtree", fail_cleanup)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(server._run_supervised_session, "finishing", 0)
        server.records["finishing"].future = future
        future.result(timeout=2)
    result = server.get_session("finishing")
    assert result["status"] == "completed"
    assert result["reward"] == 0.0
    assert result["error"] == ""
    assert result["resource_release"]["status"] == "released"


def test_failure_event_write_error_preserves_original_result(supervised_server, monkeypatch):
    server = supervised_server
    server._worker_instance_name = "mswe-test"
    server.base_config = {"model": {}, "environment": {}}
    record = server.records["finishing"]
    opened = Path.open

    def fail_write(path, *args, **kwargs):
        if path.name == "agent_server_events.jsonl":
            raise OSError("event storage unavailable")
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_write)
    monkeypatch.setattr(agent_server, "_stop_apptainer_instance", lambda *_args: "")
    # Missing SIF is a real worker setup failure. Failure logging must not
    # replace it or prevent the worker from returning its result for supervision.
    server._run_session_worker("finishing", 0)
    assert record.status == "failed"
    assert "Missing SIF" in record.error
    assert "event storage" not in record.error
    assert record.cleanup_error == ""
    assert record.resources_released is False  # Only the supervisor can prove release.


@pytest.mark.parametrize("failure", ["mkdir", "spec", "log", "launch"])
def test_start_failure_releases_only_proven_empty_allocation(supervised_server, monkeypatch, failure):
    server = supervised_server
    opened = Path.open
    mkdir = Path.mkdir
    launches = []

    def fail_mkdir(path, *args, **kwargs):
        if failure == "mkdir" and path.name == "finishing-control":
            raise OSError("control mkdir failed")
        return mkdir(path, *args, **kwargs)

    def fail_open(path, *args, **kwargs):
        if (failure == "spec" and path.name == "spec.json") or (failure == "log" and path.name == "worker.log"):
            raise OSError("control write failed")
        return opened(path, *args, **kwargs)

    def fail_launch(*_args, **_kwargs):
        launches.append(True)
        raise OSError("launch result unknown")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(agent_server.subprocess, "Popen", fail_launch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(server._run_supervised_session, "finishing", 0)
        server.records["finishing"].future = future
        future.result(timeout=2)
    result = server.get_session("finishing")
    assert result["status"] == "failed"
    assert result["error"]
    assert bool(launches) is (failure == "launch")
    assert result["resource_release"]["status"] == ("unknown" if failure == "launch" else "released")
    assert result["resource_release_failed"] is (failure == "launch")
