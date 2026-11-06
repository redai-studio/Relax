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
    server._supervised = False
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

    server._run_session = worker
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
        server.cancel_session(queued["session_id"])
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


@pytest.fixture
def supervised_server(tmp_path, monkeypatch):
    server = object.__new__(agent_server.AgentServer)
    server.lock = threading.Lock()
    server._event_lock = threading.Lock()
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


def test_slow_session_event_write_keeps_other_session_poll_responsive(supervised_server, monkeypatch):
    """Slow shared-filesystem logging must not hold the HTTP control-plane
    lock."""
    server = supervised_server
    entered, resume = threading.Event(), threading.Event()
    open_path = Path.open

    def slow_open(path, *args, **kwargs):
        if path == server.work_dir / "session_releases.jsonl":
            entered.set()
            assert resume.wait(timeout=5)
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", slow_open)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future = executor.submit(server._run_supervised_session, "finishing", 0)
        server.records["finishing"].future = future
        try:
            assert entered.wait(timeout=2)
            assert executor.submit(server.get_session, "other").result(timeout=1)["status"] == "running"
            result = executor.submit(server.get_session, "finishing").result(timeout=1)
            assert result["status"] == "running"  # Legacy clients must not accept an unfinished result either.
            assert result["resource_release"]["status"] == "unknown"
        finally:
            resume.set()
            future.result(timeout=2)
    assert server.get_session("finishing")["resource_release"]["status"] == "released"
    assert server.get_session("finishing")["status"] == "completed"


def test_session_event_write_error_preserves_failure_and_physical_proof(supervised_server, monkeypatch):
    server = supervised_server

    def fail_write(*_args):
        raise OSError("event storage unavailable")

    monkeypatch.setattr(server, "_write_session_event", fail_write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(server._run_supervised_session, "finishing", 0)
        server.records["finishing"].future = future
        future.result(timeout=2)
    result = server.get_session("finishing")
    assert result["status"] == "failed"
    assert "event storage unavailable" in result["error"]
    assert result["resource_release"]["status"] == "released"
