# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU regressions for native job status, retries and owned process cleanup."""

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.retool.evaluation import completed_evaluation, evaluation_attempt, trained_checkpoint
from examples.retool.lifecycle import close_owned_children, require_job_success


@pytest.fixture
def fake_ray(tmp_path):
    package = tmp_path / "ray"
    package.mkdir()
    (package / "__init__.py").write_text("")
    module = package / "job_submission.py"
    return module, dict(os.environ, PYTHONPATH=str(tmp_path))


@pytest.mark.parametrize("status", ["SUCCEEDED", "STOPPED", "FAILED", "RUNNING"])
def test_native_status_is_required_even_after_cli_success(tmp_path, fake_ray, status):
    module, env = fake_ray
    module.write_text(
        "class JobSubmissionClient:\n"
        "    def __init__(self, address): assert address == 'http://localhost:8265'\n"
        f"    def get_job_status(self, job): assert job == 'owned-job'; return {status!r}\n"
    )
    receipt = tmp_path / "job.json"
    env.update(RAY_ADDRESS="http://localhost:8265", RETOOL_RAY_SUBMISSION_ID="owned-job")
    if status == "SUCCEEDED":
        require_job_success(env, receipt, time.time() + 10)
    else:
        with pytest.raises(RuntimeError, match="expected SUCCEEDED"):
            require_job_success(env, receipt, time.time() + 10)
    assert json.loads(receipt.read_text())["status"] == status


@pytest.mark.parametrize("stall", ["constructor", "status"])
def test_http_stall_is_killed_and_reaped_before_cleanup(tmp_path, fake_ray, stall, monkeypatch):
    module, env = fake_ray
    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" + stall:
                entered.set()
                release.wait(10)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    module.write_text(
        "import urllib.request\n"
        "class JobSubmissionClient:\n"
        "    def __init__(self, address):\n"
        "        self.address = address\n"
        "        urllib.request.urlopen(address + '/constructor').read()\n"
        "    def get_job_status(self, job):\n"
        "        urllib.request.urlopen(self.address + '/status').read()\n"
        "        return 'SUCCEEDED'\n"
    )
    env.update(RAY_ADDRESS=f"http://127.0.0.1:{server.server_port}", RETOOL_RAY_SUBMISSION_ID="owned-job")
    children = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", start)
    cleaned = []
    started = time.monotonic()
    try:
        try:
            with pytest.raises(RuntimeError, match="run deadline"):
                require_job_success(env, tmp_path / "job.json", time.time() + 2)
        finally:
            cleaned.append(True)
        assert entered.is_set()
        assert cleaned == [True]
        assert len(children) == 1 and children[0].returncode is not None
        assert time.monotonic() - started < 5
        assert json.loads((tmp_path / "job.json").read_text())["error"] == "TimeoutExpired"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_timeout_does_not_skip_other_owned_children(monkeypatch):
    signals = []
    monkeypatch.setattr("os.killpg", lambda pgid, sig: signals.append((pgid, sig)))

    class Child:
        def __init__(self, pid, stubborn=False):
            self.pid, self.stubborn, self.waits = pid, stubborn, 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.waits += 1
            # Every sibling must be signalled before blocking on the first wait.
            assert (11, signal.SIGTERM) in signals and (12, signal.SIGTERM) in signals
            if self.stubborn and self.waits == 1:
                raise subprocess.TimeoutExpired("owned", timeout)
            return -9 if self.stubborn else 0

    ray, sandbox = Child(11), Child(12, stubborn=True)
    logs = [io.StringIO(), io.StringIO()]
    helpers = SimpleNamespace(identity=lambda pid: {"pgid": pid})
    records = close_owned_children(
        [("ray", ray, {"pgid": 11}, logs[0]), ("sandbox", sandbox, {"pgid": 12}, logs[1])], helpers
    )
    assert (12, signal.SIGKILL) in signals
    assert ray.waits == 1 and sandbox.waits == 2
    assert all(record["actual_parent_wait"] for record in records)
    assert all(log.closed for log in logs)


def test_changed_identity_is_not_killed_and_cleanup_continues(monkeypatch):
    signals = []
    monkeypatch.setattr("os.killpg", lambda pgid, sig: signals.append((pgid, sig)))

    class Child:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def wait(self, timeout):
            if self.pid == 12:
                raise subprocess.TimeoutExpired("owned", timeout)
            return 0

    helpers = SimpleNamespace(identity=lambda pid: {"pgid": pid, "start": 2})
    records = close_owned_children(
        [(str(pid), Child(pid), {"pgid": pid, "start": 1}, io.StringIO()) for pid in (11, 12)], helpers
    )
    assert signals == []
    assert records[0]["actual_parent_wait"] is False
    assert records[1]["actual_parent_wait"] is True


def test_completed_rl_recovery_uses_original_export_and_exit(tmp_path):
    original, recovery = tmp_path / "original", tmp_path / "recovery"
    (original / "gpu_rl").mkdir(parents=True)
    recovery.mkdir()
    (original / "gpu_rl/training_actual_wait.json").write_text(json.dumps(dict(actual_parent_wait=True, returncode=0)))
    (original / "checkpoint_rl.log").write_text("200\n")
    (original / "gpu_rl/ray_job_status.json").write_text(json.dumps(dict(submission_id="rl", status="SUCCEEDED")))
    (original / "gpu_sft").mkdir()
    (original / "gpu_sft/training_actual_wait.json").write_text(
        json.dumps(dict(actual_parent_wait=True, returncode=0))
    )
    (original / "gpu_sft/ray_job_status.json").write_text(json.dumps(dict(submission_id="sft", status="SUCCEEDED")))
    (original / "checkpoint_sft.log").write_text("50\n")
    export = original / "checkpoints/rl/hf_199"
    (recovery / "RECOVERY_PLAN.json").write_text(
        json.dumps(
            dict(
                rl=dict(action="SKIP_COMPLETE", completed=200, hf=dict(path=str(export)), original=str(export.parent)),
                sft=dict(action="SKIP_COMPLETE", completed=50, original=str(original / "checkpoints/sft")),
            )
        )
    )
    assert trained_checkpoint(recovery, 200) == export
    assert not (recovery / "checkpoints").exists()
    for mode in ("sft", "rl"):
        status = original / f"gpu_{mode}/ray_job_status.json"
        success = status.read_text()
        for value in (dict(submission_id=mode, status="STOPPED"), dict(status="SUCCEEDED")):
            status.write_text(json.dumps(value))
            with pytest.raises(AssertionError, match="native Ray success"):
                trained_checkpoint(recovery, 200)
        status.unlink()
        with pytest.raises(FileNotFoundError):
            trained_checkpoint(recovery, 200)
        status.write_text(success)
    (original / "gpu_rl/training_actual_wait.json").write_text(json.dumps(dict(actual_parent_wait=True, returncode=1)))
    with pytest.raises(AssertionError, match="RL did not exit successfully"):
        trained_checkpoint(recovery, 200)


def test_partial_retry_reserves_all_new_paths(tmp_path):
    root, temp, sandbox = [tmp_path / name for name in ("root", "temp", "sandbox")]
    temp.mkdir()
    first = evaluation_attempt(root, temp, sandbox, "official")
    (first[0] / "partial.txt").write_text("preserved")
    second = evaluation_attempt(root, temp, sandbox, "official")
    assert all(a != b for a, b in zip(first, second))
    assert (first[0] / "partial.txt").read_text() == "preserved"
    assert second[0].name == "attempt0001"


def test_reuse_requires_panels_model_and_normal_exit(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    value = dict(
        status="ALL_NATIVE_K8_PANELS_ACCEPTED",
        model="model",
        K=8,
        smoke=True,
        panels=dict(gsm8k=dict(questions=1, trajectories=8)),
    )
    for name, receipt in {
        "evaluation_complete": value,
        "driver_wait": dict(actual_parent_wait=True, returncode=0),
        "closed": dict(owned_PG_empty=True, children=[dict(actual_parent_wait=True)]),
    }.items():
        (output / f"{name}.json").write_text(json.dumps(receipt))
    assert completed_evaluation(output, "model", {"gsm8k": 1}, True) is None
    status = output / "ray_job_status.json"
    status.write_text(json.dumps(dict(status="STOPPED")))
    assert completed_evaluation(output, "model", {"gsm8k": 1}, True) is None
    status.write_text(json.dumps(dict(status="SUCCEEDED")))
    assert completed_evaluation(output, "model", {"gsm8k": 1}, True) == value
    assert completed_evaluation(output, "other-model", {"gsm8k": 1}, True) is None
    assert completed_evaluation(output, "model", {"gsm8k": 2}, True) is None


@pytest.fixture
def short_runtime():
    with tempfile.TemporaryDirectory(prefix="rt", dir="/tmp") as directory:
        yield Path(directory)


def test_bounded_fresh_smoke_reaches_sft_stage(tmp_path, monkeypatch, short_runtime):
    from examples.retool import recovery, workflow

    root, temp, registry = tmp_path / "run", short_runtime, tmp_path / "registry"
    for path in (root, registry):
        path.mkdir()
    helpers = SimpleNamespace(
        REG=registry,
        identity=lambda pid: {"pid": pid, "pgid": pid},
        claim_paths=lambda *args: [],
        group_members=lambda pgid: [],
        resources=lambda: {"live_unreleased_claims": []},
    )
    monkeypatch.setattr(workflow, "load_module", lambda *args: helpers)
    monkeypatch.setattr(recovery, "hf_identity", lambda path: {})
    monkeypatch.setattr(workflow.signal, "signal", lambda *args: None)
    monkeypatch.setenv("RETOOL_SOURCE_COMMIT", "test-commit")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "--root",
            str(root),
            "--temp",
            str(temp),
            "--model",
            "model",
            "--deadline",
            "9999999999",
            "--sandbox-base",
            str(tmp_path / "sandbox"),
            "--eval-config",
            str(tmp_path / "eval.json"),
            "--smoke",
            "--stop-after-updates",
            "3",
        ],
    )

    class ReachedSFT(Exception):
        pass

    def start(command, **kwargs):
        assert (root / "gpu_sft/admission.json").is_file()
        assert "--head" in command
        raise ReachedSFT

    monkeypatch.setattr(workflow.subprocess, "Popen", start)
    with pytest.raises(ReachedSFT):
        workflow.main()


@pytest.mark.parametrize("complete_mode", ["sft", "rl"])
@pytest.mark.parametrize("status", ["SUCCEEDED", "STOPPED", None])
def test_recovery_checks_completed_source_native_status(tmp_path, monkeypatch, complete_mode, status):
    from examples.retool import recovery

    parent, output = tmp_path / "parent", tmp_path / "next"
    parent.mkdir()
    (parent / "test_contract.json").write_text(
        json.dumps(
            dict(
                source_commit=recovery.COMMIT,
                sft_updates=50,
                rl_updates=200,
                sft_pool=2000,
                rl_pool=17903,
                rl_batch=64,
                K=16,
            )
        )
    )
    selected = {}
    for mode, target in (("sft", 50), ("rl", 200)):
        completed = target if mode == "sft" or complete_mode == "rl" else 10
        selected[mode] = dict(completed=completed, original=str(parent / "checkpoints" / mode))
        stage = parent / f"gpu_{mode}"
        stage.mkdir()
        (stage / "training_actual_wait.json").write_text(json.dumps(dict(actual_parent_wait=True, returncode=0)))
        (parent / f"checkpoint_{mode}.log").write_text(str(target))
        value = status if mode == complete_mode else "SUCCEEDED"
        if value is not None:
            (stage / "ray_job_status.json").write_text(json.dumps(dict(submission_id=mode, status=value)))
    monkeypatch.setattr(recovery, "select_checkpoint", lambda roots, mode, *args: selected[mode])
    run = lambda: recovery.plan(parent, output, 50, 200, hf_validator=lambda path: dict(path=str(path)))
    if status == "SUCCEEDED":
        result = run()
        assert result[complete_mode]["action"] == "SKIP_COMPLETE"
        if complete_mode == "sft":
            assert result["rl"]["action"] == "RESUME_NATIVE"
    elif status is None:
        with pytest.raises(FileNotFoundError):
            run()
    else:
        with pytest.raises(AssertionError, match="native Ray success"):
            run()
