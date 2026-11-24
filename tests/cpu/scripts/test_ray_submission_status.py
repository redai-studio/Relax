# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Public entrypoint status handling without a cluster or GPU."""

import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/entrypoint/ray-bin/ray"


@pytest.fixture
def wrapper(monkeypatch):
    pytest.importorskip("ray.dashboard.modules.job.cli")
    loader = importlib.machinery.SourceFileLoader("ray_status_wrapper", str(WRAPPER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setenv("RELAX_REAL_RAY", "/native/ray")
    return module


@pytest.mark.parametrize("code", [0, 7, -15])
@pytest.mark.parametrize("no_wait", [False, True])
def test_submit_preserves_native_failure_and_async_semantics(wrapper, monkeypatch, code, no_wait):
    argv = ["ray", "job", "submit", "--address=http://test", "--submission-id", "chosen"]
    if no_wait:
        argv.append("--no-wait")
    argv += ["--", "python", "a script.py", "--no-wait", "--submission-id=training-arg"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("RAY_NO_WAIT", "1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=(code if code >= 0 else 128 - code) if len(calls) == 1 else 1)

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    monkeypatch.setattr(wrapper, "run_native", lambda command, **kwargs: run(command, **kwargs).returncode)
    result = wrapper.main()
    assert calls[0][0] == ["/native/ray", *argv[1:]]
    if code or no_wait:
        assert result == (code if code >= 0 else 128 - code)
        assert len(calls) == 1
    else:
        assert result == 1
        payload = json.loads(calls[1][1]["input"])
        assert payload["submission_id"] == "chosen"
        assert payload["address"] == "http://test"
        assert calls[1][1]["timeout"] == 60


@pytest.mark.parametrize("id_args", [[], ["--job-id=legacy"], ["--submission-id=explicit"]])
def test_ids_auth_and_entrypoint_arguments(wrapper, monkeypatch, id_args):
    argv = ["ray", "job", "submit", *id_args, "--headers", '{"Authorization":"test"}', "--verify", "false"]
    argv += ["--", "python", "file with spaces.py", "--submission-id", "not-a-job-id"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    monkeypatch.setattr(wrapper, "run_native", lambda command, **kwargs: run(command, **kwargs).returncode)
    assert wrapper.main() == 0
    payload = json.loads(calls[1][1]["input"])
    assert payload["headers"] == '{"Authorization":"test"}'
    assert payload["verify"] is False
    if not id_args:
        assert payload["submission_id"].startswith("relax-")
        assert calls[0][0][3:5] == ["--submission-id", payload["submission_id"]]
    else:
        assert payload["submission_id"] == id_args[0].split("=")[1]
    assert calls[0][0][-5:] == argv[-5:]


@pytest.mark.parametrize("status", ["SUCCEEDED", "STOPPED", "FAILED", "RUNNING", "PENDING"])
def test_only_successful_terminal_state_passes(wrapper, monkeypatch, status):
    from ray.dashboard.modules.job import cli

    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"address": "http://test", "submission_id": "job"})))
    calls = []

    def get_status(job):
        calls.append(job)
        return status

    monkeypatch.setattr(cli, "_get_sdk_client", lambda **kwargs: SimpleNamespace(get_job_status=get_status))
    if status == "SUCCEEDED":
        wrapper.check_status()
    else:
        with pytest.raises(SystemExit, match="expected SUCCEEDED"):
            wrapper.check_status()
    assert calls == ["job"]


def test_status_timeout_is_failure(wrapper, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ray", "job", "submit", "--", "true"])

    def run(command, **kwargs):
        if "timeout" in kwargs:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    monkeypatch.setattr(wrapper, "run_native", lambda command, **kwargs: run(command, **kwargs).returncode)
    assert wrapper.main() == 1


def test_help_does_not_submit(wrapper, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ray", "job", "submit", "--help"])
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *a, **kw: pytest.fail("must not submit"))
    assert wrapper.main() == 0


def test_path_wrapper_supports_command_subshell_and_stdbuf(tmp_path):
    native = tmp_path / "ray"
    native.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nexit 7\n')
    native.chmod(0o755)
    helper = ROOT / "scripts/entrypoint/ray-status.sh"
    for invocation in ("ray status", "command ray status", "bash -c 'ray status'", "stdbuf -oL ray status"):
        result = subprocess.run(
            [
                "bash",
                "-c",
                'set -eo pipefail; source "$1"; source "$1"; ' + invocation + " | cat",
                "bash",
                str(helper),
            ],
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
            text=True,
            capture_output=True,
        )
        assert result.returncode == 7, result.stderr
        assert result.stdout == "status\n"


def test_cross_checkout_source_keeps_native_ray(tmp_path):
    import shutil

    native = tmp_path / "bin/ray"
    native.parent.mkdir()
    native.write_text("#!/bin/sh\necho native\n")
    native.chmod(0o755)
    second = tmp_path / "other"
    second.mkdir()
    shutil.copy(ROOT / "scripts/entrypoint/ray-status.sh", second / "ray-status.sh")
    shutil.copytree(WRAPPER.parent, second / "ray-bin")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -eo pipefail; source "$1"; source "$2"; ray status',
            "bash",
            str(ROOT / "scripts/entrypoint/ray-status.sh"),
            str(second / "ray-status.sh"),
        ],
        env={**os.environ, "PATH": f"{native.parent}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "native\n"


def test_sigterm_forwards_and_reaps_native_cli(tmp_path):
    import signal
    import time

    pidfile = tmp_path / "child.pid"
    child = "import os,time;from pathlib import Path;Path(%r).write_text(str(os.getpid()));time.sleep(60)" % str(
        pidfile
    )
    launcher = (
        "import runpy,sys;ns=runpy.run_path(sys.argv[1]);sys.exit(ns['run_native']([sys.executable,'-c',sys.argv[2]]))"
    )
    process = subprocess.Popen([sys.executable, "-c", launcher, str(WRAPPER), child])
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        child_pid = int(pidfile.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
