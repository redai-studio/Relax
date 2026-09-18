# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "tools" / "benchmark_sglang_router_policy.py"
_ENTRY_PATH = _REPO_ROOT / "scripts" / "training" / "multimodal" / "run-qwen3-vl-4B-8xgpu.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_sglang_router_policy", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_module()


def _port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _run_entry_script(tmp_path, *, policy=None, extra_args=()):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ray = fake_bin / "ray"
    fake_ray.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TASK24_RAY_ARGS_FILE'], 'w', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n",
        encoding="utf-8",
    )
    fake_ray.chmod(0o755)

    model_config_dir = tmp_path / "model-config"
    model_config_dir.mkdir()
    (model_config_dir / "qwen3-vl-4B.sh").write_text("MODEL_ARGS=()\n", encoding="utf-8")

    args_file = tmp_path / "ray-args.json"
    environment = os.environ.copy()
    environment.update(
        {
            "EXP_DIR": str(tmp_path / "exp"),
            "MODEL_CONFIG_DIR": str(model_config_dir),
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "RELAX_ENTRYPOINT_MODE": "1",
            "RUNTIME_ENV_JSON": "{}",
            "TASK24_RAY_ARGS_FILE": str(args_file),
        }
    )
    if policy is None:
        environment.pop("SGLANG_ROUTER_POLICY", None)
    else:
        environment["SGLANG_ROUTER_POLICY"] = policy

    result = subprocess.run(
        ["bash", str(_ENTRY_PATH), *extra_args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    args = json.loads(args_file.read_text(encoding="utf-8")) if args_file.exists() else []
    return result, args


def test_default_workload_matches_rollout_shape():
    args = benchmark.build_parser().parse_args([])
    assert args.workers == 8
    assert args.groups == 64
    assert args.samples_per_group == 8
    assert args.concurrency == 512
    assert args.groups * args.samples_per_group == 512
    assert args.policies is None


def test_parser_rejects_unknown_policy():
    with pytest.raises(SystemExit, match="2"):
        benchmark.build_parser().parse_args(["--policy", "least_busy"])


def test_entry_script_rejects_unknown_policy_before_setup():
    if shutil.which("bash") is None:
        pytest.skip("bash is not available")
    environment = os.environ.copy()
    environment["SGLANG_ROUTER_POLICY"] = "least_busy"
    result = subprocess.run(
        ["bash", str(_ENTRY_PATH)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "SGLANG_ROUTER_POLICY must be cache_aware or round_robin" in result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_entry_script_defaults_to_cache_aware(tmp_path):
    result, args = _run_entry_script(tmp_path)

    assert result.returncode == 0, result.stderr
    policy_index = args.index("--sglang-router-policy")
    assert args[policy_index + 1] == "cache_aware"
    assert args.count("--sglang-router-policy") == 1


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
def test_entry_script_forwards_round_robin_and_extra_args(tmp_path):
    extra_args = ["--rollout-seed", "7", "--sglang-enable-deterministic-inference"]
    result, args = _run_entry_script(tmp_path, policy="round_robin", extra_args=extra_args)

    assert result.returncode == 0, result.stderr
    policy_index = args.index("--sglang-router-policy")
    assert args[policy_index + 1] == "round_robin"
    assert args.count("--sglang-router-policy") == 1
    assert args[-len(extra_args) :] == extra_args


def test_mock_workers_are_closed_when_context_exits_with_error():
    pool = benchmark.MockWorkerPool(2)
    with pytest.raises(RuntimeError, match="stop now"):
        with pool:
            ports = list(pool.ports)
            for port in ports:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    assert response.status == 200
            raise RuntimeError("stop now")

    assert pool.stopped
    assert all(not thread.is_alive() for thread in pool.threads)
    assert all(not _port_is_open(port) for port in ports)


def test_worker_cleanup_runs_when_router_cleanup_raises(monkeypatch):
    events = []

    class FakeRecorder:
        def snapshot(self):
            return []

    class FakePool:
        def __init__(self, _workers, _recorder):
            self.urls = ["http://127.0.0.1:1"]
            self.ports = []
            self.stopped = False

        def start(self):
            return self

        def stop(self):
            events.append("workers")
            self.stopped = True

    class FakeRouter:
        def __init__(self, *_args, **_kwargs):
            self.port = 1
            self.stopped = False

        def start(self):
            return None

        def wait_ready(self, _workers, _timeout):
            return None

        def stop(self):
            events.append("router")
            raise RuntimeError("router cleanup failed")

    monkeypatch.setattr(benchmark, "router_available", lambda: True)
    monkeypatch.setattr(benchmark, "RequestRecorder", FakeRecorder)
    monkeypatch.setattr(benchmark, "MockWorkerPool", FakePool)
    monkeypatch.setattr(benchmark, "RouterProcess", FakeRouter)
    monkeypatch.setattr(benchmark, "_send_workload", lambda *_args: ([], [], 0.0))
    monkeypatch.setattr(
        benchmark,
        "_summarize",
        lambda *_args: {"complete": True},
    )

    with pytest.raises(RuntimeError, match="router cleanup failed"):
        benchmark.run_policy("cache_aware", groups=1, samples_per_group=1, workers=1, concurrency=1)

    assert events == ["router", "workers"]


@pytest.mark.skipif(not benchmark.router_available(), reason="sglang_router is not installed")
def test_real_router_runs_both_policies_and_leaves_no_listeners():
    result = benchmark.run_benchmark(
        ["cache_aware", "round_robin"],
        groups=8,
        samples_per_group=8,
        workers=8,
        concurrency=64,
    )

    assert result["complete"]
    assert [run["policy"] for run in result["runs"]] == ["cache_aware", "round_robin"]
    for run in result["runs"]:
        assert run["expected_requests"] == 64
        assert run["completed_requests"] == 64
        assert run["failed_requests"] == 0
        assert run["complete_groups"] == 8
        assert sum(run["request_count_per_worker"].values()) == 64
        assert run["cleanup"]["router_stopped"]
        assert run["cleanup"]["workers_stopped"]
        ports = [run["cleanup"]["router_port"], *run["cleanup"]["worker_ports"]]
        assert all(not _port_is_open(port) for port in ports)

    round_robin = result["runs"][1]
    assert list(round_robin["request_count_per_worker"].values()) == [8] * 8
