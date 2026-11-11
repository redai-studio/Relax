# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
POLICY = "qwen35-9b-elastic-autoscaler.yaml"


def _copy_recipe(tmp_path: Path, mode: str) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "checkout"
    directory = checkout / "scripts/training/text"
    directory.mkdir(parents=True)
    recipe = f"run-qwen35-9B-elastic-rollout-{mode}.sh"
    script = directory / recipe
    script.write_text((ROOT / "scripts/training/text" / recipe).read_text())
    models = checkout / "scripts/models"
    models.mkdir()
    (models / "qwen35-9B.sh").write_text((ROOT / "scripts/models/qwen35-9B.sh").read_text())
    configs = directory / "configs"
    configs.mkdir()
    policy = configs / POLICY
    policy.write_text((ROOT / "scripts/training/text/configs" / POLICY).read_text())
    return checkout, script, policy


def _environment(tmp_path: Path, checkout: Path) -> dict[str, str]:
    runtime = {"env_vars": {"ENTRYPOINT_MARKER": "1", "LD_LIBRARY_PATH": "/entry/lib"}}
    runtime["worker_process_setup_hook"] = "example.worker_setup"
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        RELAX_ENTRYPOINT_MODE="ray-job",
        RELAX=str(checkout),
        MODEL_CONFIG_DIR=str(checkout / "scripts/models"),
        RUNTIME_ENV_JSON=json.dumps(runtime),
        RAY_NO_WAIT="",
        SUBMISSION_ID="isolated-recipe-test",
        EXP_DIR=str(tmp_path / "output"),
        MODEL_DIR="/models",
        DATA_DIR="/data",
        NUM_ROLLOUT="100",
        CE_ENABLE_AFFINITY="0",
    )
    for key in ("PROJECT_NAME", "EXP_NAME", "DRY_RUN", "RAY_JOBS_ADDRESS", "HOST_IP", "WORKING_DIR"):
        env.pop(key, None)
    return env


@pytest.mark.parametrize(
    ("mode", "host_ip", "address"),
    [
        ("async", "127.0.0.1", None),
        ("hybrid", "127.0.0.1", None),
        ("hybrid", "127.0.0.1", "http://127.0.0.1:18265"),
        ("hybrid", None, None),
    ],
)
def test_elastic_recipe_submits_expected_training_mode(tmp_path, mode, host_ip, address):
    binary = tmp_path / "ray"
    binary.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    checkout, script, policy = _copy_recipe(tmp_path, mode)
    env = _environment(tmp_path, checkout)
    if host_ip:
        env["HOST_IP"] = host_ip
    if address:
        env["RAY_JOBS_ADDRESS"] = address
    if mode == "hybrid":
        env.update(RELAX="/wrong/checkout", MODEL_CONFIG_DIR="/wrong/models")
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    command = argv[argv.index("--") + 1 :]
    assert command[:3] == ["python3", "-m", "relax.entrypoints.train"]
    resource = json.loads(command[command.index("--resource") + 1])
    assert resource["actor"] == [1, 8 if mode == "hybrid" else 4]
    assert resource["rollout"] == [1, 2]
    assert ("advantages" in resource) == (mode == "async")
    assert ("--hybrid" in command) == (mode == "hybrid")
    assert ("--fully-async" in command) == (mode == "async")
    assert ("--ref-actor-config" in command) == (mode == "async")
    assert command[command.index("--tb-project-name") + 1] == f"Relax/CE/run-qwen35-9B-elastic-rollout-{mode}"
    assert command[command.index("--tb-experiment-name") + 1].startswith(f"qwen35-9B-elastic-{mode}-")
    assert command[command.index("--num-rollout") + 1] == "100"
    assert "--no-enable-affinity" in command
    assert Path(command[command.index("--autoscaler-config") + 1]) == policy
    runtime_arg = next(arg for arg in argv if arg.startswith("--runtime-env-json="))
    runtime = json.loads(runtime_arg.split("=", 1)[1])
    assert runtime["env_vars"]["NCCL_IB_DISABLE"] == "1"
    assert runtime["env_vars"]["ENTRYPOINT_MARKER"] == "1"
    assert runtime["env_vars"]["LD_LIBRARY_PATH"] == "/entry/lib"
    assert runtime["worker_process_setup_hook"] == "example.worker_setup"
    if mode == "hybrid":
        assert argv[argv.index("--working-dir") + 1] == str(checkout)
        assert argv[argv.index("--submission-id") + 1] == "isolated-recipe-test"
        if address or host_ip:
            assert argv[argv.index("--address") + 1] == (address or f"http://{host_ip}:8265")
        else:
            assert "--address" not in argv


@pytest.mark.parametrize("explicit_inputs", [True, False])
def test_hybrid_dry_run_needs_no_ray_gpu_or_output_writes(tmp_path, explicit_inputs):
    checkout, script, _ = _copy_recipe(tmp_path, "hybrid")
    for name in ("ray", "nvidia-smi"):
        binary = tmp_path / name
        binary.write_text("#!/bin/bash\nexit 97\n")
        binary.chmod(0o755)
    env = _environment(tmp_path, checkout)
    env.pop("RELAX_ENTRYPOINT_MODE")
    env.pop("MEGATRON", None)
    env.update(DRY_RUN="1", RELAX="/wrong/checkout", MODEL_CONFIG_DIR="/wrong/models")
    if not explicit_inputs:
        env.pop("MODEL_DIR")
        env.pop("DATA_DIR")
    before = set(tmp_path.rglob("*"))
    result = subprocess.run(
        ["bash", str(script), "--seed", "7"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["argv"][:3] == ["python3", "-m", "relax.entrypoints.train"]
    assert preview["argv"][-2:] == ["--seed", "7"]
    assert "--hybrid" in preview["argv"]
    model_root = "/models" if explicit_inputs else env["EXP_DIR"]
    data_root = "/data" if explicit_inputs else env["EXP_DIR"]
    assert preview["argv"][preview["argv"].index("--hf-checkpoint") + 1] == f"{model_root}/Qwen3.5-9B"
    assert preview["argv"][preview["argv"].index("--prompt-data") + 1] == (
        f"{data_root}/dapo-math-17k/dapo-math-17k.jsonl"
    )
    assert preview["runtime_env"]["worker_process_setup_hook"] == "example.worker_setup"
    assert set(tmp_path.rglob("*")) == before


def test_hybrid_delegates_to_its_checkout_entrypoint(tmp_path):
    checkout, script, _ = _copy_recipe(tmp_path, "hybrid")
    entrypoint = checkout / "scripts/entrypoint/ray-job.sh"
    entrypoint.parent.mkdir()
    entrypoint.write_text('python3 - "$RELAX" "$@" <<\'PY\'\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\nPY\n')
    env = _environment(tmp_path, checkout)
    env.pop("RELAX_ENTRYPOINT_MODE")
    env["RELAX"] = "/wrong/checkout"
    result = subprocess.run(
        ["bash", str(script), "--seed", "7"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(checkout), str(script), "--seed", "7"]
