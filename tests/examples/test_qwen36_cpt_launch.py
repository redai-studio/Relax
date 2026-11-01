# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate CPT arguments and Ray CLI submission without Ray or GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/training/cpt/run-qwen3.6-35B-A3B-cpt-256k-64xgpu.sh"


def render(**changes):
    environment = dict(os.environ, MODEL_DIR="/model", DATA_DIR="/data", SAVE_DIR="/checkpoints", EXP_NAME="review")
    environment.update(changes)
    return json.loads(subprocess.check_output(["bash", str(SCRIPT), "--dry-run"], env=environment, text=True))


def test_cpt_launch_preserves_delivery_training_configuration():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    expected = {
        "--sft-training-mode": "cpt",
        "--sft-cpt-template": "raw",
        "--input-key": "text",
        "--metadata-key": "",
        "--tool-key": "",
        "--eval-tool-key": "",
        "--sft-oversize-strategy": "skip",
        "--sft-packing-strategy": "binpack",
        "--seq-length": "262144",
        "--max-tokens-per-gpu": "262144",
        "--global-batch-size": "1600",
        "--resource": '{"sft":[1,0],"actor":[1,64]}',
        "--tensor-model-parallel-size": "4",
        "--pipeline-model-parallel-size": "1",
        "--context-parallel-size": "1",
        "--expert-model-parallel-size": "8",
        "--expert-tensor-parallel-size": "1",
        "--mtp-num-layers": "1",
        "--mtp-loss-scaling-factor": "0.1",
        "--lr": "2e-5",
        "--min-lr": "1e-6",
        "--lr-warmup-iters": "10",
        "--lr-decay-iters": "200",
        "--num-rollout": "1000",
        "--save-interval": "100",
        "--max-actor-ckpt-to-keep": "1",
        "--moe-flex-dispatcher-num-sms": "80",
        "--custom-megatron-before-train-step-hook-path": "relax.models.qwen3_5.recipe.before_step",
    }
    for output in (render(), render()):
        argv = output["argv"]
        assert "--sft-wholepack" in argv
        # Model defaults precede recipe settings, so argparse's final value wins.
        for name, value in expected.items():
            assert argv[max(i for i, arg in enumerate(argv) if arg == name) + 1] == value
        assert argv[argv.index("--load") + 1] == argv[argv.index("--save") + 1] == "/checkpoints/review"
        assert "--sft-ignore-empty-think" not in argv and "--apply-chat-template-kwargs" not in argv
        assert "--no-load-optim" not in argv and "--finetune" not in argv
        env = output["runtime_env"]["env_vars"]
        assert json.loads(argv[argv.index("--train-env-vars") + 1]) == env
        assert env["NVTE_BWD_LAYERNORM_SM_MARGIN"] == env["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "20"
        assert env["EP_DISABLE_GIN"] == "1"
        assert "B300_OPT_PREFIX" not in env  # gitleaks:allow -- compatibility literal
        assert "RELAX_SFT_WHOLEPACK" not in env


def test_cpt_launch_preserves_overrides_paths_and_worker_environment(tmp_path):
    output = render(
        MODEL_DIR="/model with spaces",
        DATA_DIR="/data with spaces.jsonl",
        SAVE_DIR=str(tmp_path / "checkpoints"),
        TRAIN_STEPS="7",
        SAVE_INTERVAL="last",
        LR="1e-5",
        MIN_LR="1e-7",
        INPUT_KEY="content",
        CPT_TEMPLATE="qwen3_5",
        RUNTIME_ENV_JSON=json.dumps({"env_vars": {"CUSTOM": "kept", "NCCL_ALGO": "Ring"}, "pip": ["example"]}),
    )
    argv = output["argv"]
    for name, value in {
        "--hf-checkpoint": "/model with spaces",
        "--prompt-data": "/data with spaces.jsonl",
        "--save": str(tmp_path / "checkpoints/review"),
        "--num-rollout": "7",
        "--save-interval": "7",
        "--lr": "1e-5",
        "--min-lr": "1e-7",
        "--input-key": "content",
        "--sft-cpt-template": "qwen3_5",
    }.items():
        assert argv[argv.index(name) + 1] == value
    assert output["runtime_env"]["pip"] == ["example"]
    assert output["runtime_env"]["env_vars"]["CUSTOM"] == "kept"
    assert "NCCL_ALGO" not in output["runtime_env"]["env_vars"]
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("name,value", [("TRAIN_STEPS", "0"), ("SAVE_INTERVAL", "0"), ("CPT_TEMPLATE", "unknown")])
def test_cpt_launch_rejects_invalid_configuration(name, value):
    with pytest.raises(subprocess.CalledProcessError):
        render(**{name: value})


@pytest.mark.parametrize(
    "submit_code,no_wait",
    [(0, ""), (7, ""), (0, "1"), (7, "1")],
)
def test_cpt_submission_propagates_ray_exit_status(tmp_path, submit_code, no_wait):
    path = tmp_path / "ray"
    path.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RAY_RECORD"\nexit "$SUBMIT_CODE"\n')
    path.chmod(0o755)
    environment = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        SUBMISSION_ID="fixed-child",
        RAY_JOBS_ADDRESS="http://ray-test",
        RUNTIME_ENV_JSON="{}",
        RAY_NO_WAIT=no_wait,
        WORKING_DIR="/source with spaces",
        SCRIPT_DIR=str(tmp_path),
        TASK_NAME="test",
        RAY_RECORD=str(tmp_path / "ray-args"),
        SUBMIT_CODE=str(submit_code),
    )
    fragment = "set -euo pipefail\nTRAIN_ARGS=(python3 -m relax.entrypoints.train)\n"
    fragment += "RAY_ARGS=(job submit" + SCRIPT.read_text().split("RAY_ARGS=(job submit", 1)[1]
    result = subprocess.run(["bash", "-c", fragment], env=environment, capture_output=True, text=True)
    assert result.returncode == submit_code, result.stderr
    args = (tmp_path / "ray-args").read_text().splitlines()
    assert args.count("job") == 1
    assert "--submission-id=fixed-child" in args
    assert "--working-dir" in args and "/source with spaces" in args
    assert ("--no-wait" in args) == bool(no_wait)


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_cpt_launch_preserves_explicit_nonfinite_policy(monkeypatch, value):
    monkeypatch.delenv("RELAX_SKIP_NONFINITE_GRAD", raising=False)
    monkeypatch.delenv("RUNTIME_ENV_JSON", raising=False)
    output = render(**({"RELAX_SKIP_NONFINITE_GRAD": value} if value is not None else {}))
    argv = output["argv"]
    driver = output["runtime_env"]["env_vars"]
    worker = json.loads(argv[argv.index("--train-env-vars") + 1])
    for env in (driver, worker):
        assert env["RELAX_SKIP_NONFINITE_GRAD"] == ("1" if value is None else value)
