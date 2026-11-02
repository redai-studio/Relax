# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check the public Qwen3.8 launch commands without starting Ray or loading
GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "examples/models/qwen3.8-flash-next/scripts"


def _dry_run(script: str, checkpoint: Path, **extra: str) -> dict:
    env = dict(
        os.environ,
        DRY_RUN="1",
        MODEL_DIR="/models",
        DATA_DIR="/data",
        SAVE_DIR=str(checkpoint),
        EXP_NAME="qwen38-test",
        MEGATRON="/image/megatron",
        RELAX_HYBRID_EP_COMMIT="abc123def",
        **extra,
    )
    result = subprocess.run(["bash", str(SCRIPTS / script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _value(argv: list[str], flag: str) -> str:
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


@pytest.mark.parametrize(
    "script,gpus,tp,pp,ep",
    [
        ("run-qwen38-qsa-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
        ("run-qwen38-flash-next-openr1mm-128xgpu.sh", 128, 8, 8, 16),
    ],
)
def test_launchers_preserve_topology_and_resumable_checkpoint(tmp_path, script, gpus, tp, pp, ep):
    first = _dry_run(script, tmp_path / "checkpoints")
    second = _dry_run(script, tmp_path / "checkpoints")
    args = first["argv"]
    resources = json.loads(_value(args, "--resource"))
    assert resources["actor"] == [1, gpus]
    assert _value(args, "--tensor-model-parallel-size") == str(tp)
    assert _value(args, "--pipeline-model-parallel-size") == str(pp)
    assert _value(args, "--expert-model-parallel-size") == str(ep)
    assert _value(args, "--num-layers") == "48"
    assert _value(args, "--save") == _value(args, "--load") == _value(second["argv"], "--load")
    assert _value(args, "--save").startswith(str(tmp_path / "checkpoints"))
    assert _value(args, "--max-actor-ckpt-to-keep") == "2"
    assert "--use-clearml" in args and "--use-metrics-service" in args
    assert not (tmp_path / "checkpoints").exists(), "Dry run must not create output directories"


def test_h800_rollout_and_driver_receive_precision_and_model_paths(tmp_path):
    config = _dry_run("run-qwen38-flash-next-openr1mm-128xgpu.sh", tmp_path / "checkpoints")
    args = config["argv"]
    assert json.loads(_value(args, "--resource"))["rollout"] == [1, 128]
    assert _value(args, "--advantage-estimator") == "grpo"
    assert _value(args, "--rollout-num-gpus-per-engine") == "8"
    assert _value(args, "--moe-token-dispatcher-type") == "alltoall"
    assert _value(args, "--prompt-data").startswith("/data/")
    env = config["runtime_env"]["env_vars"]
    assert env["MEGATRON"] == "/image/megatron"
    assert env["RELAX_QWEN38_PLE_STREAMED"] == "0"
    assert env["RELAX_QWEN38_FP32_ROUTER"] == "1"
    assert env["QWEN48_PLE_CPU_OFFLOAD"] == "0"
