# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Capture launcher arguments without contacting Ray or running training."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "balance,rotation,seed,runtime_env",
    [
        ("0", "0", None, None),
        ("1", "0", "7", '{"env_vars":{"OMP_NUM_THREADS":"2"}}'),
        ("1", "1", None, None),
    ],
)
@pytest.mark.parametrize("gpus,batch", [(8, 8), (16, 32), (32, 32)])
def test_fixed_tiers_launcher_seed_and_runtime_json(seed, runtime_env, gpus, batch, balance, rotation):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        MODEL_PATH="/model",
        DATA_ROOT="/data",
        METADATA_PATH="/metadata",
        DATASET_SIZE="1",
        SAVE_DIR="/unused",
        FIXED_PIXELS_TIERS="2073600,4194304",
        IMAGE_PREFETCH_DEPTH="4",
        IMAGE_PREFETCH_WORKERS="2",
        GPUS=str(gpus),
        GPUS_PER_NODE="8",
        GLOBAL_BATCH_SIZE=str(batch),
        JOB_ID="sft-test",
        DRIVER_PYTHONPATH="/shared/megatron:/shared/relax",
        IMAGE_DP_TIER_MICRO_BALANCE=balance,
        IMAGE_DP_TIER_RANK_ROTATION=rotation,
    )
    for key, value in (("SEED", seed), ("RUNTIME_ENV_JSON", runtime_env)):
        env.pop(key, None)
        if value is not None:
            env[key] = value
    command = """
ray() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"
}
export -f ray
bash "$1"
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "capture",
            str(root / "scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(result.stdout)
    config = json.loads(args[args.index("--i2i-sft-config") + 1])
    assert config["fixed_pixels_tiers"] == [2073600, 4194304]
    assert config["seed"] == int(seed or "42")
    assert config["image_prefetch_depth"] == 4
    assert config["image_prefetch_workers"] == 2
    if balance == "1":
        assert config["dp_tier_micro_balance"] is True
    else:
        assert "dp_tier_micro_balance" not in config
    if rotation == "1":
        assert config["dp_tier_rank_rotation"] is True
    else:
        assert "dp_tier_rank_rotation" not in config
    assert int(args[args.index("--seed") + 1]) == config["seed"]
    assert json.loads(args[args.index("--runtime-env-json") + 1]) == json.loads(runtime_env or "{}")
    assert int(args[args.index("--global-batch-size") + 1]) == batch
    assert int(args[args.index("--rollout-batch-size") + 1]) == batch
    assert int(args[args.index("--actor-num-nodes") + 1]) == gpus // 8
    assert int(args[args.index("--actor-num-gpus-per-node") + 1]) == 8
    assert int(args[args.index("--num-gpus-per-node") + 1]) == 8
    assert json.loads(args[args.index("--resource") + 1])["actor"] == [1, gpus]
    assert args[args.index("--submission-id") + 1] == "sft-test"
    assert args[args.index("--") + 1 : args.index("--") + 3] == ["env", "PYTHONPATH=/shared/megatron:/shared/relax"]


@pytest.mark.parametrize(
    "rotation,balance,error",
    [
        *(("0", value, "IMAGE_DP_TIER_MICRO_BALANCE must be 0 or 1") for value in ("true", "2", "typo")),
        *((value, "1", "IMAGE_DP_TIER_RANK_ROTATION must be 0 or 1") for value in ("true", "2", "typo")),
        ("1", "0", "IMAGE_DP_TIER_RANK_ROTATION requires IMAGE_DP_TIER_MICRO_BALANCE=1"),
    ],
)
def test_fixed_tiers_launcher_rejects_invalid_tier_flags_before_submission(rotation, balance, error):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        MODEL_PATH="/model",
        DATA_ROOT="/data",
        METADATA_PATH="/meta",
        DATASET_SIZE="1",
        IMAGE_DP_TIER_MICRO_BALANCE=balance,
        IMAGE_DP_TIER_RANK_ROTATION=rotation,
        GPUS="16",
        GPUS_PER_NODE="8",
        GLOBAL_BATCH_SIZE="32",
    )
    result = subprocess.run(
        ["bash", str(root / "scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert error in result.stderr
