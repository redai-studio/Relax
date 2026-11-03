# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/training/diffusion/run-sensenova-condition-cache.sh"


def preview(tmp_path, **overrides):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MODEL_PATH": "/models/with spaces/SenseNova",
            "DATA_ROOT": "/data/images",
            "METADATA_PATH": "/data/train.json",
            "DATASET_SIZE": "128",
            "SAVE_DIR": str(tmp_path / "save"),
            "RELAX_CONDITION_CACHE_RUN_DIR": str(tmp_path / "metrics"),
            "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
            "RUNTIME_ENV_JSON": '{"env_vars":{"EXISTING":"yes","PATH":"/worker/bin"},"pip":["existing"]}',
            "DRIVER_PYTHONPATH": "/imports/with spaces",
            "RAY_NO_WAIT": "",
            "RELAX_CONDITION_CACHE": "0",
            "USE_CLEARML": "0",
            "GPUS": "8",
            "GPUS_PER_NODE": "8",
        }
    )
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("cache", ["0", "1"])
def test_preview_propagates_cache_controls_and_preserves_runtime_paths(tmp_path, cache):
    result = preview(tmp_path, RELAX_CONDITION_CACHE=cache, RELAX_CONDITION_CACHE_DIR="/shared/cache dir")
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    runtime = json.loads(args[args.index("--runtime-env-json") + 1])
    assert runtime["env_vars"]["PATH"] == "/worker/bin"
    assert runtime["env_vars"]["EXISTING"] == "yes"
    assert runtime["env_vars"]["RELAX_CONDITION_CACHE"] == cache
    assert runtime["env_vars"]["RELAX_CONDITION_PROFILE"] == "1"
    assert runtime["env_vars"]["RELAX_CONDITION_VERIFY"] == "0"
    assert runtime["env_vars"]["RELAX_CONDITION_VERIFY_RTOL"] == "0.01"
    assert runtime["env_vars"]["RELAX_CONDITION_CACHE_MAX_BYTES"] == "8589934592"
    assert runtime["env_vars"]["RELAX_CONDITION_CACHE_DIR"] == "/shared/cache dir"
    assert runtime["pip"] == ["existing"]
    assert args[args.index("--model-path") + 1] == "/models/with spaces/SenseNova"
    assert args[args.index("--global-batch-size") + 1] == "8"
    assert args[args.index("--micro-batch-size") + 1] == "1"
    assert args[args.index("--fsdp-trainable-mode") + 1] == "lora"
    assert "--no-wait" not in args
    assert not (tmp_path / "metrics").exists()
    assert not (tmp_path / "save").exists()


def test_smaller_gpu_count_preserves_effective_batch_and_explicit_nowait(tmp_path):
    result = preview(tmp_path, GPUS="2", GPUS_PER_NODE="2", RAY_NO_WAIT="1")
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert "--no-wait" in args
    assert args[args.index("--global-batch-size") + 1] == "8"
    assert json.loads(args[args.index("--resource") + 1])["actor"] == [1, 2]
    assert args[args.index("--actor-num-gpus-per-node") + 1] == "2"


def test_condition_prefetch_controls_reach_ray_workers(tmp_path):
    result = preview(
        tmp_path,
        IMAGE_PREFETCH_DEPTH="4",
        RELAX_CONDITION_PREFETCH_DEPTH="2",
        RELAX_CONDITION_PREFETCH_MAX_BYTES="536870912",
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    env = json.loads(args[args.index("--runtime-env-json") + 1])["env_vars"]
    assert env["RELAX_CONDITION_PREFETCH_DEPTH"] == "2"
    assert env["RELAX_CONDITION_PREFETCH_MAX_BYTES"] == "536870912"


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_branch_groups_are_opt_in_without_changing_legacy_contract(tmp_path, enabled):
    result = preview(tmp_path, FSDP_BRANCH_GROUPS=enabled)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    config = json.loads(args[args.index("--i2i-sft-config") + 1])
    if enabled == "1":
        assert config["fsdp_branch_groups"] is True
    else:
        assert "fsdp_branch_groups" not in config


def test_invalid_branch_group_control_fails_before_submission(tmp_path):
    result = preview(tmp_path, FSDP_BRANCH_GROUPS="yes")
    assert result.returncode != 0
    assert "FSDP_BRANCH_GROUPS" in result.stderr


@pytest.mark.parametrize("overrides", [{"IMAGE_PREFETCH_DEPTH": "0"}, {"RELAX_CONDITION_PREFETCH_MAX_BYTES": "0"}])
def test_condition_prefetch_requires_decoded_lookahead_and_capacity(tmp_path, overrides):
    result = preview(tmp_path, **({"IMAGE_PREFETCH_DEPTH": "4", "RELAX_CONDITION_PREFETCH_DEPTH": "2"} | overrides))
    assert result.returncode != 0


def test_verify_flags_propagate_and_perf_can_disable_checkpoints(tmp_path):
    result = preview(
        tmp_path,
        RELAX_CONDITION_VERIFY="1",
        RELAX_CONDITION_VERIFY_RTOL="0",
        RELAX_CONDITION_VERIFY_SNAPSHOT="1",
        SAVE_CHECKPOINTS="0",
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    env = json.loads(args[args.index("--runtime-env-json") + 1])["env_vars"]
    assert env["RELAX_CONDITION_VERIFY"] == "1"
    assert env["RELAX_CONDITION_VERIFY_RTOL"] == "0"
    assert env["RELAX_CONDITION_VERIFY_SNAPSHOT"] == "1"
    assert "--save" not in args
    assert "--save-interval" not in args
    assert not (tmp_path / "save").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"GPUS": "3", "GPUS_PER_NODE": "3"},
        {"RELAX_CONDITION_CACHE": "on"},
        {"RELAX_CONDITION_CACHE": "1", "RELAX_CONDITION_CACHE_MAX_BYTES": "0"},
    ],
)
def test_invalid_budget_modes_and_batch_fail_without_submission(tmp_path, overrides):
    result = preview(tmp_path, **overrides)
    assert result.returncode != 0
    assert not (tmp_path / "metrics").exists()


def test_extra_arguments_cannot_override_the_fixed_effective_batch(tmp_path):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL_PATH": "/model",
        "DATA_ROOT": "/data",
        "DATASET_SIZE": "128",
        "SAVE_DIR": str(tmp_path / "save"),
        "RAY_DASHBOARD_ADDRESS": "http://ray-head:8265",
    }
    result = subprocess.run(["bash", str(SCRIPT), "--global-batch-size=4"], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "effective batch 8" in result.stderr
