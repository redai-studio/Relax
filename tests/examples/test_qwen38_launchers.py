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
    )
    env.update(extra)
    result = subprocess.run(["bash", str(SCRIPTS / script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _value(argv: list[str], flag: str) -> str:
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


@pytest.mark.parametrize(
    "script,gpus,tp,pp,ep",
    [
        ("run-qwen38-qsa-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
        ("run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
        ("run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
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


NV_RECIPES = [
    ("run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh", "Et*12|t*14|t*14|t*8mL", "2023", True),
    ("run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh", "Et*14|t*12|t*12|t*10mL", "996", False),
]


@pytest.mark.parametrize(
    "script,connections",
    [("run-qwen38-qsa-sft-32xgpu-blackwell.sh", "32"), *[(recipe[0], "1") for recipe in NV_RECIPES]],
)
def test_recipe_connections_survive_ray_entrypoint_defaults(tmp_path, script, connections):
    config = _dry_run(
        script, tmp_path / "checkpoints", RELAX_ENTRYPOINT_MODE="ray-job", CUDA_DEVICE_MAX_CONNECTIONS="1"
    )
    assert config["runtime_env"]["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == connections
    actor = json.loads(_value(config["argv"], "--train-env-vars"))
    assert actor["CUDA_DEVICE_MAX_CONNECTIONS"] == connections


@pytest.mark.parametrize("script,pp,decay,multimodal", NV_RECIPES)
@pytest.mark.parametrize("nvte", ["0", "1"])
def test_nv_recipes_keep_topology_modality_and_runtime_in_sync(tmp_path, script, pp, decay, multimodal, nvte):
    from relax.utils.env import validate_env

    config = _dry_run(
        script,
        tmp_path / "checkpoints",
        NVTE_USE_CUTLASS_GROUPED_GEMM=nvte,
        RUNTIME_ENV_JSON=json.dumps({"env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}}),
    )
    args = config["argv"]
    runtime = config["runtime_env"]["env_vars"]
    actor = json.loads(_value(args, "--train-env-vars"))
    assert actor == {key: runtime[key] for key in actor}
    validate_env(actor)
    for key, value in {
        "NVTE_USE_CUTLASS_GROUPED_GEMM": nvte,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RELAX_QWEN38_QSA_MIXED_LONG_BACKEND": "cudnn_pr1765",
        "RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN": "2051",
        "RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN": "0",
        "RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS": "0",
        "RELAX_QWEN38_RECORD_FUNCTION_TAGS": "0",
        "RELAX_AUTOTUNE_LOG_DIR": "",
    }.items():
        assert runtime[key] == value
    assert _value(args, "--pipeline-model-parallel-layout") == pp
    assert _value(args, "--lr-decay-iters") == decay
    assert _value(args, "--num-attention-heads") == "24"
    assert _value(args, "--num-query-groups") == "2"
    assert _value(args, "--kv-channels") == "256"
    assert _value(args, "--global-batch-size") == "512"
    assert _value(args, "--micro-batch-size") == "1"
    assert _value(args, "--recompute-num-layers") == "2"
    assert _value(args, "--sft-tq-timeout-minutes") == "120"
    assert _value(args, "--max-global-restart") == "0"
    for flag in ("--multimodal-keys", "--image-max-token-num", "--sft-image-preprocess-on-rank"):
        assert (flag in args) == multimodal
    assert "--use-health-check" not in args and "--use-pytorch-profiler" not in args


@pytest.mark.parametrize("script,pp,decay,multimodal", NV_RECIPES)
def test_grouped_control_changes_only_long_backend(tmp_path, script, pp, decay, multimodal):
    nv = _dry_run(script, tmp_path / "checkpoints")
    grouped = _dry_run(script, tmp_path / "checkpoints", RELAX_QWEN38_QSA_MIXED_LONG_BACKEND="grouped")
    for config in (nv, grouped):
        config["runtime_env"]["env_vars"].pop("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND")
        actor = json.loads(_value(config["argv"], "--train-env-vars"))
        actor.pop("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND")
        index = config["argv"].index("--train-env-vars") + 1
        config["argv"][index] = json.dumps(actor, sort_keys=True)
        config["argv"][config["argv"].index("--tb-experiment-name") + 1] = "run"
    assert nv == grouped
