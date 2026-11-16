# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check the public Qwen3.8 launch commands without starting Ray or loading
GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
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
        ("run-qwen38-qsa-sft-8xgpu-blackwell.sh", 8, 2, 1, 8),
        ("run-qwen38-qsa-sft-openr1mm-8xgpu-blackwell.sh", 8, 2, 1, 8),
        ("run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
        ("run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh", 32, 2, 4, 8),
        ("run-qwen38-flash-next-openr1mm-128xgpu.sh", 128, 8, 8, 16),
        ("run-qwen38-flash-next-openr1mm-8xgpu-blackwell.sh", 8, 2, 1, 8),
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


@pytest.mark.parametrize("backend", ["hybridep", "mok"])
def test_eight_gpu_grpo_preserves_colocation_data_and_backend_contract(tmp_path, backend):
    config = _dry_run(
        "run-qwen38-flash-next-openr1mm-8xgpu-blackwell.sh",
        tmp_path / "checkpoints",
        MOE_BACKEND=backend,
        DATA_DIR="/data/openr1mm-rl.parquet",
        SAVE_CHECKPOINTS="0",
        NUM_ROLLOUT="2",
        PYTHONPATH="/runtime/mok",
    )
    args = config["argv"]
    assert json.loads(_value(args, "--resource")) == {"actor": [1, 8], "rollout": [1, 8]}
    assert "--colocate" in args and "--no-offload-train" not in args
    assert "--selective-offload" in args
    assert _value(args, "--rollout-num-gpus-per-engine") == "8"
    assert _value(args, "--advantage-estimator") == "grpo"
    assert _value(args, "--rm-type") == "openr1mm"
    assert _value(args, "--prompt-data") == "/data/openr1mm-rl.parquet"
    assert _value(args, "--input-key") == "prompt"
    assert _value(args, "--label-key") == "label"
    assert json.loads(_value(args, "--multimodal-keys")) == {"image": "image"}
    assert int(_value(args, "--rollout-batch-size")) * int(_value(args, "--n-samples-per-prompt")) == 128
    assert _value(args, "--global-batch-size") == "128"
    assert _value(args, "--data-pad-size-multiplier") == "4096"
    assert _value(args, "--max-tokens-per-gpu") == "8192"
    assert _value(args, "--log-probs-max-tokens-per-gpu") == "8192"
    assert "--moe-hybridep-pad-variable-tokens" not in args
    assert "--get-mismatch-metrics" not in args
    assert _value(args, "--num-rollout") == "2"
    assert _value(args, "--cuda-graph-impl") == "none"
    assert "--sglang-disable-cuda-graph" in args
    assert "--save" not in args and "--save-interval" not in args
    assert "--load" in args and "--loss-type" not in args
    env = config["runtime_env"]["env_vars"]
    assert env["QWEN38_PLE_CPU_OFFLOAD"] == env["QWEN48_PLE_CPU_OFFLOAD"] == "0"
    assert env["RELAX_QWEN38_PLE_STREAMED"] == "0"
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "/runtime/mok" in env["PYTHONPATH"]
    if backend == "mok":
        assert _value(args, "--moe-megakernel-backend") == "mok"
        assert json.loads(_value(args, "--moe-megakernel-backend-config")) == {"shared_expert_mode": "native"}
        assert "--moe-shared-expert-overlap" not in args
    else:
        assert _value(args, "--moe-flex-dispatcher-backend") == "hybridep"
        assert "--moe-megakernel-backend" not in args


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


def test_eight_gpu_moe_backends_keep_workload_and_worker_paths_in_sync(tmp_path):
    script = "run-qwen38-qsa-sft-8xgpu-blackwell.sh"
    configs = {
        backend: _dry_run(
            script,
            tmp_path / "checkpoints",
            MOE_BACKEND=backend,
            EXP_NAME="",
            PYTHONPATH="/runtime/mok:/runtime/cublas",
            CUDA_HOME="/runtime/cuda",
            RELAX_HYBRID_EP_COMMIT="image-pinned" if backend == "hybridep" else "",
            SEQ_LENGTH="4096",
            MAX_TOKENS_PER_GPU="4096",
            MOK_BACKEND_CONFIG='{"shared_expert_mode":"native","fwd_num_comm_sms":40}',
        )
        for backend in ("hybridep", "mok")
    }
    for backend, config in configs.items():
        args = config["argv"]
        assert _value(args, "--seq-length") == _value(args, "--max-tokens-per-gpu") == "4096"
        assert _value(args, "--data-pad-size-multiplier") == "2048"
        assert json.loads(_value(args, "--multimodal-keys")) == {"image": "images"}
        assert "--enable-mtp-training" in args
        assert _value(args, "--cuda-graph-impl") == "none"
        assert "--no-store-param-remainders" not in args
        assert _value(args, "--exp-avg-dtype") == "bf16"
        assert _value(args, "--exp-avg-sq-dtype") == "bf16"
        assert "--accumulate-allreduce-grads-in-fp32" in args
        assert _value(args, "--load").endswith(backend)
        assert _value(args, "--save") == _value(args, "--load")
        worker = json.loads(_value(args, "--train-env-vars"))
        assert worker == config["runtime_env"]["env_vars"]
        assert worker["PYTHONPATH"].split(":")[:3] == [str(ROOT), "/image/megatron", "/runtime/mok"]
        assert worker["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
        assert worker["CUDA_HOME"] == "/runtime/cuda"
        assert _value(args, "--max-global-restart") == "0"
    native, mok = (configs[name]["argv"] for name in ("hybridep", "mok"))
    assert _value(native, "--moe-flex-dispatcher-backend") == "hybridep"
    assert "--moe-shared-expert-overlap" in native
    assert "--moe-megakernel-backend" not in native
    assert _value(mok, "--moe-megakernel-backend") == "mok"
    assert json.loads(_value(mok, "--moe-megakernel-backend-config")) == {
        "shared_expert_mode": "native",
        "fwd_num_comm_sms": 40,
    }
    assert _value(mok, "--moe-token-dispatcher-type") == "alltoall"
    for flag in ("--moe-shared-expert-overlap", "--moe-hybridep-pad-variable-tokens", "--moe-flex-dispatcher-backend"):
        assert flag not in mok
    for flag in ("--global-batch-size", "--recompute-granularity", "--recompute-num-layers", "--seq-length"):
        assert _value(native, flag) == _value(mok, flag)


@pytest.mark.parametrize(
    "extra", [{"MOE_BACKEND": "unknown"}, {"MAX_TOKENS_PER_GPU": "768"}, {"DATA_PAD_SIZE_MULTIPLIER": "2048"}]
)
def test_eight_gpu_recipe_rejects_unsupported_backend_or_variable_ep_shapes(tmp_path, extra):
    env = dict(
        os.environ,
        DRY_RUN="1",
        MODEL_DIR="/models",
        DATA_DIR="/data",
        SAVE_DIR=str(tmp_path),
        MEGATRON="/runtime/megatron",
        MOE_BACKEND="mok",
        MAX_TOKENS_PER_GPU="8192",
    )
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run-qwen38-qsa-sft-8xgpu-blackwell.sh")],
        env=env | extra,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2


def test_eight_gpu_smoke_disables_final_checkpoint_writes(tmp_path):
    config = _dry_run(
        "run-qwen38-qsa-sft-8xgpu-blackwell.sh",
        tmp_path / "checkpoints",
        SAVE_CHECKPOINTS="0",
        NUM_STEPS="2",
        EVAL_SIZE="1",
    )
    args = config["argv"]
    assert _value(args, "--num-rollout") == "2"
    assert _value(args, "--eval-size") == "1"
    assert "--load" in args
    assert "--save" not in args
    assert "--save-interval" not in args
