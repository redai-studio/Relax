# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render the image-native recipe without submitting a Ray job."""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_optimized_launcher_uses_stable_checkpoints_and_image_dependencies():
    environment = dict(
        os.environ,
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/checkpoints",
        EXP_NAME="cpu-validation",
        DRY_RUN="1",
    )
    script = ROOT / "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    outputs = [
        json.loads(subprocess.check_output(["bash", str(script)], env=environment, text=True)) for _ in range(2)
    ]
    for output in outputs:
        argv = output["argv"]
        assert argv[argv.index("--load") + 1] == argv[argv.index("--save") + 1] == "/checkpoints/cpu-validation"
        assert "--sft-wholepack" in argv
        assert "--sft-packing-strategy" not in argv
        assert "--save-interval" in argv
        assert argv[argv.index("--max-tokens-per-gpu") + 1] == "262144"
        worker_env = output["runtime_env"]["env_vars"]
        assert json.loads(argv[argv.index("--train-env-vars") + 1]) == worker_env
        assert "B300_OPT_PREFIX" not in worker_env  # gitleaks:allow -- existing runtime environment compatibility key
        assert "RELAX_SFT_WHOLEPACK" not in worker_env
        assert worker_env["EP_DISABLE_GIN"] == "1"
        assert worker_env["LD_PRELOAD"].startswith("/opt/relax/cublas13.3/nvidia/cu13/lib/libcublasLt.so.13:")


def test_optimized_shell_preserves_worker_environment_and_argument_overrides():
    environment = dict(
        os.environ,
        MODEL_DIR="/model with spaces",
        DATA_DIR="/data with spaces.jsonl",
        SAVE_DIR="/checkpoints with spaces",
        EXP_NAME="cpu-validation",
        DRY_RUN="1",
        RUNTIME_ENV_JSON=json.dumps(
            {"env_vars": {"CUSTOM_WORKER_KEY": "preserved", "NCCL_ALGO": "Ring"}, "pip": ["example-package"]}
        ),
    )
    script = ROOT / "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh"
    output = json.loads(
        subprocess.check_output(["bash", str(script), "--num-rollout", "3"], env=environment, text=True)
    )
    argv = output["argv"]
    assert argv[-2:] == ["--num-rollout", "3"]
    assert argv[argv.index("--hf-checkpoint") + 1] == "/model with spaces"
    assert argv[argv.index("--prompt-data") + 1] == "/data with spaces.jsonl"
    assert argv[argv.index("--save") + 1] == "/checkpoints with spaces/cpu-validation"
    # The shared model definition sets alltoall; the recipe's later flex setting must win.
    dispatch = [argv[i + 1] for i, value in enumerate(argv) if value == "--moe-token-dispatcher-type"]
    assert dispatch[-1] == "flex"
    assert argv[argv.index("--moe-flex-dispatcher-backend") + 1] == "deepepv2"
    assert argv[argv.index("--custom-megatron-before-train-step-hook-path") + 1] == (
        "relax.models.qwen3_5.recipe.before_step"
    )
    assert output["runtime_env"]["pip"] == ["example-package"]
    assert output["runtime_env"]["env_vars"]["CUSTOM_WORKER_KEY"] == "preserved"
    assert "NCCL_ALGO" not in output["runtime_env"]["env_vars"]


def test_multimodal_recipe_separates_preprocessing_and_training_threads():
    environment = dict(os.environ, MODEL_DIR="/model", DATA_DIR="/data", SAVE_DIR="/checkpoints", DRY_RUN="1")
    script = ROOT / "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh"
    output = json.loads(
        subprocess.check_output(
            ["bash", str(script), "--multimodal", "--num-rollout", "15"], env=environment, text=True
        )
    )
    argv = output["argv"]
    assert "--multimodal" not in argv
    assert argv[-2:] == ["--num-rollout", "15"]
    values = {arg: argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg.startswith("--")}
    assert json.loads(values["--multimodal-keys"]) == {"image": "images"}
    assert values["--image-max-token-num"] == "16384"
    assert values["--sft-invalid-multimodal-strategy"] == "error"
    assert values["--sft-max-in-flight-steps"] == "3"
    assert values["--sft-prefetch-buffer-size"] == "2048"
    assert values["--sft-prefetch-num-workers"] == "96"
    runtime = output["runtime_env"]["env_vars"]
    train = json.loads(values["--train-env-vars"])
    assert runtime["RELAX_PROCESSOR_POOL_SIZE"] == "96"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert runtime[key] == "1"
        assert train[key] == "24"
        assert key in runtime["RELAX_PROPAGATE_ENV_VARS"].split(",")
    assert train["RELAX_DISABLE_VISION_BRANCH_SYNC"] == "0"
    assert train["RELAX_LOGPROB_FAST_PATH"] == "0"
