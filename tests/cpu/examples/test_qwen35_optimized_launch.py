# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render the image-native recipe without submitting a Ray job."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


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


@pytest.mark.parametrize(
    "entry",
    [
        "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh",
        "scripts/training/cpt/run-qwen3.6-35B-A3B-cpt-256k-64xgpu.sh",
    ],
)
@pytest.mark.parametrize("wholepack", ["0", "1"])
def test_wholepack_can_be_disabled_without_disabling_model_optimizations(entry, wholepack):
    environment = dict(
        os.environ,
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/checkpoints",
        DRY_RUN="1",
        QWEN35_WHOLEPACK=wholepack,
        GLOBAL_BATCH_SIZE="2048",
    )
    script = ROOT / entry
    output = json.loads(subprocess.check_output(["bash", str(script)], env=environment, text=True))
    argv = output["argv"]
    assert argv[argv.index("--global-batch-size") + 1] == "2048"
    assert ("--sft-wholepack" in argv) is (wholepack == "1")
    if wholepack == "0":
        assert argv[argv.index("--sft-packing-strategy") + 1] == "binpack"
    assert argv[argv.index("--custom-megatron-before-train-step-hook-path") + 1] == (
        "relax.models.qwen3_5.recipe.before_step"
    )
    assert "--overlap-dispatch-backward-with-experts-wgrad" in argv


def _write_fake_nvcc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'preload': os.environ.get('LD_PRELOAD'), "
        "'target_override': os.environ.get('RELAX_NVCC_REAL')}))\n"
        "sys.exit(int(os.environ.get('TEST_NVCC_EXIT_CODE', '0')))\n"
    )
    path.chmod(0o755)


@pytest.mark.parametrize("custom_compiler,exit_code", [(False, 0), (True, 9)])
def test_qwen36_launcher_isolates_nvcc_without_clearing_training_preloads(tmp_path, custom_compiler, exit_code):
    cuda_home = tmp_path / "cuda toolkit"
    compiler = tmp_path / "custom compiler" if custom_compiler else cuda_home / "bin/nvcc"
    _write_fake_nvcc(compiler)
    environment = dict(
        os.environ,
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/checkpoints",
        DRY_RUN="1",
        CUDA_HOME=str(cuda_home),
        LD_PRELOAD="",
        RUNTIME_ENV_JSON="{}",
    )
    environment.pop("EP_JIT_NVCC_COMPILER", None)
    if custom_compiler:
        environment["EP_JIT_NVCC_COMPILER"] = str(compiler)
    script = ROOT / "scripts/training/sft/run-qwen36-35B-A3B-32xgpu-muon.sh"
    output = json.loads(subprocess.check_output(["bash", str(script)], env=environment, text=True))
    argv = output["argv"]
    train = json.loads(argv[argv.index("--train-env-vars") + 1])
    runtime = output["runtime_env"]["env_vars"]
    assert train["LD_PRELOAD"] == runtime["LD_PRELOAD"]
    assert "libcublasLt.so.13" in train["LD_PRELOAD"]
    for key in ("EP_JIT_NVCC_COMPILER", "RELAX_NVCC_REAL", "EP_JIT_PRINT_COMPILER_COMMAND"):
        assert train[key] == runtime[key]
    assert train["EP_JIT_PRINT_COMPILER_COMMAND"] == "1"

    args = ["--version", "source with spaces.cu", "-o", "output.cubin"]
    result = subprocess.run(
        [train["EP_JIT_NVCC_COMPILER"], *args],
        env={**os.environ, **train, "TEST_NVCC_EXIT_CODE": str(exit_code)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == {"argv": args, "preload": None, "target_override": None}
    assert "libcublasLt.so.13" in train["LD_PRELOAD"]


def test_nvcc_wrapper_keeps_image_fallback_and_supports_wrapped_target(tmp_path):
    wrapper = ROOT / "scripts/tools/nvcc-without-preload.sh"
    installed = tmp_path / "cuda/bin/nvcc"
    _write_fake_nvcc(installed.with_name("nvcc.real"))
    installed.write_bytes((ROOT / "docker/scripts/nvcc-without-preload.sh").read_bytes())
    installed.chmod(0o755)
    environment = dict(os.environ, LD_PRELOAD="relax-test-preload.so")
    environment.pop("RELAX_NVCC_REAL", None)

    for executable, extra_env in (
        (installed, {}),
        (wrapper, {"RELAX_NVCC_REAL": str(installed)}),
    ):
        result = subprocess.run(
            [str(executable), "--version"],
            env={**environment, **extra_env},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"argv": ["--version"], "preload": None, "target_override": None}
    assert environment["LD_PRELOAD"] == "relax-test-preload.so"
