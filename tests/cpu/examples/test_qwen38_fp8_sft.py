# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the FP8 SFT and BF16 reference launcher without a Ray cluster."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-flash-next-fp8-32xgpu.sh"


def _dry_run(checkpoint: Path, **extra: str) -> dict:
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
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _value(argv: list[str], flag: str) -> str:
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


@pytest.mark.parametrize("precision,storage", [("mxfp8", "fp8"), ("bf16", "bf16")])
def test_fp8_sft_reference_uses_optimizer_that_accepts_original_master_values(tmp_path, precision, storage):
    result = _dry_run(
        tmp_path / "checkpoints",
        PRECISION=precision,
        PARAM_STORAGE=storage,
        RAY_JOBS_ADDRESS="http://ray-head.invalid:8265",
        GLOO_SOCKET_IFNAME="test-interface",
    )
    args = result["argv"]
    assert "--use-distributed-optimizer" in args
    assert ("--use-precision-aware-optimizer" in args) == (precision == "mxfp8")
    assert ("--fp8-param-gather" in args) == (storage == "fp8")
    assert _value(args, "--hf-checkpoint") == "/models"
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("local_cache", ["", "/node-local/qwen38-cache"])
def test_fp8_sft_compiler_caches_are_separate_from_shared_checkpoints(tmp_path, local_cache):
    result = _dry_run(
        tmp_path / "shared-checkpoints",
        RAY_JOBS_ADDRESS="http://ray-head.invalid:8265",
        GLOO_SOCKET_IFNAME="test-interface",
        RELAX_KERNEL_CACHE_LOCAL_DIR=local_cache,
        SUBMISSION_ID="qwen38-cache-test",
    )
    runtime = result["runtime_env"]["env_vars"]
    actor = json.loads(_value(result["argv"], "--train-env-vars"))
    root = Path(runtime["RELAX_KERNEL_CACHE_LOCAL_DIR"])
    if local_cache:
        assert root == Path(local_cache)
    else:
        assert root.is_relative_to("/image/megatron")
    for name, subdir in {
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton",
        "TILELANG_CACHE_DIR": "tilelang",
        "CUDA_CACHE_PATH": "cuda",
        "TORCH_EXTENSIONS_DIR": "extensions",
    }.items():
        assert runtime[name] == actor[name] == str(root / subdir)
    assert actor["RELAX_KERNEL_CACHE_LOCAL_DIR"] == str(root)
    assert _value(result["argv"], "--save") == str(tmp_path / "shared-checkpoints/qwen38-test")
    assert not (tmp_path / "shared-checkpoints").exists()
