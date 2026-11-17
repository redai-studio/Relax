# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise MoK backend selection and capacity calculation without Ray or
GPUs."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-openr1mm-32xgpu-blackwell.sh"


def _dry_run(checkpoint: Path, **extra: str) -> dict:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ}
    env.update(
        DRY_RUN="1",
        MODEL_DIR="/models",
        DATA_DIR="/data/openr1mm-sft.jsonl",
        SAVE_DIR=str(checkpoint),
        EXP_NAME="qwen38-mok-test",
        MEGATRON="/image/megatron",
    )
    env.update(extra)
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _value(argv: list[str], flag: str) -> str:
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


@pytest.mark.parametrize("backend", ["hybridep", "mok"])
def test_backend_selection_and_worker_dependencies(tmp_path: Path, backend: str) -> None:
    config = _dry_run(
        tmp_path / "checkpoints",
        MOE_BACKEND=backend,
        PYTHONPATH="/runtime/extra",
        CUDA_HOME="/runtime/cuda",
        RELAX_HYBRID_EP_COMMIT="image-pinned" if backend == "hybridep" else "",
    )
    args = config["argv"]
    worker = json.loads(_value(args, "--train-env-vars"))
    assert worker["PYTHONPATH"].split(":")[:3] == [str(ROOT), "/image/megatron", "/runtime/extra"]
    assert worker["CUDA_HOME"] == "/runtime/cuda"
    assert config["runtime_env"]["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert worker["CUDA_DEVICE_MAX_CONNECTIONS"] == "32"
    assert not (tmp_path / "checkpoints").exists()
    if backend == "mok":
        assert _value(args, "--moe-megakernel-backend") == "mok"
        assert _value(args, "--moe-token-dispatcher-type") == "alltoall"
        assert _value(args, "--cuda-graph-impl") == "none"
        for flag in (
            "--moe-shared-expert-overlap",
            "--moe-hybridep-pad-variable-tokens",
            "--moe-flex-dispatcher-backend",
        ):
            assert flag not in args
    else:
        assert _value(args, "--moe-flex-dispatcher-backend") == "hybridep"
        assert "--moe-shared-expert-overlap" in args
        assert "--moe-megakernel-backend" not in args


def test_capacity_and_user_options_preserve_reference_workload(tmp_path: Path) -> None:
    configs = {
        backend: _dry_run(
            tmp_path / "checkpoints",
            MOE_BACKEND=backend,
            RELAX_HYBRID_EP_COMMIT="image-pinned" if backend == "hybridep" else "",
            SEQ_LENGTH="65537",
            MAX_TOKENS_PER_GPU="65536",
            GBS="64",
            MOK_BACKEND_CONFIG='{"fwd_num_comm_sms":40,"macrobatch_size":1048576}',
        )["argv"]
        for backend in ("hybridep", "mok")
    }
    native, mok = configs["hybridep"], configs["mok"]
    assert json.loads(_value(mok, "--moe-megakernel-backend-config")) == {
        "shared_expert_mode": "native",
        "fixed_num_local_tokens": 49152,
        "fwd_num_comm_sms": 40,
        "macrobatch_size": 1048576,
    }
    for flag in (
        "--seq-length",
        "--max-tokens-per-gpu",
        "--global-batch-size",
        "--recompute-num-layers",
        "--prompt-data",
    ):
        assert _value(native, flag) == _value(mok, flag)
