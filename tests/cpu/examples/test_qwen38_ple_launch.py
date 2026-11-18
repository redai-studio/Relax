# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The PLE example must forward its switches to both driver and training
actors."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh"


@pytest.mark.parametrize("enabled", [None, "1"])
def test_ple_launch_environment(enabled, tmp_path, monkeypatch):
    env = dict(os.environ)
    env.pop("QWEN38_PLE_INPUT_HALO_CP", None)
    env.pop("QWEN38_PLE_INPUT_HALO_TRANSPORT", None)
    env.update(
        DRY_RUN="1",
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR=str(tmp_path / "save"),
        MEGATRON="/megatron",
        RELAX_HYBRID_EP_COMMIT="test-revision",
        QWEN38_CP_SIZE="8",
        QWEN38_PP_LAYOUT="Et*24|t*24mL",
        QWEN38_SFT_MODALITY="text",
        RELAX_QWEN38_QSA_MIXED_LONG_BACKEND="cute",
        RELAX_QWEN38_QSA_CP="1",
    )
    if enabled is not None:
        env["QWEN38_PLE_INPUT_HALO_CP"] = enabled
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--tensor-model-parallel-size",
            "1",
            "--pipeline-model-parallel-size",
            "2",
            "--linear-cp-mode",
            "headwise",
            "--global-batch-size",
            "32",
            "--rollout-batch-size",
            "32",
            "--max-tokens-per-gpu",
            "16384",
            "--chunked-optimizer-state-offload",
            "--optimizer-state-offload-fraction",
            "1.0",
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    config = json.loads(result.stdout)
    argv = config["argv"]
    actor_env = json.loads(argv[argv.index("--train-env-vars") + 1])
    for worker_env in (config["runtime_env"]["env_vars"], actor_env):
        assert worker_env["QWEN38_PLE_INPUT_HALO_CP"] == (enabled or "0")
        assert worker_env["QWEN38_PLE_INPUT_HALO_TRANSPORT"] == "adaptive"
    for flag, value in [
        ("--tensor-model-parallel-size", "1"),
        ("--pipeline-model-parallel-size", "2"),
        ("--context-parallel-size", "8"),
    ]:
        assert [argv[i + 1] for i, arg in enumerate(argv) if arg == flag][-1] == value
    assert not (tmp_path / "save").exists()

    if enabled is not None:
        pytest.importorskip("megatron.training.arguments")
        import sys
        from unittest.mock import patch

        import torch

        from relax.utils.arguments import parse_args

        for key, value in config["runtime_env"]["env_vars"].items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(
            sys, "argv", ["ple-example", *argv[3:], "--skip-hf-validate", "--disable-s3-model-download"]
        )
        with patch("megatron.training.arguments.get_device_arch_version", return_value=10):
            args = parse_args()
        assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.context_parallel_size) == (
            1,
            2,
            8,
        )
        assert args.expert_model_parallel_size == 8 and args.expert_tensor_parallel_size == 1
        assert not args.sequence_parallel
        assert args.linear_cp_mode == "headwise"
        assert args.global_batch_size == 32 and args.max_tokens_per_gpu == 16384
        assert not torch.cuda.is_initialized()
