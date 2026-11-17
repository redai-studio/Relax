# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


LAUNCHER = (
    Path(__file__).resolve().parents[3] / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-sft-32xgpu-muon.sh"
)


@pytest.mark.parametrize("actor_override", [None, "16"])
def test_cpu_driver_and_gpu_actor_connections(tmp_path: Path, actor_override: str | None) -> None:
    env = {
        "PATH": os.environ["PATH"],
        "DRY_RUN": "1",
        "MODEL_DIR": "/model",
        "DATA_DIR": "/data",
        "SAVE_DIR": str(tmp_path),
        "MEGATRON": "/megatron",
        "RELAX_HYBRID_EP_COMMIT": "image-pinned",
        "RUNTIME_ENV_JSON": json.dumps({"env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}}),
    }
    if actor_override is not None:
        env["QWEN38_CUDA_DEVICE_MAX_CONNECTIONS"] = actor_override

    config = json.loads(subprocess.check_output(["bash", str(LAUNCHER)], env=env, text=True, timeout=10))
    argv = config["argv"]
    actor_env = json.loads(argv[argv.index("--train-env-vars") + 1])
    assert config["runtime_env"]["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert actor_env["CUDA_DEVICE_MAX_CONNECTIONS"] == (actor_override or "32")
