# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
import sys
from pathlib import Path


def test_task11_recipe_carries_supervision_and_worker_env(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    dump = tmp_path / "argv.bin"
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("RELAX_STRAGGLER_"):
            del env[key]
    env.update(
        {
            "DRY_RUN": "1",
            "RELAX_ENTRYPOINT_MODE": "ray-job",
            "RUNTIME_ENV_JSON": "{}",
            "TRAIN_VENV": sys.prefix,
            "RELAX": str(root),
            "TRAIN_ARGS_DUMP": str(dump),
            "TRAIN_RAY_ADDRESS": "localhost:6379",
            "NO_PROXY": "*",
            "HTTP_PROXY": "",
            "RAY_JOB_SUBMISSION_ID": "unit-test-owned-job",
            "TENSORBOARD_DIR": str(tmp_path / "tensorboard"),
        }
    )
    result = subprocess.run(
        [
            "bash",
            "scripts/training/sft/run-qwen3-0.6B-4xgpu-dp4-observer.sh",
            "--tensor-model-parallel-size",
            "2",
            "--use-pytorch-profiler",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    args = dump.read_bytes().decode().rstrip("\0").split("\0")
    assert args[-3:] == ["--tensor-model-parallel-size", "2", "--use-pytorch-profiler"]
    assert "--submission-id unit-test-owned-job" in result.stdout
    payload = result.stdout.split("=== DRY_RUN resolved RUNTIME_ENV_JSON (not submitted) ===\n")[1].split(
        "=== DRY_RUN final command"
    )[0]
    worker = json.loads(payload)["env_vars"]
    assert worker["RAY_ADDRESS"] == "localhost:6379"
    assert worker["NO_PROXY"] == "*"
    assert worker["HTTP_PROXY"] == ""
    assert worker["TENSORBOARD_DIR"] == str(tmp_path / "tensorboard")
