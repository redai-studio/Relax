# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = Path("examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-128xh800.sh")
MODEL = Path("scripts/models/qwen38-flash-next.sh")


@pytest.mark.parametrize("cleanup_setting", [None, "0", "1"])
def test_h800_sft_delegates_cluster_cleanup_to_ray_entrypoint(tmp_path: Path, cleanup_setting: str | None) -> None:
    launcher = tmp_path / LAUNCHER
    launcher.parent.mkdir(parents=True)
    shutil.copy2(ROOT / LAUNCHER, launcher)
    model = tmp_path / MODEL
    model.parent.mkdir(parents=True)
    shutil.copy2(ROOT / MODEL, model)

    entry = tmp_path / "scripts/entrypoint/ray-job.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text('printf "%s\\n" "$RELAX_SKIP_CLUSTER_CLEANUP" "$RAY_JOBS_ADDRESS"\n')
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ}
    env.update(
        MODEL_DIR="/models",
        DATA_DIR="/data",
        SAVE_DIR=str(tmp_path / "output"),
        MEGATRON="/runtime/megatron",
        RAY_JOBS_ADDRESS="http://127.0.0.1:8265",
    )
    if cleanup_setting is not None:
        env["RELAX_SKIP_CLUSTER_CLEANUP"] = cleanup_setting

    fake_ray = tmp_path / "bin/ray"
    fake_ray.parent.mkdir()
    ray_trace = tmp_path / "ray-called"
    env["MOCK_RAY_TRACE"] = str(ray_trace)
    fake_ray.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$MOCK_RAY_TRACE"\nexit 1\n')
    fake_ray.chmod(0o755)
    env["PATH"] = str(fake_ray.parent) + os.pathsep + env["PATH"]

    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [cleanup_setting or "", env["RAY_JOBS_ADDRESS"]]
    assert not ray_trace.exists(), "The recipe must delegate Ray operations to the entrypoint"
