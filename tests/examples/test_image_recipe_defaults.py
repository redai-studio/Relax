# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Render recipes with externally configured runtime paths."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SFT = "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh"
CPT = "scripts/training/cpt/run-qwen3.6-35B-A3B-cpt-256k-64xgpu.sh"
QSA = "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh"


@pytest.mark.parametrize("script", [SFT, CPT, QSA])
def test_image_recipe_three_path_environment(script):
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "LANG", "TMPDIR")}
    env.update(
        DRY_RUN="1",
        MEGATRON="/runtime/megatron",
        PYTHONPATH="/runtime/cublas",
        MODEL_DIR="/models/exact checkpoint",
        DATA_DIR="/datasets/exact input",
        SAVE_DIR="/outputs/new experiment",
        RELAX_GPU_ARCH="blackwell",
        RELAX_FLA_CONV_NB_MODE="auto",
        RELAX_HYBRID_EP_COMMIT="image-pinned-commit",
    )
    result = subprocess.run(["bash", str(ROOT / script)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    args = rendered["argv"]
    worker = json.loads(args[args.index("--train-env-vars") + 1])
    assert args[args.index("--hf-checkpoint") + 1] == env["MODEL_DIR"]
    assert args[args.index("--prompt-data") + 1] == env["DATA_DIR"]
    assert worker["RELAX_SKIP_NONFINITE_GRAD"] == "1"
    if script == QSA:
        assert worker["MEGATRON"] == env["MEGATRON"]
    else:
        assert worker["RELAX_FLA_CONV_NB_MODE"] == "none"
        assert "/runtime/cublas" in worker["PYTHONPATH"].split(":")
    assert rendered["runtime_env"]["env_vars"]["RELAX_SKIP_NONFINITE_GRAD"] == "1"


@pytest.mark.parametrize("script,busy", [(SFT, False), (CPT, False), (QSA, False), (QSA, True)])
def test_recipe_delegates_without_cluster_cleanup(tmp_path, script, busy):
    import shutil

    target = tmp_path / script
    target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / script, target)
    model = tmp_path / "scripts/models/qwen38-flash-next.sh"
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/models/qwen38-flash-next.sh", model)
    entry = tmp_path / "scripts/entrypoint/ray-job.sh"
    entry.parent.mkdir(parents=True)
    entry.write_text('printf "%s\\n" "$RELAX_SKIP_CLUSTER_CLEANUP" "$RAY_JOBS_ADDRESS"\n')
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "LANG", "TMPDIR")}
    env.update(MODEL_DIR="/models", DATA_DIR="/data", SAVE_DIR="/output", RELAX_HYBRID_EP_COMMIT="image-pinned")
    env.update(MEGATRON="/runtime/megatron", RAY_JOBS_ADDRESS="http://127.0.0.1:8265")
    fake_ray = tmp_path / "bin/ray"
    fake_ray.parent.mkdir()
    fake_ray.write_text("#!/bin/sh\nprintf RUNNING\n" if busy else "#!/bin/sh\nexit 0\n")
    fake_ray.chmod(0o755)
    env["PATH"] = str(fake_ray.parent) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True, timeout=10)
    if busy:
        assert result.returncode == 2
        assert "refusing concurrent training" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1", "http://127.0.0.1:8265"]
