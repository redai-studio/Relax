# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the FP8 SFT and BF16 reference launcher without a Ray cluster."""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-flash-next-fp8-32xgpu.sh"


def test_dry_run_previews_training_without_submitting_or_creating_checkpoints(tmp_path):
    marker = tmp_path / "ray-called"
    ray = tmp_path / "ray"
    ray.write_text('#!/bin/sh\ntouch "$RAY_MARKER"\nexit 1\n')
    ray.chmod(0o755)
    checkpoints = tmp_path / "checkpoints"
    env = dict(
        {k: os.environ[k] for k in ("HOME", "LANG", "TMPDIR") if k in os.environ},
        DRY_RUN="1",
        PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        RAY_MARKER=str(marker),
        MODEL_DIR="/models",
        DATA_DIR="/data",
        SAVE_DIR=str(checkpoints),
        RAY_JOBS_ADDRESS="http://ray-head.invalid:8265",
        GLOO_SOCKET_IFNAME="test-interface",
        RELAX_HYBRID_EP_COMMIT="abc123def",
    )
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout)["argv"]
    assert not marker.exists()
    assert not checkpoints.exists()
