# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise recipe submission with isolated entrypoint and Ray boundaries.

The existing Ray wrapper tests do not execute recipe environment setup. These
short Bash subprocesses need no cluster, GPU, or model files.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPES = [
    "run-glm5.3-744B-A40B-dapo-lora-40k-32xgpu.sh",
    "run-glm5.3-744B-A40B-openthoughts3-lora-40k-32xgpu.sh",
    "eval-glm5.3-744B-A40B-aime-32xgpu.sh",
]


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("mode", ["local", "local_override", "remote", "remote_missing"])
def test_recipe_resolves_jobs_address_after_entrypoint(tmp_path, recipe, mode):
    scripts = tmp_path / "scripts"
    for directory in ("training/sft", "entrypoint", "models", "bin"):
        (scripts / directory).mkdir(parents=True)
    target = scripts / "training/sft" / recipe
    shutil.copyfile(ROOT / "scripts/training/sft" / recipe, target)
    shutil.copyfile(ROOT / "scripts/models/glm5.3-744B-A40B.sh", scripts / "models/glm5.3-744B-A40B.sh")
    marker = tmp_path / "bootstrapped"
    (scripts / "entrypoint/local.sh").write_text(
        'touch "$BOOTSTRAP_MARKER"\nexport RELAX_ENTRYPOINT_MODE=local MASTER_ADDR=127.0.0.1\n'
    )
    capture = tmp_path / "submission.json"
    ray = scripts / "bin/ray"
    ray.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        'with open(os.environ["CAPTURE"], "w") as output:\n    json.dump(sys.argv[1:], output)\n'
    )
    ray.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAY_", "RELAX_"))}
    env.update(
        PATH=f"{scripts / 'bin'}:{env['PATH']}",
        MODEL_DIR=str(tmp_path / "models"),
        MODEL_NAME="glm",
        DATA_DIR=str(tmp_path / "data"),
        SAVE_DIR=str(tmp_path / "save"),
        RUNTIME_ENV_JSON="{}",
        BOOTSTRAP_MARKER=str(marker),
        CAPTURE=str(capture),
    )
    if mode.startswith("remote"):
        env["RELAX_ENTRYPOINT_MODE"] = "ray-job"
    explicit = "http://ray-head.example:9123"
    if mode in ("local_override", "remote"):
        env["RAY_JOB_ADDRESS"] = explicit
    result = subprocess.run(["bash", str(target)], env=env, capture_output=True, text=True, timeout=15)
    if mode == "remote_missing":
        assert result.returncode != 0
        assert "Set RAY_JOB_ADDRESS" in result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    assert marker.exists() == (mode != "remote")
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["job", "submit"]
    expected = "http://127.0.0.1:8265" if mode == "local" else explicit
    assert f"--address={expected}" in argv
    assert "--no-wait" not in argv
