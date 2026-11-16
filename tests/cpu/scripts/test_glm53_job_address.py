# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise recipe submission with isolated entrypoint and Ray boundaries.

The existing Ray wrapper tests do not execute recipe environment setup or data
selection. These short Bash subprocesses need no cluster, GPU, or model files.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SFT_RECIPE = "run-glm5.3-744B-A40B-lora-40k-32xgpu.sh"
RECIPES = [SFT_RECIPE, "eval-glm5.3-744B-A40B-aime-32xgpu.sh"]


@pytest.fixture
def run_recipe(tmp_path):
    scripts = tmp_path / "scripts"
    for directory in ("training/sft", "entrypoint", "models", "bin"):
        (scripts / directory).mkdir(parents=True)
    for recipe in RECIPES:
        shutil.copyfile(ROOT / "scripts/training/sft" / recipe, scripts / "training/sft" / recipe)
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
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAY_", "RELAX_", "EVAL_"))}
    env.update(
        PATH=f"{scripts / 'bin'}:{env['PATH']}",
        MODEL_DIR=str(tmp_path / "models"),
        MODEL_NAME="glm",
        DATA_DIR=str(tmp_path / "data"),
        PROMPT_DATA=str(tmp_path / "training messages.jsonl"),
        SAVE_DIR=str(tmp_path / "save"),
        RUNTIME_ENV_JSON="{}",
        BOOTSTRAP_MARKER=str(marker),
        CAPTURE=str(capture),
    )

    def run(recipe, **overrides):
        capture.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        launch_env = dict(env, **overrides)
        result = subprocess.run(
            ["bash", str(scripts / "training/sft" / recipe)],
            env=launch_env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        argv = json.loads(capture.read_text()) if capture.exists() else None
        return result, argv, marker.exists()

    return run


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("mode", ["local", "local_override", "remote", "remote_missing"])
def test_recipe_resolves_jobs_address_after_entrypoint(run_recipe, recipe, mode):
    overrides = {}
    if mode.startswith("remote"):
        overrides["RELAX_ENTRYPOINT_MODE"] = "ray-job"
    explicit = "http://ray-head.example:9123"
    if mode in ("local_override", "remote"):
        overrides["RAY_JOB_ADDRESS"] = explicit
    result, argv, bootstrapped = run_recipe(recipe, **overrides)
    if mode == "remote_missing":
        assert result.returncode != 0
        assert "Set RAY_JOB_ADDRESS" in result.stderr
        assert argv is None
        return
    assert result.returncode == 0, result.stderr
    assert bootstrapped == (mode != "remote")
    assert argv[:2] == ["job", "submit"]
    expected = "http://127.0.0.1:8265" if mode == "local" else explicit
    assert f"--address={expected}" in argv
    assert "--no-wait" not in argv


@pytest.mark.parametrize(
    "eval_data,interval", [("", "50"), ("/data/validation messages.jsonl", "50"), ("/data/eval.jsonl", "0")]
)
def test_sft_uses_external_datasets_and_optional_evaluation(run_recipe, eval_data, interval):
    result, argv, _ = run_recipe(
        SFT_RECIPE,
        DATA_DIR="",
        PROMPT_DATA="/data/custom training.jsonl",
        EVAL_DATA=eval_data,
        EVAL_INTERVAL=interval,
        EVAL_DATASET_NAME="custom-validation",
        EXP_NAME="custom-sft",
        SAVE_DIR="/checkpoints",
        LEARNING_RATE="2e-6",
        NUM_ROLLOUT="50",
    )
    assert result.returncode == 0, result.stderr
    for flag, value in {
        "--prompt-data": "/data/custom training.jsonl",
        "--lr": "2e-6",
        "--num-rollout": "50",
        "--save": "/checkpoints/custom-sft",
        "--load": "/checkpoints/custom-sft",
    }.items():
        assert argv[argv.index(flag) + 1] == value
    if eval_data and interval != "0":
        position = argv.index("--eval-prompt-data")
        assert argv[position + 1 : position + 3] == ["custom-validation", eval_data]
        assert argv[argv.index("--eval-interval") + 1] == interval
    else:
        assert "--eval-prompt-data" not in argv
        assert "--eval-interval" not in argv


def test_sft_requires_explicit_training_data_before_bootstrap(run_recipe):
    result, argv, bootstrapped = run_recipe(SFT_RECIPE, PROMPT_DATA="")
    assert result.returncode != 0
    assert "Set PROMPT_DATA" in result.stderr
    assert argv is None
    assert not bootstrapped
