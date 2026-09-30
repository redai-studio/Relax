# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise public launch recipes with a recording Ray CLI; never submit
jobs."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(path for path in (ROOT / "examples/models/kimi-k3/scripts").glob("*.sh") if "-grpo" not in path.stem)


def _environment(tmp_path):
    data = tmp_path / "data with spaces"
    data.mkdir()
    (data / "READY.json").write_text("{}")
    capture = tmp_path / "ray.json"
    ray = tmp_path / "ray"
    ray.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    ray.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "HF_CHECKPOINT",
            "HELLASWAG_DATA_DIR",
            "OPENR1MM_DATA_DIR",
            "POKEMON_DATA_DIR",
            "LLAVA_DATA_DIR",
            "SAVE_DIR",
            "LOAD_DIR",
            "DRY_RUN",
            "RAY_NO_WAIT",
            "CLEARML_CONFIG_FILE",
            "FLA_TILELANG",
            "NUM_ROLLOUT",
            "SAVE_INTERVAL",
        }
    }
    env.update(
        MODEL_DIR=str(tmp_path / "model with spaces"),
        DATA_DIR=str(data),
        RELAX_ENTRYPOINT_MODE="test",
        RUNTIME_ENV_JSON='{"env_vars":{}}',
        CAPTURE=str(capture),
        PATH=f"{tmp_path}:{env['PATH']}",
        EXP_NAME="public-recipe-test",
    )
    return env, capture


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
@pytest.mark.parametrize("saving", [False, True])
def test_kimi_k3_recipe_resolves_shared_paths_and_worker_settings(tmp_path, script, saving):
    env, capture = _environment(tmp_path)
    if saving:
        env["SAVE_DIR"] = str(tmp_path / "checkpoint with spaces")
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["job", "submit"]
    assert "--no-wait" not in argv
    assert argv[argv.index("--hf-checkpoint") + 1] == env["MODEL_DIR"]
    assert env["DATA_DIR"] in argv[argv.index("--prompt-data") + 1]
    assert argv.count("--train-env-vars") == 1
    train_env = json.loads(argv[argv.index("--train-env-vars") + 1])
    assert train_env["FLA_TILELANG"] == "0"
    if "openr1mm" in script.name or "llava-onevision" in script.name:
        assert train_env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "--use-clearml" in argv
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    assert "CLEARML_CONFIG_FILE" not in runtime.get("env_vars", {})
    assert "clearml" not in runtime.get("working_dir", "").lower()
    if saving:
        expected = str(Path(env["SAVE_DIR"]) / env["EXP_NAME"])
        assert argv[argv.index("--save") + 1] == expected
        assert argv[argv.index("--load") + 1] == expected
    else:
        assert "--save" not in argv
    if "rawtext" in script.name:
        assert runtime["env_vars"]["RELAX_SFT_RAW_TEXT_CONCAT"] == "1"
    if "lora" in script.name:
        assert runtime["env_vars"]["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] == "false"
        assert ("--save-lora-only" in argv) is saving


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_kimi_k3_recipe_dry_run_never_calls_ray(tmp_path, script):
    env, capture = _environment(tmp_path)
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=30
    )
    assert not capture.exists()
    argv = shlex.split(result.stdout)
    assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
    assert argv[argv.index("--hf-checkpoint") + 1] == env["MODEL_DIR"]


@pytest.mark.parametrize("no_wait", ["", "1"])
def test_kimi_k3_recipe_preserves_explicit_wait_and_optimizer_save(tmp_path, no_wait):
    env, capture = _environment(tmp_path)
    env.update(RAY_NO_WAIT=no_wait, SAVE_OPTIMIZER="1", SAVE_DIR=str(tmp_path / "save"), FLA_TILELANG="1")
    script = ROOT / "examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh"
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    assert ("--no-wait" in argv) is bool(no_wait)
    assert "--no-save-optim" not in argv
    assert json.loads(argv[argv.index("--train-env-vars") + 1])["FLA_TILELANG"] == "1"
