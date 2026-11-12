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


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(
    path for path in (ROOT / "examples/models/kimi-k3/scripts").glob("run-*.sh") if "-grpo" not in path.stem
)


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
            "SAVE_OPTIMIZER",
            "ATTENTION_BACKEND",
            "TRAIN_ENV_VARS",
            "RELAX_KIMI_K3_EXPERT_SPLIT",
            "RELAX_KIMI_K3_FUSED_SITU",
            "RELAX_KIMI_K3_FUSED_ATTN_RES",
        }
    }
    env.update(
        MODEL_DIR=str(tmp_path / "model with spaces"),
        DATA_DIR=str(data),
        RELAX_ENTRYPOINT_MODE="test",
        WARMUP_HF_CACHE="0",
        RAY_DASHBOARD_ADDRESS="http://ray-dashboard.test:8265",
        RUNTIME_ENV_JSON='{"env_vars":{}}',
        RELAX_KERNEL_CACHE_DIR="",  # Launch recording must not prepare a real Ray cache.
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
    if "kimi-k3.1-pro" in script.name and not saving:
        result = subprocess.run(["bash", str(script)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
        assert "Set SAVE_DIR" in result.stderr
        assert not capture.exists()
        return
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["job", "submit"]
    assert "--no-wait" not in argv
    assert argv[argv.index("--hf-checkpoint") + 1] == env["MODEL_DIR"]
    assert env["DATA_DIR"] in argv[argv.index("--prompt-data") + 1]
    assert argv.count("--train-env-vars") == 1
    train_env = json.loads(argv[argv.index("--train-env-vars") + 1])
    assert train_env["FLA_TILELANG"] == "0"
    assert train_env["RELAX_KIMI_K3_FUSED_SITU"] == "1"
    assert train_env["RELAX_KIMI_K3_FUSED_ATTN_RES"] == "1"
    if "kimi-k3.1" not in script.name:
        assert argv[max(i for i, arg in enumerate(argv) if arg == "--attention-backend") + 1] == "fused"
    if "openr1mm" in script.name or "llava-onevision" in script.name:
        assert train_env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert "--use-clearml" in argv
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    assert "CLEARML_CONFIG_FILE" not in runtime.get("env_vars", {})
    assert "clearml" not in runtime.get("working_dir", "").lower()
    if "kimi-k3.1-pro" in script.name and "128xblackwell" in script.name:
        aim = "aim-0825" in script.name
        assert runtime["env_vars"]["RELAX_SFT_EVAL_FIRST_STEP"] == ("0" if aim else "1")
        assert "RELAX_SFT_EVAL_FIRST_STEP" in runtime["env_vars"]["RELAX_PROPAGATE_ENV_VARS"].split(",")
        if aim:
            assert "--eval-interval" not in argv
        else:
            assert argv[argv.index("--eval-interval") + 1] == "100"
    if saving:
        expected = str(Path(env["SAVE_DIR"]) / env["EXP_NAME"])
        assert argv[argv.index("--save") + 1] == expected
        assert argv[argv.index("--load") + 1] == expected
        if "kimi-k3.1-pro" in script.name:
            weights_only = "hellaswag" in script.name or "aim-0825" in script.name
            assert ("--no-save-optim" in argv) is weights_only
            assert ("--no-load-optim" in argv) is weights_only
            expected_interval = "10" if "lora" in script.name else "100"
            assert argv[argv.index("--save-interval") + 1] == expected_interval
            assert argv[argv.index("--max-actor-ckpt-to-keep") + 1] == "1"
    else:
        assert "--save" not in argv
    if "rawtext" in script.name:
        assert runtime["env_vars"]["RELAX_SFT_RAW_TEXT_CONCAT"] == "1"
        assert "RELAX_SFT_RAW_TEXT_CONCAT" in runtime["env_vars"]["RELAX_PROPAGATE_ENV_VARS"].split(",")
        assert "--multimodal-keys" not in argv
        assert argv[argv.index("--global-batch-size") + 1] == "256"
        assert argv[argv.index("--eval-prompt-data") + 2] == str(Path(env["DATA_DIR"]) / "validation.jsonl")
    if "lora" in script.name:
        assert train_env["RELAX_KIMI_K3_EXPERT_SPLIT"] == "1"
        assert runtime["env_vars"]["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] == "false"
        assert ("--save-lora-only" in argv) is saving
    if "dsa-optimized" in script.name or "dsa-fused" in script.name:
        assert train_env["RELAX_KIMI_K31_DSA_SELECTION_REUSE"] == "1"
        assert train_env["RELAX_KIMI_K31_DSA_FUSED_INDEXER"] == ("1" if "dsa-fused" in script.name else "0")
        assert train_env["RELAX_KIMI_K31_DSA_QUERY_CHUNK_SIZE"] == "8192"
        for flag, expected in (
            ("--decoder-first-pipeline-num-layers", "24"),
            ("--decoder-last-pipeline-num-layers", "21"),
        ):
            assert argv[max(i for i, arg in enumerate(argv) if arg == flag) + 1] == expected
        import yaml

        config = yaml.safe_load(Path(argv[argv.index("--custom-config-path") + 1]).read_text())
        assert config["save"] is None and config["save_interval"] is None


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_kimi_k3_recipe_dry_run_never_calls_ray(tmp_path, script):
    env, capture = _environment(tmp_path)
    env["DRY_RUN"] = "1"
    if "kimi-k3.1-pro" in script.name:
        env["SAVE_DIR"] = str(tmp_path / "checkpoint with spaces")
    result = subprocess.run(
        ["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=30
    )
    assert not capture.exists()
    argv = shlex.split(result.stdout)
    assert argv[:3] == ["python3", "-m", "relax.entrypoints.train"]
    assert argv[argv.index("--hf-checkpoint") + 1] == env["MODEL_DIR"]


@pytest.mark.parametrize("no_wait", ["", "1"])
@pytest.mark.parametrize("model", ["kimi-k3", "kimi-k3.1-pro"])
def test_kimi_k3_recipe_preserves_explicit_wait_and_optimizer_save(tmp_path, no_wait, model):
    env, capture = _environment(tmp_path)
    env.update(RAY_NO_WAIT=no_wait, SAVE_OPTIMIZER="1", SAVE_DIR=str(tmp_path / "save"), FLA_TILELANG="1")
    script = ROOT / (
        "examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-sft.sh"
        if model == "kimi-k3.1-pro"
        else f"examples/models/kimi-k3/scripts/run-{model}-openr1mm-128xblackwell.sh"
    )
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    assert ("--no-wait" in argv) is bool(no_wait)
    assert "--no-save-optim" not in argv
    assert json.loads(argv[argv.index("--train-env-vars") + 1])["FLA_TILELANG"] == "1"


@pytest.mark.parametrize("model", ["kimi-k3", "kimi-k3.1-pro"])
def test_kimi_k3_optimizations_can_be_disabled(tmp_path, model):
    env, capture = _environment(tmp_path)
    env.update(
        SAVE_DIR=str(tmp_path / "save"),
        ATTENTION_BACKEND="flash",
        RELAX_KIMI_K3_EXPERT_SPLIT="0",
        RELAX_KIMI_K3_FUSED_SITU="0",
        RELAX_KIMI_K3_FUSED_ATTN_RES="0",
    )
    recipe = "pokemon" if model == "kimi-k3" else "openr1mm"
    script = ROOT / (
        "examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-lora.sh"
        if model == "kimi-k3.1-pro"
        else f"examples/models/kimi-k3/scripts/run-{model}-{recipe}-lora-64xblackwell.sh"
    )
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    train_env = json.loads(argv[argv.index("--train-env-vars") + 1])
    for key in ("RELAX_KIMI_K3_EXPERT_SPLIT", "RELAX_KIMI_K3_FUSED_SITU", "RELAX_KIMI_K3_FUSED_ATTN_RES"):
        assert train_env[key] == "0"
    if model == "kimi-k3":
        assert argv[max(i for i, arg in enumerate(argv) if arg == "--attention-backend") + 1] == "flash"


@pytest.mark.parametrize(
    "cache_mode", ["default", "explicit", "disabled", "prepared", "mismatch", "disabled_prepared"]
)
def test_kimi_k3_recipe_prepares_cache_after_outer_entrypoint(tmp_path, cache_mode):
    env, capture = _environment(tmp_path)
    env["SAVE_DIR"] = str(tmp_path / "save with spaces")
    for key in list(env):
        if key.startswith("RELAX_KERNEL_CACHE_"):
            del env[key]
    if cache_mode == "disabled":
        env["RELAX_KERNEL_CACHE_DIR"] = ""
    elif cache_mode in ("explicit", "prepared", "mismatch", "disabled_prepared"):
        env["RELAX_KERNEL_CACHE_DIR"] = str(tmp_path / "explicit cache")
    if cache_mode in ("prepared", "mismatch", "disabled_prepared"):
        env.update(
            RELAX_KERNEL_CACHE_SESSION_ID="existing",
            RELAX_KERNEL_CACHE_LOCAL_DIR="/tmp/test-local",
            RELAX_KERNEL_CACHE_KEY="test-key",
            RELAX_KERNEL_CACHE_BUILD_FINGERPRINT="test-build",
        )
        runtime = {key: value for key, value in env.items() if key.startswith("RELAX_KERNEL_CACHE_")}
        if cache_mode == "mismatch":
            runtime["RELAX_KERNEL_CACHE_SESSION_ID"] = "another-session"
        env["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": runtime})
        if cache_mode == "disabled_prepared":
            env["RELAX_KERNEL_CACHE_DIR"] = ""
    # Substitute only the cluster preparation calls. Use the real runtime-env
    # injection and the actual recipe submit command to catch ordering mistakes.
    wrapper = tmp_path / "record-cache.sh"
    wrapper.write_text(
        """set -eo pipefail
source "$ROOT/scripts/entrypoint/kernel-cache.sh"
relax_kernel_cache_configure() {
    export RELAX_KERNEL_CACHE_SESSION_ID=created
    export RELAX_KERNEL_CACHE_LOCAL_DIR=/tmp/test-local
    export RELAX_KERNEL_CACHE_KEY=test-key
    export RELAX_KERNEL_CACHE_BUILD_FINGERPRINT=test-build
}
relax_kernel_cache_start_agents() {
    echo started >> "$CACHE_CALLS"
}
source "$RECIPE"
"""
    )
    calls = tmp_path / "cache-calls"
    env.update(
        ROOT=str(ROOT),
        CACHE_CALLS=str(calls),
        RECIPE=str(ROOT / "examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-sft.sh"),
    )
    run = subprocess.run(["bash", str(wrapper)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    if cache_mode in ("mismatch", "disabled_prepared"):
        assert run.returncode != 0 and "inconsistent" in run.stderr
        assert not capture.exists() and not calls.exists()
        return
    assert run.returncode == 0, run.stderr
    argv = json.loads(capture.read_text())
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    if cache_mode == "disabled":
        assert "RELAX_KERNEL_CACHE_DIR" not in runtime["env_vars"]
        assert not calls.exists()
    else:
        expected = (
            str(Path(env["SAVE_DIR"]) / ".kernel-cache") if cache_mode == "default" else env["RELAX_KERNEL_CACHE_DIR"]
        )
        assert runtime["env_vars"]["RELAX_KERNEL_CACHE_DIR"] == expected
        assert runtime["env_vars"]["RELAX_KERNEL_CACHE_LOCAL_DIR"] == "/tmp/test-local"
        if cache_mode == "prepared":
            assert runtime["env_vars"]["RELAX_KERNEL_CACHE_SESSION_ID"] == "existing"
            assert not calls.exists()
        else:
            assert runtime["env_vars"]["RELAX_KERNEL_CACHE_SESSION_ID"] == "created"
            assert calls.read_text().splitlines() == ["started"]


@pytest.mark.parametrize("mode", ["sft", "lora"])
@pytest.mark.parametrize("raw", ["0", "1"])
def test_kimi_k31_external_data_steps_parallelism_and_resume(tmp_path, mode, raw):
    env, capture = _environment(tmp_path)
    env.update(
        SAVE_DIR=str(tmp_path / "save"),
        NUM_ROLLOUT="37",
        NUM_GPUS="64",
        CP_SIZE="4",
        EP_SIZE="16",
        RAW_TEXT=raw,
        RELAX_LORA_SHARE_EXPERT_ADAPTERS="true",
        TRAIN_FILE="train.jsonl",
        EVAL_INTERVAL="5",
        EVAL_FILE="heldout.jsonl",
        EVAL_NAME="heldout",
        LOAD_DIR=str(tmp_path / "resume"),
    )
    script = ROOT / f"examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-{mode}.sh"
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    assert argv[argv.index("--num-rollout") + 1] == "37"
    assert json.loads(argv[argv.index("--resource") + 1])["actor"] == [1, 64]
    assert argv[argv.index("--load") + 1] == env["LOAD_DIR"]
    assert argv[argv.index("--prompt-data") + 1] == str(Path(env["DATA_DIR"]) / "train.jsonl")
    assert argv[argv.index("--eval-prompt-data") + 2] == str(Path(env["DATA_DIR"]) / "heldout.jsonl")
    assert ("--multimodal-keys" in argv) is (raw == "0")
    assert ("--save-lora-only" in argv) is (mode == "lora")
    runtime = json.loads(next(a.split("=", 1)[1] for a in argv if a.startswith("--runtime-env-json=")))
    assert runtime["env_vars"].get("RELAX_SFT_RAW_TEXT_CONCAT") == ("1" if raw == "1" else None)
    if mode == "lora":
        assert runtime["env_vars"]["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] == "false"


@pytest.mark.parametrize("mode", ["sft", "lora"])
@pytest.mark.parametrize("mapping", [None, '{"image":"pictures","video":"clips"}'])
def test_kimi_k31_multimodal_mapping_json_is_preserved(tmp_path, mode, mapping):
    env, capture = _environment(tmp_path)
    env.update(SAVE_DIR=str(tmp_path / "save"), RAW_TEXT="0")
    env.pop("MULTIMODAL_KEYS", None)
    if mapping is not None:
        env["MULTIMODAL_KEYS"] = mapping
    script = ROOT / f"examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-{mode}.sh"
    subprocess.run(["bash", str(script)], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    actual = argv[argv.index("--multimodal-keys") + 1]
    assert json.loads(actual) == json.loads(mapping or '{"image":"images"}')
    if mapping is not None:
        assert actual == mapping
