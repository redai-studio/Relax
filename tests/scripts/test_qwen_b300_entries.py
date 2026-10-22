# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise concrete recipes without importing GPU packages or contacting
Ray."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MM16 = ROOT / "scripts/training/sft/run-qwen3.6-35B-A3B-sft-16xgpu-multimodal-b300.sh"
SFT64 = ROOT / "scripts/training/sft/run-qwen3.5-397B-A17B-sft-64xB300.sh"
CPT64 = ROOT / "scripts/training/cpt/run-qwen3.5-397B-A17B-cpt-64xB300.sh"
ENTRIES = [MM16, SFT64, CPT64]


def run_entry(entry, *arguments, **overrides):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("B300_", "TRAIN_", "RELAX_", "MOE_", "SFT_"))
    }
    for key in ("MULTIMODAL_KEYS", "CONVERSATION_KEY_MAP", "SAVE_CHECKPOINT", "LOAD_DIR", "HF_CHECKPOINT"):
        env.pop(key, None)
    env.update(MODEL_DIR="/model", DATA_DIR="/data", SAVE_DIR="/checkpoints", DRY_RUN="1", EXP_NAME="stable")
    env.update(overrides)
    return subprocess.run(["bash", str(entry), *arguments], env=env, capture_output=True, text=True, timeout=20)


def render(entry, *arguments, **overrides):
    result = run_entry(entry, *arguments, **overrides)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def value(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.stem)
def test_resume_and_checkpoint_paths_are_stable(entry):
    first = render(entry)["args"]
    second = render(entry)["args"]
    for args in (first, second):
        assert value(args, "--load") == value(args, "--save") == "/checkpoints/stable"
        assert value(args, "--save-interval") == "50"
        assert int(value(args, "--max-actor-ckpt-to-keep")) > 0
    changed = render(entry, EXP_NAME="other")["args"]
    assert value(changed, "--load") == value(changed, "--save") == "/checkpoints/other"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.stem)
@pytest.mark.parametrize("mapping", [None, '{"image":"images","video":"videos"}'])
def test_multimodal_keys_preserve_json(entry, mapping):
    overrides = {} if mapping is None else {"MULTIMODAL_KEYS": mapping}
    args = render(entry, *([] if entry == MM16 else ["--multimodal"]), **overrides)["args"]
    assert json.loads(value(args, "--multimodal-keys")) == json.loads(mapping or '{"image":"images"}')


def test_multimodal_baseline_keeps_numerical_and_environment_defaults():
    config = render(MM16)
    args = config["args"]
    expected = {
        "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "2",
        "--context-parallel-size": "4",
        "--expert-model-parallel-size": "4",
        "--moe-flex-dispatcher-backend": "hybridep",
        "--sft-prefetch-buffer-size": "128",
        "--data-pad-size-multiplier": "8192",
        "--sft-packing-strategy": "binpack",
        "--recompute-num-layers": "20",
        "--image-max-token-num": "512",
    }
    for name, expected_value in expected.items():
        assert value(args, name) == expected_value
    assert "--moe-hybridep-pad-variable-tokens" in args
    assert "--freeze-vision-model" in args
    env = config["runtime_env"]["env_vars"]
    assert env["RELAX_LEGACY_FLA_ROOT"] == "/opt/relax/compat/fla-0.4.2"
    assert env["PYTHONPATH"].split(":")[0] == env["RELAX_LEGACY_FLA_ROOT"]
    assert env["RELAX_DISABLE_VISION_BRANCH_SYNC"] == "0"
    assert env["RELAX_LOGPROB_CE_IMPL"] == env["RELAX_MTP_CE_IMPL"] == "native"


def test_multimodal_override_paths_and_optional_saving():
    args = render(
        MM16,
        MODEL_DIR="/other-model",
        DATA_DIR="/other-data",
        SAVE_CHECKPOINT="0",
        LOAD_DIR="/resume",
        IMAGE_MAX_TOKEN_NUM="1024",
        FREEZE_VISION="0",
    )["args"]
    assert value(args, "--hf-checkpoint") == "/other-model/Qwen3.6-35B-A3B"
    assert value(args, "--prompt-data") == "/other-data/multimodal-mixture.yaml"
    assert value(args, "--load") == "/resume"
    assert value(args, "--image-max-token-num") == "1024"
    assert "--save" not in args
    assert "--freeze-vision-model" not in args


@pytest.mark.parametrize("entry", [SFT64, CPT64], ids=lambda entry: entry.stem)
def test_64_gpu_recipes_reject_hybridep_before_submission(entry):
    result = run_entry(entry, MOE_FLEX_DISPATCHER_BACKEND="hybridep")
    assert result.returncode != 0
    assert "deepep" in result.stderr.lower()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.stem)
def test_scripts_have_no_hpc_dependency_and_valid_shell(entry):
    subprocess.run(["bash", "-n", str(entry)], check=True)
    assert "hpc/qwen3_moe" not in entry.read_text()


@pytest.mark.parametrize(
    "flag",
    [
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--resource",
    ],
)
def test_multimodal_topology_cannot_be_overridden(flag):
    result = run_entry(MM16, flag, "8")
    assert result.returncode != 0
    assert flag in result.stderr


@pytest.mark.parametrize("entry,mode", [(SFT64, "sft"), (CPT64, "cpt")])
def test_397b_topology_and_mode_defaults(entry, mode):
    args = render(entry)["args"]
    for name, expected in {
        "--tensor-model-parallel-size": "2",
        "--pipeline-model-parallel-size": "4",
        "--context-parallel-size": "4",
        "--expert-model-parallel-size": "8",
        "--expert-tensor-parallel-size": "1",
        "--moe-flex-dispatcher-backend": "deepep",
    }.items():
        assert value(args, name) == expected
    assert json.loads(value(args, "--resource"))["actor"] == [1, 64]
    if mode == "cpt":
        assert value(args, "--sft-training-mode") == "cpt"
        assert value(args, "--mtp-num-layers") == "0"
    else:
        assert "--enable-mtp-training" in args
    assert "--moe-hybridep-pad-variable-tokens" not in args
    text = render(entry, EXP_NAME="")["args"]
    multimodal = render(entry, "--multimodal", EXP_NAME="")["args"]
    assert value(text, "--save") != value(multimodal, "--save")


def test_multimodal_runtime_preserves_base_environment():
    config = render(MM16, RUNTIME_ENV_JSON=json.dumps({"env_vars": {"EXISTING": "yes", "PYTHONPATH": "/base"}}))
    env = config["runtime_env"]["env_vars"]
    assert env["EXISTING"] == "yes"
    assert env["PYTHONPATH"] == "/opt/relax/compat/fla-0.4.2:/base"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.stem)
def test_entry_disables_cluster_cleanup_before_setup(entry):
    source = entry.read_text()
    assert source.index("export RELAX_SKIP_CLUSTER_CLEANUP=1") < source.index("/scripts/entrypoint/ray-job.sh")
