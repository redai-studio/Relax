# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise concrete recipes without importing GPU packages or contacting
Ray."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MM16 = ROOT / "scripts/training/sft/run-qwen3.6-35B-A3B-sft-16xgpu-multimodal-blackwell.sh"
SFT64 = ROOT / "scripts/training/sft/run-qwen3.5-397B-A17B-sft-64xblackwell.sh"
CPT64 = ROOT / "scripts/training/cpt/run-qwen3.5-397B-A17B-cpt-64xblackwell.sh"
ENTRIES = [MM16, SFT64, CPT64]


def run_entry(entry, *arguments, **overrides):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("B300_", "TRAIN_", "RELAX_", "MOE_", "SFT_")  # gitleaks:allow -- compatibility literal
        )
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


def test_397b_real_entry_delegates_without_inheriting_nounset(tmp_path):
    entry = tmp_path / SFT64.relative_to(ROOT)
    entry.parent.mkdir(parents=True)
    entry.write_text(SFT64.read_text())
    launcher = tmp_path / "scripts/entrypoint/ray-job.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        'unused="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}"\n'
        "python3 - \"$@\" <<'PY'\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "PY\n"
    )
    env = os.environ.copy()
    for key in ("RELAX_ENTRYPOINT_MODE", "NCCL_SOCKET_IFNAME", "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"):
        env.pop(key, None)
    env.update(MODEL_DIR="/model", DATA_DIR="/data", SAVE_DIR="/checkpoints", DRY_RUN="0")
    arguments = ["--multimodal", "--num-rollout", "25", "--save", "path with spaces"]
    result = subprocess.run(["bash", str(entry), *arguments], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [str(entry), *arguments]


def test_397b_multimodal_preserves_delivered_vision_defaults():
    args = render(SFT64, "--multimodal")["args"]
    assert value(args, "--image-max-token-num") == "512"
    assert value(args, "--image-min-token-num") == "4"
    assert "--freeze-vision-model" in args
    assert "--freeze-vision-projection" in args
    overridden = render(SFT64, "--multimodal", FREEZE_VISION="0", IMAGE_MAX_TOKEN_NUM="1024", IMAGE_MIN_TOKEN_NUM="8")[
        "args"
    ]
    assert value(overridden, "--image-max-token-num") == "1024"
    assert value(overridden, "--image-min-token-num") == "8"
    assert "--freeze-vision-model" not in overridden
    assert "--freeze-vision-projection" not in overridden


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


def test_397b_accepts_direct_paths_and_resume_overrides(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    data = tmp_path / "dataset.jsonl"
    data.write_text("")
    extra = ["--ckpt-step", "19", "--num-rollout", "40", "--use-checkpoint-opt-param-scheduler"]
    args = render(SFT64, "--multimodal", *extra, MODEL_DIR=str(model), DATA_DIR=str(data))["args"]
    assert value(args, "--hf-checkpoint") == value(args, "--ref-load") == str(model)
    assert value(args, "--prompt-data") == str(data)
    assert args[-len(extra) :] == extra


def test_397b_preserves_legacy_dataset_parent(tmp_path):
    data = tmp_path / "multimodal.jsonl"
    data.write_text("")
    args = render(SFT64, "--multimodal", DATA_DIR=str(tmp_path))["args"]
    assert value(args, "--prompt-data") == str(data)


@pytest.mark.parametrize("flag", ["--tensor-model-parallel-size=8", "--moe-flex-dispatcher-backend"])
def test_397b_forwarding_does_not_override_fixed_topology(flag):
    result = run_entry(SFT64, flag, "hybridep")
    assert result.returncode != 0
    assert "fixed topology" in result.stderr


def test_397b_submission_id_reaches_ray_without_real_submission(tmp_path):
    entry = tmp_path / "scripts/training/sft" / SFT64.name
    entry.parent.mkdir(parents=True)
    entry.write_text(SFT64.read_text())
    model_script = tmp_path / "scripts/models/qwen35-397B-A17B.sh"
    model_script.parent.mkdir(parents=True)
    model_script.write_text((ROOT / "scripts/models/qwen35-397B-A17B.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ray = bin_dir / "ray"
    fake_ray.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE_PATH"\n')
    fake_ray.chmod(0o755)
    captured = tmp_path / "ray-args"
    result = run_entry(
        entry,
        "--ckpt-step",
        "19",
        DRY_RUN="0",
        RELAX_ENTRYPOINT_MODE="test",
        RAY_ADDRESS="http://unused.invalid",
        SUBMISSION_ID="resume-test",
        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
        CAPTURE_PATH=str(captured),
    )
    assert result.returncode == 0, result.stderr
    args = captured.read_text().split("\0")[:-1]
    assert value(args, "--submission-id") == "resume-test"
    assert args[-2:] == ["--ckpt-step", "19"]
