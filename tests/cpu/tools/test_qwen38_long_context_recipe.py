# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Execute recipe shells at the CLI boundary without contacting Ray or CUDA."""

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/training/text"


def run_recipe(tmp_path, *, script="run-qwen38-long-context.sh", extra=(), **overrides):
    env = os.environ.copy()
    for name in (
        "SCENARIO",
        "DATA_MODE",
        "RESUME",
        "EXP_NAME",
        "MAX_TOKENS_PER_GPU",
        "QWEN38_PLE_INPUT_HALO_TRANSPORT",
        "RUNTIME_ENV_JSON",
        "RAY_NO_WAIT",
        "RAY_API_SERVER_ADDRESS",
        "RAY_JOBS_ADDRESS",
    ):
        env.pop(name, None)
    env.update(
        DRY_RUN="1",
        MODEL_DIR=str(tmp_path / "model"),
        DATA_DIR=str(tmp_path / "data.jsonl"),
        SAVE_DIR=str(tmp_path / "save"),
        MEGATRON=env.get("MEGATRON", "/run/ci-megatron"),
        RELAX_HYBRID_EP_COMMIT="recipe-test",
        RELAX_ENTRYPOINT_MODE="test",
        SUBMISSION_ID="recipe-test-job",
    )
    env.update(overrides)
    return subprocess.run(["bash", str(SCRIPTS / script), *extra], env=env, capture_output=True, text=True, timeout=60)


def option(argv, name):
    return argv[len(argv) - 1 - argv[::-1].index(name) + 1]


@pytest.mark.parametrize(
    "scenario,seq,cp,budget,mode,dataset",
    [
        ("mixed512", 524288, 8, 65536, "full", "RepeatedSFTDataset"),
        ("filled512", 524288, 8, 65536, "full", "FilledSFTDataset"),
        ("mixed1m", 1048576, 16, 32768, "adaptive", "DoubledSFTDataset"),
        ("filled1m", 1048576, 16, 32768, "adaptive", "FilledSFTDataset"),
    ],
)
def test_scenario_cli_and_real_parser(tmp_path, monkeypatch, scenario, seq, cp, budget, mode, dataset):
    result = run_recipe(tmp_path, SCENARIO=scenario)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    argv, env = payload["argv"], payload["runtime_env"]["env_vars"]
    assert option(argv, "--seq-length") == str(seq)
    assert option(argv, "--context-parallel-size") == str(cp)
    assert option(argv, "--max-tokens-per-gpu") == str(budget)
    assert option(argv, "--custom-dataset-class") == "examples.long_context_repeat." + dataset
    assert option(argv, "--num-rollout") == "200"
    assert option(argv, "--load").endswith(f"qwen38-{scenario}-32gpu")
    assert "--save" not in argv and "--use-clearml" not in argv
    assert env["QWEN38_PLE_INPUT_HALO_TRANSPORT"] == mode
    assert json.loads(option(argv, "--train-env-vars"))["QWEN38_PLE_INPUT_HALO_TRANSPORT"] == mode
    assert not (tmp_path / "save").exists()
    # Parse actual emitted arguments. Only HF files and the GPU architecture
    # query are bypassed: no weights, CUDA context, or cluster is needed.
    pytest.importorskip("megatron.training.arguments")
    import sys
    from unittest.mock import patch

    import torch

    from relax.utils.arguments import parse_args

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["recipe-test", *argv[3:], "--skip-hf-validate", "--disable-s3-model-download"])
    with patch("megatron.training.arguments.get_device_arch_version", return_value=10):
        args = parse_args()
    assert (args.seq_length, args.context_parallel_size, args.pipeline_model_parallel_size) == (seq, cp, 2)
    assert (args.tensor_model_parallel_size, args.expert_model_parallel_size, args.global_batch_size) == (1, 8, 32)
    assert args.num_layers == 48 and args.mtp_num_layers == 1
    assert args.empty_unused_memory_level == 0
    assert args.num_epoch is None
    assert args.max_tokens_per_gpu == budget
    assert args.resource["actor"] == [1, 32]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "script,mode,scenario",
    [
        ("run-qwen38-512k-cp.sh", "mixed", "mixed512"),
        ("run-qwen38-512k-cp.sh", "filled", "filled512"),
        ("run-qwen38-1m-cp.sh", "double", "mixed1m"),
        ("run-qwen38-1m-cp.sh", "filled", "filled1m"),
    ],
)
def test_wrappers_and_argument_forwarding(tmp_path, script, mode, scenario):
    result = run_recipe(tmp_path, script=script, DATA_MODE=mode, extra=("--lr", "1e-5"))
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["argv"]
    assert option(argv, "--load").endswith(f"qwen38-{scenario}-32gpu")
    assert option(argv, "--lr") == "1e-5"


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"SCENARIO": "unknown"}, "SCENARIO must"),
        ({"MAX_TOKENS_PER_GPU": "0"}, "Require positive"),
        ({"MAX_TOKENS_PER_GPU": "1048576"}, "Require positive"),
        ({"RESUME": "yes"}, "RESUME must"),
        ({"RESUME": "1"}, "Set EXP_NAME"),
        ({"RESUME": "1", "EXP_NAME": "saved"}, "original checkpoint tracker"),
    ],
)
def test_invalid_recipe_rejected_without_submission(tmp_path, overrides, fragment):
    result = run_recipe(tmp_path, **overrides)
    assert result.returncode != 0
    assert fragment in result.stderr


def test_checkpoint_identity_and_resume_scheduler(tmp_path):
    # Capture the real submission command at the Ray executable boundary.
    # All recipe setup and checkpoint-identity handling execute normally.
    executable = tmp_path / "bin/ray"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE_RAY']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    capture = tmp_path / "ray.json"
    result = run_recipe(
        tmp_path,
        DRY_RUN="0",
        EXP_NAME="saved",
        SAVE_INTERVAL="100",
        CAPTURE_RAY=str(capture),
        PATH=str(executable.parent) + os.pathsep + os.environ["PATH"],
    )
    assert result.returncode == 0, result.stderr
    submitted = json.loads(capture.read_text())
    assert submitted[:2] == ["job", "submit"]
    assert "--no-wait" not in submitted
    checkpoint = tmp_path / "save/saved"
    assert (checkpoint / "recipe-config.json").is_file()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("99\n")
    fresh = run_recipe(tmp_path, EXP_NAME="saved")
    assert fresh.returncode != 0 and "Checkpoint already exists" in fresh.stderr
    resumed = run_recipe(tmp_path, RESUME="1", EXP_NAME="saved", NUM_STEPS="210", extra=("--ckpt-step", "99"))
    assert resumed.returncode == 0, resumed.stderr
    argv = json.loads(resumed.stdout)["argv"]
    assert "--use-checkpoint-opt_param-scheduler" in argv
    assert option(argv, "--ckpt-step") == "99"
    assert option(argv, "--num-rollout") == "210"
    for changed in ({"SCENARIO": "filled512"}, {"GBS": "16"}, {"SEED": "1"}, {"MAX_TOKENS_PER_GPU": "32768"}):
        mismatch = run_recipe(tmp_path, RESUME="1", EXP_NAME="saved", **changed)
        assert mismatch.returncode != 0 and "differs from the saved" in mismatch.stderr


@pytest.mark.parametrize("extra", [("--load", "/other"), ("--load=/other",), ("--save", "/other"), ("--save=/other",)])
def test_checkpoint_path_override_rejected(tmp_path, extra):
    result = run_recipe(tmp_path, extra=extra)
    assert result.returncode != 0
    assert "Use SAVE_DIR, EXP_NAME and SAVE_INTERVAL" in result.stderr
    assert not (tmp_path / "save").exists()


@pytest.mark.parametrize("no_wait,exit_code", [(None, 0), ("", 0), ("1", 0), (None, 23)])
def test_submission_wait_and_failure_status(tmp_path, no_wait, exit_code):
    executable = tmp_path / "bin/ray"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE_RAY']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ['FAKE_RAY_EXIT']))\n"
    )
    executable.chmod(0o755)
    capture = tmp_path / "ray.json"
    overrides = {} if no_wait is None else {"RAY_NO_WAIT": no_wait}
    result = run_recipe(
        tmp_path,
        DRY_RUN="0",
        CAPTURE_RAY=str(capture),
        FAKE_RAY_EXIT=str(exit_code),
        PATH=str(executable.parent) + os.pathsep + os.environ["PATH"],
        **overrides,
    )
    assert result.returncode == exit_code, result.stderr
    submitted = json.loads(capture.read_text())
    assert ("--no-wait" in submitted) == bool(no_wait)


def test_node_discovery_and_submission_share_ray_address(tmp_path):
    # Run the real shared entrypoint; replace only external executables. No Ray
    # service, GPU query, cleanup, or kernel-cache agent is started by this test.
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "ray"
    executable.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "with Path(os.environ['CAPTURE_RAY']).open('a') as f:\n"
        " f.write(json.dumps({'argv':sys.argv[1:],'address':os.environ.get('RAY_API_SERVER_ADDRESS')})+'\\n')\n"
        "if sys.argv[1:3]==['list','nodes']:\n"
        " print(json.dumps([{'state':'ALIVE','resources_total':{'GPU':8},'is_head_node':True,'node_ip':'192.0.2.1'}]))\n"
    )
    executable.chmod(0o755)
    gpu = binary / "nvidia-smi"
    gpu.write_text("#!/bin/sh\nexit 1\n")
    gpu.chmod(0o755)
    jq = binary / "jq"
    jq.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdin.read()\nprint('192.0.2.1')\n")
    jq.chmod(0o755)
    capture = tmp_path / "ray.jsonl"
    address = "http://192.0.2.20:8265"
    result = run_recipe(
        tmp_path,
        DRY_RUN="0",
        RELAX_ENTRYPOINT_MODE="",
        RELAX_SKIP_CLUSTER_CLEANUP="1",
        RELAX_KERNEL_CACHE_DIR="",
        RAY_NO_WAIT="1",
        RAY_API_SERVER_ADDRESS=address,
        CAPTURE_RAY=str(capture),
        PATH=str(binary) + os.pathsep + os.environ["PATH"],
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [call["argv"][:2] for call in calls] == [["list", "nodes"], ["job", "submit"]]
    assert all(call["address"] == address for call in calls)
