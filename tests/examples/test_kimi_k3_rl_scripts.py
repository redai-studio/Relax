# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Record RL launch commands without submitting jobs or importing GPU code."""

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.examples.test_kimi_k3_scripts import ROOT, _environment


FULL = ROOT / "examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh"
SMOKE = ROOT / "scripts/training/text/run-kimi-k3-5l-8xgpu-grpo.sh"


def _launch(script, env, capture, extra_args=()):
    subprocess.run(["bash", str(script), *extra_args], cwd=ROOT, env=env, check=True, capture_output=True, timeout=30)
    argv = json.loads(capture.read_text())
    runtime = json.loads(next(item.split("=", 1)[1] for item in argv if item.startswith("--runtime-env-json=")))
    return argv, runtime["env_vars"]


@pytest.mark.parametrize("script", [FULL, SMOKE], ids=lambda p: p.stem)
@pytest.mark.parametrize("wait", [False, True])
def test_kimi_k3_rl_recipe_preserves_layout_and_worker_settings(tmp_path, monkeypatch, script, wait):
    monkeypatch.setenv("NUM_ROLLOUT", "4")
    monkeypatch.setenv("SAVE_INTERVAL", "2")
    env, capture = _environment(tmp_path)
    env.update(SAVE_DIR=str(tmp_path / "save with spaces"), RAY_NO_WAIT="" if wait else "1")
    argv, worker_env = _launch(script, env, capture)
    value = lambda flag: argv[max(i for i, item in enumerate(argv) if item == flag) + 1]
    assert ("--no-wait" in argv) is not wait
    expected_model = str(Path(env["MODEL_DIR"]) / "Kimi-K3") if script == FULL else env["MODEL_DIR"]
    assert value("--hf-checkpoint") == expected_model
    assert env["DATA_DIR"] in value("--prompt-data")
    assert value("--save") == str(Path(env["SAVE_DIR"]) / env["EXP_NAME"])
    assert "--use-clearml" in argv and "--use-metrics-service" in argv
    assert "CLEARML_CONFIG_FILE" not in worker_env
    train_env = json.loads(value("--train-env-vars"))
    assert train_env["FLA_TILELANG"] == "0"
    assert train_env["OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG"] == "1"
    assert value("--advantage-estimator") == "grpo" and "--use-tis" in argv
    if script == FULL:
        assert json.loads(value("--resource")) == {"actor": [1, 128], "rollout": [1, 128]}
        for flag, expected in {
            "--tensor-model-parallel-size": "4",
            "--pipeline-model-parallel-size": "4",
            "--context-parallel-size": "2",
            "--expert-model-parallel-size": "32",
            "--decoder-first-pipeline-num-layers": "21",
            "--decoder-last-pipeline-num-layers": "24",
            "--sglang-dp-size": "2",
            "--sglang-ep-size": "16",
            "--rollout-num-gpus-per-engine": "16",
            "--optimizer-offload-fraction": "0.9",
            "--num-rollout": "200",
            "--save-interval": "200",
            "--sglang-load-format": "dummy",
            "--sglang-moe-runner-backend": "flashinfer_mxfp4",
            "--custom-rm-path": "examples.models.kimi-k3.rewards.openr1mm.reward_func",
        }.items():
            assert value(flag) == expected
        assert "--rm-type" not in argv
        assert "--use-kl-loss" not in argv and "--ref-load" not in argv
        assert "--load" not in argv and "--no-save-optim" in argv
        assert worker_env["MEGATRON_SYNC_SAVE_BOUNDED_STAGING"] == "1"
        assert worker_env["MEGATRON_SYNC_SAVE_STAGE_BYTES"] == "1073741824"
        assert worker_env["RAY_SERVE_RUN_SYNC_IN_THREADPOOL"] == "1"
        assert worker_env["PYTORCH_ALLOC_CONF"] == worker_env["PYTORCH_CUDA_ALLOC_CONF"] == ""
        assert {"MEGATRON_SYNC_SAVE_BOUNDED_STAGING", "MEGATRON_SYNC_SAVE_STAGE_BYTES"} <= set(
            worker_env["RELAX_PROPAGATE_ENV_VARS"].split(",")
        )
    else:
        assert json.loads(value("--resource")) == {"actor": [1, 8], "rollout": [1, 8]}
        assert value("--num-layers") == "5" and value("--num-experts") == "128"
        assert value("--tensor-model-parallel-size") == "2"
        assert value("--expert-model-parallel-size") == "4"
        assert value("--entropy-coef") == "0.01"
        assert value("--load") == value("--save")
        assert "--no-save-optim" not in argv


def test_kimi_k3_full_rl_preserves_explicit_run_length(tmp_path):
    env, capture = _environment(tmp_path)
    env.update(SAVE_DIR=str(tmp_path / "save"), NUM_ROLLOUT="4", SAVE_INTERVAL="2")
    argv, _ = _launch(FULL, env, capture)
    assert argv[argv.index("--num-rollout") + 1] == "4"
    assert argv[argv.index("--save-interval") + 1] == "2"


@pytest.mark.parametrize("script", [FULL, SMOKE], ids=lambda p: p.stem)
def test_kimi_k3_rl_recipe_dry_run_keeps_checkpoint_path_stable(tmp_path, script):
    env, capture = _environment(tmp_path)
    env.update(DRY_RUN="1", SAVE_DIR=str(tmp_path / "save"))
    env.pop("EXP_NAME")
    outputs = []
    for stamp in ("first-start", "second-start"):
        date = tmp_path / "date"
        date.write_text(f"#!/bin/sh\necho {stamp}\n")
        date.chmod(0o755)
        run = subprocess.run(["bash", str(script)], env=env, cwd=ROOT, check=True, capture_output=True, text=True)
        outputs.append(shlex.split(run.stdout))
    assert not capture.exists()
    assert outputs[0][outputs[0].index("--save") + 1] == outputs[1][outputs[1].index("--save") + 1]
    assert (
        outputs[0][outputs[0].index("--tb-experiment-name") + 1]
        != outputs[1][outputs[1].index("--tb-experiment-name") + 1]
    )


def test_kimi_k3_full_rl_can_roll_back_bounded_staging(tmp_path):
    env, capture = _environment(tmp_path)
    env.update(
        SAVE_DIR=str(tmp_path / "save"), MEGATRON_SYNC_SAVE_BOUNDED_STAGING="0", MEGATRON_SYNC_SAVE_STAGE_BYTES="0"
    )
    _, worker_env = _launch(FULL, env, capture)
    assert worker_env["MEGATRON_SYNC_SAVE_BOUNDED_STAGING"] == "0"
    assert worker_env["MEGATRON_SYNC_SAVE_STAGE_BYTES"] == "0"


def test_kimi_k3_full_rl_requires_persistent_save_directory(tmp_path):
    env, capture = _environment(tmp_path)
    env.pop("CHECKPOINT_DIR", None)
    run = subprocess.run(["bash", str(FULL)], env=env, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0 and "Set SAVE_DIR" in run.stderr
    assert not capture.exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_k3_full_rl_dynamic_filter_is_explicit(tmp_path, enabled):
    env, capture = _environment(tmp_path)
    env["SAVE_DIR"] = str(tmp_path / "save")
    filter_path = "relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std"
    extra_args = (
        ["--dynamic-sampling-filter-path", filter_path, "--over-sampling-batch-size", "256"] if enabled else []
    )
    argv, _ = _launch(FULL, env, capture, extra_args)
    if enabled:
        assert argv[argv.index("--dynamic-sampling-filter-path") + 1] == filter_path
        assert argv[argv.index("--over-sampling-batch-size") + 1] == "256"
    else:
        assert "--dynamic-sampling-filter-path" not in argv
        assert "--over-sampling-batch-size" not in argv


@pytest.mark.parametrize("enabled", [False, True])
def test_kimi_k3_full_rl_expert_routing_override(tmp_path, enabled):
    env, capture = _environment(tmp_path)
    env["SAVE_DIR"] = str(tmp_path / "save")
    env["COLOCATE_EXPERT_WEIGHT_ROUTING"] = "1" if enabled else "0"
    argv, _ = _launch(FULL, env, capture)
    assert ("--colocate-expert-weight-routing" in argv) is enabled
