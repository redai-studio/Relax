# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/mini_swe_agent"


@pytest.fixture
def submission(tmp_path):
    version = subprocess.check_output(["bash", "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'], text=True)
    if tuple(map(int, version.split())) < (5, 1):
        pytest.skip("The Mini SWE PD launcher requires Bash 5.1+")
    repo = tmp_path / "repo"
    example = repo / "examples/mini_swe_agent"
    example.mkdir(parents=True)
    for name in (
        "run_mini_swe_agent_pd.sh",
        "run_mini_swe_agent_hicache.sh",
        "run_mini_swe_agent_hicache_blackwell.sh",
    ):
        shutil.copy2(EXAMPLE / name, example / name)
    (example / "scripts").mkdir()
    shutil.copy2(EXAMPLE / "scripts/run-qwen35-35B-A3B-8xgpu-hicache-fully-async.sh", example / "scripts")
    models = repo / "scripts/models"
    models.mkdir(parents=True)
    (models / "qwen35-35B-A3B.sh").write_text("MODEL_ARGS=()\n")
    checkpoint = tmp_path / "models/Qwen3.5-35B-A3B"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "ray").write_text(
        f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
    )
    (binary / "ray").chmod(0o755)
    os.symlink(sys.executable, binary / "python")
    env = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "CAPTURE": str(tmp_path / "captured.json"),
        "RELAX_ENTRYPOINT_MODE": "ray-job",
        "RUNTIME_ENV_JSON": json.dumps({"env_vars": {"PYTHONPATH": "isolated-sglang"}}),
        "MINI_SWE_AGENT_VENV": str(tmp_path),
        "MODEL_DIR": str(tmp_path / "models"),
        "R2E_DATA_PATH": str(tmp_path / "data"),
        "R2E_SIF_DIR": str(tmp_path / "sif"),
        "SAVE_DIR": str(tmp_path / "save"),
        "EXP_ROOT": str(tmp_path / "exp"),
        "RUN_ID": "test-hicache",
        "ROLLOUT_BATCH_SIZE": "1",
        "AGENT_SERVER_TRAIN_CONCURRENCY": "8",
        "ENABLE_EVAL": "0",
    }
    env.pop("ENABLE_CHECKPOINT_SAVE", None)
    return example, env


@pytest.mark.parametrize("deployment", [None, "regular", "pd"])
def test_hicache_submission_selects_topology_and_preserves_runtime(submission, deployment):
    example, env = submission
    env.pop("HICACHE_DEPLOYMENT", None)
    if deployment is not None:
        env["HICACHE_DEPLOYMENT"] = deployment
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())
    assert "--save" in args and "--save-interval" in args
    config_values = [args[i + 1] for i, arg in enumerate(args) if arg == "--sglang-config"]
    config = "pd-hicache-cpu-2p2d-tp2.yaml" if deployment == "pd" else "hicache-cpu-4x-tp2.yaml"
    assert config_values[-1] == str(example / config)
    runtime = json.loads(args[args.index("--runtime-env-json") + 1])
    assert runtime["env_vars"]["PYTHONPATH"].endswith("isolated-sglang")
    assert args[args.index("--") + 1 : args.index("--") + 4] == [
        "bash",
        str(example / "run_mini_swe_agent_pd.sh"),
        "--ray-driver",
    ]


def test_hicache_submission_rejects_unknown_deployment(submission):
    example, env = submission
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache.sh")],
        env={**env, "HICACHE_DEPLOYMENT": "unknown"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "regular or pd" in result.stderr
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize("scheduling", ["group", "incremental"])
def test_blackwell_submission_preserves_128k_and_complete_group_batch(submission, scheduling):
    example, env = submission
    env.update(GLOBAL_BATCH_SIZE="128", AGENT_SERVER_TRAIN_CONCURRENCY="64", AGENTIC_SCHEDULING=scheduling)
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache_blackwell.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())

    def last_value(flag):
        return [args[i + 1] for i, item in enumerate(args) if item == flag][-1]

    assert last_value("--global-batch-size") == "128"
    assert last_value("--rollout-batch-size") == "16"
    assert last_value("--n-samples-per-prompt") == "8"
    assert last_value("--seq-length") == last_value("--rollout-max-context-len") == "131072"
    assert last_value("--sglang-context-length") == last_value("--max-tokens-per-gpu") == "131072"
    assert last_value("--agentic-concurrency") == ("8" if scheduling == "group" else "16")
    assert ("--agentic-fixed-step-eager" in args) == (scheduling == "incremental")
    assert "--save" not in args and "--save-interval" not in args and "--save-hf" not in args
    assert last_value("--rollout-result-dir") == str(Path(env["EXP_ROOT"]) / "rollout_result")
    assert not Path(env["SAVE_DIR"]).exists()
    runtime = json.loads(last_value("--runtime-env-json"))["env_vars"]
    assert runtime["AGENT_SERVER_MAX_TURNS"] == "100"
    assert runtime["AGENT_SERVER_SUPERVISED_SESSIONS"] == "1"


@pytest.mark.parametrize("scheduling", ["group", "incremental"])
def test_fully_async_submission_uses_disjoint_four_gpu_roles_and_oversampling(submission, scheduling):
    example, env = submission
    env.update(GLOBAL_BATCH_SIZE="256", AGENT_SERVER_TRAIN_CONCURRENCY="64", AGENTIC_SCHEDULING=scheduling)
    env.pop("RAY_NO_WAIT", None)
    result = subprocess.run(
        ["bash", str(example / "scripts/run-qwen35-35B-A3B-8xgpu-hicache-fully-async.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())

    def last_value(flag):
        return [args[i + 1] for i, item in enumerate(args) if item == flag][-1]

    assert json.loads(last_value("--resource")) == {"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}
    assert "--fully-async" in args and "--colocate" not in args
    assert "--no-wait" not in args and "--use-clearml" in args
    assert last_value("--global-batch-size") == "256"
    assert last_value("--rollout-batch-size") == "32"
    assert last_value("--over-sampling-batch-size") == "48"
    assert last_value("--num-iters-per-train-update") == "8"
    assert last_value("--num-steps-per-rollout") == "1"
    assert last_value("--tensor-model-parallel-size") == last_value("--expert-model-parallel-size") == "4"
    assert last_value("--context-parallel-size") == "1"
    assert last_value("--seq-length") == last_value("--sglang-context-length") == "131072"
    assert last_value("--max-tokens-per-gpu") == "131072"
    assert last_value("--agentic-concurrency") == ("8" if scheduling == "group" else "48")
    assert ("--agentic-eager-session-generation" in args) == (scheduling == "incremental")
    assert "--agentic-fixed-step-eager" not in args
    assert "--save" not in args and "--save-interval" not in args and "--save-hf" not in args
    runtime = json.loads(last_value("--runtime-env-json"))["env_vars"]
    assert runtime["AGENT_SERVER_MAX_TURNS"] == "100"
    assert runtime["AGENT_SERVER_RESUMABLE_GENERATION"] == "1"
    assert runtime["AGENT_SERVER_SESSION_MEMORY_MIB"] == "8192"
    model = yaml.safe_load((EXAMPLE / "hicache-cpu-2x-tp2-blackwell.yaml").read_text())["sglang"][0]
    assert model["num_gpus_per_engine"] == 2
    assert sum(group["num_gpus"] for group in model["engine_groups"]) == 4


@pytest.mark.parametrize(
    ("filename", "roles", "gpus", "engine_gpus", "attention_dp"),
    [
        ("hicache-cpu-4x-tp2.yaml", ["regular"], [8], 2, 1),
        ("pd-hicache-cpu-2p2d-tp2.yaml", ["prefill", "decode"], [4, 4], 2, 1),
        ("pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml", ["prefill", "decode"], [4, 4], 4, 2),
    ],
)
def test_cpu_cache_examples_support_both_topologies(filename, roles, gpus, engine_gpus, attention_dp):
    model = yaml.safe_load((EXAMPLE / filename).read_text())["sglang"][0]
    assert model["num_gpus_per_engine"] == engine_gpus
    groups = model["engine_groups"]
    assert [group["worker_type"] for group in groups] == roles
    assert [group["num_gpus"] for group in groups] == gpus
    for group in groups:
        overrides = group["overrides"]
        assert overrides["hicache_storage_backend"] is None
        assert overrides["enable_hierarchical_cache"] == (group["worker_type"] != "decode")
        if attention_dp > 1:
            assert overrides["dp_size"] == attention_dp
            assert overrides["enable_dp_attention"] is True
            assert overrides["ep_size"] == 4
            assert engine_gpus // attention_dp == 2
        if group["worker_type"] != "decode":
            assert overrides["mamba_radix_cache_strategy"] == "extra_buffer"
            assert overrides["mamba_track_interval"] % overrides["page_size"] == 0
            assert overrides["hicache_write_policy"] == "write_through"
            assert overrides["chunked_prefill_size"] // attention_dp == 8192
            assert overrides["max_mamba_cache_size"] // attention_dp == 666
