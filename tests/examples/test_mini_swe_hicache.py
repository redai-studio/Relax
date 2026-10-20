# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/mini_swe_agent"


@pytest.fixture
def submission(tmp_path):
    version = subprocess.check_output(["bash", "-c", 'echo "${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"'], text=True)
    if tuple(map(int, version.split())) < (5, 1):
        pytest.skip("The Mini SWE PD launcher requires Bash 5.1+")
    repo = tmp_path / "repo"
    example = repo / "examples/mini_swe_agent"
    example.mkdir(parents=True)
    for name in ("run_mini_swe_agent_pd.sh", "run_mini_swe_agent_hicache.sh"):
        shutil.copy2(EXAMPLE / name, example / name)
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
