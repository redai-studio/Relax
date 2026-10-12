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
        "RELAX_HICACHE_FILE_ROOT": str(tmp_path / "ssd"),
    }
    return example, env


@pytest.mark.parametrize("tier", ["cpu", "ssd"])
def test_hicache_submission_preserves_pd_driver_and_forwards_storage_root(submission, tier):
    example, env = submission
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache.sh")],
        env={**env, "HICACHE_TIER": tier},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())
    config_values = [args[i + 1] for i, arg in enumerate(args) if arg == "--sglang-config"]
    assert config_values[-1] == str(example / f"pd-hicache-{tier}-2p2d-tp2.yaml")
    runtime = json.loads(args[args.index("--runtime-env-json") + 1])
    assert runtime["env_vars"]["RELAX_HICACHE_FILE_ROOT"] == env["RELAX_HICACHE_FILE_ROOT"]
    assert runtime["env_vars"]["PYTHONPATH"].endswith("isolated-sglang")
    assert args[args.index("--") + 1 : args.index("--") + 4] == [
        "bash",
        str(example / "run_mini_swe_agent_pd.sh"),
        "--ray-driver",
    ]
    assert args[args.index("--sglang-router-decode-policy") + 1] == "power_of_two"


def test_ssd_launch_rejects_reusing_a_nonempty_run_directory(submission):
    example, env = submission
    cache = Path(env["RELAX_HICACHE_FILE_ROOT"])
    cache.mkdir()
    (cache / "existing.bin").write_bytes(b"keep")
    result = subprocess.run(
        ["bash", str(example / "run_mini_swe_agent_hicache.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "empty cache directory" in result.stderr
    assert (cache / "existing.bin").read_bytes() == b"keep"
    assert not Path(env["CAPTURE"]).exists()


def test_cpu_and_ssd_examples_differ_only_in_native_storage_settings():
    configs = [yaml.safe_load((EXAMPLE / f"pd-hicache-{tier}-2p2d-tp2.yaml").read_text()) for tier in ("cpu", "ssd")]
    for config in configs:
        model = config["sglang"][0]
        assert model["num_gpus_per_engine"] == 2
        prefill, decode = model["engine_groups"]
        assert prefill["num_gpus"] == decode["num_gpus"] == 4
        assert prefill["overrides"]["enable_hierarchical_cache"]
        assert not decode["overrides"]["enable_hierarchical_cache"]
        assert decode["overrides"]["hicache_storage_backend"] is None
        for group in (prefill, decode):
            assert not any("session" in key for key in group["overrides"])
        for key in list(prefill["overrides"]):
            if key.startswith("hicache_storage_"):
                del prefill["overrides"][key]
    assert configs[0] == configs[1]
