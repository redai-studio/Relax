# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Offline checks for private configuration loading and dataset selection."""

import os
import subprocess
from pathlib import Path

import pytest

from examples.osworld_agentic.scripts.prepare_data import _select


RECIPE = Path(__file__).resolve().parents[2] / "examples/osworld_agentic"


@pytest.mark.parametrize("options,enabled", [([], "1"), (["--no-save-trajectories"], "0")])
def test_trajectory_options_default_to_valid_json_without_task_file(options, enabled):
    import json

    launcher = (RECIPE / "scripts/run-qwen3vl-8B-osworld-8xgpu.sh").read_text()
    setup = launcher.split("# Recipe-only options", 1)[1].split(': "${OSWORLD_REPO:', 1)[0]
    command = "# Recipe-only options" + setup + "\npython3 -c 'import os, json; print(json.dumps(dict(os.environ)))'"
    result = subprocess.run(
        ["bash", "-eu", "-c", command, "test", *options],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    settings = json.loads(result.stdout)
    assert json.loads(settings["CUA_ARCHIVE_TASK_LIMITS"]) == {}
    assert settings["CUA_ARCHIVE_ENABLE"] == enabled
    assert settings["CUA_ARCHIVE_MAX_BYTES"] == "10737418240"
    assert settings["CUA_ARCHIVE_PER_TASK_LIMIT"] == "5"


def test_private_env_preserves_launch_roots_without_tracing_secrets(tmp_path):
    # Execute only configuration setup, before the cluster entrypoint can run.
    launcher = (RECIPE / "scripts/run-qwen3vl-8B-osworld-8xgpu.sh").read_text()
    setup = launcher.split("# Keep launch-time roots authoritative", 1)[1].split("# Recipe-only options", 1)[0]
    setup = "# Keep launch-time roots authoritative" + setup
    (tmp_path / "setup.sh").write_text('set +x\nSCRIPT_DIR="$1"\n' + setup)
    secret = "test-only-secret-do-not-log"
    (tmp_path / "env.local.sh").write_text(
        f'export NEX_API_KEY="{secret}"\n'
        'export OSWORLD_REPO="/example/osworld"\n'
        'export MODEL_DIR="/private/models" DATA_DIR="/private/data" SAVE_DIR="/private/save"\n'
    )
    env = dict(os.environ, MODEL_DIR="/launch/models", DATA_DIR="/launch/data", SAVE_DIR="/launch/save")
    result = subprocess.run(
        [
            "bash",
            "-euxc",
            'source "$1/setup.sh"; '
            'test "$MODEL_DIR" = /launch/models; test "$DATA_DIR" = /launch/data; '
            'test "$SAVE_DIR" = /launch/save; test -n "${NEX_API_KEY:+present}"',
            "test",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr.replace(secret, "<redacted>")
    assert secret not in result.stdout + result.stderr


@pytest.mark.parametrize("content, message", [("missing\n", "unknown task_id"), ("# empty\n", "no tasks")])
def test_manifest_rejects_incomplete_dataset(tmp_path, content, message):
    manifest = tmp_path / "tasks.txt"
    manifest.write_text(content)
    tasks = {"known": (tmp_path / "known.json", {"id": "known", "instruction": "test"})}
    with pytest.raises(ValueError, match=message):
        _select(tasks, manifest)


def test_manifest_selects_requested_tasks(tmp_path):
    manifest = tmp_path / "tasks.txt"
    manifest.write_text("known # comment\n")
    tasks = {"known": (tmp_path / "known.json", {"id": "known", "instruction": "test"})}
    rows = _select(tasks, manifest)
    assert len(rows) == 1
    assert rows[0]["extra_info"]["task_id"] == "known"


def test_agent_entry_keeps_launch_storage_settings_over_private_defaults(tmp_path):
    import json
    import shutil

    runner = tmp_path / "run_agent_app.sh"
    shutil.copyfile(RECIPE / runner.name, runner)
    (tmp_path / "env.local.sh").write_text(
        'export OSWORLD_REPO="/example/external" NEX_API_BASE="http://unused"\n'
        'export NEX_WORKSPACE_ID="test" NEX_API_KEY="test" NEX_TEMPLATE="test"\n'
        "export CUA_ARCHIVE_ENABLE=0 CUA_ARCHIVE_BACKEND=http CUA_ARCHIVE_DIR=/old\n"
    )
    python = tmp_path / "agent-python"
    python.write_text(
        "#!/bin/bash\nexec python3 -c 'import json, os; "
        'print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("CUA_ARCHIVE_")}))\'\n'
    )
    python.chmod(0o755)
    env = dict(
        os.environ,
        OSWORLD_AGENT_PYTHON=str(python),
        CUA_ARCHIVE_ENABLE="1",
        CUA_ARCHIVE_BACKEND="directory",
        CUA_ARCHIVE_DIR=str(tmp_path / "new"),
        RELAX_BASE_URL="http://unused",
        RELAX_SESSION_ID="session",
        RELAX_INPUT_JSON="unused",
        RELAX_OUTPUT_JSON="unused",
    )
    env.pop("AGENT_DEBUG_LOG_DIR", None)
    result = subprocess.run(["bash", str(runner)], env=env, capture_output=True, text=True, check=True)
    settings = json.loads(result.stdout)
    assert settings["CUA_ARCHIVE_ENABLE"] == "1"
    assert settings["CUA_ARCHIVE_BACKEND"] == "directory"
    assert settings["CUA_ARCHIVE_DIR"] == str(tmp_path / "new")


@pytest.mark.parametrize("gpu_count", [8, 16])
@pytest.mark.parametrize("no_wait", [False, True])
@pytest.mark.parametrize("external_entrypoint", [False, True])
def test_launcher_builds_submission_without_touching_cluster(tmp_path, no_wait, external_entrypoint, gpu_count):
    """Run the real recipe against fake entrypoint/Ray executables only."""
    import json
    import shutil

    repo = tmp_path / "repo with spaces"
    recipe = repo / "examples/osworld_agentic"
    (recipe / "scripts").mkdir(parents=True)
    launcher = recipe / "scripts/run-qwen3vl-8B-osworld-8xgpu.sh"
    shutil.copyfile(RECIPE / "scripts" / launcher.name, launcher)
    (repo / "scripts/models").mkdir(parents=True)
    shutil.copyfile(RECIPE.parents[1] / "scripts/models/qwen3-vl-8B.sh", repo / "scripts/models/qwen3-vl-8B.sh")
    (repo / "scripts/entrypoint").mkdir()
    (repo / "scripts/entrypoint/local.sh").write_text(
        'test "$RAY_ADDRESS" = auto\n'
        'test -z "${RELAX_ENTRYPOINT_MODE:-}"\n'
        "export RELAX_ENTRYPOINT_MODE=local\n"
        'export RUNTIME_ENV_JSON=\'{"env_vars":{"ENTRY_SENTINEL":"kept"},"working_dir":"kept"}\'\n'
    )
    shutil.copyfile(repo / "scripts/entrypoint/local.sh", repo / "scripts/entrypoint/ray-job.sh")
    (repo / "ray.py").write_text(
        "import os\nfrom types import SimpleNamespace\n"
        "def init(**kwargs): return SimpleNamespace(address_info={'webui_url':'head:8265'})\n"
        "def cluster_resources(): return {'GPU':int(os.environ['TEST_GPUS'])}\n"
        "def nodes(): return [{'Alive':True,'Resources':{'GPU':8}}] * (int(os.environ['TEST_GPUS'])//8)\n"
        "def get_runtime_context(): return SimpleNamespace(gcs_address='head:6379')\n"
        "def shutdown(): pass\n"
    )
    (recipe / "scripts/prepare_agent_env.sh").write_text(
        'mkdir -p "$DATA_DIR/osworld-agent-venv/bin"\n'
        'ln -s "$(command -v python3)" "$DATA_DIR/osworld-agent-venv/bin/python"\n'
    )
    (recipe / "scripts/cache_agent_env.py").write_text(
        "import sys\nassert sys.argv[1].endswith('/osworld-agent-venv')\n"
        "print('/tmp/test-local-agent-env/bin/python')\n"
    )
    secret = "private-test-api-key-never-log"
    (recipe / "env.local.sh").write_text(f'export OSWORLD_REPO="{tmp_path}/external"\nexport NEX_API_KEY="{secret}"\n')
    data = tmp_path / "data with spaces"
    (data / "osworld").mkdir(parents=True)
    (data / "osworld/train.parquet").write_text("placeholder; no training is executed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ray = bin_dir / "ray"
    ray.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'if sys.argv[1:] == ["status"]: sys.exit(0)\n'
        'assert sys.argv[1:3] == ["job", "submit"]\n'
        'with open(os.environ["CAPTURE_ARGS"], "w") as f: json.dump(sys.argv[1:], f)\n'
    )
    ray.chmod(0o755)
    capture = tmp_path / "argv.json"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "MODEL_DIR": str(tmp_path / "models with spaces"),
        "DATA_DIR": str(data),
        "SAVE_DIR": str(tmp_path / "outputs with spaces"),
        "TEST_GPUS": str(gpu_count),
        "CAPTURE_ARGS": str(capture),
        "WORKING_DIR": str(repo),
    }
    if no_wait:
        env["RAY_NO_WAIT"] = "1"
    if external_entrypoint:
        env["RELAX_ENTRYPOINT_MODE"] = "ray-job"
        env["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": {"ENTRY_SENTINEL": "kept"}, "working_dir": "kept"})
    limits = tmp_path / "task limits.json"
    limits.write_text('{"custom-task": 2, "disabled-task": 0}')
    result = subprocess.run(
        [
            "bash",
            "-x",
            str(launcher),
            "--save-trajectories",
            "--trajectory-max-bytes",
            "20000000",
            "--trajectory-per-task-limit",
            "3",
            "--trajectory-task-limits",
            str(limits),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr.replace(secret, "<redacted>")
    argv = json.loads(capture.read_text())
    assert ("--no-wait" in argv) == no_wait
    assert argv[argv.index("--working-dir") + 1] == str(repo)
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["OSWORLD_SANDBOX_HEALTH"] == "1"
    assert runtime["env_vars"]["OSWORLD_HEALTH_DIR"].endswith(runtime["env_vars"]["OSWORLD_RUN_TAG"])
    assert runtime["working_dir"] == "kept"
    assert runtime["env_vars"]["ENTRY_SENTINEL"] == "kept"
    assert runtime["env_vars"]["OSWORLD_AGENT_PYTHON"] == "/tmp/test-local-agent-env/bin/python"
    assert runtime["env_vars"]["CUA_ARCHIVE_ENABLE"] == "1"
    assert runtime["env_vars"]["CUA_ARCHIVE_BACKEND"] == "directory"
    assert runtime["env_vars"]["CUA_ARCHIVE_MAX_BYTES"] == "20000000"
    assert runtime["env_vars"]["CUA_ARCHIVE_PER_TASK_LIMIT"] == "3"
    assert json.loads(runtime["env_vars"]["CUA_ARCHIVE_TASK_LIMITS"]) == {"custom-task": 2, "disabled-task": 0}
    assert runtime["env_vars"]["CUA_ARCHIVE_DIR"].endswith(runtime["env_vars"]["OSWORLD_RUN_TAG"] + "/sessions")
    assert secret not in json.dumps(argv) + result.stdout + result.stderr
    train = argv[argv.index("--") + 1 :]
    assert train[:3] == ["python3", "-m", "examples.osworld_agentic.app.health_runner"]
    for flag, expected in {
        "--lr": "5e-6",
        "--num-rollout": "200",
        "--num-steps-per-rollout": "2",
        "--max-tokens-per-gpu": "24576",
        "--log-probs-max-tokens-per-gpu": "24576",
        "--kl-loss-coef": "0.001",
        "--tb-project-name": "Relax/osworld-agentic",
        "--prompt-data": str(data / "osworld/train.parquet"),
    }.items():
        assert train[train.index(flag) + 1] == expected
    assert json.loads(train[train.index("--resource") + 1]) == {"actor": [1, gpu_count], "rollout": [1, gpu_count]}
    assert "--save" not in train and "--load" not in train
    assert "--agentic-prelaunch" in train
    assert "--freeze-vision-model" in train
    assert "--dynamic-sampling-filter-path" not in train
