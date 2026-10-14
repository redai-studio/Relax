# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Offline checks for private configuration loading and dataset selection."""

import os
import subprocess
from pathlib import Path

import pytest

from examples.osworld_agentic.scripts.prepare_data import _select


RECIPE = Path(__file__).resolve().parents[2] / "examples/osworld_agentic"


def test_private_env_preserves_launch_roots_without_tracing_secrets(tmp_path):
    # Execute only configuration setup, before the cluster entrypoint can run.
    launcher = (RECIPE / "scripts/run-qwen3vl-8B-osworld-8xgpu.sh").read_text()
    setup = launcher.split("# Keep launch-time roots authoritative", 1)[1].split("export MEGATRON=", 1)[0]
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


@pytest.mark.parametrize("no_wait", [False, True])
@pytest.mark.parametrize("external_entrypoint", [False, True])
def test_launcher_builds_submission_without_touching_cluster(tmp_path, no_wait, external_entrypoint):
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
        "CAPTURE_ARGS": str(capture),
        "WORKING_DIR": str(repo),
    }
    if no_wait:
        env["RAY_NO_WAIT"] = "1"
    if external_entrypoint:
        env["RELAX_ENTRYPOINT_MODE"] = "ray-job"
        env["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": {"ENTRY_SENTINEL": "kept"}, "working_dir": "kept"})
    result = subprocess.run(["bash", "-x", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr.replace(secret, "<redacted>")
    argv = json.loads(capture.read_text())
    assert ("--no-wait" in argv) == no_wait
    assert argv[argv.index("--working-dir") + 1] == str(repo)
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--runtime-env-json=")))
    assert runtime["working_dir"] == "kept"
    assert runtime["env_vars"]["ENTRY_SENTINEL"] == "kept"
    assert secret not in json.dumps(argv) + result.stdout + result.stderr
    train = argv[argv.index("--") + 1 :]
    assert train[:3] == ["python3", "-m", "relax.entrypoints.train"]
    for flag, expected in {
        "--lr": "5e-6",
        "--num-rollout": "200",
        "--num-steps-per-rollout": "2",
        "--max-tokens-per-gpu": "49152",
        "--log-probs-max-tokens-per-gpu": "49152",
        "--kl-loss-coef": "0.001",
        "--tb-project-name": "Relax/osworld-agentic",
        "--prompt-data": str(data / "osworld/train.parquet"),
    }.items():
        assert train[train.index(flag) + 1] == expected
    assert "--save" not in train and "--load" not in train
    assert "--agentic-prelaunch" in train
    assert "--freeze-vision-model" in train
    assert "--dynamic-sampling-filter-path" not in train
