# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.fixture
def launch(tmp_path):
    recipes = tmp_path / "scripts/training/sft"
    recipes.mkdir(parents=True)
    for recipe in (SCRIPTS / "training/sft").glob("run-mimo-v2.6-flash-*.sh"):
        shutil.copy(recipe, recipes)
    entry = tmp_path / "scripts/entrypoint"
    entry.mkdir()
    # A regression must never run the local launcher's real process cleanup.
    (entry / "local.sh").write_text("echo local-entrypoint >&2; exit 23\n")
    (entry / "ray-job.sh").write_text('export RELAX_ENTRYPOINT_MODE=ray-job RUNTIME_ENV_JSON="{}"; exec bash "$@"\n')
    for name, code in {
        "timeout": 'shift; exec "$@"',
        "ray": 'if [ "$1" = status ]; then exit "${STATUS:-0}"; fi\nprintf "%s\\n" "$@"',
    }.items():
        cmd = tmp_path / name
        cmd.write_text("#!/bin/sh\n" + code + "\n")
        cmd.chmod(0o755)
    checkpoint = tmp_path / "full checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(dict(num_hidden_layers=48, n_routed_experts=256, moe_layer_freq=[1] * 48))
    )

    def run(recipe, **env):
        return subprocess.run(
            [
                "bash",
                recipes / f"run-mimo-v2.6-flash-{recipe}-sft-48layer-64xgpu.sh",
                checkpoint,
                "input data",
                "save dir",
                "--seed",
                "42",
            ],
            env={
                "PATH": f"{tmp_path}:{os.defpath}",
                "PYTHON": sys.executable,
                "MODEL_CONFIG_DIR": str(SCRIPTS / "models"),
                **env,
            },
            text=True,
            capture_output=True,
        )

    return run


@pytest.mark.parametrize("recipe", ["opensearch-vl", "kimi"])
def test_multinode_avoids_local_startup(launch, recipe):
    for env in ({}, {"RAY_ADDRESS": "head:6379", "STATUS": "1"}, {"RELAX_ENTRYPOINT_MODE": "local"}):
        result = launch(recipe, **env)
        assert result.returncode == 1 and "Multi-node MiMo SFT" in result.stderr
        assert "local-entrypoint" not in result.stderr and "submit" not in result.stdout

    for env in ({"RAY_ADDRESS": "head:6379"}, {"RELAX_ENTRYPOINT_MODE": "spmd-multinode", "RUNTIME_ENV_JSON": "{}"}):
        result = launch(recipe, RAY_DASHBOARD_ADDRESS="http://head:8265", **env)
        assert result.returncode == 0, result.stderr
        args = result.stdout.splitlines()
        assert args.count("submit") == 1 and "--address=http://head:8265" in args
        assert json.loads(args[args.index("--resource") + 1])["actor"] == [1, 64]
        assert args[args.index("--prompt-data") + 1] == "input data"
        assert args[-2:] == ["--seed", "42"]
