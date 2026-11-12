# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "nemo_gym_agentic"
REPO_ROOT = EXAMPLE_DIR.parents[1]


def test_shared_scripts_remain_under_example_scripts() -> None:
    scripts_dir = EXAMPLE_DIR / "scripts"
    expected = {
        "run_agent_app.sh",
        "run_gateway.sh",
        "run_training.sh",
        "convert_dataset.py",
    }

    assert expected <= {path.name for path in scripts_dir.iterdir()}


def test_each_recipe_owns_its_scripts() -> None:
    expected = {
        "calendar": {
            "prepare_calendar.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-calendar.sh",
            "start_calendar_gym.sh",
            "verify_calendar.py",
        },
        "gsm8k": {
            "prepare_gsm8k.sh",
            "run-qwen3-4B-8xgpu-nemo-gym.sh",
            "start_gsm8k_gym.sh",
            "verify_gsm8k.py",
        },
        "workplace-assistant": {
            "prepare_workplace_assistant.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-workplace.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-workplace-hybrid-async.sh",
            "start_workplace_assistant_gym.sh",
            "start_workplace_assistant_gym_remote.sh",
            "verify_workplace_assistant.py",
            "verify_workplace_assistant_trial.py",
        },
        "r2e-gym": {
            "prepare_r2e_gym.py",
            "prepare_r2e_gym.sh",
            "run-qwen35-9B-8xgpu-nemo-gym-r2e.sh",
            "start_r2e_gym_local.sh",
            "start_r2e_gym_remote.sh",
            "submit_r2e_gym.sh",
            "verify_r2e_gym_trial.py",
        },
        "multienv-math-workplace": {
            "README.md",
            "prepare_multienv.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-multienv.sh",
            "start_multienv_gym.sh",
            "start_multienv_gym_remote.sh",
        },
        "reasoning-gym-cc": {
            "README.md",
            "prepare_reasoning_gym_cc.sh",
            "run-qwen3-4B-8xgpu-nemo-gym-reasoning-gym-cc.sh",
            "start_reasoning_gym_cc_gym.sh",
            "start_reasoning_gym_cc_gym_remote.sh",
            "verify_reasoning_gym_cc_trial.py",
        },
    }

    for recipe_name, expected_files in expected.items():
        recipe_dir = EXAMPLE_DIR / "recipes" / recipe_name
        assert expected_files <= {path.name for path in recipe_dir.iterdir()}


def test_generic_recipe_local_entrypoint_exists() -> None:
    entrypoint = EXAMPLE_DIR / "scripts" / "../../../scripts/entrypoint/local.sh"

    assert entrypoint.resolve() == REPO_ROOT / "scripts/entrypoint/local.sh"
    assert entrypoint.is_file()


def test_r2e_local_launcher_help_does_not_require_ray() -> None:
    launcher = EXAMPLE_DIR / "recipes" / "r2e-gym" / "start_r2e_gym_local.sh"

    result = subprocess.run(
        ["bash", str(launcher), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--data-dir <prepared-r2e-directory>" in result.stdout
    assert "--mode <golden|train>" in result.stdout
    assert "--proxy <http-proxy-url>" in result.stdout
    assert "--callback-proxy <http-proxy-url>" in result.stdout
    assert "--callback-timeout-s <positive-integer>" in result.stdout
    assert "RAY_ADDRESS" in result.stdout


def test_r2e_local_launcher_rejects_non_gcs_ray_addresses() -> None:
    launcher = EXAMPLE_DIR / "recipes" / "r2e-gym" / "start_r2e_gym_local.sh"

    for ray_address in ("http://127.0.0.1:8265", "ray://127.0.0.1:10001"):
        env = os.environ.copy()
        env["RAY_ADDRESS"] = ray_address
        result = subprocess.run(
            [
                "bash",
                str(launcher),
                "--data-dir",
                "/tmp/not-used",
                "--mode",
                "train",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 2
        assert "must be a Ray GCS address" in result.stderr
