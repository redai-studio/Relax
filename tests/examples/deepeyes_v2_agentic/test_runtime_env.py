# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Launch-script checks for DeepEyesV2 runtime env propagation."""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepeyes_v2_agentic"

SEARCH_ENV_VARS = {
    "DEEPEYES_V2_SEARCH_CACHE_PATHS",
    "DEEPEYES_V2_SEARCH_BACKEND",
    "DEEPEYES_V2_SEARCH_RETRIEVER_URL",
    "DEEPEYES_V2_SEARCH_TOPK",
    "DEEPEYES_V2_SEARCH_BRAVE_API_KEY",
    "DEEPEYES_V2_SEARCH_BRAVE_ENDPOINT",
    "DEEPEYES_V2_SEARCH_TRUST_ENV",
    "DEEPEYES_V2_SEARCH_TIMEOUT",
    "DEEPEYES_V2_SEARCH_MAX_RETRIES",
    "DEEPEYES_V2_SEARCH_RETRY_BUDGET",
}


def test_training_scripts_forward_search_env_vars_to_ray_runtime_env():
    scripts = [
        EXAMPLE_DIR / "run_deepeyes_v2_agentic.sh",
        EXAMPLE_DIR / "run_deepeyes_v2_agentic_klx.sh",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "env_vars" in text or "EXTRA_ENV_VARS_JSON" in text
        missing = [name for name in SEARCH_ENV_VARS if name not in text]
        assert not missing, f"{script.name} does not forward: {missing}"


@pytest.mark.parametrize(
    ("script_name", "judge_source", "runtime_env_source"),
    [
        ("run_deepeyes_v2_agentic.sh", 'source "${SCRIPT_DIR}/sglang_judge_service.sh"', None),
        (
            "run_deepeyes_v2_agentic_klx.sh",
            'source "${SCRIPT_DIR}/sglang_judge_service_klx.sh"',
            'source "${SCRIPT_DIR}/../../scripts/entrypoint/runtime-env-klx.sh"',
        ),
    ],
)
def test_training_script_xtrace_does_not_leak_brave_key(tmp_path, script_name, judge_source, runtime_env_source):
    secret = "test-brave-key-should-not-appear"
    script = tmp_path / script_name
    text = (EXAMPLE_DIR / script_name).read_text(encoding="utf-8")
    text = text.replace(
        judge_source,
        textwrap.dedent(
            """
            export DEEPEYES_JUDGE_API_KEY="EMPTY"
            export DEEPEYES_JUDGE_BASE_URL="http://127.0.0.1:30000/v1"
            export DEEPEYES_JUDGE_MODELS="Qwen2.5-1.5B-Instruct"
            """
        ).strip(),
    )
    if runtime_env_source is not None:
        text = text.replace(
            runtime_env_source,
            'export RUNTIME_ENV_JSON="{\\"env_vars\\": {${EXTRA_ENV_VARS_JSON}}}"',
        )
    script.write_text(text, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    model_config_dir = tmp_path / "model-config"
    model_config_dir.mkdir()
    (model_config_dir / "qwen36-35B-A3B.sh").write_text("MODEL_ARGS=()\n", encoding="utf-8")

    data_dir = tmp_path / "data"
    (data_dir / "sif").mkdir(parents=True)
    (data_dir / "sif" / "deepeyes_v2_kernel.sif").write_text("", encoding="utf-8")

    app_env = tmp_path / "app-env"
    app_python = app_env / ".venv" / "bin" / "python"
    app_python.parent.mkdir(parents=True)
    app_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    app_python.chmod(app_python.stat().st_mode | stat.S_IXUSR)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_ray = bin_dir / "ray"
    fake_ray.write_text("#!/bin/sh\necho ray-submit-stub\nexit 0\n", encoding="utf-8")
    fake_ray.chmod(fake_ray.stat().st_mode | stat.S_IXUSR)

    (tmp_path / "env.sh").write_text(
        textwrap.dedent(
            f"""
            export DEEPEYES_V2_SEARCH_BACKEND=brave
            export DEEPEYES_V2_SEARCH_BRAVE_API_KEY={secret}
            export DEEPEYES_V2_SEARCH_BRAVE_ENDPOINT=https://proxy.example/search
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RELAX_ENTRYPOINT_MODE": "ray-job",
        "MODEL_CONFIG_DIR": str(model_config_dir),
        "MODEL_DIR": str(tmp_path / "models"),
        "DATA_DIR": str(data_dir),
        "SAVE_DIR": str(tmp_path / "save"),
        "DEEPEYES_V2_APP_ENV_ROOT": str(app_env),
        "WANDB_API_KEY": "YOUR-KEY",
    }
    result = subprocess.run(["bash", "-x", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert secret not in result.stderr
    assert secret not in result.stdout
