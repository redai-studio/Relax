# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the launch scripts with deployment and cleanup commands
replaced."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("suffix", ["", "_klx"])
def test_search_settings_reach_workers_without_xtrace(tmp_path, suffix):
    example = tmp_path / "examples" / "deepeyes_v2_agentic"
    example.mkdir(parents=True)
    filename = f"run_deepeyes_v2_agentic{suffix}.sh"
    shutil.copyfile(ROOT / "examples" / "deepeyes_v2_agentic" / filename, example / filename)
    (example / f"sglang_judge_service{suffix}.sh").write_text(":\n")
    entrypoint = tmp_path / "scripts" / "entrypoint"
    entrypoint.mkdir(parents=True)
    # Use the real runtime-env builder; no cluster operations in this file.
    shutil.copyfile(ROOT / "scripts" / "entrypoint" / "runtime-env-klx.sh", entrypoint / "runtime-env-klx.sh")
    model_config = tmp_path / "model-config"
    model_config.mkdir()
    (model_config / "qwen36-35B-A3B.sh").write_text(":\n")
    app_env = tmp_path / "app-env"
    (app_env / ".venv" / "bin").mkdir(parents=True)
    (app_env / ".venv" / "bin" / "python").symlink_to(sys.executable)
    sif = tmp_path / "test.sif"
    sif.touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    # Prevent the launcher's startup find -exec cleanup from touching /tmp.
    (bin_dir / "find").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "find").chmod(0o755)
    (bin_dir / "ray").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        'payload = json.loads(sys.argv[sys.argv.index("--runtime-env-json") + 1])\n'
        'with open(os.environ["CAPTURE_PATH"], "w") as f:\n    json.dump(payload, f)\n'
    )
    (bin_dir / "ray").chmod(0o755)
    secret = 'test-key-"quoted"\\value'
    config_path = str(tmp_path / 'search "quoted".yaml')
    (example / "env.sh").write_text(
        f"export DEEPEYES_V2_SEARCH_CONFIG={shlex.quote(config_path)}\n"
        f"export DEEPEYES_V2_SEARCH_API_KEY={shlex.quote(secret)}\n"
    )
    capture = tmp_path / "runtime.json"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "WORKDIR": str(tmp_path),
        "MODEL_DIR": str(tmp_path),
        "DATA_DIR": str(tmp_path),
        "SAVE_DIR": str(tmp_path),
        "CPU_THREADS_PER_ACTOR": "4",
        "MODEL_CONFIG_DIR": str(model_config),
        "RELAX_ENTRYPOINT_MODE": "test",
        "APPTAINER_IMAGE_PATH": str(sif),
        "DEEPEYES_V2_APP_ENV_ROOT": str(app_env),
        "CAPTURE_PATH": str(capture),
        "PYTHONPATH": "",
    }
    result = subprocess.run(
        ["bash", str(example / filename)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(capture.read_text())
    assert payload["env_vars"]["DEEPEYES_V2_SEARCH_CONFIG"] == config_path
    assert payload["env_vars"]["DEEPEYES_V2_SEARCH_API_KEY"] == secret
    assert payload["env_vars"]["SANDBOX_BACKEND"] == "apptainer_jupyter"
    for output in [result.stdout, result.stderr, *(p.read_text() for p in (tmp_path / "logs").glob("*.log"))]:
        assert "test-key-" not in output
