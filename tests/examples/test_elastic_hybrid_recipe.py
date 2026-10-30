# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["async", "hybrid"])
def test_elastic_recipe_submits_expected_training_mode(tmp_path, mode):
    binary = tmp_path / "ray"
    binary.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        RELAX_ENTRYPOINT_MODE="ray-job",
        RELAX=str(ROOT),
        MODEL_CONFIG_DIR=str(ROOT / "scripts/models"),
        RUNTIME_ENV_JSON="{}",
        HOST_IP="127.0.0.1",
        MODEL_DIR="/models",
        DATA_DIR="/data",
        NUM_ROLLOUT="100",
        CE_ENABLE_AFFINITY="0",
    )
    env.pop("RELAX_ELASTIC_TRAIN_MODE", None)
    script = ROOT / f"scripts/training/text/run-qwen35-9B-elastic-rollout-{mode}.sh"
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout.splitlines()[-1])
    command = argv[argv.index("--") + 1 :]
    assert command[:3] == ["python3", "-m", "relax.entrypoints.train"]
    resource = json.loads(command[command.index("--resource") + 1])
    assert resource["actor"] == [1, 8 if mode == "hybrid" else 4]
    assert resource["rollout"] == [1, 2]
    assert ("advantages" in resource) == (mode == "async")
    assert ("--hybrid" in command) == (mode == "hybrid")
    assert ("--fully-async" in command) == (mode == "async")
    assert ("--ref-actor-config" in command) == (mode == "async")
    assert command[command.index("--num-rollout") + 1] == "100"
    assert "--no-enable-affinity" in command
    autoscaler = Path(command[command.index("--autoscaler-config") + 1])
    assert autoscaler == ROOT / "scripts/training/text/configs/qwen35-9b-elastic-autoscaler.yaml"
    assert autoscaler.is_file()
    runtime_arg = next(arg for arg in argv if arg.startswith("--runtime-env-json="))
    runtime = json.loads(runtime_arg.split("=", 1)[1])
    assert runtime["env_vars"]["NCCL_IB_DISABLE"] == "1"
