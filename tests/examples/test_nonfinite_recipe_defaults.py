# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
KEY = "RELAX_SKIP_NONFINITE_GRAD"
SFT = "scripts/training/sft/run-qwen35-optimized-64xblackwell.sh"
SFT397 = "scripts/training/sft/run-qwen3.5-397B-A17B-sft-64xblackwell.sh"
CPT = "scripts/training/cpt/run-qwen3.6-35B-A3B-cpt-256k-64xgpu.sh"
QWEN38 = "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh"


def render(script, args, value, stale_runtime=False):
    env = dict(
        os.environ,
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/checkpoints",
        EXP_NAME="skip-policy-test",
        MEGATRON="/image/megatron",
        RELAX_HYBRID_EP_COMMIT="test-image",
        DRY_RUN="1",
    )
    env.pop(KEY, None)
    env.pop("RUNTIME_ENV_JSON", None)
    if value is not None:
        env[KEY] = value
    if stale_runtime:
        env["RUNTIME_ENV_JSON"] = json.dumps({"env_vars": {KEY: "1"}})
    result = subprocess.run(["bash", str(ROOT / script), *args], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    argv = output["args"] if script == SFT397 else output["argv"]
    index = max(i for i, arg in enumerate(argv) if arg == "--train-env-vars")
    return output["runtime_env"]["env_vars"], json.loads(argv[index + 1])


@pytest.mark.parametrize(
    "script,args",
    [
        (SFT, []),
        (SFT, ["--multimodal"]),
        (SFT397, []),
        (SFT397, ["--multimodal"]),
        (QWEN38, ["--multimodal-keys", "null"]),
        (QWEN38, []),
    ],
)
@pytest.mark.parametrize("value", [None, "0", "1"])
def test_recipe_skip_default_and_explicit_override(script, args, value):
    runtime, train = render(script, args, value)
    assert runtime[KEY] == train[KEY] == ("1" if value is None else value)
    if script == QWEN38:
        assert KEY in runtime["RELAX_PROPAGATE_ENV_VARS"].split(",")


@pytest.mark.parametrize("script", [SFT, SFT397, CPT, QWEN38])
def test_explicit_diagnostic_disable_overrides_base_runtime(script):
    runtime, train = render(script, [], "0", stale_runtime=True)
    assert runtime[KEY] == train[KEY] == "0"
