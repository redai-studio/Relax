# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Capture launcher arguments without contacting Ray or starting training."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(shutil.which("jq") is None, reason="launcher requires jq")
_ROOT = Path(__file__).resolve().parents[2]
_BASE = "run-gemma4-31B-sft-64k-8xgpu.sh"
_MULTI = "run-gemma4-31B-multiturn-flex64k.sh"


def _launch(tmp_path, *, script=_BASE, **overrides):
    capture = tmp_path / "arguments.json"
    env = os.environ.copy()
    for name in ("ACTOR_GPUS", "EXTRA_TRAIN_ARGS", "LABEL_KEY", "RAY_NO_WAIT", "WORKING_DIR"):
        env.pop(name, None)
    env.update(
        RELAX_ENTRYPOINT_MODE="ray-job",
        MODEL_CONFIG_DIR=str(_ROOT / "scripts/models"),
        RUNTIME_ENV_JSON='{"env_vars":{"EXISTING":"preserved"}}',
        CKPT="/model/gemma4",
        PROMPT_DATA="/data/messages.jsonl",
        SAVE_DIR=str(tmp_path / "save"),
        NUM_NODES="1",
        GPUS_PER_NODE="8",
        TP_SIZE="8",
        WORLD_SIZE="1",
        RAY_ADDRESS="http://ray-head:8265",
        CAPTURE_FILE=str(capture),
        GEMMA4_CONVERSION_MODE="vl",
        GEMMA4_SFT_THINKING="0",
        GEMMA4_SFT_TOOL_TURN_BOUNDARY="1",
        GEMMA4_SFT_MS_SWIFT_COMPAT="0",
        DISABLE_CHECKPOINT_SAVE="0",
    )
    env.update(overrides)
    command = """
ray() {
    python3 -c 'import json,os,sys; open(os.environ["CAPTURE_FILE"],"w").write(json.dumps(sys.argv[1:]))' "$@"
}
export -f ray
bash "$1"
"""
    result = subprocess.run(
        ["bash", "-c", command, "capture", str(_ROOT / "scripts/training/sft" / script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    args = json.loads(capture.read_text()) if capture.exists() else None
    return result, args


@pytest.mark.parametrize("nodes", [1, 2])
@pytest.mark.parametrize("script", [_BASE, _MULTI])
def test_text_64k_launcher_preserves_required_runtime_and_loss_options(tmp_path, nodes, script):
    result, args = _launch(tmp_path, script=script, NUM_NODES=str(nodes))
    assert result.returncode == 0, result.stderr
    runtime = json.loads(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--runtime-env-json=")))
    assert runtime["env_vars"]["EXISTING"] == "preserved"
    assert runtime["env_vars"]["GEMMA4_CONVERSION_MODE"] == "text"
    assert runtime["env_vars"]["RELAX_GEMMA4_FORCE_FLEX_GLOBAL"] == "1"
    assert runtime["env_vars"]["GEMMA4_SFT_THINKING"] == "0"
    assert runtime["env_vars"]["GEMMA4_SFT_TOOL_TURN_BOUNDARY"] == "1"
    assert "--address=http://ray-head:8265" in args
    assert args[args.index("--seq-length") + 1] == "65536"
    assert args[args.index("--global-batch-size") + 1] == str(nodes)
    assert args[args.index("--context-parallel-size") + 1] == "1"
    assert args[args.index("--pipeline-model-parallel-size") + 1] == "1"
    assert json.loads(args[args.index("--resource") + 1])["actor"] == [1, nodes * 8]
    assert args[args.index("--max-seqlen-per-dp-cp-rank") + 1] == "65536"
    assert "--sft-chunked-logits" in args and "--recompute-loss-function" in args
    assert "--overlap-grad-reduce" not in args and "--defer-embedding-wgrad-compute" not in args
    assert "--optimizer-cpu-offload" in args
    assert args[args.index("--save-hf-post-hook-path") + 1] == (
        "scripts.tools.gemma4_chat_template_hook.patch_gemma4_chat_template"
    )
    assert "--label-key" not in args
    if script == _MULTI:
        assert "--no-wait" in args


def test_text_64k_launcher_can_disable_saves_and_cpu_offload(tmp_path):
    result, args = _launch(tmp_path, DISABLE_CHECKPOINT_SAVE="1", OPTIMIZER_CPU_OFFLOAD="0")
    assert result.returncode == 0, result.stderr
    assert "--save" not in args and "--save-hf" not in args and "--save-hf-post-hook-path" not in args
    assert "--optimizer-cpu-offload" not in args
    assert "--recompute-loss-function" in args


@pytest.mark.parametrize(
    "overrides",
    [
        {"TP_SIZE": "0"},
        {"TP_SIZE": "3"},
        {"ACTOR_GPUS": "7"},
        {"ACTOR_GPUS": "16"},
        {"NUM_NODES": "0"},
    ],
)
def test_text_64k_launcher_rejects_invalid_topology_before_submission(tmp_path, overrides):
    result, args = _launch(tmp_path, **overrides)
    assert result.returncode != 0
    assert "ERROR:" in result.stderr
    assert args is None
