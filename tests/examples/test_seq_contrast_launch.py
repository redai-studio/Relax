# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the example's actual submission command without a Ray cluster."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-nli-pt-8xgpu.sh"


@pytest.mark.parametrize("initializer", [None, "xavier"])
def test_example_passes_initializer_and_preserves_ray_runtime(tmp_path, initializer):
    model = tmp_path / "models/Qwen2-VL-2B-Instruct-emb"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "vocab_size": 151665}))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "submission.json"
    ray = binaries / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    ray.chmod(0o755)
    runtime = {"env_vars": {"EXISTING_OPTION": "preserved"}, "worker_process_setup_hook": "existing.setup"}
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "MODEL_DIR": str(tmp_path / "models"),
        "DATA_DIR": str(tmp_path / "data"),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "RAY_ADDRESS": "http://127.0.0.1:8265",
        "RELAX_ENTRYPOINT_MODE": "ray-job",
        "RUNTIME_ENV_JSON": json.dumps(runtime),
        "RELAX_RUN_LOG_DIR": str(tmp_path / "logs"),
        "RAY_NO_WAIT": "",
    }
    env.pop("EXP_NAME", None)
    env.pop("WORKING_DIR", None)
    env.pop("RELAX_LORA_A_INIT_METHOD", None)
    if initializer is not None:
        env["RELAX_LORA_A_INIT_METHOD"] = initializer
        env["EXP_NAME"] = "explicit-xavier-run"
    subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    args = json.loads(capture.read_text())
    actual_runtime = json.loads(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--runtime-env-json=")))
    assert actual_runtime == {
        **runtime,
        "env_vars": {**runtime["env_vars"], "RELAX_LORA_A_INIT_METHOD": initializer or "kaiming"},
    }
    assert args[args.index("--custom-dataset-class") + 1] == "relax.engine.sft.dataset.retrieval.NLIPairDataset"
    assert "--data-parallel-random-init" in args
    assert "--colocate" not in args
    assert args[args.index("--sft-max-in-flight-steps") + 1] == "2"
    assert "--custom-megatron-before-train-step-hook-path" not in args
    assert "--use-tensorboard" not in args
    assert "--no-wait" not in args
    experiment = "explicit-xavier-run" if initializer else "qwen2-vl-2b-nli-bridge-lora-kaiming-dp8"
    expected_checkpoint = str(tmp_path / "checkpoints" / experiment)
    assert args[args.index("--load") + 1] == args[args.index("--save") + 1] == expected_checkpoint
