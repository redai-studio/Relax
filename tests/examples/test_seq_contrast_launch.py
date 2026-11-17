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


@pytest.mark.parametrize(
    ("model_type", "initializer", "shape"),
    [
        ("qwen2_vl", None, None),
        ("qwen2_vl", "xavier", None),
        ("qwen3_5", None, ("0.8b", 1024, 3584, 24, 8, 2, True)),
        ("qwen3_5", None, ("2b", 2048, 6144, 24, 8, 2, True)),
        ("qwen3_5", None, ("4b", 2560, 9216, 32, 16, 4, True)),
        ("qwen3_5", None, ("9b", 4096, 12288, 32, 16, 4, False)),
    ],
)
def test_example_passes_initializer_and_preserves_ray_runtime(tmp_path, model_type, initializer, shape):
    qwen35 = shape is not None
    model = tmp_path / "models" / ("Qwen3.5-emb" if qwen35 else "Qwen2-VL-2B-Instruct-emb")
    model.mkdir(parents=True)
    config = {"model_type": model_type, "text_config": {"vocab_size": 151665}}
    if qwen35:
        size, hidden, ffn, layers, heads, groups, tied = shape
        config["tie_word_embeddings"] = tied
        config["text_config"] = {
            "hidden_size": hidden,
            "intermediate_size": ffn,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": groups,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "vocab_size": 248320,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
            },
        }
    (model / "config.json").write_text(json.dumps(config))
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
        "MODEL_DIR": str(model if qwen35 else model.parent),
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
        "env_vars": {
            **runtime["env_vars"],
            "RELAX_LORA_A_INIT_METHOD": initializer or "kaiming",
        },
    }
    assert args[args.index("--task-type") + 1] == "embedding"
    assert args[args.index("--custom-loss-function-path") + 1] == "relax.utils.training.embedding.infonce_loss"
    assert args[args.index("--custom-dataset-class") + 1] == "examples.seq_contrast_sft.nli.NLIPairDataset"
    assert "--data-parallel-random-init" in args
    assert "--colocate" not in args
    assert args[args.index("--sft-max-in-flight-steps") + 1] == "2"
    assert "--custom-megatron-before-train-step-hook-path" not in args
    assert "--use-tensorboard" not in args
    assert "--no-wait" not in args
    profile = f"qwen35-{size}" if qwen35 else "qwen2-vl-2b"
    experiment = "explicit-xavier-run" if initializer else f"{profile}-nli-bridge-lora-kaiming-dp8"
    if qwen35:
        assert args[args.index("--hidden-size") + 1] == str(hidden)
        assert args[args.index("--ffn-hidden-size") + 1] == str(ffn)
        assert args[args.index("--num-layers") + 1] == str(layers)
        assert args[args.index("--num-attention-heads") + 1] == str(heads)
        assert args[args.index("--num-query-groups") + 1] == str(groups)
        assert ("--untie-embeddings-and-output-weights" in args) == (not tied)
        assert args[args.index("--lora-target-modules") + 1 : args.index("--lora-scope")] == [
            "linear_proj",
            "linear_fc2",
        ]
    expected_checkpoint = str(tmp_path / "checkpoints" / experiment)
    assert args[args.index("--load") + 1] == args[args.index("--save") + 1] == expected_checkpoint
