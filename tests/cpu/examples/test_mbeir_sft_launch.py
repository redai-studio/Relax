# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the actual M-BEIR launcher against an isolated submission recorder."""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/seq_contrast_sft/scripts/run-qwen2-vl-2B-mbeir-sft-8xgpu.sh"


def test_mbeir_launch_uses_the_prepared_dataset_horizon(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "vocab_size": 151658}))
    data = tmp_path / "data/mbeir"
    data.mkdir(parents=True)
    (data / "pairs.meta.json").write_text(
        json.dumps(
            {
                "stage": "prepared",
                "raw_queries": 17,
                "global_batch_size": 8,
                "sampling": "native_runtime",
                "prepared_rows": 17,
            }
        )
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    capture = tmp_path / "submission.json"
    ray = binaries / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({'args':sys.argv[1:],"
        "'skip_cleanup':os.environ['RELAX_SKIP_CLUSTER_CLEANUP']}))\n"
    )
    ray.chmod(0o755)
    runtime = {"env_vars": {"EXISTING_OPTION": "preserved"}, "worker_process_setup_hook": "existing.setup"}
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "MODEL_DIR": str(model),
        "DATA_DIR": str(data.parent),
        "SAVE_DIR": str(tmp_path / "checkpoints"),
        "RAY_ADDRESS": "http://127.0.0.1:8265",
        "RELAX_ENTRYPOINT_MODE": "ray-job",
        "RUNTIME_ENV_JSON": json.dumps(runtime),
        "RELAX_RUN_LOG_DIR": str(tmp_path / "logs"),
        "RAY_NO_WAIT": "1",
    }
    env.pop("EXP_NAME", None)
    env.pop("WORKING_DIR", None)
    env.pop("RELAX_LORA_A_INIT_METHOD", None)
    result = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    submission = json.loads(capture.read_text())
    args = submission["args"]
    actual_runtime = json.loads(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--runtime-env-json=")))
    assert actual_runtime == {
        **runtime,
        "env_vars": {
            **runtime["env_vars"],
            "RELAX_LORA_A_INIT_METHOD": "kaiming",
        },
    }
    assert submission["skip_cleanup"] == "1"
    assert args[args.index("--num-rollout") + 1] == "3"
    assert args[args.index("--lr-warmup-iters") + 1] == "1"
    assert args[args.index("--global-batch-size") + 1] == "8"
    assert args[args.index("--task-type") + 1] == "embedding"
    assert args[args.index("--custom-loss-function-path") + 1] == "relax.utils.training.embedding.infonce_loss"
    assert args[args.index("--custom-dataset-class") + 1] == "examples.seq_contrast_sft.mbeir.MbeirPairDataset"
    assert "--freeze-vision-model" in args and "--freeze-vision-projection" in args
    assert args[args.index("--image-max-token-num") + 1] == "300"
    assert "--no-wait" in args
    assert "--use-clearml" not in args and "--use-tensorboard" not in args
    assert "--no-use-tensorboard" in args
    checkpoint = str(tmp_path / "checkpoints/qwen2-vl-2b-mbeir-from-pt-r128-dp8")
    assert args[args.index("--load") + 1] == args[args.index("--save") + 1] == checkpoint
