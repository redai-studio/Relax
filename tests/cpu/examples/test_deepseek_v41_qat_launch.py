# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import shlex
import subprocess
from pathlib import Path


def test_vision_recipe_only_requests_qat_when_enabled(tmp_path: Path) -> None:
    recipe = Path(__file__).resolve().parents[3] / (
        "examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-vision-sft.sh"
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "deepseek_v41", "vision_config": {"depth": 1}, "text_config": {"num_hidden_layers": 4}}
        )
    )
    env = {
        "PATH": os.environ["PATH"],
        "MODEL_DIR": str(tmp_path),
        "HF_CKPT": str(tmp_path),
        "DATA_DIR": str(tmp_path),
        "SAVE_DIR": str(tmp_path),
        "EXP_NAME": "test-sft",
        "DRY_RUN": "1",
    }

    def launch(**overrides: str) -> list[str]:
        output = subprocess.check_output(["bash", str(recipe)], env={**env, **overrides}, text=True)
        argv = shlex.split(output)
        name = argv.index("--tb-experiment-name") + 1
        argv[name] = argv[name].rsplit("-", 2)[0]  # Exclude the launch timestamp.
        return argv

    baseline = launch()
    assert launch(V41_MXFP4_QAT="0") == baseline
    assert "--custom-megatron-init-path" not in baseline
    assert "--bf16" in baseline

    enabled = launch(V41_MXFP4_QAT="1")
    hook = enabled.index("--custom-megatron-init-path")
    assert enabled[hook + 1] == "relax.models.deepseekv41.qat.enable_routed_mxfp4_qat"
    del enabled[hook : hook + 2]
    assert enabled == [arg.replace("test-sft", "test-sft-mxfp4-qat") for arg in baseline]
