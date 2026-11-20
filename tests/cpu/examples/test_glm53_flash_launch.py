# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/training/sft/run-glm5.3-flash-320B-A18B-lora-32xgpu.sh"


def _launch(tmp_path: Path, visible_gpus: int, **environment: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "train.jsonl").touch()
    checkpoint = tmp_path / "GLM-5.3-Flash"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {
                    "layer_types": ["linear_attention"],
                    "mlp_layer_types": ["sparse"],
                    "num_hidden_layers": 1,
                    "n_routed_experts": 16,
                },
            }
        )
    )
    (tmp_path / "torch.py").write_text(
        "import os\n\n"
        "class Cuda:\n"
        "    def device_count(self):\n"
        "        return int(os.environ['VISIBLE_GPUS'])\n\n"
        "cuda = Cuda()\n"
    )
    for command, script in {
        "ray": 'printf "%s\\n" "$*"',
        "pkill": "true",
        "sleep": "true",
        "nvidia-smi": "false",
        "tee": "cat",
    }.items():
        executable = tmp_path / command
        executable.write_text(f"#!/bin/bash\n{script}\n")
        executable.chmod(0o755)

    env = {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(tmp_path),
        "MODEL_DIR": str(tmp_path),
        "DATA_DIR": str(tmp_path),
        "SAVE_DIR": str(tmp_path),
        "VISIBLE_GPUS": str(visible_gpus),
        **environment,
    }
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize(("visible_gpus", "num_gpus"), [(8, ""), (32, "8"), (16, "32")])
def test_flash_local_launch_rejects_an_unsatisfiable_gpu_request(tmp_path, visible_gpus, num_gpus):
    result = _launch(tmp_path, visible_gpus, NUM_GPUS=num_gpus)

    assert result.returncode != 0
    assert "use an existing Ray cluster" in result.stderr
    assert "Cleaning up stale processes" not in result.stdout
    assert "start --head" not in result.stdout
    assert "job submit" not in result.stdout


def test_flash_local_launch_registers_visible_gpus(tmp_path):
    result = _launch(tmp_path, 32)

    assert result.returncode == 0, result.stderr
    assert any(line.startswith("start --head ") and "--num-gpus 32" in line for line in result.stdout.splitlines())
    assert "job submit --address=http://127.0.0.1:8265" in result.stdout


def test_flash_existing_cluster_launch_accepts_a_cpu_submitter(tmp_path):
    result = _launch(
        tmp_path,
        0,
        RELAX_ENTRYPOINT_MODE="ray-job",
        RAY_JOB_ADDRESS="http://ray-head:8265",
        RUNTIME_ENV_JSON="{}",
    )

    assert result.returncode == 0, result.stderr
    assert "start --head" not in result.stdout
    assert "job submit --address=http://ray-head:8265" in result.stdout
