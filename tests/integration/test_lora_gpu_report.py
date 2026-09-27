# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.lora_gpu.resources import reclaim_order
from tests.integration.lora_gpu.run import compare


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_lora_gpu_comparison_rejects_nonfinite_logprobs(invalid):
    baseline = {"tokens": [1, 2, 3, 4], "logprobs": [-1.0] * 4}
    actual = {"tokens": [1, 2, 3, 4], "logprobs": [-1.0, invalid, -1.0, -1.0]}
    assert not compare(actual, baseline)["passed"]
    assert not compare(baseline, actual)["passed"]


def test_lora_gpu_comparison_rejects_wrong_tokens_and_truncated_scores():
    baseline = {"tokens": [1, 2, 3, 4], "logprobs": [-1.0] * 4}
    assert compare(baseline, baseline)["passed"]
    assert not compare({**baseline, "tokens": [5, 2, 3, 4]}, baseline)["passed"]
    assert not compare({**baseline, "logprobs": [-1.0] * 3}, baseline)["passed"]


@pytest.mark.parametrize("unload_time", [2.5, 3.5, 5.0])
def test_lora_gpu_reclamation_requires_terminal_release_before_unload(unload_time: float) -> None:
    trace = [
        {"ev": "req.resolve.exit", "rids": ["retained-A"], "lora_id": "a", "mono": 1.0},
        {"ev": "registry.wait_for_unload.enter", "id": "a", "counts": {"a": 1}, "mono": 2.0},
        {"ev": "registry.release", "ids": ["a"], "counts": {"a": 0}, "mono": 3.0},
        {"ev": "registry.wait_for_unload.exit", "id": "a", "mono": 4.0},
        {"ev": "manager.unload.exit", "name": "A", "mono": unload_time},
    ]
    if unload_time < 4.0:
        with pytest.raises(AssertionError):
            reclaim_order(trace, "A")
    else:
        assert reclaim_order(trace, "A")["physical_unload"]["mono"] == unload_time


@pytest.mark.skipif(os.environ.get("RELAX_LORA_GPU_ACCEPTANCE") != "1", reason="opt-in three-GPU experiment")
def test_lora_gpu_joint_acceptance(tmp_path):
    import torch

    if torch.cuda.device_count() < 3:
        pytest.skip("three visible GPUs required")
    root = Path(__file__).resolve().parents[2]
    required = ["LORA_GPU_MODEL", "LORA_GPU_ADAPTER_A", "LORA_GPU_ADAPTER_B"]
    if any(name not in os.environ for name in required):
        pytest.skip("set LORA_GPU_MODEL, LORA_GPU_ADAPTER_A and LORA_GPU_ADAPTER_B")
    command = [
        sys.executable,
        "-m",
        "tests.integration.lora_gpu.run",
        "--model",
        os.environ[required[0]],
        "--adapter-a",
        os.environ[required[1]],
        "--adapter-b",
        os.environ[required[2]],
        "--output",
        str(tmp_path / "gpu"),
        "--deterministic",
    ]
    if os.environ.get("LORA_GPU_AOT") == "1":
        command.append("--aot")
    subprocess.run(command, cwd=root, check=True, timeout=1800)
