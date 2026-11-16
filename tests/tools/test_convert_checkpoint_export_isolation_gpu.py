# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU export must preserve a caller's initialized CUDA and process group."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an initialized CUDA parent")
def test_cpu_export_isolates_initialized_parent(tmp_path: Path) -> None:
    code = """
import sys
import pytest
import torch

if torch.cuda.is_available():
    torch.cuda.init()
cuda_initialized = torch.cuda.is_initialized()
torch.distributed.init_process_group("gloo", init_method=sys.argv[2], rank=0, world_size=1)
group = torch.distributed.group.WORLD
try:
    result = pytest.main([sys.argv[1] + "::test_cpu_export_round_trip_without_model_or_cuda", "-q", "--tb=short", "--show-capture=no"])
    assert torch.distributed.is_initialized(), "CPU tests changed the parent process group"
    assert torch.distributed.group.WORLD is group, "CPU tests replaced the parent process group"
    torch.distributed.all_reduce(torch.ones(1), group=group)
    assert torch.cuda.is_initialized() == cuda_initialized, "CPU tests changed the parent CUDA state"
finally:
    torch.distributed.destroy_process_group()
sys.exit(result)
"""
    env = dict(os.environ)
    env.pop("_RELAX_CPU_EXPORT_TEST_CHILD", None)
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve().parents[1] / "cpu/tools/test_convert_checkpoint_to_hf_cpu.py"),
            (tmp_path / "store").as_uri(),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
