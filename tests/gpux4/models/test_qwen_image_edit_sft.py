# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from pathlib import Path

import pytest
import torch

from tests.models.qwen_image_edit_fsdp import run_fsdp_processes


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices for Qwen FSDP2 execution")
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_qwen_two_gpu_fsdp_and_checkpoint(tmp_path: Path, mode: str) -> None:
    run_fsdp_processes(tmp_path, 2, mode)
