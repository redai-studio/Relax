# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def kernel():
    path = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_fused_attention.py"
    spec = importlib.util.spec_from_file_location("qsa_tile_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dim,expected", [(64, (128, 64, 128, 64)), (128, (128, 64, 128, 64)), (256, (128, 64, 64, 32))]
)
def test_other_hardware_keeps_original_tile_profiles(kernel, capability, dtype, dim, expected):
    assert kernel._select_tiles(dim, capability, dtype) == expected


@pytest.mark.parametrize("dim,expected", [(128, (64, 64, 64, 32)), (256, (32, 32, 32, 32))])
def test_sm103_bf16_uses_bounded_forward_and_backward_tiles(kernel, dim, expected):
    assert kernel._select_tiles(dim, (10, 3), torch.bfloat16) == expected


@pytest.mark.parametrize(
    "dim,dtype,expected",
    [
        (64, torch.bfloat16, (128, 64, 128, 64)),
        (128, torch.float16, (128, 64, 128, 64)),
        (256, torch.float16, (128, 64, 64, 32)),
    ],
)
def test_sm103_other_inputs_keep_original_tiles(kernel, dim, dtype, expected):
    assert kernel._select_tiles(dim, (10, 3), dtype) == expected
