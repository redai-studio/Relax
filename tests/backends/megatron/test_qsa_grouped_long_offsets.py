# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real grouped-QSA pointers beyond signed 32-bit element offsets."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _check_long_offsets(heads: int, routes: int) -> None:
    source = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_mixed_ops/grouped.py"
    spec = importlib.util.spec_from_file_location("qsa_grouped_long_offsets", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attention_bf16 = module.attention_bf16

    length = 1 << 20
    # A single self route gives an exact output/gradient oracle without a
    # quadratic reference. The unused route columns still exercise t * S.
    q = torch.zeros((length, heads, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros((length, 1, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.ones((length, 1, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.full((length, routes), -1, device="cuda", dtype=torch.int32)
    indices[:, 0] = torch.arange(length, device="cuda", dtype=torch.int32)
    output = attention_bf16(q, k, v, indices, 256**-0.5)
    # Probe both sides of each overflowing stride, as well as the final row.
    boundary = (1 << 31) // max(heads * 256, routes)
    rows = torch.tensor([0, boundary - 1, boundary, boundary + 1, length - 1], device="cuda")
    torch.testing.assert_close(output[rows], torch.ones_like(output[rows]), rtol=0, atol=0)
    dq, dk, dv = torch.autograd.grad(output, (q, k, v), torch.ones_like(output))
    torch.testing.assert_close(dq[rows], torch.zeros_like(dq[rows]), rtol=0, atol=0)
    torch.testing.assert_close(dk[rows], torch.zeros_like(dk[rows]), rtol=0, atol=0)
    torch.testing.assert_close(dv[rows], torch.full_like(dv[rows], heads), rtol=0, atol=0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Long QSA offsets require CUDA")
@pytest.mark.parametrize("heads,routes", [(12, 1), (1, 2052)])
def test_grouped_qsa_long_offsets(heads: int, routes: int) -> None:
    # Existing short numerical tests cannot cross either pointer boundary.
    # The largest case uses about 52 GiB; skip devices without this capacity.
    if torch.cuda.mem_get_info()[0] < 64 * 2**30:
        pytest.skip("Long QSA pointer regression requires 64 GiB free GPU memory")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(heads), str(routes)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_long_offsets(int(sys.argv[1]), int(sys.argv[2]))
