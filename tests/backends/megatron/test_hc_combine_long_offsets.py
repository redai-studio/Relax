# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate HC residual pointer arithmetic at 512k/1M TP2 sequence lengths."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _check_long_offsets(rows: int) -> None:
    source = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/hc_combine.py"
    spec = importlib.util.spec_from_file_location("hc_combine_long_offsets", source)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    residual = torch.zeros((rows, 1, 10240), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = torch.full((rows, 1, 2560), 3.0, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gates = torch.full((rows, 1, 4), 2.0, device="cuda", requires_grad=True)
    actual = ops.combine(residual, output, gates, 4)
    boundary = (1 << 31) // 10240
    samples = torch.tensor([0, boundary - 1, boundary, boundary + 1, rows - 1], device="cuda")
    # Exact analytic oracle: R + A*Y = 6; dR=1, dY=4*2, dA=2560*3.
    torch.testing.assert_close(actual[samples], torch.full_like(actual[samples], 6), rtol=0, atol=0)
    grads = torch.autograd.grad(actual, (residual, output, gates), torch.ones_like(actual))
    for grad, expected in zip(grads, (1, 8, 7680), strict=True):
        torch.testing.assert_close(grad[samples], torch.full_like(grad[samples], expected), rtol=0, atol=0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HC pointer regression requires CUDA")
@pytest.mark.parametrize("rows", [262144, 524288])
def test_hc_combine_long_offsets(rows: int) -> None:
    # Short parity tests never cross row*4*2560 >= 2**31. The largest case
    # needs about 35 GiB; isolate a potential illegal access from other tests.
    if torch.cuda.mem_get_info()[0] < 40 * 2**30:
        pytest.skip("HC long-offset regression requires 40 GiB free GPU memory")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(rows)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_long_offsets(int(sys.argv[1]))
