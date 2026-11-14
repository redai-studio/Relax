# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent slot-wise oracle for packed expanded attention and its
gradients."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("reordered", [False, True])
def test_packed_expanded_attention_preserves_selected_slots_and_gradients(reordered):
    path = Path(__file__).resolve().parents[3] / "relax/models/kimi_k3/expanded_dsa.py"
    spec = importlib.util.spec_from_file_location("_k31_expanded_attention_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator().manual_seed(19)
    tensors = [
        torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in ((4, 1, 2, 3), (8, 1, 2, 3), (8, 1, 2, 2))
    ]
    reference = [value.detach().clone().requires_grad_() for value in tensors]
    indices = torch.tensor([[[0, 0, -1, 7], [1, 2, 3, -1], [4, 6, 6, -1], [-1, -1, -1, -1]]])
    starts, ends = torch.tensor([0, 0, 4, 4]), torch.tensor([1, 4, 7, 8])
    positions = torch.tensor([3, 2, 1, 0, 7, 6, 5, 4]) if reordered else torch.arange(8)
    actual = module.packed_expanded_attention(
        *tensors,
        indices,
        0.6,
        varlen_starts=starts,
        varlen_ends=ends,
        key_positions=positions,
    )
    q, k, v = reference
    rows = []
    for row in range(4):
        slots = [idx for idx in indices[0, row] if idx >= 0 and starts[row] <= positions[idx] < ends[row]]
        if slots:
            keys, values = k[slots, 0], v[slots, 0]
            scores = torch.einsum("hd,khd->hk", q[row, 0], keys) * 0.6
            rows.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values))
        else:
            rows.append(q[row, 0, :, :2] * 0 + (k.sum() + v.sum()) * 0)
    expected = torch.stack(rows).unsqueeze(1).flatten(-2)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_grads = torch.autograd.grad((actual * cotangent).sum(), tensors)
    expected_grads = torch.autograd.grad((expected * cotangent).sum(), reference)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for observed, wanted in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual[-1]) == 0
    assert torch.count_nonzero(actual_grads[0][-1]) == 0
