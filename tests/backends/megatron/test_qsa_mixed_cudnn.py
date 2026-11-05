# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Blackwell numerical checks for the PR1765 adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py"
SPEC = importlib.util.spec_from_file_location("qsa_mixed_cudnn_under_test", SOURCE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PR1765 numerics require Blackwell CUDA and optional cuDNN QSA"
)
@pytest.mark.parametrize("lengths", [(2051,), (2052,), (4097,), (7, 2053, 5, 2052)])
def test_cuda_output_and_qkv_gradients_against_fp32_reference(monkeypatch, lengths):
    """Synthetic numerical regression, not a TP/PP integration or performance
    test."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("This integration gate targets Blackwell SM100/SM103")
    pytest.importorskip("cudnn.QSA.api", reason="Install the pinned PR1765 optional dependency")
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    monkeypatch.setenv("RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS", "0")
    torch.manual_seed(1234)
    total = sum(lengths)
    values = [torch.randn((1, total, h, 256), device="cuda", dtype=torch.bfloat16) / 4 for h in (12, 1, 1)]
    actual_inputs = [v.detach().requires_grad_() for v in values]
    references = [v.float().detach().requires_grad_() for v in values]
    bounds, selections, expected = [0], [], []
    for length in lengths:
        start = bounds[-1]
        bounds.append(start + length)
        position = torch.arange(length, device="cuda")
        complete = (position + 1) // 4
        block_count = (length + 3) // 4
        scores = torch.rand((length, block_count), device="cuda")
        scores.masked_fill_(torch.arange(block_count, device="cuda")[None, :] >= complete[:, None], -torch.inf)
        selected_scores, blocks = scores.topk(min(512, block_count), dim=-1, sorted=False)
        blocks = blocks.to(torch.int32).masked_fill(~torch.isfinite(selected_scores), -1)
        blocks = torch.nn.functional.pad(blocks, (0, 512 - blocks.shape[1]), value=-1)
        selections.append(blocks)
        # Independent token-level oracle: selected complete blocks plus only the
        # query's partial current block, bounded by the current document.
        selected = torch.zeros((length, block_count), device="cuda", dtype=torch.bool)
        rows = position[:, None].expand_as(blocks)
        valid = blocks >= 0
        selected[rows[valid], blocks[valid].long()] = True
        key = position[None, :]
        mask = selected[:, position // 4] | (((position[:, None] + 1) % 4 != 0) & (key // 4 == position[:, None] // 4))
        mask &= key <= position[:, None]
        if length > 4096:
            # An always-dense or first-512-block implementation must not pass
            # merely because only one boundary row differs from Flash.
            assert selected[-1, 512:].any() and not selected[-1, :512].all()
            assert mask[-1, :-1].sum() == 2048
        q, k, v = [x[:, start : bounds[-1]].transpose(1, 2) for x in references]
        with sdpa_kernel(SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0, scale=256**-0.5, enable_gqa=True
            )
        expected.append(out.transpose(1, 2))
    selection = SimpleNamespace(compress_ratio=4, selected_ids=torch.cat(selections))
    actual = adapter.attention_for_selection(*actual_inputs, selection, 256**-0.5, tuple(bounds))
    expected = torch.cat(expected, dim=1)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)
    expected_grads = torch.autograd.grad(expected, references, upstream.float())
    for got, want in zip((actual, *actual_grads), (expected, *expected_grads)):
        assert torch.isfinite(got).all()
        error = torch.linalg.vector_norm(got.float() - want) / torch.linalg.vector_norm(want).clamp_min(1e-12)
        assert error < 0.02
