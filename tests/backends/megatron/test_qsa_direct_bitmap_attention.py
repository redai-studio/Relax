# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare compact mixed attention with the existing expanded-token route."""

import importlib

import pytest
import torch
from megatron.core.transformer.experimental_attention_variant.qsa import QSASelection

from relax.models.qwen3_8.ops.qsa_mixed_ops import attention_for_selection
from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes
from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import bind_document_bounds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA mixed QSA")
@pytest.mark.parametrize(
    "lengths",
    [(2056, 2056), (16, 2056), (16, 48), (16388,), (0, 2055, 7, 130), (17, 2056, 16388), (32772,), (17, 2056, 32772)],
)
def test_direct_bitmap_mixed_preserves_output_and_all_qkv_gradients(monkeypatch, lengths):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", "-1")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", "16384")
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "grouped")
    flex = importlib.import_module("relax.models.qwen3_8.ops.qsa_mixed_ops.flex")
    torch.manual_seed(341)
    total = sum(lengths)
    ids = torch.full((total, 512), -1, device="cuda", dtype=torch.int32)
    positions = torch.empty((1, total), device="cuda", dtype=torch.int32)
    docs = torch.empty_like(positions)
    bounds = [0]
    for doc, length in enumerate(lengths):
        start = bounds[-1]
        p = torch.arange(length, device="cuda", dtype=torch.int32)
        complete = (p + 1) // 4
        base = torch.arange(512, device="cuda", dtype=torch.int32)[None]
        candidate = torch.where(complete[:, None] > 512, base * complete[:, None] // 512, base)
        ids[start : start + length] = candidate.masked_fill(candidate >= complete[:, None], -1)
        positions[0, start : start + length] = p
        docs[0, start : start + length] = doc
        bounds.append(start + length)
    selection = bind_document_bounds(
        QSASelection(
            doc_ids=docs,
            positions=positions,
            selected_bits=None,
            bits_per_row=0,
            bits_per_row_t=torch.zeros((), device="cuda", dtype=torch.int64),
            compress_ratio=4,
            all_selected=False,
            selected_ids=ids,
            max_complete_blocks=max(lengths) // 4,
        ),
        tuple(bounds),
    )
    routes = torch.nn.functional.pad(token_routes(selection, 1, total)[0], (0, 1), value=-1)
    words = (max(lengths) // 4 + 1 + 31) // 32
    starts = torch.arange(total, device="cuda", dtype=torch.int32) - positions[0]
    bits = torch.zeros((total, words), device="cuda", dtype=torch.int32)
    flex._selected_bits[(total,)](routes, starts, bits, 2052, 512, words, 4, 512, num_warps=4)
    from dataclasses import replace

    compact = replace(
        selection,
        selected_ids=None,
        selected_bits=bits.view(torch.uint8).flatten(),
        bits_per_row=words * 4,
        bits_per_row_t=torch.full((), words * 4, device="cuda", dtype=torch.int64),
    )
    inputs = [torch.randn((1, total, heads, 256), device="cuda", dtype=torch.bfloat16) for heads in (6, 1, 1)]
    grad = torch.randn_like(inputs[0])

    def run(value):
        tensors = [tensor.detach().requires_grad_(True) for tensor in inputs]
        output = attention_for_selection(*tensors, value, 256**-0.5, True)
        gradients = torch.autograd.grad(output, tensors, grad)
        return output.detach(), gradients

    expected, expected_gradients = run(selection)
    dedicated = importlib.import_module("relax.models.qwen3_8.ops.qsa_mixed_ops.bitmap_attention")
    original_attention = dedicated.attention
    calls = []

    def record_attention(q, k, v, bits, scale, bounds):
        calls.append((q.shape[0], tuple(bounds)))
        return original_attention(q, k, v, bits, scale, bounds)

    monkeypatch.setattr(dedicated, "attention", record_attention)
    actual, actual_gradients = run(compact)
    middle = [length for length in lengths if 2051 < length <= 32768]
    assert bool(calls) == bool(middle)
    if middle:
        assert calls[0][0] == sum(middle)
        assert calls[0][1][-1] == sum(middle)
    if middle:
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-10)
        assert error < 0.008
    else:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for got, want in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(got).all()
        if max(lengths) > 32768:
            # The unchanged Grouped backend uses non-deterministic FP32 KV atomics.
            torch.testing.assert_close(got, want, atol=0.01, rtol=0.01)
        elif max(lengths) > 2051:
            error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-10)
            assert error < 0.008
        else:
            torch.testing.assert_close(got, want, atol=0, rtol=0)
