# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Flex sparse GQA with selected-block and causal-tail masks."""

import os

import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


_compiled_attention = torch.compile(flex_attention, dynamic=True)


@triton.jit
def _selected_bits(
    Indices,
    Starts,
    Bits,
    STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    WORDS: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    ids = tl.load(Indices + row * STRIDE + offsets * RATIO, offsets < TOPK, other=-1)
    start = tl.load(Starts + row)
    block = (ids - start) // RATIO
    valid = (offsets < TOPK) & (ids >= start)
    tl.atomic_or(Bits + row * WORDS + tl.maximum(block, 0) // 32, 1 << (block % 32), valid, sem="relaxed")


def layout(bounds: tuple[int, ...] | list[int], device: torch.device) -> tuple:
    total = bounds[-1]
    rows = torch.arange(total, device=device, dtype=torch.int32)
    starts = torch.tensor(bounds[:-1], device=device, dtype=torch.int32)
    docs = torch.searchsorted(starts, rows, right=True) - 1
    token_starts = starts[docs]
    return docs, rows - token_starts, token_starts, max(b - a for a, b in zip(bounds, bounds[1:]))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    assert q.ndim == k.ndim == v.ndim == 3
    tokens = q.shape[0]
    ratio = 4
    assert (indices.shape[1] - ratio) % ratio == 0
    bounds = getattr(indices, "_qsa_bounds", (0, tokens))
    maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
    budget = indices.shape[1] - ratio
    if maximum <= budget + ratio - 1:
        from flash_attn import flash_attn_varlen_func

        cu = torch.tensor(bounds, device=q.device, dtype=torch.int32)
        return flash_attn_varlen_func(
            q.contiguous(), k.contiguous(), v.contiguous(), cu, cu, maximum, maximum, softmax_scale=scale, causal=True
        )
    metadata = getattr(indices, "_qsa_layout", None)
    if metadata is None:
        metadata = layout(bounds, q.device)
    docs, positions, starts, maximum = metadata
    words = (maximum // ratio + 1 + 31) // 32
    with torch.no_grad():
        bits = torch.zeros((tokens, words), device=q.device, dtype=torch.int32)
        topk = budget // ratio
        _selected_bits[(tokens,)](
            indices, starts, bits, indices.shape[1], topk, words, ratio, triton.next_power_of_2(topk), num_warps=4
        )
        stride = torch.tensor(words, device=q.device, dtype=torch.int64)
        flat_bits = bits.flatten()

        def mask(batch, head, qi, ki):
            safe_q, safe_k = qi.clamp(max=tokens - 1), ki.clamp(max=tokens - 1)
            qp, kp = positions[safe_q], positions[safe_k]

            complete = (qp + 1) // 4
            key_block = kp // 4
            selected = ((flat_bits[safe_q * stride + key_block // 32] >> (key_block % 32)) & 1) != 0
            return (
                (qi < tokens)
                & (ki < tokens)
                & (docs[safe_q] == docs[safe_k])
                & (kp <= qp)
                & (((key_block < complete) & selected) | (key_block == complete))
            )

        if os.environ.get("RELAX_QWEN38_QSA_FLEX_SPARSE_MASK", "1") == "1":
            from relax.models.qwen3_8.ops.qsa_mixed_ops.block_mask import create

            block_mask = create(bits, starts, mask)
        else:
            block_mask = create_block_mask(mask, 1, None, tokens, tokens, device=q.device, _compile=True)
    out = _compiled_attention(
        q.transpose(0, 1).unsqueeze(0).contiguous(),
        k.transpose(0, 1).unsqueeze(0).contiguous(),
        v.transpose(0, 1).unsqueeze(0).contiguous(),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
    )
    return out.squeeze(0).transpose(0, 1)


def attention_auto(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float
) -> torch.Tensor:
    return attention(q, k, v, indices, scale)
