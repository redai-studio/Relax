# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""QSA computation helpers; no model-method installation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_r4(
    raw,
    offsets,
    output,
    TOTAL: tl.constexpr,
    COLUMNS: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    channel = index % D
    block = (index // D) % COLUMNS
    doc = index // (D * COLUMNS)
    active = index < TOTAL
    start = tl.load(offsets + doc, active, 0)
    end = tl.load(offsets + doc + 1, active, 0)
    complete = active & (4 * (block + 1) <= end - start)
    first = start + 4 * block
    a = tl.load(raw + first * S0 + channel * S1, complete, 0).to(tl.float64)
    b = tl.load(raw + (first + 1) * S0 + channel * S1, complete, 0).to(tl.float64)
    c = tl.load(raw + (first + 2) * S0 + channel * S1, complete, 0).to(tl.float64)
    d = tl.load(raw + (first + 3) * S0 + channel * S1, complete, 0).to(tl.float64)
    result = ((a + b) + c) + d
    tl.store(output + index, result * 0.25, active)


def fixed_pool(raw_keys: torch.Tensor, offsets: tuple[int, ...], max_doc_len: int) -> tuple:
    """Packed documents in increasing ID order; CPU offsets are the layout
    authority."""
    if (
        raw_keys.ndim != 2
        or raw_keys.dtype != torch.bfloat16
        or not raw_keys.is_cuda
        or raw_keys.shape[0] <= 0
        or raw_keys.shape[1] <= 0
        or any(s <= 0 for s in raw_keys.stride())
        or raw_keys.requires_grad
    ):
        raise ValueError("Fixed QSA pool requires a frozen CUDA BF16 matrix")
    if (
        not isinstance(offsets, tuple)
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != raw_keys.shape[0]
        or any(type(x) is not int for x in offsets)
        or any(a > b for a, b in zip(offsets, offsets[1:]))
        or max_doc_len != max(b - a for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("Invalid packed QSA offsets")
    # Match max(doc_ids)+1: trailing empty documents have no token IDs.
    while offsets[-2] == offsets[-1]:
        offsets = offsets[:-1]
    n_docs, columns, dim = len(offsets) - 1, max_doc_len // 4, raw_keys.shape[1]
    bounds = torch.tensor(offsets, device=raw_keys.device, dtype=torch.int64)
    positions = (torch.arange(columns, device=raw_keys.device, dtype=torch.long) * 4).unsqueeze(0).expand(n_docs, -1)
    valid = torch.arange(columns, device=raw_keys.device).unsqueeze(0) < ((bounds[1:] - bounds[:-1]) // 4).unsqueeze(1)
    pooled = torch.empty((n_docs, columns, dim), device=raw_keys.device, dtype=raw_keys.dtype)
    if pooled.numel():
        _pool_r4[(triton.cdiv(pooled.numel(), 256),)](
            raw_keys,
            bounds,
            pooled,
            pooled.numel(),
            columns,
            dim,
            raw_keys.stride(0),
            raw_keys.stride(1),
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return pooled, positions, valid
