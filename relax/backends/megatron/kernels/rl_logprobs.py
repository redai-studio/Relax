# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""FP32-accumulator linear log-prob/entropy tiles.

Reference: verl utils/kernel/kernels.py at
0938fcb3190bc8eca3091f77aead3cd44c4d8869 (Apache-2.0).
Only row statistics leave the forward GEMM; backward writes a bounded BF16
dlogits workspace. No BF16 rounding of forward logits or probabilities.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _project(
    X, W, rows, vocab: tl.constexpr, hidden: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(hidden, BK)):
        kk = start * BK + k
        x = tl.load(X + m[:, None] * hidden + kk[None, :], (m[:, None] < rows) & (kk[None, :] < hidden), 0)
        w = tl.load(W + n[None, :] * hidden + kk[:, None], (n[None, :] < vocab) & (kk[:, None] < hidden), 0)
        acc = tl.dot(x, w, acc)
    return acc


@triton.jit
def _forward(
    X,
    W,
    Y,
    Stats,
    rows,
    vocab: tl.constexpr,
    hidden: tl.constexpr,
    offset: tl.constexpr,
    temperature: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = _project(X, W, rows, vocab, hidden, BM, BN, BK) / temperature
    z = tl.where(n[None, :] < vocab, z, -float("inf"))
    maximum = tl.max(z, 1)
    shifted = z - maximum[:, None]
    p = tl.exp(shifted)
    denom = tl.sum(p, 1)
    weighted = tl.sum(p * tl.where(n[None, :] < vocab, shifted, 0.0), 1)
    y = tl.load(Y + m, m < rows, -1)
    target = tl.sum(tl.where((n[None, :] < vocab) & (n[None, :] + offset == y[:, None]), z, 0.0), 1)
    base = (tl.program_id(1) * rows + m) * 4
    tl.store(Stats + base, maximum, m < rows)
    tl.store(Stats + base + 1, denom, m < rows)
    tl.store(Stats + base + 2, weighted, m < rows)
    tl.store(Stats + base + 3, target, m < rows)


@triton.jit
def _backward(
    X,
    W,
    Y,
    Maximum,
    Denom,
    Mean,
    A,
    B,
    D,
    rows,
    vocab: tl.constexpr,
    hidden: tl.constexpr,
    offset: tl.constexpr,
    temperature: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_B: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    z = _project(X, W, rows, vocab, hidden, BM, BN, BK) / temperature
    maximum = tl.load(Maximum + m, m < rows, 0)
    denom = tl.load(Denom + m, m < rows, 1)
    p = tl.exp(z - maximum[:, None]) / denom[:, None]
    y = tl.load(Y + m, m < rows, -1)
    dz = tl.full((BM, BN), 0, tl.float32)
    if HAS_A:
        a = tl.load(A + m, m < rows, 0)
        dz = a[:, None] * ((n[None, :] + offset == y[:, None]).to(tl.float32) - p)
    if HAS_B:
        b = tl.load(B + m, m < rows, 0)
        mean = tl.load(Mean + m, m < rows, 0)
        dz += b[:, None] * p * (mean[:, None] - (z - maximum[:, None]))
    # Match native low-precision backward: cast the combined scaled gradient once.
    tl.store(D + m[:, None] * vocab + n[None, :], dz / temperature, (m[:, None] < rows) & (n[None, :] < vocab))


def forward_stats(
    x: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, offset: int, temperature: float
) -> torch.Tensor:
    stats = torch.empty((triton.cdiv(weight.shape[0], 256), x.shape[0], 4), device=x.device, dtype=torch.float32)
    _forward[(triton.cdiv(x.shape[0], 16), stats.shape[0])](
        x,
        weight,
        labels,
        stats,
        x.shape[0],
        weight.shape[0],
        weight.shape[1],
        offset,
        temperature,
        16,
        256,
        32,
        num_warps=4,
    )
    return stats


def backward_dlogits(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    denom: torch.Tensor,
    mean: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    offset: int,
    temperature: float,
) -> torch.Tensor:
    result = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    _backward[(triton.cdiv(x.shape[0], 16), triton.cdiv(weight.shape[0], 256))](
        x,
        weight,
        labels,
        maximum,
        denom,
        mean,
        a,
        b,
        result,
        x.shape[0],
        weight.shape[0],
        weight.shape[1],
        offset,
        temperature,
        a is not None,
        b is not None,
        16,
        256,
        32,
        num_warps=4,
    )
    return result
