# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (Miles).
# Copyright 2025 Zhipu AI (Slime).
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax from Miles (https://github.com/radixark/miles), commit 3a7659a4.
# Miles includes software developed from Slime (https://github.com/THUDM/slime).

"""MSA block scores for training: each query's maximum logit over every causal
key block of its sequence."""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor


_BQ = 64


@triton.jit
def _block_max_scores_kernel(
    Q,
    K,
    TILES,
    OUT,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_oh,
    stride_ot,
    scale,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BLK: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    q_start = tl.load(TILES + tile * 5)
    q_end = tl.load(TILES + tile * 5 + 1)
    global_q_start = tl.load(TILES + tile * 5 + 2)
    seq_start = tl.load(TILES + tile * 5 + 3)
    seq_end = tl.load(TILES + tile * 5 + 4)

    q_rows = tl.arange(0, BQ)
    offs_q = q_start + q_rows
    global_q = global_q_start + q_rows
    offs_d = tl.arange(0, D)
    q_in = offs_q < q_end
    q = tl.load(Q + offs_q[:, None] * stride_qt + head * stride_qh + offs_d[None, :], mask=q_in[:, None], other=0.0)

    # a tile never straddles sequences, so every query shares the key blocks; the last query bounds them
    last_block = (global_q_start + tl.minimum(BQ, q_end - q_start) - 1 - seq_start) // BLK
    for block in range(0, last_block + 1):
        offs_k = seq_start + block * BLK + tl.arange(0, BLK)
        k_in = offs_k < seq_end
        k = tl.load(K + offs_k[:, None] * stride_kt + offs_d[None, :], mask=k_in[:, None], other=0.0)
        logits = tl.dot(q, tl.trans(k))
        causal = (offs_k[None, :] <= global_q[:, None]) & k_in[None, :]
        best = tl.max(tl.where(causal, logits, float("-inf")), axis=1) * scale
        tl.store(OUT + head * stride_oh + offs_q * stride_ot + block, best, mask=q_in)


def msa_block_max_scores(
    q: Tensor, k: Tensor, bounds: Sequence[int], block_size: int, scale: float, *, cp_size: int = 1, cp_rank: int = 0
) -> Tensor:
    """``q`` [T, H, D], ``k`` [S, D] -> ``[H, T, NB]`` fp32 block scores,
    ``-inf`` where no key is causal.

    ``NB`` is the block count of the longest sequence; blocks start at each
    sequence's first token.
    """
    T, H, D = q.shape
    tiles = []
    local_start = 0
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        if cp_size == 1:
            segments = ((start, end),)
        else:
            chunk = (end - start) // (2 * cp_size)
            heads = (cp_rank, 2 * cp_size - cp_rank - 1)
            segments = tuple((start + head * chunk, start + (head + 1) * chunk) for head in heads)
        for first, last in segments:
            local_end = local_start + last - first
            tiles.extend(
                (row, local_end, first + row - local_start, start, end) for row in range(local_start, local_end, _BQ)
            )
            local_start = local_end
    if local_start != T or bounds[-1] != k.shape[0]:
        raise ValueError("MSA query and key lengths must match the packed document boundaries")
    num_blocks = max(-(-(end - start) // block_size) for start, end in zip(bounds[:-1], bounds[1:], strict=True))
    tiles = torch.tensor(tiles, dtype=torch.int32, device=q.device)
    out = torch.full((H, T, num_blocks), float("-inf"), dtype=torch.float32, device=q.device)
    qc, kc = q.contiguous(), k.contiguous()
    _block_max_scores_kernel[(tiles.shape[0], H)](
        qc,
        kc,
        tiles,
        out,
        qc.stride(0),
        qc.stride(1),
        kc.stride(0),
        out.stride(0),
        out.stride(1),
        scale,
        D=D,
        BQ=_BQ,
        BLK=block_size,
        num_warps=4,
        num_stages=2,
    )
    return out
