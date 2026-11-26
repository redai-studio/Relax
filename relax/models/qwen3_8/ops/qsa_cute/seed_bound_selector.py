# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in QSA DoSA, with the existing radix selector's FP32/tie contract.

Only four-head BF16 queries on SM90 and SM103 are optimized. Unsupported
geometry and short documents return None so select_ids retains its radix
implementation. Scratch is bounded by a 128-row chunk, not the CP rank's full
query count.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def dosa_select_ids(
    query: Tensor,
    keys: Tensor,
    visible: Tensor,
    budget: int,
    *,
    seed_size: int = 4096,
    candidate_capacity: int = 32768,
    row_chunk: int = 128,
    stats: dict[str, Tensor] | None = None,
) -> Tensor | None:
    """Screen safely, refine with the original radix scorer, return block IDs.

    Shared uniform seeds are causally masked per row. Insufficient seeds yield
    -inf threshold (no pruning). Candidate overflow triggers full causal
    scoring on GPU; compaction never silently drops a live candidate.
    Statistics are device tensors and cause no synchronization in the training
    hot path.
    """
    if (
        query.device.type != "cuda"
        or torch.version.hip is not None
        or query.dtype != torch.bfloat16
        or keys.dtype != query.dtype
        or query.ndim != 3
        or query.shape[1:] != (4, 128)
        or keys.ndim != 2
        or keys.shape[1] != 128
        or keys.device != query.device
        or visible.device != query.device
        or visible.shape != query.shape[:1]
        or visible.dtype not in (torch.int32, torch.int64)
        or torch.cuda.get_device_capability(query.device) not in ((9, 0), (10, 3))
    ):
        return None
    if seed_size < budget or seed_size % 128 or candidate_capacity < budget or row_chunk <= 0:
        raise ValueError(
            "QSA DoSA requires seed_size >= budget (a multiple of 128), capacity >= budget, row_chunk > 0"
        )
    rows, blocks = query.shape[0], keys.shape[0]
    if blocks <= seed_size or blocks >= (1 << 21) or not 0 < budget <= 512:
        return None
    import triton

    from relax.models.qwen3_8.ops.qsa_cute import score_bound_kernels as kernels
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import (
        _qsa_select_score_kernel,
        _qsa_select_topk_kernel,
    )

    query, keys = query.contiguous(), keys.contiguous()
    visible = visible.clamp(0, blocks).int().contiguous()
    seed_ids = torch.arange(seed_size, device=query.device) * blocks // seed_size
    seed_keys = keys[seed_ids].contiguous()
    seed_visible = ((visible.long() * seed_size + blocks - 1) // blocks).int()
    output = torch.empty((rows, budget), dtype=torch.int32, device=query.device)
    norms = torch.empty(blocks, device=query.device)
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    kernels.key_norm[(triton.cdiv(blocks, 32),)](keys, norms, blocks, **launch)
    chunk = min(rows, row_chunk)
    capacity = min(blocks, candidate_capacity)
    tiles = triton.cdiv(blocks, 128)
    kp = triton.next_power_of_2(budget)
    anchor = torch.empty((chunk, 128), dtype=query.dtype, device=query.device)
    fold = torch.empty(chunk, device=query.device)
    error = torch.empty_like(fold)
    seeds = torch.empty((chunk, seed_size), device=query.device)
    flags = torch.empty((chunk, blocks), dtype=torch.bool, device=query.device)
    counts = torch.empty((chunk, tiles), dtype=torch.int32, device=query.device)
    ids = torch.empty((chunk, capacity), dtype=torch.int32, device=query.device)
    scores = torch.empty((chunk, blocks), device=query.device)
    lengths = torch.empty(chunk, dtype=torch.int32, device=query.device)
    codes = torch.empty((chunk, kp), dtype=torch.int64, device=query.device)
    if stats is not None:
        survivors = torch.empty(rows, dtype=torch.int32, device=query.device)
        refined = torch.empty_like(survivors)
        thresholds = torch.empty(rows, device=query.device)
    for start in range(0, rows, row_chunk):
        end = min(start + row_chunk, rows)
        n, q, vis = end - start, query[start:end], visible[start:end]
        kernels.prepare[(n,)](q, anchor, fold, error, **launch)
        seeds[:n].fill_(-torch.inf)
        # Reuse the exact dense scorer with shared seed keys. This retains
        # inter-query key reuse rather than padding each query to 64 rows.
        _qsa_select_score_kernel.fn[(triton.cdiv(n, 64), seed_size // 128)](
            q,
            seed_keys,
            seeds,
            seed_visible[start:end],
            vis,
            n,
            seed_size,
            seed_size,
            128**-0.5,
            NH=4,
            DI=128,
            BM=64,
            BN=128,
            BK=16,
            HAS_LO=False,
            **launch,
        )
        tau = seeds[:n].topk(budget, dim=-1).values[:, -1].contiguous()
        kernels.screen[(triton.cdiv(n, 16), tiles)](
            keys, norms, anchor, fold, error, vis, tau, flags, counts, n, blocks, **launch
        )
        prefix = counts[:n].cumsum(-1, dtype=torch.int32)
        kernels.compact[(n, tiles)](flags, prefix, ids, blocks, capacity, **launch)
        kernels.refine[(n, tiles)](q, keys, ids, prefix, vis, scores, lengths, blocks, capacity, **launch)
        dense_visible = torch.where(prefix[:, -1] > capacity, vis, 0)
        # Zero visible lengths skip all scoring for non-overflow rows and
        # preserve their compacted scores in the same scratch buffer.
        _qsa_select_score_kernel.fn[(triton.cdiv(n, 64), tiles)](
            q,
            keys,
            scores,
            dense_visible,
            vis,
            n,
            blocks,
            blocks,
            128**-0.5,
            NH=4,
            DI=128,
            BM=64,
            BN=128,
            BK=16,
            HAS_LO=False,
            **launch,
        )
        _qsa_select_topk_kernel[(n,)](
            scores,
            lengths,
            codes,
            lengths,
            lengths,
            budget,
            blocks,
            (1 << 21) - 1,
            TILE=2048,
            KP=kp,
            IDB=21,
            HAS_LO=False,
            GIVEN_THR=False,
            num_warps=8,
        )
        kernels.restore[(n,)](codes, ids, prefix, output[start:end], blocks, capacity, kp, budget, **launch)
        if stats is not None:
            survivors[start:end] = prefix[:, -1]
            refined[start:end] = lengths[:n]
            thresholds[start:end] = tau
    if stats is not None:
        stats.update(
            visible=visible, survivors=survivors, refined=refined, overflow=survivors > capacity, threshold=thresholds
        )
    return output
