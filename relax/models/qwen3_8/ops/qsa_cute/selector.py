# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exact block-ID selection with the PR7553 radix kernels and a CPU oracle."""

from __future__ import annotations

import functools
import json
import os

import torch
from torch import Tensor


@functools.lru_cache(maxsize=1)
def _interval_config():
    from relax.models.qwen3_8.ops.qsa_cute.score_interval_selector import IntervalConfig

    return IntervalConfig(**json.loads(os.environ.get("QWEN38_DOSA_INTERVAL_CONFIG", "{}")))


def select_ids(query: Tensor, keys: Tensor, positions: Tensor, budget: int, impl: str = "auto") -> Tensor:
    """Select complete causal blocks. Radix, DoSA and CPU ties use ascending
    block ID; torch keeps its original GEMM and topk tie behavior.

    ``dosa-set`` preserves radix membership and may reorder the valid IDs. It
    is only used by the CuTe CP set consumer.

    The CPU implementation is also useful for small numerical oracles. It uses
    the same FP32 scoring contract; it is not a fallback for unsupported GPUs.
    """
    if impl not in ("auto", "triton", "cudnn", "torch", "dosa", "dosa-set") or budget <= 0:
        raise ValueError("QSA radix selection requires a positive block budget and a supported implementation")
    if query.ndim != 3 or keys.ndim != 2 or query.shape[-1] != keys.shape[-1]:
        raise ValueError("QSA selector expects [queries, heads, dim] and [blocks, dim]")
    if positions.shape != (query.shape[0],):
        raise ValueError("QSA selector positions must match its query rows")
    rows, heads, dim = query.shape
    blocks = keys.shape[0]
    if blocks == 0 or rows == 0:
        return torch.full((rows, budget), -1, dtype=torch.int32, device=query.device)
    count = min(budget, blocks)
    visible = ((positions + 1) // 4).clamp(min=0, max=blocks)
    if query.is_cuda and impl in ("dosa", "dosa-set"):
        boundary_only = impl == "dosa-set"
        variant = os.environ.get("QWEN38_DOSA_VARIANT")
        if boundary_only or variant in ("interval", "boundary"):
            from relax.models.qwen3_8.ops.qsa_cute.score_interval_selector import select_interval_ids

            config = _interval_config()
            if not boundary_only and (config.boundary_bins or config.boundary_reuse_topk):
                config = config._replace(half_scores=False, boundary_bins=0, boundary_reuse_topk=False)
            selected = select_interval_ids(query, keys, visible, budget, config=config, boundary_only=boundary_only)
        else:
            from relax.models.qwen3_8.ops.qsa_cute.seed_bound_selector import dosa_select_ids

            selected = dosa_select_ids(query, keys, visible, budget)
        if selected is not None:
            return selected
        impl = "triton" if boundary_only else "auto"
    output = torch.full((rows, budget), -1, dtype=torch.int32, device=query.device)
    if query.is_cuda and impl == "torch":
        from relax.models.qwen3_8.ops.qsa_mixed_ops.selector import index_scores

        # Same score/topk baseline as the existing mixed selector, bounded to
        # 512 query rows so the CP control cannot allocate a full score matrix.
        for start in range(0, rows, 512):
            end = min(start + 512, rows)
            scores = index_scores(query[start:end], keys).relu().sum(-1) * dim**-0.5
            valid = torch.arange(blocks, device=query.device)[None, :] < visible[start:end, None]
            values, ids = scores.masked_fill(~valid, -torch.inf).topk(count, dim=-1)
            output[start:end, :count] = torch.where(values.isfinite(), ids, -1).int()
    elif query.is_cuda:
        from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import ID_BITS, qsa_select_topk_keys

        encoded = qsa_select_topk_keys(query.contiguous(), keys.contiguous(), visible, count, impl=impl)
        mask = (1 << ID_BITS) - 1
        output[:, :count] = torch.where(encoded > 0, mask - (encoded & mask), -1).int()
    else:
        scores = torch.zeros(rows, blocks, dtype=torch.float32, device=query.device)
        for head in range(heads):
            scores += (query[:, head].float() @ keys.float().T).relu()
        scores *= dim**-0.5
        valid = torch.arange(blocks, device=query.device)[None, :] < visible[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        ids = torch.argsort(scores, descending=True, stable=True)[:, :count]
        output[:, :count] = torch.where(valid.gather(1, ids), ids, -1).int()
    return output
