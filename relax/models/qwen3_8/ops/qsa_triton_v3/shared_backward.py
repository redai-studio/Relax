# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Backward-local workspace; overflow stays on GPU and recomputes dQ."""

from dataclasses import dataclass
from typing import Any

import torch
import triton

from relax.models.qwen3_8.ops import qsa_fused_attention as base
from relax.models.qwen3_8.ops.qsa_triton_v3 import shared_kernels as ops


DEFAULT_CAPACITY_BYTES = 24 * 2**30


@dataclass
class Workspace:
    ds: torch.Tensor
    edge_map: torch.Tensor
    prefix: torch.Tensor
    capacity_edges: int
    requested_bytes: int

    @property
    def allocated_bytes(self) -> int:
        return self.ds.numel() * self.ds.element_size()


def workspace_shape(q_len: int, kv_len: int, heads: int, capacity_bytes: int) -> tuple[int, int, int]:
    """CPU shape arithmetic only; does not inspect device counts."""
    if type(capacity_bytes) is not int:
        raise TypeError("Scratch capacity must be a CPU integer byte budget")
    if min(q_len, kv_len, heads) <= 0 or not 0 <= capacity_bytes <= DEFAULT_CAPACITY_BYTES:
        raise ValueError("Positive shapes and scratch budget in [0,24GiB] required")
    # Original v2 overflow branch retains int32 contiguous Q/K/LSE offsets.
    # This prototype is bounded to the current sequence budget and D256.
    if max(q_len, kv_len) > 131072 or max(q_len, kv_len) * heads * 256 >= 2**31:
        raise ValueError("Prototype sequence/address range exceeds the validated int32 bound")
    rows, columns = triton.cdiv(q_len, 128), triton.cdiv(kv_len, 128)
    if rows * columns >= 2**31:
        raise ValueError("Edge lookup IDs exceed int32 range")
    capacity = min(rows * columns, capacity_bytes // (heads * 128 * 128 * 4))
    return rows, columns, capacity


def allocate_workspace(
    q: torch.Tensor,
    k: torch.Tensor,
    plan: base.QSAFusedPlan,
    capacity_bytes: int = DEFAULT_CAPACITY_BYTES,
) -> Workspace:
    rows, columns, capacity = workspace_shape(q.shape[0], k.shape[0], q.shape[1], capacity_bytes)
    if plan.kv_indices.shape != (rows, columns) or plan.kv_counts.numel() != rows:
        raise ValueError("Plan shape does not match classifier shape")
    return Workspace(
        torch.empty(capacity * q.shape[1] * 128 * 128, device=q.device, dtype=torch.float32),
        torch.empty((rows, columns), device=q.device, dtype=torch.int32),
        torch.empty(rows + 1, device=q.device, dtype=torch.int64),
        capacity,
        capacity_bytes,
    )


def prepare_lookup(plan: base.QSAFusedPlan, workspace: Workspace) -> dict[str, Any]:
    rows, columns = workspace.edge_map.shape
    prefix = ops._prefix_counts[(1,)](plan.kv_counts, workspace.prefix, rows, triton.next_power_of_2(rows))
    lookup = ops._edge_lookup[(rows,)](
        plan.kv_counts,
        plan.kv_indices,
        workspace.prefix,
        workspace.edge_map,
        columns,
        triton.next_power_of_2(columns),
    )
    return {"prefix": prefix, "lookup": lookup}


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: Any,
    plan: base.QSAFusedPlan,
    scale: float,
    config: Any,
    lse: torch.Tensor,
    do: torch.Tensor,
    delta: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    out: torch.Tensor,
    residual: torch.Tensor,
    *,
    capacity_bytes: int = DEFAULT_CAPACITY_BYTES,
) -> None:
    from relax.models.qwen3_8.ops.qsa_triton_v3 import launch_delta

    workspace = allocate_workspace(q, k, plan, capacity_bytes)
    prepare_lookup(plan, workspace)
    launch_delta(out, do, delta, residual)
    t, h, d = q.shape
    n, hk = k.shape[:2]
    metadata = (selection.positions.view(-1), selection.doc_ids.view(-1), plan.bits32, None)
    scratch = (workspace.ds, workspace.edge_map, workspace.prefix, workspace.capacity_edges)
    common = (t, n, plan.bits32.shape[1])
    for owner, kernels in (("dkv", (ops._shared_dkv,)), ("dq", (ops._fit_dq, ops._fallback_dq))):
        bm, bn, warps, stages = getattr(config, owner)
        tail = (h * d, d, hk * d, d, scale, h, hk, d, plan.ratio, 128, bm, bn, False)
        if owner == "dkv":
            args = (
                q,
                k,
                v,
                do,
                dk,
                dv,
                lse,
                delta,
                *metadata,
                plan.query_counts,
                plan.query_indices,
                *scratch,
                *common,
                plan.query_indices.shape[1],
                *tail,
            )
            grid = (triton.cdiv(n, bn), hk)
        else:
            args = (
                q,
                k,
                v,
                do,
                dq,
                lse,
                delta,
                *metadata,
                plan.kv_counts,
                plan.kv_indices,
                *scratch,
                *common,
                plan.kv_indices.shape[1],
                *tail,
            )
            grid = (triton.cdiv(t, bm), h)
        for kernel in kernels:
            kernel[grid](*args, num_warps=warps, num_stages=stages)
