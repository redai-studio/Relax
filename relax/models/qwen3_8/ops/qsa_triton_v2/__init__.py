# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental deterministic R4 bitset attention backend.

Uses rolled loops, selection-word broadcasting and KV-major dKV intermediates.
Reuses the original 128-token plan. Each pass has an independent (query tile,
key tile, warps, stages) configuration. No runtime autotuning, global
monkeypatching, atomics, or selector changes are involved.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
import triton

from relax.models.qwen3_8.ops import qsa_fused_attention as base
from relax.models.qwen3_8.ops.qsa_triton_v2 import kernels as ops


@dataclass(frozen=True)
class Config:
    # Measured D256 candidates; configurable explicitly for offline experiments.
    fwd: tuple[int, int, int, int] = (64, 32, 4, 1)
    dq: tuple[int, int, int, int] = (128, 32, 8, 1)
    dkv: tuple[int, int, int, int] = (128, 32, 8, 1)


DEFAULT = Config()
_SM103_BF16_D256 = Config(dq=(64, 32, 8, 1))


def _select_config(dim: int, capability: tuple[int, int], dtype: torch.dtype) -> Config:
    # SM103: D256 high/low dS accumulation needs 592 TMEM columns at BM128,
    # above the 512-column limit. BM64 was validated at 512 columns.
    if capability == (10, 3) and dim == 256 and dtype == torch.bfloat16:
        return _SM103_BF16_D256
    return DEFAULT


# Identical frozen-selection plans are shared with the original backend.
build_plan = base.build_plan
supports = base.supports


def launch(
    pass_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: Any,
    plan: base.QSAFusedPlan,
    query_ids: torch.Tensor | None,
    scale: float,
    config: Config,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    do: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    implementation: ModuleType = ops,
    out_residual: torch.Tensor | None = None,
) -> Any:
    """Launch into supplied buffers; return compiled metadata for
    experiments."""
    bm, bn, warps, stages = getattr(config, pass_name)
    if bm not in (16, 32, 64, 128) or bn not in (16, 32, 64, 128):
        raise ValueError("Tiles must divide the 128-token classifier block")
    t, h, d = q.shape
    n, hk = k.shape[:2]
    metadata = (selection.positions.view(-1), selection.doc_ids.view(-1), plan.bits32, query_ids)
    common = (t, n, plan.bits32.shape[1])
    tail = (h * d, d, hk * d, d, scale, h, hk, d, plan.ratio, base._BLOCK, bm, bn, query_ids is not None)
    kwargs = dict(num_warps=warps, num_stages=stages)
    if pass_name == "fwd":
        if out_residual is not None:
            kwargs["OutResidual"] = out_residual
        return implementation._qsa_fused_fwd[(triton.cdiv(t, bm), h)](
            q,
            k,
            v,
            out,
            lse,
            *metadata,
            plan.kv_counts,
            plan.kv_indices,
            *common,
            plan.kv_indices.shape[1],
            *tail,
            **kwargs,
        )
    if pass_name == "dq":
        return implementation._qsa_fused_bwd_dq[(triton.cdiv(t, bm), h)](
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
            *common,
            plan.kv_indices.shape[1],
            *tail,
            **kwargs,
        )
    if pass_name == "dkv":
        return implementation._qsa_fused_bwd_dkv[(triton.cdiv(n, bn), hk)](
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
            *common,
            plan.query_indices.shape[1],
            *tail,
            **kwargs,
        )
    raise ValueError(pass_name)


def launch_delta(
    out: torch.Tensor, do: torch.Tensor, delta: torch.Tensor, residual: torch.Tensor | None = None
) -> Any:
    t, h, d = out.shape
    if residual is not None:
        return ops._qsa_fused_delta_residual[(triton.cdiv(t, 32), h)](
            out, residual, do, delta, t, h, d, 32, num_warps=4
        )
    return base._qsa_fused_delta[(triton.cdiv(t, 32), h)](out, do, delta, t, h * d, d, h, d, 32, num_warps=4)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selection: Any,
        plan: base.QSAFusedPlan,
        query_ids: torch.Tensor | None,
        scale: float,
        config: Config,
        save_residual: bool,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
        residual = torch.empty_like(q) if save_residual else None
        launch("fwd", q, k, v, selection, plan, query_ids, scale, config, out=out, lse=lse, out_residual=residual)
        ctx.save_for_backward(q, k, v, out, residual, lse)
        ctx.args = selection, plan, query_ids, scale, config
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, out, residual, lse = ctx.saved_tensors
        assert residual is not None
        do = do.contiguous()
        delta = torch.empty_like(lse)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        launch_delta(out, do, delta, residual)
        args = (q, k, v, *ctx.args)
        launch("dq", *args, lse=lse, do=do, delta=delta, dq=dq)
        launch("dkv", *args, lse=lse, do=do, delta=delta, dk=dk, dv=dv)
        return dq, dk, dv, None, None, None, None, None, None


def qsa_fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selection: Any,
    plan: base.QSAFusedPlan,
    query_ids: torch.Tensor | None,
    scale: float,
    config: Config | None = None,
) -> torch.Tensor:
    if query.shape[0] != plan.q_len:
        raise ValueError("Plan/query length mismatch")
    if key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("Inconsistent Q/K/V shapes")
    if query.shape[1] % key.shape[1]:
        raise ValueError("Q heads must be a multiple of KV heads")
    if config is None:
        config = _select_config(query.shape[-1], torch.cuda.get_device_capability(query.device), query.dtype)
    return _Attention.apply(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        selection,
        plan,
        query_ids,
        scale,
        config,
        torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)),
    )
