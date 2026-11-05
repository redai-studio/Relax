# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in v3: shared FP32 dS and SM103 backward tiles, reusing v2 forward."""

from typing import Any

import torch

from relax.models.qwen3_8.ops import qsa_fused_attention as base
from relax.models.qwen3_8.ops import qsa_triton_v2 as v2


Config = v2.Config
DEFAULT = v2.DEFAULT
_SM103_BF16_D256 = Config(dq=(64, 32, 8, 1), dkv=(128, 64, 8, 1))
build_plan = v2.build_plan
supports = v2.supports
launch = v2.launch
launch_delta = v2.launch_delta


def _select_config(dim: int, capability: tuple[int, int], dtype: torch.dtype) -> Config:
    if capability == (10, 3) and dim == 256 and dtype == torch.bfloat16:
        return _SM103_BF16_D256
    return v2._select_config(dim, capability, dtype)


def _can_share_ds(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: base.QSAFusedPlan,
    query_ids: torch.Tensor | None,
    config: Config,
) -> bool:
    # Keep unvalidated devices, dimensions and CP-local queries on the original path.
    return (
        config == _SM103_BF16_D256
        and query_ids is None
        and plan.ratio == 4
        and q.is_cuda
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == 256
        and 0 < q.shape[0] <= 131072
        and 0 < k.shape[0] <= 131072
        and 0 < q.shape[1] * max(q.shape[0], k.shape[0]) * 256 < 2**31
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )


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
        selection, plan, query_ids, scale, config = ctx.args
        if _can_share_ds(q, k, v, plan, query_ids, config):
            from relax.models.qwen3_8.ops.qsa_triton_v3.shared_backward import backward

            backward(q, k, v, selection, plan, scale, config, lse, do, delta, dq, dk, dv, out, residual)
        else:
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
