# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Default-on compilation of Qwen3.8 gated-residual HC reads and write gates.

Only the tensor math is compiled. Parameter publication stays outside the
graph; optional inner recomputation bounds the saved HC activations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def gated_residual_mix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Match MCore's streamwise norm, BF16 projections and FP32 gates."""
    streams = x.float().unflatten(-1, (n, -1))
    normed = streams * torch.rsqrt(streams.pow(2).mean(-1, keepdim=True) + eps)
    xn = (normed.flatten(-2) * (1.0 + norm_weight.float())).to(x.dtype)
    down = F.silu(F.linear(xn, down_weight).float() / n)
    gate = torch.sigmoid(F.linear(down.to(xn.dtype), up_weight).float())
    block_input = (gate.unflatten(-1, (n, -1)) * xn.float().unflatten(-1, (n, -1))).mean(-2).to(x.dtype)
    inject = None if inject_weight is None else 2.0 * torch.sigmoid(F.linear(xn, inject_weight).float() / n)
    return block_input, inject


class _SGLangGroupedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, n: int, eps: float) -> torch.Tensor:
        from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm

        ctx.save_for_backward(x, weight)
        ctx.n, ctx.eps = n, eps
        return grouped_gemma_rmsnorm(x, weight, x.shape[-1] // n, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        x, weight = ctx.saved_tensors
        streams = x.float().unflatten(-1, (ctx.n, -1))
        rstd = torch.rsqrt(streams.square().mean(-1, keepdim=True) + ctx.eps)
        normalized = streams * rstd
        grad = grad.float().unflatten(-1, (ctx.n, -1))
        scaled = grad * (1.0 + weight.float().unflatten(-1, (ctx.n, -1)))
        dx = rstd * (scaled - normalized * (scaled * normalized).mean(-1, keepdim=True))
        dw = (grad * normalized).reshape(-1, weight.numel()).sum(0)
        return dx.reshape_as(x).to(x.dtype), dw.to(weight.dtype), None, None


class _BatchInvariantLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mm_batch_invariant

        ctx.save_for_backward(x, weight)
        return mm_batch_invariant(x, weight.t())

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight = ctx.saved_tensors
        return grad.mm(weight), grad.t().mm(x)


class _BatchInvariantStreamMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim

        ctx.streams = x.shape[-2]
        return mean_dim(x, -2)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor]:
        return ((grad.unsqueeze(-2) / ctx.streams).expand(-1, ctx.streams, -1),)


def _sglang_mix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # SGLang's deterministic eager HC rounds at each BF16 operator boundary.
    xn = _SGLangGroupedRMSNorm.apply(x.flatten(0, 1).contiguous(), norm_weight, n, eps)
    down = F.silu(_BatchInvariantLinear.apply(xn, down_weight) / n)
    gate = torch.sigmoid(_BatchInvariantLinear.apply(down, up_weight))
    product = (gate * xn).unflatten(-1, (n, -1))
    block = _BatchInvariantStreamMean.apply(product.float()).to(x.dtype).reshape(*x.shape[:2], -1)
    inject = None
    if inject_weight is not None:
        inject = (2 * torch.sigmoid(_BatchInvariantLinear.apply(xn, inject_weight) / n)).float()
        inject = inject.reshape(*x.shape[:2], n)
    return block, inject


def _qwen38_mix(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor | None,
    n: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Keep token count dynamic while exposing the production GEMM dimensions
    # to Inductor. Fully symbolic weight dimensions generate slower kernels.
    assert x.ndim == 3 and x.shape[1:] == (1, 10240)
    assert norm_weight.shape == (10240,)
    assert down_weight.shape == (320, 10240) and up_weight.shape == (10240, 320)
    assert inject_weight is None or inject_weight.shape == (4, 10240)
    assert n == 4
    return gated_residual_mix(x, norm_weight, down_weight, up_weight, inject_weight, n, eps)


def _supported(hc: Any, x: torch.Tensor, ensure_ready: Callable | None) -> bool:
    if x.device.type != "cuda" or x.dtype != torch.bfloat16 or x.ndim != 3 or hc.n != 4:
        return False
    # Functional linear must not bypass adapters or hooks that transform its
    # inputs/outputs. Native DDP publication is handled explicitly below.
    from torch.nn.modules import module as module_hooks

    if any(
        getattr(module_hooks, name)
        for name in (
            "_global_forward_pre_hooks",
            "_global_forward_hooks",
            "_global_backward_pre_hooks",
            "_global_backward_hooks",
        )
    ):
        return False
    for linear in (hc.input_mix_weight_down, hc.input_mix_weight_up, hc.block_inject_weight):
        if linear is None:
            continue
        if type(linear) is not torch.nn.Linear or linear.bias is not None:
            return False
        if linear.weight.dtype != x.dtype or linear.weight.device != x.device:
            return False
        if linear._forward_hooks or linear._backward_hooks or linear._backward_pre_hooks:
            return False
        for hook in linear._forward_pre_hooks.values():
            if (
                ensure_ready is None
                or getattr(hook, "__module__", "") != "megatron.core.distributed.distributed_data_parallel"
                or getattr(hook, "__qualname__", "") != "DistributedDataParallel._make_forward_pre_hook.<locals>.hook"
            ):
                return False
    return True


@lru_cache(maxsize=2)
def compiled_mix(fixed_width: bool) -> Callable:
    # Other models and CPU fallbacks need not initialize Inductor at all.
    options = {"emulate_precision_casts": True}
    if fixed_width:
        options.update(shape_padding=False, unroll_reductions_threshold=0)
    return torch.compile(
        _qwen38_mix if fixed_width else gated_residual_mix,
        dynamic=True,
        # The fused graph has several valid training, replay and output-mixer
        # signatures. Dynamo can run eager for unseen signatures at its cache limit.
        fullgraph=False,
        options=options,
    )


def fused_mix(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    from megatron.core import utils as mcore_utils

    ensure_ready = getattr(mcore_utils, "ensure_params_ready", None)
    weights = [self.hc_norm.weight, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight]
    if self.block_inject_weight is not None:
        weights.append(self.block_inject_weight.weight)
    if ensure_ready is not None:
        ensure_ready(weights)
    # Inductor assumes static parameter inputs are 16-byte aligned. A sharded
    # backend can republish the same Parameter at a different flat-buffer offset.
    # Clone the affected views after publication so gradients reach their owners.
    # Give Dynamo ordinary tensor inputs for every layout. During the initial
    # checkpoint forward, detach all of them so layout-dependent clones do not
    # introduce additional requires_grad signatures.
    if torch.is_grad_enabled():
        weights = [weight.view_as(weight) for weight in weights]
    else:
        weights = [weight.detach() for weight in weights]
    weights = [weight.clone() if weight.data_ptr() % 16 else weight for weight in weights]
    inject = weights[3] if len(weights) == 4 else None
    if os.environ.get("RELAX_QWEN38_HC_BITWISE") == "1":
        return _sglang_mix(hidden_states, *weights[:3], inject, self.n, self.norm_eps)
    fixed_width = (
        hidden_states.shape[1:] == (1, 10240)
        and weights[0].shape == (10240,)
        and weights[1].shape == (320, 10240)
        and weights[2].shape == (10240, 320)
        and (inject is None or inject.shape == (4, 10240))
    )
    fn = compiled_mix(fixed_width)
    if os.environ.get("RELAX_QWEN38_HC_RECOMPUTE", "0") == "1" and torch.is_grad_enabled():
        from megatron.core.tensor_parallel.random import is_checkpointing
        from torch.utils.checkpoint import checkpoint

        max_local_rows = int(os.environ.get("RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS", "0"))
        # Small decoder/MTP inputs covered by the outer uniform checkpoint
        # can retain intermediates during replay to avoid another recomputation.
        # Heads and longer inputs keep the inner non-reentrant checkpoint.
        outer_covers_mix = (
            fixed_width
            and inject is not None
            and self.config.recompute_granularity == "full"
            and self.config.recompute_method == "uniform"
            and is_checkpointing()
            and hidden_states.shape[0] <= max_local_rows
        )
        if not outer_covers_mix:
            return checkpoint(fn, hidden_states, *weights[:3], inject, self.n, self.norm_eps, use_reentrant=False)
    return fn(hidden_states, *weights[:3], inject, self.n, self.norm_eps)


def supported(hc: Any, x: torch.Tensor) -> bool:
    from megatron.core import utils as mcore_utils

    return _supported(hc, x, getattr(mcore_utils, "ensure_params_ready", None))
