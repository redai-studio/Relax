# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Default-on compilation of Qwen3.8 gated-residual HC reads and write gates.

Only the tensor math is compiled. Parameter publication stays outside the
graph; parameter objects, residual writes and checkpoint policy stay native.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn.functional as F


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
        fullgraph=True,
        options=options,
    )


def fused_mix(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    from megatron.core import utils as mcore_utils

    ensure_ready = getattr(mcore_utils, "ensure_params_ready", None)
    weights = [self.hc_norm.weight, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight]
    inject = self.block_inject_weight.weight if self.block_inject_weight is not None else None
    if inject is not None:
        weights.append(inject)
    if ensure_ready is not None:
        ensure_ready(weights)
    fixed_width = (
        hidden_states.shape[1:] == (1, 10240)
        and weights[0].shape == (10240,)
        and weights[1].shape == (320, 10240)
        and weights[2].shape == (10240, 320)
        and (inject is None or inject.shape == (4, 10240))
    )
    fn = compiled_mix(fixed_width)
    return fn(hidden_states, *weights[:3], inject, self.n, self.norm_eps)


def supported(hc: Any, x: torch.Tensor) -> bool:
    from megatron.core import utils as mcore_utils

    return _supported(hc, x, getattr(mcore_utils, "ensure_params_ready", None))
