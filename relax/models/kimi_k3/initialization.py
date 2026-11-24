# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Initialize Bridge's empty KDA decay tensors using FLA's safe-gate rule."""

import math

import torch


@torch.no_grad()
def initialize_kda_decay(
    a_log: torch.Tensor, dt_bias: torch.Tensor, *, seed: int, layer: int, tp_rank: int, tp_size: int
) -> None:
    """Generate a global per-layer state before selecting the actual TP slice.

    FLA 0.5.2 KDAPreTrainedModel initializes safe-gate A_log to zero and
    inverse-softplus dt from log-uniform [0.001, 0.1]. Preserve FP32 storage
    and model RNG state; data/context replicas use the same global sample.
    """
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("KDA decay state must remain FP32 during initialization")
    generator = torch.Generator(device="cpu").manual_seed(seed + 104729 * layer)
    dt = torch.rand(dt_bias.numel() * tp_size, dtype=torch.float32, generator=generator)
    dt = (dt * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).exp().clamp_(min=1e-4)
    bias = dt + torch.log(-torch.expm1(-dt))
    dt_bias.copy_(bias.narrow(0, tp_rank * dt_bias.numel(), dt_bias.numel()))
    a_log.zero_()
