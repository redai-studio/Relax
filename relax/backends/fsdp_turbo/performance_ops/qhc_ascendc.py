# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Adapter for the specified CANN TextGatedResidual training wheel."""

from __future__ import annotations

from typing import Any

import torch


def text_gated_residual(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor | None,
    gamma: torch.Tensor,
    channels: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Preserve token layout and links to the original FSDP parameters.

    The wheel owns first-order autograd, missing-output-gradient handling and
    checkpoint cache allocation. Its operator applies ``1 + gamma`` internally.
    """
    from cann_ops_transformer_qhcofficial.ops.mhc.text_gated_residual.text_gated_residual import (
        text_gated_residual as native_forward,
    )

    if x.device.type not in ("npu", "privateuseone"):
        raise ValueError("AscendC QHC requires NPU input")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("AscendC QHC requires BF16 or FP16 input")
    if channels <= 0 or x.ndim < 2 or x.shape[-1] % channels:
        raise ValueError("AscendC QHC requires channels dividing the feature width")
    token_shape = x.shape[:-1]
    # Cast/contiguous outside the wheel's autograd Function so gradients return
    # to the original FSDP parameters and their original dtype. Never retain
    # materialized parameter copies across forwards.
    result = native_forward(
        x.reshape(1, -1, x.shape[-1]).contiguous(),
        w_down.contiguous(),
        w_up.contiguous(),
        w_gate=w_gate.contiguous() if w_gate is not None else None,
        channels=channels,
        gamma=gamma.float().contiguous(),
        eps=eps,
    )
    if w_gate is None:
        return result.reshape(*token_shape, x.shape[-1] // channels), None
    mixed, inject = result
    return (
        mixed.reshape(*token_shape, x.shape[-1] // channels),
        inject.reshape(*token_shape, channels),
    )


def qhc_forward(self: Any, hyper_input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Drop-in Qwen4ExpTextGatedResidual.forward; bind with MethodType."""
    if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
        raise ValueError("QHC input feature width does not match hc_count * hidden_size")
    gate = self.block_inject_weight
    mixed, inject = text_gated_residual(
        hyper_input,
        self.input_mix_weight_down.weight,
        self.input_mix_weight_up.weight,
        gate.weight if gate is not None else None,
        self.hc_norm.weight,
        self.hc_count,
        self.hc_norm.eps,
    )
    return (mixed, hyper_input, inject) if gate is not None else mixed
