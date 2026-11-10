# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Make legacy gated-residual norms visible to Megatron-FSDP hooks."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class _GatedResidualNorm(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        num_streams: int,
        eps: float,
        norm_impl: Callable[[Tensor, Tensor, int, float], Tensor],
    ) -> None:
        super().__init__()
        # Reuse the Parameter, including sequence_parallel and requires_grad.
        self.weight = weight
        self.num_streams = num_streams
        self.eps = eps
        self.norm_impl = norm_impl

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.norm_impl(hidden_states, self.weight, self.num_streams, self.eps)


def _fsdp_safe_mix(self: Any, hidden_states: Tensor) -> tuple[Tensor, Tensor | None]:
    """Keep MCore's gated-residual arithmetic; enter the norm via
    Module.__call__."""
    s, b, _ = hidden_states.shape
    n, C = self.n, self.hidden_size
    xn = self.hc_norm(hidden_states)
    down = F.silu(self.input_mix_weight_down(xn).float() / n)
    gate = torch.sigmoid(self.input_mix_weight_up(down.to(xn.dtype)).float())
    block_input = (gate.view(s, b, n, C) * xn.float().view(s, b, n, C)).mean(dim=2)
    block_input = block_input.to(hidden_states.dtype)
    if self.block_inject_weight is None:
        return block_input, None
    inject = 2.0 * torch.sigmoid(self.block_inject_weight(xn).float() / n)
    return block_input, inject


def patch_gated_residual_norms_for_fsdp(model: nn.Module) -> int:
    """Repair legacy modules before FSDP builds its buckets and hooks.

    MCore's legacy _mix reads hc_norm.weight without calling hc_norm. This
    bypasses the gather hook when the mixer is outside a TransformerLayer FSDP
    unit (notably decoder.final_layernorm). Keep the parameter tree and
    upstream norm arithmetic intact, but make the owning module callable.
    """
    module_path = "megatron.core.transformer.hyper_connection"
    try:
        hyper_connection = import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            return 0
        raise

    connection_type = getattr(hyper_connection, "GatedResidualHyperConnection", None)
    if connection_type is None:
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, connection_type):
            continue
        norm = module.hc_norm
        # Newer upstream implementations with a real norm need no adapter.
        if type(norm) is not nn.Module or not isinstance(getattr(norm, "weight", None), nn.Parameter):
            continue
        # Do not overwrite a downstream subclass's custom mixing algorithm.
        if getattr(module._mix, "__func__", None) is not connection_type._mix:
            continue
        module.hc_norm = _GatedResidualNorm(
            norm.weight, module.n, module.norm_eps, hyper_connection.gated_residual_group_rmsnorm
        )
        module.hc_norm.train(norm.training)
        module._mix = MethodType(_fsdp_safe_mix, module)
        patched += 1

    if patched:
        logger.info("Enabled FSDP-safe hc_norm forward for %d gated-residual module(s).", patched)
    return patched
