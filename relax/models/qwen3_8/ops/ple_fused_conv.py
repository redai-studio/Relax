# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# SPDX-License-Identifier: Apache-2.0
# Adapted for native PLE parameters, packing and compiled rounding.
"""Compile PLE convolution, activation and add; keep TP halo exchange outside
the graph."""

import math
from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn.functional as F

from relax.utils.env import Envs


def convolution_values(
    extended: torch.Tensor,
    normed: torch.Tensor,
    gated: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Preserve per-tap FP32 accumulation and both BF16 roundings before the
    add."""
    kernel = weight.shape[-1]
    halo = (kernel - 1) * dilation
    result = torch.zeros_like(normed, dtype=torch.float32)
    weight = weight.squeeze(1).to(normed.dtype)
    for tap in range(kernel):
        offset = (kernel - 1 - tap) * dilation
        source = extended[halo - offset : halo - offset + normed.shape[0]]
        result = (
            result + torch.where((positions >= offset).unsqueeze(-1), source.float(), 0.0) * weight[:, tap].float()
        )
    return gated + F.silu(result.to(normed.dtype).float()).to(gated.dtype)


@lru_cache(maxsize=1)
def _compiled_conv() -> Callable:
    return torch.compile(
        convolution_values,
        dynamic=True,
        fullgraph=True,
        options={
            "emulate_precision_casts": True,
            # Torch 2.11 joint_graph.pointless_convert otherwise deletes the
            # required FP32 -> BF16 -> FP32 rounding before SiLU.
            "use_joint_graph_passes": False,
            "shape_padding": False,
            "unroll_reductions_threshold": 0,
        },
    )


def supported(self: Any, hidden: torch.Tensor) -> bool:
    from megatron.core import parallel_state

    return not (
        Envs.RELAX_QWEN38_PLE_STREAMED
        or not hidden.is_cuda
        or hidden.dtype != torch.bfloat16
        or parallel_state.get_context_parallel_world_size() > 1
        or getattr(self, "_cp_packed_seq_params", None) is not None
    )


def fused_forward(self: Any, hidden: torch.Tensor) -> torch.Tensor:

    if self._ngram_ids is None:
        raise RuntimeError("PerLayerEmbedding.prepare() must be called before forward().")
    from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm as norm

    seq, batch, width = hidden.shape
    n, channels = self.n, self.hidden_size
    embeddings = self.ple_embedding(self._ngram_ids).to(hidden.dtype)
    if embeddings.shape[0] != seq:
        raise ValueError("PLE embedding rows do not match the local hidden sequence")
    key = norm(self.key_proj(embeddings), self.norm_key.weight, n, self.norm_eps)
    value = self.value_proj(embeddings)
    query = norm(hidden, self.norm_query.weight, n, self.norm_eps)
    gate = (key.float().view(seq, batch, n, channels) * query.float().view(seq, batch, n, channels)).sum(
        -1, keepdim=True
    ) / math.sqrt(channels)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (gate.sigmoid() * value.float().unsqueeze(-2)).to(hidden.dtype).view(seq, batch, width)
    normed = norm(gated, self.norm_conv.weight, n, self.norm_eps)
    positions = self._local_positions(seq)
    if self.halo and self.config.sequence_parallel and self.tp_group.size() > 1:
        from torch.distributed.nn.functional import all_gather

        if seq < self.halo:
            raise ValueError("PLE sequence-parallel chunk is shorter than its halo")
        tails = all_gather(normed[-self.halo :].contiguous(), group=self.tp_group)
        rank = self.tp_group.rank()
        # Rank zero must participate in the all-gather backward as well.
        previous = tails[rank - 1] if rank else tails[-1] * 0.0
        extended = torch.cat((previous, normed), dim=0)
    else:
        extended = F.pad(normed, (0, 0, 0, 0, self.halo, 0))
    return _compiled_conv()(extended, normed, gated, self.conv1d.weight, positions, self.conv_dilation)
