# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in bounded-memory PLE forward for packed Qwen4Exp context parallelism.

The four residual streams are independent inside PLE. Checkpointing each stream
keeps its FP32 gate, RMSNorm, and convolution temporaries out of the other
streams' forward activations. The original implementation remains the default.
"""

import math

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm
from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
from torch.utils.checkpoint import checkpoint


def _stream_conv(ple: PerLayerEmbedding, x: torch.Tensor, positions: torch.Tensor, stream: int) -> torch.Tensor:
    """The original PLE dilated depthwise convolution for one HC stream."""
    channels = ple.hidden_size
    start = stream * channels
    weight = ple.conv1d.weight[start : start + channels].squeeze(1).to(x.dtype)
    s_local = x.shape[0]
    x_ext = F.pad(x, (0, 0, 0, 0, ple.halo, 0))
    result = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(ple.conv_kernel_size):
        offset = (ple.conv_kernel_size - 1 - tap) * ple.conv_dilation
        source = x_ext[ple.halo - offset : ple.halo - offset + s_local]
        valid = (positions >= offset).unsqueeze(-1)
        result = result + torch.where(valid, source.float(), 0.0) * weight[:, tap].float()
    return result.to(x.dtype)


def _stream_forward(
    ple: PerLayerEmbedding,
    hidden: torch.Tensor,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    stream: int,
) -> torch.Tensor:
    channels = ple.hidden_size
    start = stream * channels
    stop = start + channels
    key = gated_residual_group_rmsnorm(key_raw, ple.norm_key.weight[start:stop], 1, ple.norm_eps)
    query = gated_residual_group_rmsnorm(hidden, ple.norm_query.weight[start:stop], 1, ple.norm_eps)
    gate = (key.float() * query.float()).sum(dim=-1, keepdim=True) / math.sqrt(channels)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = (torch.sigmoid(gate) * value.float()).to(hidden.dtype)
    normed = gated_residual_group_rmsnorm(gated_value, ple.norm_conv.weight[start:stop], 1, ple.norm_eps)
    conv_out = _stream_conv(ple, normed, positions, stream)
    return gated_value + F.silu(conv_out.float()).to(hidden.dtype)


def _streamed_forward(self: PerLayerEmbedding, hidden_states: torch.Tensor) -> torch.Tensor:
    if self._ngram_ids is None:
        raise RuntimeError("PerLayerEmbedding.prepare() must be called before forward().")
    if any(parameter.requires_grad for parameter in self.parameters()):
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED requires all PLE parameters to be frozen")

    from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1 or self._cp_packed_seq_params is None:
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED currently requires packed THD with CP>1")

    sp_replicated = self.config.sequence_parallel and self.tp_group.size() > 1
    if sp_replicated:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.tp_group)
    hidden_states = reconstruct_tensor_cp(hidden_states, self._cp_packed_seq_params, dim=0, differentiable=True)
    sequence_length, batch_size, hc_hidden_size = hidden_states.shape
    channels = self.hidden_size
    if hc_hidden_size != self.n * channels:
        raise RuntimeError("PLE hidden stream width mismatch")

    embeddings = self.ple_embedding(self._ngram_ids).to(hidden_states.dtype)
    if sp_replicated:
        embeddings = gather_from_sequence_parallel_region(embeddings, group=self.tp_group)
    if embeddings.shape[0] != sequence_length:
        raise RuntimeError("PLE embeddings and hidden sequence lengths differ")
    key_raw = self.key_proj(embeddings)
    value = self.value_proj(embeddings)
    positions = self._local_positions(sequence_length)

    pieces = []
    for stream in range(self.n):
        start = stream * channels
        stop = start + channels
        hidden_stream = hidden_states[..., start:stop]
        key_stream = key_raw[..., start:stop]
        piece = checkpoint(
            lambda h, k, v, p, stream=stream: _stream_forward(self, h, k, v, p, stream),
            hidden_stream,
            key_stream,
            value,
            positions,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        pieces.append(piece)
    result = torch.cat(pieces, dim=-1)
    result = split_tensor_cp(result, self._cp_packed_seq_params, dim=0)
    if sp_replicated:
        # Keep each TP replica's loss local; the input gather already sums gradients.
        result = result.chunk(self.tp_group.size(), dim=0)[self.tp_group.rank()].contiguous()
    return result
