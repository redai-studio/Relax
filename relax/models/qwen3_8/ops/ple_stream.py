# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in bounded-memory PLE for Qwen4Exp at CP=1 or with packed CP.

The four residual streams are independent inside PLE. Checkpoint sequence
chunks with a left convolution halo inside each stream, so backward replay does
not rebuild full-sequence FP32 gate, RMSNorm, and convolution tensors. The
original implementation remains the default.
"""

import math
from typing import Optional

import torch
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.transformer.hyper_connection import gated_residual_group_rmsnorm
from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
from torch.utils.checkpoint import checkpoint


_STREAM_CHUNK_ROWS = 8192


def _stream_conv(
    ple: PerLayerEmbedding,
    x: torch.Tensor,
    positions: torch.Tensor,
    stream: int,
    tp_halo: Optional[torch.Tensor],
) -> torch.Tensor:
    """The original PLE dilated depthwise convolution for one HC stream."""
    channels = ple.hidden_size
    start = stream * channels
    weight = ple.conv1d.weight[start : start + channels].squeeze(1).to(x.dtype)
    s_local = x.shape[0]
    x_ext = F.pad(x, (0, 0, 0, 0, ple.halo, 0)) if tp_halo is None else torch.cat((tp_halo, x), dim=0)
    result = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(ple.conv_kernel_size):
        offset = (ple.conv_kernel_size - 1 - tap) * ple.conv_dilation
        source = x_ext[ple.halo - offset : ple.halo - offset + s_local]
        valid = (positions >= offset).unsqueeze(-1)
        result = result + torch.where(valid, source.float(), 0.0) * weight[:, tap].float()
    return result.to(x.dtype)


def _stream_gated_values(
    ple: PerLayerEmbedding,
    hidden: torch.Tensor,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    stream: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = ple.hidden_size
    start = stream * channels
    stop = start + channels
    key = gated_residual_group_rmsnorm(key_raw, ple.norm_key.weight[start:stop], 1, ple.norm_eps)
    query = gated_residual_group_rmsnorm(hidden, ple.norm_query.weight[start:stop], 1, ple.norm_eps)
    gate = (key.float() * query.float()).sum(dim=-1, keepdim=True) / math.sqrt(channels)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = (torch.sigmoid(gate) * value.float()).to(hidden.dtype)
    normed = gated_residual_group_rmsnorm(gated_value, ple.norm_conv.weight[start:stop], 1, ple.norm_eps)
    return gated_value, normed


def _stream_forward(
    ple: PerLayerEmbedding,
    hidden: torch.Tensor,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    stream: int,
    tp_halo: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    gated_value, normed = _stream_gated_values(ple, hidden, key_raw, value, stream)
    conv_out = _stream_conv(ple, normed, positions, stream, tp_halo)
    return gated_value + F.silu(conv_out.float()).to(hidden.dtype)


def _tp_left_halo(
    ple: PerLayerEmbedding, hidden: torch.Tensor, key_raw: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    halo, channels = ple.halo, ple.hidden_size
    if hidden.shape[0] < halo:
        raise ValueError("PLE sequence-parallel chunk is shorter than its halo")
    hidden_tails = hidden[-halo:].split(channels, dim=-1)
    key_tails = key_raw[-halo:].split(channels, dim=-1)
    normed_tails = [
        _stream_gated_values(ple, h, k, value[-halo:], stream)[1]
        for stream, (h, k) in enumerate(zip(hidden_tails, key_tails, strict=True))
    ]
    tails = dist_nn.all_gather(torch.cat(normed_tails, dim=-1), group=ple.tp_group)
    rank = ple.tp_group.rank()
    # Rank zero must join the all-gather backward even without a predecessor.
    return tails[rank - 1] if rank else tails[-1] * 0.0


def _streamed_forward(self: PerLayerEmbedding, hidden_states: torch.Tensor) -> torch.Tensor:
    if self._ngram_ids is None:
        raise RuntimeError("PerLayerEmbedding.prepare() must be called before forward().")
    if any(parameter.requires_grad for parameter in self.parameters()):
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED requires all PLE parameters to be frozen")

    from megatron.core.ssm.mamba_context_parallel import reconstruct_tensor_cp, split_tensor_cp
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size > 1 and self._cp_packed_seq_params is None:
        raise RuntimeError("RELAX_QWEN38_PLE_STREAMED requires packed THD when CP>1")

    sp_enabled = self.config.sequence_parallel and self.tp_group.size() > 1
    sp_replicated = cp_size > 1 and sp_enabled
    if sp_replicated:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.tp_group)
    if cp_size > 1:
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
    tp_halo = None
    if cp_size == 1 and sp_enabled and self.halo:
        # Exchange only the normalized tail, once outside the chunk checkpoints.
        tp_halo = _tp_left_halo(self, hidden_states, key_raw, value)

    pieces = []
    for stream in range(self.n):
        start = stream * channels
        stop = start + channels
        hidden_stream = hidden_states[..., start:stop]
        key_stream = key_raw[..., start:stop]
        chunks = []
        for begin in range(0, max(sequence_length, 1), _STREAM_CHUNK_ROWS):
            end = min(begin + _STREAM_CHUNK_ROWS, sequence_length)
            left = max(0, begin - self.halo)
            trim = begin - left
            prefix = tp_halo[..., start:stop] if tp_halo is not None and begin < self.halo else None
            # Keep document-relative positions unchanged: the original mask
            # still prevents halo reads across packed document boundaries.
            # Bind trim/stream because checkpoint replay occurs after this loop.
            piece = checkpoint(
                lambda h, k, v, p, prefix, stream=stream, trim=trim: _stream_forward(self, h, k, v, p, stream, prefix)[
                    trim:
                ],
                hidden_stream[left:end],
                key_stream[left:end],
                value[left:end],
                positions[left:end],
                prefix,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            chunks.append(piece)
        pieces.append(torch.cat(chunks, dim=0))
        del chunks, piece
    result = torch.cat(pieces, dim=-1)
    if cp_size > 1:
        result = split_tensor_cp(result, self._cp_packed_seq_params, dim=0)
    if sp_replicated:
        # Keep each TP replica's loss local; the input gather already sums gradients.
        result = result.chunk(self.tp_group.size(), dim=0)[self.tp_group.rank()].contiguous()
    return result
