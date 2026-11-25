# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Static packed PLE routing for token-local projections and depthwise
convolution."""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


if TYPE_CHECKING:
    from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding


@dataclass(frozen=True)
class ShardedPLERoute:
    bounds: tuple[int, ...]
    dilation: int
    cp_rows: torch.Tensor  # CP-local tokens, in the order required by the TP vocabulary lookup.
    taps: torch.Tensor  # [kernel, local sequence], indices into [local rows, gathered halos, zero].
    send_rows: torch.Tensor
    capacity: int  # Same padded boundary count on every rank, including ranks with nothing to send.
    group: dist.ProcessGroup


@dataclass(frozen=True)
class ShardedPLEState:
    local_ngram_ids: torch.Tensor
    route: ShardedPLERoute


@lru_cache(maxsize=1)
def _group_coordinates(
    group: dist.ProcessGroup, cp: dist.ProcessGroup, tp: dist.ProcessGroup, device: torch.device
) -> tuple[tuple[int, int], ...]:
    # Resolve NCCL rank order once per group, when preparing the first microbatch.
    here = torch.tensor([cp.rank(), tp.rank()], dtype=torch.int32, device=device)
    peers = torch.empty(2 * group.size(), dtype=here.dtype, device=device)
    dist.all_gather_into_tensor(peers, here, group=group)
    return tuple(map(tuple, peers.reshape(-1, 2).cpu().tolist()))


def _build_route(
    bounds: tuple[int, ...], coords: tuple[tuple[int, int], ...], rank: int, kernel: int, dilation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cp_size = max(cp for cp, _ in coords) + 1
    tp_size = max(tp for _, tp in coords) + 1
    if len(set(coords)) != len(coords) or len(coords) != cp_size * tp_size:
        raise ValueError("PLE needs a complete TP x CP group")
    cu = np.asarray(bounds, dtype=np.int64)
    lengths = np.diff(cu)
    if (
        not lengths.size
        or cu[0] != 0
        or cu[-1] <= 0
        or cu[-1] % len(coords)
        or np.any(lengths < 0)
        or np.any(lengths % (2 * cp_size))
        or kernel < 1
        or dilation < 1
    ):
        raise ValueError("Invalid physical THD layout or PLE convolution geometry")

    # Each document is divided into 2*CP pieces. CP rank c owns c and its mirror;
    # SP splits that concatenated per-CP stream into TP equally sized chunks.
    global_rows = np.arange(cu[-1])
    docs = np.searchsorted(cu[1:], global_rows, side="right")
    start = cu[docs]
    width = lengths[docs] // (2 * cp_size)
    piece, offset = np.divmod(global_rows - start, width)
    cp = np.minimum(piece, 2 * cp_size - piece - 1)
    cp_offset = start // cp_size + (piece >= cp_size) * width + offset
    local_size = len(global_rows) // len(coords)
    tp, local_offset = np.divmod(cp_offset, local_size)
    ranks = np.empty((cp_size, tp_size), dtype=np.int64)
    ranks[tuple(np.asarray(coords).T)] = np.arange(len(coords))
    owner = ranks[cp, tp]
    owned = global_rows[owner == rank]
    cp_rows = global_rows[cp == coords[rank][0]]

    # Mark a source only if a valid tap reads it from another rank. This also
    # handles a short piece whose dilated taps cross more than one boundary.
    shifts = dilation * np.arange(kernel - 1, -1, -1)
    boundary = np.zeros(len(global_rows), dtype=bool)
    for shift in shifts:
        if 0 < shift < len(global_rows):
            boundary[:-shift] |= (docs[:-shift] == docs[shift:]) & (owner[:-shift] != owner[shift:])
    counts = np.bincount(owner[boundary], minlength=len(coords))
    capacity = int(counts.max())
    slots = np.full(len(global_rows), -1, dtype=np.int64)
    for peer, count in enumerate(counts):
        slots[boundary & (owner == peer)] = local_size + peer * capacity + np.arange(count)
    slots[owned] = local_offset[owned]
    send_rows = local_offset[boundary & (owner == rank)]
    sources = owned[None, :] - shifts[:, None]
    taps = np.where(sources >= start[owned], slots[sources.clip(min=0)], local_size + len(coords) * capacity)
    return cp_rows, taps, send_rows, capacity


def prepare_sharded_ple(
    layer: "PerLayerEmbedding", cu_seqlens: tuple[int, ...], route: ShardedPLERoute | None = None
) -> ShardedPLEState:
    ids = layer._ngram_ids
    if ids.shape[0] != 1 or ids.shape[1] != cu_seqlens[-1]:
        raise ValueError("PLE hash rows do not match the packed physical boundaries")
    if (
        route is None
        or route.bounds != cu_seqlens
        or route.taps.shape[0] != layer.conv_kernel_size
        or route.dilation != layer.conv_dilation
    ):
        from megatron.core import parallel_state

        group = parallel_state.get_tensor_and_context_parallel_group()
        coords = _group_coordinates(group, parallel_state.get_context_parallel_group(), layer.tp_group, ids.device)
        cp_rows, taps, send, capacity = _build_route(
            cu_seqlens, coords, group.rank(), layer.conv_kernel_size, layer.conv_dilation
        )
        route = ShardedPLERoute(
            cu_seqlens,
            layer.conv_dilation,
            torch.as_tensor(cp_rows, device=ids.device),
            torch.as_tensor(taps, device=ids.device),
            torch.as_tensor(send, device=ids.device),
            capacity,
            group,
        )
    # The layout can be shared by PLE layers in one microbatch; hash values are layer-specific.
    return ShardedPLEState(ids.index_select(1, route.cp_rows), route)


def _silu_residual(conv: torch.Tensor, gated: torch.Tensor) -> torch.Tensor:
    return gated + F.silu(conv.to(gated.dtype).float()).to(gated.dtype)


@lru_cache(maxsize=1)
def _compiled_silu_residual() -> Callable:
    # Match the native compiled PLE convolution's SiLU backward at BF16 rounding boundaries.
    return torch.compile(
        _silu_residual,
        dynamic=True,
        fullgraph=True,
        options={"emulate_precision_casts": True, "use_joint_graph_passes": False, "shape_padding": False},
    )


def _convolve(
    normed: torch.Tensor, halo: torch.Tensor, gated: torch.Tensor, weight: torch.Tensor, taps: torch.Tensor
) -> torch.Tensor:
    rows = torch.cat((normed, halo, torch.zeros_like(normed[:1])))
    weight = weight.squeeze(1).to(normed.dtype)
    conv = torch.zeros_like(normed, dtype=torch.float32)
    for tap in range(weight.shape[1]):
        source = rows.index_select(0, taps[tap])
        conv = conv + source.float() * weight[:, tap].float()
    silu_residual = _compiled_silu_residual() if conv.is_cuda else _silu_residual
    return silu_residual(conv, gated)


def sharded_convolution(
    state: ShardedPLEState, normed: torch.Tensor, gated: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    route = state.route
    if normed.shape[0] != route.taps.shape[1] or normed.shape != gated.shape:
        raise ValueError("PLE window and local SP activations have different shapes")
    halo = normed[:0]
    if route.capacity:
        outgoing = normed.index_select(0, route.send_rows)
        outgoing = F.pad(outgoing, (0, 0, 0, 0, 0, route.capacity - outgoing.shape[0]))
        halo = gather_from_sequence_parallel_region(outgoing, tensor_parallel_output_grad=True, group=route.group)
    # The gathered tensor participates in autograd even on a rank with no incoming
    # boundaries. All peers must enter its reduce-scatter in backward.
    return _convolve(normed, halo, gated, weight, route.taps)


def sharded_forward(layer: "PerLayerEmbedding", hidden: torch.Tensor, state: ShardedPLEState) -> torch.Tensor:
    from .ple_fused_conv import _gate_for_training

    if hidden.dtype != torch.bfloat16:
        raise ValueError("Sharded PLE requires BF16 hidden states")
    embeddings = layer.ple_embedding(state.local_ngram_ids).to(hidden.dtype)
    if embeddings.shape[:2] != hidden.shape[:2]:
        raise ValueError("PLE embedding and hidden states must own the same local token rows")
    gated, normed = _gate_for_training(
        layer.key_proj(embeddings),
        hidden,
        layer.value_proj(embeddings),
        layer.norm_key.weight,
        layer.norm_query.weight,
        layer.norm_conv.weight,
        layer.n,
        layer.norm_eps,
    )
    return sharded_convolution(state, normed, gated, layer.conv1d.weight)
