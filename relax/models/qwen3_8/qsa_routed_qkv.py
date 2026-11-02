# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Gated-QSA TP routing; explicit process groups and opt-in execution."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist


@lru_cache(maxsize=2)
def route_table(tp: int) -> tuple:
    """[source][consumer] -> (source column, output column, width)
    intersections."""
    if tp not in (4, 8):
        raise ValueError("Only TP4/8, Hq24, Hkv2, D256 gated projection supported")
    shard, per_group, local_q = 13312 // tp, tp // 2, 6144 // tp
    table = []
    for source in range(tp):
        destinations = []
        start, end = source * shard, (source + 1) * shard
        for consumer in range(tp):
            group, head_rank = divmod(consumer, per_group)
            base = group * 6656
            intervals = (
                (base + head_rank * local_q, local_q),
                (base + 3072 + head_rank * local_q, local_q),
                (base + 6144, 256),
                (base + 6400, 256),
            )
            pieces, output_start = [], 0
            for first, width in intervals:
                left, right = max(first, start), min(first + width, end)
                if left < right:
                    pieces.append((left - start, output_start + left - first, right - left))
                output_start += width
            destinations.append(tuple(pieces))
        table.append(tuple(destinations))
    return tuple(table)


def pack(matrix: torch.Tensor, routes: tuple, *, inverse: bool) -> torch.Tensor:
    rows = matrix.shape[0]
    result = matrix.new_empty(rows * sum(piece[2] for route in routes for piece in route))
    offset = 0
    for route in routes:
        for source, destination, width in route:
            column = destination if inverse else source
            result[offset : offset + rows * width].view(width, rows).copy_(
                matrix[:, column : column + width].transpose(0, 1)
            )
            offset += rows * width
    return result


def unpack(buffer: torch.Tensor, routes: tuple, rows: int, width: int, *, inverse: bool) -> torch.Tensor:
    result = buffer.new_zeros((rows, width))
    offset = 0
    for group in routes:
        for source, destination, count in group:
            values = buffer[offset : offset + rows * count].view(count, rows).transpose(0, 1)
            column = source if inverse else destination
            if inverse:
                # Rank-ordered SUM for shared KV; unique Q/gate intervals are added once.
                result[:, column : column + count].add_(values)
            else:
                result[:, column : column + count].copy_(values)
            offset += rows * count
    if offset != buffer.numel():
        raise RuntimeError("Route payload size mismatch")
    return result


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, projection: torch.Tensor, group: Any) -> torch.Tensor:
        if group is None:
            raise ValueError("An existing explicit TP process group is required")
        tp, rank = dist.get_world_size(group), dist.get_rank(group)
        table = route_table(tp)
        if projection.ndim != 3 or projection.shape[-1] != 13312 // tp:
            raise ValueError("Unexpected gated QKV projection shape")
        shape = tuple(projection.shape)
        rows = shape[0] * shape[1]
        outgoing, incoming = table[rank], tuple(table[source][rank] for source in range(tp))
        send_sizes = [rows * sum(piece[2] for piece in route) for route in outgoing]
        recv_sizes = [rows * sum(piece[2] for piece in route) for route in incoming]
        send = pack(projection.reshape(rows, shape[-1]), outgoing, inverse=False)
        receive = projection.new_empty(sum(recv_sizes))
        dist.all_to_all_single(receive, send, recv_sizes, send_sizes, group=group)
        result = unpack(receive, incoming, rows, 12288 // tp + 512, inverse=False)
        ctx.group, ctx.shape, ctx.outgoing, ctx.incoming = group, shape, outgoing, incoming
        ctx.send_sizes, ctx.recv_sizes = send_sizes, recv_sizes
        return result.view(*shape[:2], -1)

    @staticmethod
    def backward(ctx: Any, output_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        rows = ctx.shape[0] * ctx.shape[1]
        send = pack(output_grad.reshape(rows, -1), ctx.incoming, inverse=True)
        receive = output_grad.new_empty(sum(ctx.send_sizes))
        dist.all_to_all_single(receive, send, ctx.send_sizes, ctx.recv_sizes, group=ctx.group)
        result = unpack(receive, ctx.outgoing, rows, ctx.shape[-1], inverse=True)
        return result.view(ctx.shape), None


def routed_projection(projection: torch.Tensor, group: Any) -> torch.Tensor:
    return _Route.apply(projection, group)


def warmup_collectives(group: Any, device: torch.device) -> None:
    """Initialize both TP routing directions before model activation peaks.

    All TP ranks must call this initialization-only helper. It creates no model
    gradients and consumes no RNG; large-message NCCL resources may still be
    lazy.
    """
    if group is None:
        raise ValueError("An existing explicit TP process group is required")
    tp = dist.get_world_size(group)
    route_table(tp)
    with torch.enable_grad():
        projection = torch.zeros((1, 1, 13312 // tp), device=device, dtype=torch.bfloat16, requires_grad=True)
        _Route.apply(projection, group).sum().backward()
    torch.cuda.synchronize(device)


def forward(
    self: Any,
    hidden_states: torch.Tensor,
    key_value_states: Any = None,
    output_gate: bool = False,
    head_wise_gate: bool = False,
    split_qkv: bool = True,
) -> tuple:
    from megatron.core.transformer.attention import SelfAttention, apply_module

    config, tp_group = self.config, self.pg_collection.tp
    supported = (
        output_gate
        and not head_wise_gate
        and split_qkv
        and key_value_states is None
        and type(self.core_attention).__name__ == "QSACoreAttention"
        and self.world_size in (4, 8)
        and config.num_attention_heads == 24
        and config.num_query_groups == 2
        and config.kv_channels == 256
        and config.attention_output_gate
        and config.context_parallel_size == 1
        and hidden_states.dtype in (torch.bfloat16, torch.float32)
        and not getattr(config, "fp8", None)
        and not getattr(config, "fp4", None)
        and not getattr(config, "test_mode", False)
    )
    if not supported:
        return SelfAttention._relax_native_get_query_key_value_tensors(
            self, hidden_states, key_value_states, output_gate, head_wise_gate, split_qkv
        )
    mixed, _ = apply_module(self.linear_qkv)(hidden_states)
    local = routed_projection(mixed, tp_group)
    q_width = 6144 // self.world_size
    query, gate, key, value = local.split((q_width, q_width, 256, 256), dim=-1)
    shape = (*local.shape[:2], -1, 256)
    query, gate, key, value = (tensor.reshape(shape) for tensor in (query, gate, key, value))
    if self.q_layernorm is not None:
        query = apply_module(self.q_layernorm)(query)
    if self.k_layernorm is not None:
        key = apply_module(self.k_layernorm)(key)
    return query, key, value, gate
