# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CuTe sparse attention adapted to Relax's document-relative selection IDs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from relax.models.qwen3_8.ops.qsa_cute.layout import block_metadata, validate_bounds
from relax.models.qwen3_8.ops.qsa_cute.profiling import profile_range


def sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    ids: Tensor,
    bounds: tuple[int, ...],
    scale: float,
    query_positions: Tensor | None = None,
) -> Tensor:
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("CuTe QSA requires [batch, tokens, heads, dim] Q/K/V")
    if query.shape[0] != key.shape[0] or key.shape[1] != bounds[-1]:
        raise ValueError("CuTe QSA global K/V must match its document boundaries")
    if ids.shape[:2] != query.shape[:2]:
        raise ValueError("CuTe QSA selected IDs must describe exactly the local query rows")
    from relax.models.qwen3_8.ops.qsa_mixed_ops.geometry import validate_geometry

    validate_geometry("cute", query.shape[2], key.shape[2], query.shape[3], key.shape[3])
    if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("CuTe QSA requires matching BF16 or FP16 Q/K/V")
    if not query.is_cuda or any(t.device != query.device for t in (key, value, ids)):
        raise ValueError("CuTe QSA requires Q/K/V and IDs on the same CUDA device")
    if torch.cuda.get_device_capability(query.device)[0] != 10:
        raise ValueError("This CuTe QSA adapter targets Blackwell SM10x")
    from relax.models.qwen3_8.ops.qsa_cute.cute_qsa_attn import qsa_sparse_attention

    with profile_range("metadata/blocks"):
        metadata = block_metadata(ids, bounds, query_positions)
    return qsa_sparse_attention(
        query,
        key,
        value,
        metadata.indices,
        metadata.counts,
        metadata.bases,
        metadata.ends,
        scale=scale,
        q_positions=metadata.query_positions,
        static_shapes=True,
    )


def attention_for_selection(
    query: Tensor, key: Tensor, value: Tensor, selection: Any, scale: float, bounds: tuple[int, ...]
) -> Tensor:
    """Keep short documents batched in Flash; send long queries to CuTe."""
    from relax.models.qwen3_8.ops.qsa_mixed_ops.cudnn import _flash
    from relax.utils.env import Envs

    batch, length, heads, dim = query.shape
    validate_bounds(bounds, length)
    ids = selection.selected_ids.view(batch, length, -1)
    maximum = ids.shape[-1] * 4 + 3
    threshold = Envs.RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN
    threshold = maximum if threshold == -1 else threshold
    if not 0 <= threshold <= maximum:
        raise ValueError("CuTe mixed Flash threshold exceeds the exact dense-causal budget")
    short = [(a, b) for a, b in zip(bounds, bounds[1:]) if 0 < b - a <= threshold]
    long = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a > threshold]
    outputs = []
    for row in range(batch):
        pieces, positions = [], []
        if short:
            short_rows = torch.cat([torch.arange(a, b, device=query.device) for a, b in short])
            tensors = [tensor[row].index_select(0, short_rows) for tensor in (query, key, value)]
            pieces.append(_flash(*tensors, tuple(b - a for a, b in short), scale))
            positions.append(short_rows)
        if long:
            long_rows = torch.cat([torch.arange(a, b, device=query.device) for a, b in long])
            pieces.append(
                sparse_attention(
                    query[row : row + 1].index_select(1, long_rows),
                    key[row : row + 1],
                    value[row : row + 1],
                    ids[row : row + 1].index_select(1, long_rows),
                    bounds,
                    scale,
                    long_rows,
                )[0]
            )
            positions.append(long_rows)
        output = query.new_zeros((length, heads, dim))
        if pieces:
            output = output.index_copy(0, torch.cat(positions), torch.cat(pieces))
        outputs.append(output)
    return torch.stack(outputs)
