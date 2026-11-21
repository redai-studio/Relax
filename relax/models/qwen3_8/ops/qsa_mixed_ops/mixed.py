# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Dispatch Flash/Flex/Grouped sparse GQA by document length."""

import torch

from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import _bounds_tensor, attention_auto
from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import attention as flex
from relax.models.qwen3_8.ops.qsa_mixed_ops.geometry import validate_geometry
from relax.models.qwen3_8.ops.qsa_mixed_ops.grouped import attention_bf16
from relax.utils.env import Envs


def long_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float
) -> torch.Tensor:

    validate_geometry("grouped", q.shape[1], k.shape[1], q.shape[-1], k.shape[-1])
    assert k.shape == v.shape
    return attention_bf16(q, k, v, indices, scale, block_k=64, gradient_group=None)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    if q.shape[0] == 0:
        return q.clone()
    bounds = getattr(indices, "_qsa_bounds", (0, q.shape[0]))
    budget = indices.shape[1] - 4
    flash_max_len = Envs.RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN
    explicit_flash_boundary = flash_max_len != -1
    if not explicit_flash_boundary:
        flash_max_len = budget + 3
    if not 0 <= flash_max_len <= budget + 3:
        raise ValueError("Mixed QSA Flash maximum length exceeds the dense selector budget")
    flex_max_len = Envs.RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN
    if not 0 <= flex_max_len <= 16384:
        raise ValueError("Mixed QSA Flex maximum length must be in [0, 16384]")
    if explicit_flash_boundary and flex_max_len:
        raise ValueError("An explicit mixed Flash boundary requires Flex maximum length zero")
    groups = [[], [], []]
    for start, end in zip(bounds, bounds[1:]):
        length = end - start
        if length == 0:
            continue
        kind = 0 if length <= flash_max_len else (1 if length <= flex_max_len else 2)
        groups[kind].append((start, end))
    if sum(bool(group) for group in groups) <= 1:
        if groups[2]:
            return long_attention(q, k, v, indices, scale)
        return attention_auto(q, k, v, indices, scale)

    outputs, order = [], []
    for kind, group in enumerate(groups):
        if not group:
            continue
        rows = torch.cat([torch.arange(a, b, device=q.device) for a, b in group])
        local_bounds = [0]
        for a, b in group:
            local_bounds.append(local_bounds[-1] + b - a)
        local = [tensor.index_select(0, rows) for tensor in (q, k, v)]
        if kind == 0:
            from flash_attn import flash_attn_varlen_func

            cu = _bounds_tensor(local_bounds, q.device)
            maximum = max(b - a for a, b in group)
            output = flash_attn_varlen_func(*local, cu, cu, maximum, maximum, softmax_scale=scale, causal=True)
        else:
            selected = indices.index_select(0, rows)
            offsets = torch.cat(
                [
                    torch.full((b - a, 1), new_start - a, device=q.device, dtype=indices.dtype)
                    for (a, b), new_start in zip(group, local_bounds)
                ]
            )
            selected = torch.where(selected >= 0, selected + offsets, selected)
            valid_tokens = local_bounds[-1]
            if kind == 1:
                padding = (-valid_tokens) % 16384
                if padding:
                    local = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padding)) for x in local]
                    selected = torch.nn.functional.pad(selected, (0, 0, 0, padding), value=-1)
                    local_bounds.append(valid_tokens + padding)
                from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import layout

                metadata = layout(local_bounds, q.device)
                selected._qsa_layout = (*metadata[:3], 16384)
            selected._qsa_bounds = tuple(local_bounds)
            output = (flex if kind == 1 else long_attention)(*local, selected, scale)
            output = output[:valid_tokens]
        outputs.append(output)
        order.append(rows)
    permutation = torch.cat(order)
    inverse = torch.empty_like(permutation)
    inverse.scatter_(0, permutation, torch.arange(q.shape[0], device=q.device))
    return torch.cat(outputs).index_select(0, inverse)


def _block_ids_from_bits(bits: torch.Tensor) -> torch.Tensor:
    from relax.distributed.replay_manager._bitmap import decode_contiguous_into

    encoded = torch.nn.functional.pad(bits, (1, 0))
    ids = torch.empty((bits.shape[0], 512), device=bits.device, dtype=torch.int32)
    decode_contiguous_into(encoded, 0, 0, bits.shape[0], ids)
    return ids


def attention_from_bits(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bits: torch.Tensor,
    bounds: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    """Keep Flash/Flex/Grouped routing without expanding Flex token indices."""
    if not q.shape[0]:
        return q.clone()
    flash_max = Envs.RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN
    explicit = flash_max != -1
    flash_max = 2051 if not explicit else flash_max
    flex_max = Envs.RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN
    if not 0 <= flash_max <= 2051 or not 0 <= flex_max <= 16384 or (explicit and flex_max):
        raise ValueError("Invalid mixed QSA Flash/Flex boundaries")
    # The bitmap-native kernel is validated through 32K. Keep explicit
    # non-default cutoffs and the Flash-only override unchanged.
    bitmap_max = 32768 if flex_max == 16384 else flex_max
    groups = [[], [], []]
    for start, end in zip(bounds, bounds[1:]):
        if end > start:
            kind = 0 if end - start <= flash_max else (1 if end - start <= bitmap_max else 2)
            groups[kind].append((start, end))
    multiple = sum(bool(group) for group in groups) > 1
    outputs, order = [], []
    for kind, group in enumerate(groups):
        if not group:
            continue
        rows = torch.cat([torch.arange(a, b, device=q.device) for a, b in group]) if multiple else None
        local = [tensor.index_select(0, rows) if multiple else tensor for tensor in (q, k, v)]
        local_bits = bits.index_select(0, rows) if multiple else bits
        local_bounds = [0]
        for a, b in group:
            local_bounds.append(local_bounds[-1] + b - a)
        maximum = max(b - a for a, b in group)
        if kind == 0:
            from flash_attn import flash_attn_varlen_func

            cu = _bounds_tensor(local_bounds, q.device)
            output = flash_attn_varlen_func(
                *(tensor.contiguous() for tensor in local),
                cu,
                cu,
                maximum,
                maximum,
                softmax_scale=scale,
                causal=True,
            )
        elif kind == 1:
            from relax.models.qwen3_8.ops.qsa_mixed_ops.bitmap_attention import attention as bitmap_attention

            output = bitmap_attention(*local, local_bits.contiguous(), scale, local_bounds)
        else:
            from types import SimpleNamespace

            from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import layout
            from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes

            # Only Grouped needs explicit token routes. Flex never constructs
            # full-size block IDs, expanded routes, or padded route tensors.
            ids = _block_ids_from_bits(local_bits)
            positions = layout(local_bounds, q.device)[1].unsqueeze(0)
            selection = SimpleNamespace(selected_ids=ids, positions=positions, compress_ratio=4)
            routes = torch.nn.functional.pad(token_routes(selection, 1, local_bounds[-1])[0], (0, 1), value=-1)
            output = long_attention(*local, routes, scale)
        if not multiple:
            return output
        outputs.append(output)
        order.append(rows)
    permutation = torch.cat(order)
    inverse = torch.empty_like(permutation)
    inverse.scatter_(0, permutation, torch.arange(q.shape[0], device=q.device))
    return torch.cat(outputs).index_select(0, inverse)
