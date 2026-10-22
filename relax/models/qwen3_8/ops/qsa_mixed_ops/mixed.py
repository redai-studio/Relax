# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""Dispatch Flash/Flex/Grouped sparse GQA by document length."""

import torch

from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import attention as flex
from relax.models.qwen3_8.ops.qsa_mixed_ops.flex import attention_auto
from relax.models.qwen3_8.ops.qsa_mixed_ops.grouped import attention_bf16
from relax.utils.env import Envs


def long_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float
) -> torch.Tensor:

    assert q.shape[1:] == (12, 256) and k.shape[1:] == (1, 256)
    assert k.shape == v.shape
    return attention_bf16(q, k, v, indices, scale, block_k=64, gradient_group=None)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    if q.shape[0] == 0:
        return q.clone()
    bounds = getattr(indices, "_qsa_bounds", (0, q.shape[0]))
    budget = indices.shape[1] - 4
    flex_max_len = Envs.RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN
    if not 0 <= flex_max_len <= 16384:
        raise ValueError("Mixed QSA Flex maximum length must be in [0, 16384]")
    groups = [[], [], []]
    for start, end in zip(bounds, bounds[1:]):
        length = end - start
        if length == 0:
            continue
        kind = 0 if length <= budget + 3 else (1 if length <= flex_max_len else 2)
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

            cu = torch.tensor(local_bounds, device=q.device, dtype=torch.int32)
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
