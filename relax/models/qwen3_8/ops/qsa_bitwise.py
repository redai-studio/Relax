# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Numerical preparation shared by Qwen3.8 attention and its indexer."""

import torch


def mrope_inv_freq(rotary):
    dim = rotary.rotary_dim
    exponents = torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim
    return 1.0 / (rotary.base**exponents)


def mrope_arguments(positions, inv_freq, rotary):
    from megatron.core.fusions.fused_mrope import mrope_freqs_to_rotary_emb

    axes = positions.expand(3, -1) if positions.ndim == 1 else positions
    freqs = (axes[:, None, :, None].float() * inv_freq).contiguous()
    cu_seqlens = torch.arange(2, dtype=torch.int32, device=positions.device) * positions.shape[-1]
    materialized = mrope_freqs_to_rotary_emb(freqs, rotary.mrope_section, interleaved_mrope=rotary.mrope_interleaved)
    return materialized, cu_seqlens


def normalize_and_rotate(tensor, norm, freqs, cu_seqlens):
    from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb_thd
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine_torch import rmsnorm_fwd

    normalized, _, _ = rmsnorm_fwd(
        tensor.reshape(-1, tensor.shape[-1]).contiguous(),
        norm.weight.data,
        norm.variance_epsilon,
        None,
        None,
        TE_DType[tensor.dtype],
        0,
        True,
    )
    return fused_apply_rotary_pos_emb_thd(
        normalized.view(tensor.shape), cu_seqlens, freqs, cp_size=1, cp_rank=0, interleaved=False
    )


def select_blocks(query, keys, starts, ends, topk):
    from sglang.srt.batch_invariant_ops import matmul_persistent

    from .router_bitwise import stable_topk

    rows, heads, width = query.shape
    columns = keys.shape[0]
    if columns == 0:
        return torch.full((rows, topk), -1, device=query.device, dtype=torch.int32)

    dot = matmul_persistent(query.float().reshape(-1, width), keys.float().T.contiguous())
    a, b, c, d = dot.view(rows, heads, columns).relu_().unbind(dim=1)
    logits = ((a + b) + (c + d)) * (width**-0.5)
    ids = torch.arange(columns, device=query.device)
    visible = (ids >= starts[:, None]) & (ids < ends[:, None])
    logits.masked_fill_(~visible, -torch.inf)
    values, indices = stable_topk(logits, min(topk, columns))
    selected = (indices - starts[:, None]).masked_fill_(~values.isfinite(), -1).to(torch.int32)
    return torch.nn.functional.pad(selected, (0, topk - selected.shape[1]), value=-1)


def select_training_blocks(query, pooled_keys, doc_ids, positions, ratio, topk, output_format):
    lengths = torch.bincount(doc_ids.long(), minlength=pooled_keys.shape[0]).tolist()
    device = query.device
    chunks = []
    for q, pos, keys in zip(query.split(lengths), positions.split(lengths), pooled_keys):
        keys = keys[: len(q) // ratio]
        ends = ((pos + 1) // ratio).clamp_max(len(keys))
        if len(keys) <= topk:
            ids = torch.arange(topk, device=device, dtype=torch.int32)
            chunks.append(torch.where(ids < ends[:, None], ids, -1))
            continue
        # Keep the one-document FP32 dot and sort workspace bounded on long input.
        tile = max(1, (128 << 20) // (len(keys) * query.shape[1] * 4))
        for rows, visible in zip(q.split(tile), ends.split(tile)):
            chunks.append(select_blocks(rows, keys, torch.zeros_like(visible), visible, topk))

    selected = torch.cat(chunks)
    all_selected = max(lengths) // ratio <= topk
    if output_format == "ids":
        return selected, all_selected

    nbytes = (pooled_keys.shape[1] + 8) // 8
    valid = selected >= 0
    blocks = selected.masked_fill(~valid, 0).long()
    bits = torch.zeros((len(query), nbytes), device=device, dtype=torch.uint8)
    bits.scatter_add_(1, blocks // 8, ((1 << (blocks % 8)) * valid).to(torch.uint8))
    return bits, all_selected
