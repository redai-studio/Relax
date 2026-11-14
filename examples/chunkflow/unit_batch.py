# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from collections.abc import Sequence

import torch

from examples.chunkflow import rope as cf_rope
from examples.chunkflow.metadata import int32_metadata
from examples.chunkflow.scheduler import Segment
from examples.chunkflow.units import Unit


def _rows_for_unit(parent_cu: Sequence[int], unit: Unit) -> list[tuple[int, int]]:
    return [
        (parent_cu[s.seq_id] + s.prefix, parent_cu[s.seq_id] + s.prefix + s.length)
        for chunk in unit.chunks
        for s in chunk.segments
    ]


def _rope_offsets(unit: Unit) -> list[int]:
    segments = [seg for chunk in unit.chunks for seg in chunk.segments]
    if unit.is_dependent and len(segments) > 1:
        raise NotImplementedError(
            "chunkflow lockstep units with multiple dependent segments need per-segment RoPE offsets; "
            "use --chunkflow-pack-policy staggered for numerically correct training"
        )
    return [seg.prefix for seg in segments]


def slice_parent_batch(
    parent: dict,
    unit: Unit,
    *,
    pad_token_id: int,
    args=None,
) -> dict:
    tokens = parent["tokens"]
    assert tokens.dim() == 2 and tokens.size(0) == 1, f"expected [1, T], got {tokens.shape}"
    flat_tokens = tokens[0]
    flat_mask = parent["full_loss_masks"][0]
    parent_cu = parent["packed_seq_params"].cu_seqlens_q_cpu

    rows = _rows_for_unit(parent_cu, unit)
    lengths = [e - s for s, e in rows]
    segments = [seg for chunk in unit.chunks for seg in chunk.segments]

    token_parts = [flat_tokens[s:e] for s, e in rows]
    # Shift targets within sequence boundaries.
    target_parts = []
    pre_shifted_labels = parent.get("unconcat_pre_shifted_labels")
    for (s, e), length, seg in zip(rows, lengths, segments, strict=True):
        if pre_shifted_labels is not None:
            labels = pre_shifted_labels[seg.seq_id][seg.prefix : seg.prefix + length]
            target_parts.append(labels.masked_fill(labels.eq(-100), pad_token_id))
            continue
        seq_end = parent_cu[seg.seq_id + 1]
        target = flat_tokens[s + 1 : min(e + 1, seq_end)]
        if target.size(0) < length:
            target = torch.cat(
                [
                    target,
                    torch.full((length - target.size(0),), pad_token_id, dtype=target.dtype, device=target.device),
                ]
            )
        target_parts.append(target)
    mask_parts = [flat_mask[s:e] for s, e in rows]

    unit_tokens = token_parts[0] if len(token_parts) == 1 else torch.cat(token_parts)
    unit_targets = target_parts[0] if len(target_parts) == 1 else torch.cat(target_parts)
    unit_mask = mask_parts[0] if len(mask_parts) == 1 else torch.cat(mask_parts)

    full_mask = unit_mask
    pad = (-unit_tokens.size(0)) % 128
    if pad:
        unit_tokens = torch.cat(
            [unit_tokens, torch.full((pad,), pad_token_id, dtype=unit_tokens.dtype, device=unit_tokens.device)]
        )
        full_mask = torch.cat([full_mask, torch.zeros(pad, dtype=unit_mask.dtype, device=unit_mask.device)])

    # RoPE uses sequence-local prefixes, not packed row offsets.
    offsets = _rope_offsets(unit)
    use_mrope = getattr(args, "uses_unsplit_forward", False)
    if use_mrope and parent.get("multimodal_train_inputs"):
        raise NotImplementedError("ChunkFlow explicit MRoPE positions currently support text-only batches")
    packed = _build_packed_seq_params(unit_tokens, lengths, [0] * len(offsets) if use_mrope else offsets)
    physical_cu = getattr(parent["packed_seq_params"], "qsa_select_offsets_cpu", None)
    if physical_cu is not None:
        packed.chunkflow_parent_physical_cu = tuple(physical_cu)
        packed.chunkflow_parent_max_seqlen = max(b - a for a, b in zip(physical_cu[:-1], physical_cu[1:]))
    # Native PLE hashes the original tokens, including n-grams crossing a chunk
    # boundary. Views keep this metadata cheap and replayable during backward.
    if unit.is_dependent:
        start, end = rows[0]
        packed.chunkflow_prefix_tokens = flat_tokens[start - offsets[0] : end].unsqueeze(0)
    else:
        packed.chunkflow_prefix_tokens = unit_tokens[: sum(lengths)].unsqueeze(0)

    from relax.backends.megatron.data import PrepackedBatch

    batch: dict = PrepackedBatch(
        {
            "tokens": unit_tokens.unsqueeze(0),
            # The native loss slicer expects one prompt token followed by T
            # targets for total=T+1, response=T. This prefix is loss metadata
            # only; the model still receives exactly the T input tokens.
            "unconcat_tokens": [torch.cat([unit_tokens[:1], unit_targets])],
            "full_loss_masks": full_mask.unsqueeze(0),
            "loss_masks": [unit_mask],
            "total_lengths": [sum(lengths) + 1],
            "response_lengths": [sum(lengths)],
            "max_seq_lens": [max(lengths)],
            "packed_seq_params": packed,
        }
    )
    if use_mrope:
        # Bridge consumes physical packing boundaries and explicit [3, B, T] positions separately.
        physical_cu = list(packed.cu_seqlens_q_cpu)
        physical_cu[-1] = unit_tokens.size(0)
        packed.cu_seqlens_q_padded = int32_metadata(physical_cu, unit_tokens.device) if pad else packed.cu_seqlens_q
        packed.cu_seqlens_kv_padded = packed.cu_seqlens_q_padded
        packed.max_seqlen_q = packed.max_seqlen_kv = max([*lengths[:-1], lengths[-1] + pad])
        position_parts = [
            torch.arange(offset, offset + length, device=unit_tokens.device)
            for offset, length in zip(offsets, lengths)
        ]
        positions = position_parts[0] if len(position_parts) == 1 else torch.cat(position_parts)
        if pad:
            positions = torch.cat([positions, positions.new_ones(pad)])
        batch["position_ids"] = positions.view(1, 1, -1).expand(3, 1, -1)

    _slice_response_aligned_keys(parent, batch, segments)
    for key in ("__is_dummy__", "__loss_scale__", "replay_micro_batch_index"):
        if key in parent:
            batch[key] = parent[key]
    return batch


def _build_packed_seq_params(unit_tokens: torch.Tensor, lengths: list[int], offsets: list[int]):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    device = unit_tokens.device

    cu_tensor = int32_metadata(cu, device)
    packed = PackedSeqParams(
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        qkv_format="thd",
    )
    packed.cu_seqlens_q_cpu = cu

    if any(offset for offset in offsets):
        cf_rope.apply_rope_offset(packed, offsets=offsets, lengths=lengths)
    return packed


def _slice_response_aligned_keys(parent: dict, batch: dict, segments: list[Segment]) -> None:
    total_lengths = parent.get("total_lengths") or []
    response_lengths = parent.get("response_lengths") or []
    keys = ("log_probs", "ref_log_probs", "values", "advantages", "returns", "rollout_log_probs")
    if not any(parent.get(k) is not None for k in keys):
        return

    for key in keys:
        vals = parent.get(key)
        if vals is None:
            continue
        picked = []
        for segment in segments:
            k = segment.seq_id
            tensor = vals[k]
            prompt_len = total_lengths[k] - response_lengths[k]
            # RL values are indexed by target token; a logit at row r predicts
            # r+1. SFT vectors already use predictor coordinates (CP=1).
            offset = segment.prefix + (1 - prompt_len if prompt_len else 0)
            start = max(0, -offset)
            end = min(segment.length, tensor.size(0) - offset)
            part = tensor.new_zeros((segment.length, *tensor.shape[1:]))
            if end > start:
                part[start:end] = tensor[offset + start : offset + end]
            picked.append(part)
        batch[key] = [torch.cat(picked)]
