# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from collections.abc import Sequence

import torch

from relax.backends.megatron.chunkflow import rope as cf_rope
from relax.backends.megatron.chunkflow.metadata import int32_metadata
from relax.backends.megatron.chunkflow.scheduler import Segment
from relax.backends.megatron.chunkflow.units import Unit


def _segment_rows(parent_cu: Sequence[int], segment: Segment) -> tuple[int, int]:
    start = parent_cu[segment.seq_id] + segment.prefix
    return start, start + segment.length


def _rows_for_unit(parent_cu: Sequence[int], unit: Unit) -> list[tuple[int, int]]:
    return [_segment_rows(parent_cu, s) for seg in unit.chunks for s in seg.segments]


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
    for (s, e), length, seg in zip(rows, lengths, segments, strict=True):
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

    from relax.backends.megatron.data import PrepackedBatch

    batch: dict = PrepackedBatch(
        {
            "tokens": unit_tokens.unsqueeze(0),
            "unconcat_tokens": [unit_targets],
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

    _slice_response_aligned_keys(parent, batch, rows)
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


def _slice_response_aligned_keys(parent: dict, batch: dict, rows: list[tuple[int, int]]) -> None:
    total_lengths = parent.get("total_lengths") or []
    response_lengths = parent.get("response_lengths") or []
    keys = ("log_probs", "ref_log_probs", "values", "advantages", "returns", "rollout_log_probs")
    if not any(parent.get(k) is not None for k in keys):
        return

    parent_cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    sample_of_row: dict[int, tuple[int, int]] = {}
    for k, (cu_lo, cu_hi) in enumerate(zip(parent_cu[:-1], parent_cu[1:], strict=True)):
        prompt_len = (total_lengths[k] - response_lengths[k]) if k < len(total_lengths) else 0
        for r in range(int(cu_lo), int(cu_hi)):
            sample_of_row[r] = (k, max(r - int(cu_lo) - prompt_len, 0))

    for key in keys:
        vals = parent.get(key)
        if vals is None:
            continue
        picked = []
        for s, e in rows:
            idx = [sample_of_row.get(r, (0, 0)) for r in range(s, e)]
            tensor = vals[idx[0][0]] if idx else vals[0]
            picked.append(tensor[[i for _, i in idx]])
        batch[key] = [torch.cat(picked)]
