# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed sequence dimensions and microbatch layout binding for V4.1."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .layout import ThdLayout


if TYPE_CHECKING:
    from ..runtime import V41CarryPlan


def bind_layout(packed_seq_params: Any, cp_group: Any, local_length: int | None = None) -> ThdLayout:
    """Read the layout retained by this forward's checkpoint closure.

    Model.forward attaches it to a shallow copy of PackedSeqParams, so cached
    device metadata cannot remain attached to the caller's reusable batch.
    Direct layer calls can construct the same layout from its CPU boundaries.
    """
    cp_rank = cp_group.rank() if cp_group is not None else 0
    cp_size = cp_group.size() if cp_group is not None else 1
    layout = getattr(packed_seq_params, "v41_thd_layout", None)
    if layout is None:
        layout = ThdLayout.from_packed_seq_params(
            packed_seq_params, cp_rank=cp_rank, cp_size=cp_size, seqlen_local=local_length
        )
    if layout.cp_size != cp_size or layout.global_start != cp_rank * layout.local_length:
        raise ValueError("DeepSeek V4.1 packed layout does not match the model's context-parallel group.")
    if local_length is not None and layout.local_length != local_length:
        raise ValueError("DeepSeek V4.1 packed layout does not match the layer's local sequence length.")
    return layout


def carry_dimensions(plan: V41CarryPlan, s: int, b: int, layout: ThdLayout) -> tuple[int, dict[int, int], int]:
    local_length = s * plan.tp_size if plan.sequence_parallel else s
    if b != 1:
        raise ValueError("DeepSeek V4.1 THD carry requires a singleton batch dimension.")
    if layout.local_length != local_length or layout.cp_size != plan.cp_size:
        raise ValueError("DeepSeek V4.1 THD carry does not match the packed sequence/CP layout.")
    compressed_lengths = {
        key: layout.compressed_count(plan.compress_ratios[key]) for key in (*plan.latent_keys, *plan.index_k_keys)
    }
    candidate_width = (
        layout.candidate_width(plan.candidate_ratio) if plan.candidates and plan.candidate_ratio > 0 else 0
    )
    return local_length, compressed_lengths, candidate_width
