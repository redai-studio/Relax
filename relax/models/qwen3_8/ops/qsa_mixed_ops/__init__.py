# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright 2025-2026 RadixArk, Inc. (miles).
# Copyright 2025 Zhipu AI.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Relax.

"""QSA adapter dispatching Flash/Flex/Grouped sparse GQA by document length."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor


def attention_for_selection(
    query: Tensor, key: Tensor, value: Tensor, selection: Any, scale: float, is_thd: bool
) -> Tensor:
    """Preserve native selected blocks, expanding the causal tail exactly
    once."""
    from relax.models.qwen3_8.ops.qsa_mixed_ops.mixed import attention
    from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes

    batch, length, heads, width = query.shape
    if width != 256 or heads != 12 or key.shape[2:] != (1, 256):
        raise NotImplementedError("Mixed QSA requires TP2 local geometry 12Q/1KV/D256")
    if selection.compress_ratio != 4 or selection.selected_ids is None:
        raise ValueError("Mixed QSA requires native selected four-token block IDs")
    bounds = getattr(selection, "_relax_qsa_mixed_bounds", None) if is_thd else (0, length)
    if bounds is None or bounds[0] != 0 or bounds[-1] != length:
        raise ValueError("Mixed packed QSA requires producer CPU document boundaries")
    # The source defines budget=width-4 and reads each selected block at stride 4.
    # token_routes has three possible tail tokens; reserve the fourth column.
    indices = torch.nn.functional.pad(token_routes(selection, batch, length), (0, 1), value=-1)
    outputs = []
    for row in range(batch):
        selected = indices[row].contiguous()
        selected._qsa_bounds = bounds
        outputs.append(attention(query[row], key[row], value[row], selected, scale))
    return torch.stack(outputs)


def forward(
    self: Any,
    hidden_states: Tensor,
    rotary_pos_emb: Any,
    packed_seq_params: Any = None,
    output_format: str = "bits",
) -> Any:
    from relax.backends.megatron.qsa_layout_metadata import valid_physical_layout
    from relax.models.qwen3_8.qsa_dispatch import selector_forward
    from relax.utils.env import Envs

    if Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION:
        raise ValueError("Mixed QSA must not be combined with fused attention")
    if self.config.context_parallel_size != 1 or self.config.qsa_indexer_loss_coeff != 0:
        raise NotImplementedError("Mixed QSA supports CP1 frozen Stage-1 indexers")
    if any(parameter.requires_grad for parameter in self.parameters()):
        raise NotImplementedError("Mixed QSA requires a frozen indexer")
    selection = selector_forward(self, hidden_states, rotary_pos_emb, packed_seq_params, output_format)
    # Keep all-short batches on the source's Flash varlen path as well.
    # Native all_selected would otherwise bypass this backend through TE.
    selection = replace(selection, all_selected=False)
    bounds = None
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        record = valid_physical_layout(packed_seq_params)
        bounds = record[2] if record is not None else getattr(packed_seq_params, "qsa_select_offsets_cpu", None)
        if (
            not isinstance(bounds, tuple)
            or len(bounds) < 2
            or bounds[0] != 0
            or bounds[-1] != selection.positions.shape[1]
            or any(type(value) is not int for value in bounds)
            or any(left > right for left, right in zip(bounds, bounds[1:]))
        ):
            raise ValueError("Mixed QSA requires exact producer qsa_select_offsets_cpu metadata")
    from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import bind_document_bounds

    return bind_document_bounds(selection, bounds)
