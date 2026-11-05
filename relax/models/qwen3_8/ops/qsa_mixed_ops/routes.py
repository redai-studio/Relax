# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Expand selected document-local blocks for mixed QSA attention."""

from typing import Any

import torch
from torch import Tensor


def token_routes(selection: Any, batch: int, sequence_length: int) -> Tensor:
    """Expand each selected four-token block and the current incomplete
    tail."""
    ids = selection.selected_ids.view(batch, sequence_length, -1)
    positions = selection.positions
    physical = torch.arange(sequence_length, device=ids.device, dtype=torch.int32).view(1, -1)
    document_start = physical - positions
    offsets = torch.arange(selection.compress_ratio, device=ids.device, dtype=torch.int32)
    selected = document_start[..., None, None] + ids[..., None] * selection.compress_ratio + offsets
    selected = selected.masked_fill(ids[..., None] < 0, -1).flatten(-2)
    tail_offsets = torch.arange(selection.compress_ratio - 1, device=ids.device, dtype=torch.int32)
    tail_count = (positions + 1) % selection.compress_ratio
    tail_start = document_start + (positions // selection.compress_ratio) * selection.compress_ratio
    tail = (tail_start[..., None] + tail_offsets).masked_fill(tail_offsets >= tail_count[..., None], -1)
    return torch.cat((selected, tail), dim=-1).contiguous()
