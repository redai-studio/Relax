# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed Engram hashing with the same CP ownership as the attention stream."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .layout import ThdLayout


if TYPE_CHECKING:
    import torch


def hash_ids_thd(engram: Any, input_ids: torch.Tensor, layout: ThdLayout) -> torch.Tensor:
    """Return CP-local hashes, blocking the n-gram history at each sample start.

    Every CP rank participates, including a rank with no valid query. Gathering
    integer IDs preserves the existing Engram communication and avoids moving
    embedding tables or activations between ranks.
    """
    import torch

    if input_ids.shape != (1, layout.local_length):
        raise ValueError("V4.1 THD Engram expects one packed, CP-local token stream.")
    if engram.cp_size != layout.cp_size:
        raise ValueError("V4.1 THD Engram CP group disagrees with its packed layout.")
    if layout.cp_size > 1:
        if engram.cp_group.rank() != layout.cp_rank:
            raise ValueError("V4.1 THD Engram CP rank disagrees with its packed layout.")
        chunks = [torch.empty_like(input_ids) for _ in range(layout.cp_size)]
        torch.distributed.all_gather(chunks, input_ids.contiguous(), group=engram.cp_group)
        global_ids = torch.cat(chunks, dim=1)
    else:
        global_ids = input_ids
    hashes = engram.hash_ids(global_ids, seq_positions=layout.positions(input_ids.device, global_stream=True))
    start = layout.global_start
    return hashes[:, start : start + layout.local_length].transpose(0, 1)
