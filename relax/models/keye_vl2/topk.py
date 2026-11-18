# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Shared DSA selection with the same boundary tie-break for train and
decode."""

from collections.abc import Callable

import torch


KEYE_INDEXER_TOPK = 2048


def dsa_topk_indices(scores: torch.Tensor) -> torch.Tensor:
    """Select up to 2048 keys, preferring earlier tokens when scores tie."""
    width = scores.shape[-1]
    if width <= KEYE_INDEXER_TOPK:
        return torch.arange(width, device=scores.device).expand(scores.shape[0], -1)

    from flashinfer import top_k

    return top_k(
        scores.detach().contiguous(), KEYE_INDEXER_TOPK, deterministic=True, tie_break=1, dsa_graph_safe=True
    )[1]


def copy_capture_rows(destination: torch.Tensor, selected: torch.Tensor, valid: torch.Tensor) -> None:
    """Capture logical token positions, with -1 for unused slots."""
    destination.fill_(-1)
    destination[:, : selected.shape[1]].copy_(selected.masked_fill(~valid, -1))


def resolve_replayed_selection(
    selected: torch.Tensor,
    causal: torch.Tensor,
    replay: torch.Tensor,
    row_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use captured positions for present rows, preserving the causal
    support."""
    fallback = torch.nn.functional.pad(selected, (0, replay.shape[1] - selected.shape[1]), value=-1)
    indices = torch.where(row_valid[:, None], replay, fallback).long()
    in_range = (indices >= 0) & (indices < causal.shape[1])
    indices = indices.clamp(0, causal.shape[1] - 1)
    return indices, in_range & causal.gather(1, indices)


def _resolve_replayed_rows(
    causal: torch.Tensor,
    replay: torch.Tensor,
    row_valid: torch.Tensor,
    fallback_ranges: tuple[tuple[int, int], ...],
    *,
    row_offset: int,
    score_rows: Callable[[int, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate missing rows from the same CPU layout that produced row_valid.

    Covered chunks can read the immutable replay tensor directly. Copy only
    when replacing a missing row; the trusted layout already selects the
    replay/fallback source, so a second tensor-wide where is unnecessary.
    """
    selected = replay
    for begin, end in fallback_ranges:
        first, stop = max(begin - row_offset, 0), min(end - row_offset, len(replay))
        if first >= stop:
            continue
        if selected is replay:
            selected = replay.clone()
        if causal.shape[1] <= KEYE_INDEXER_TOPK:
            fallback = torch.arange(causal.shape[1], device=replay.device).expand(stop - first, -1)
        else:
            scores = score_rows(first, stop)
            scores = scores.masked_fill(~causal[first:stop], torch.finfo(scores.dtype).min)
            fallback = dsa_topk_indices(scores)
        selected[first:stop] = torch.nn.functional.pad(fallback, (0, replay.shape[1] - fallback.shape[1]), value=-1)
    indices = selected.long()
    in_range = (indices >= 0) & (indices < causal.shape[1])
    indices = indices.clamp(0, causal.shape[1] - 1)
    return indices, in_range & causal.gather(1, indices)
