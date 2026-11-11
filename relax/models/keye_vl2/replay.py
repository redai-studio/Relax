# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keye replay uses document-local token IDs and fixed-width wire rows."""

import torch
import torch.nn.functional as F


INDEXER_TOPK = 2048


def capture_rows(indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pad short prefixes without turning invalid selections into key zero."""
    if indices.ndim != 2 or indices.shape != valid.shape or indices.shape[1] > INDEXER_TOPK:
        raise ValueError("Keye capture requires matching [query, up to 2048] indices and validity")
    return F.pad(indices.to(torch.int32).masked_fill(~valid, -1), (0, INDEXER_TOPK - indices.shape[1]), value=-1)


def replay_selection(
    computed: torch.Tensor,
    replay: tuple[torch.Tensor, torch.Tensor],
    causal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep IDs document-local; replace only real rollout rows, not padding."""
    recorded, row_valid = replay
    if recorded.shape != (computed.shape[0], INDEXER_TOPK) or row_valid.shape != computed.shape[:1]:
        raise ValueError("Keye replay rows must match local queries with top-k 2048")
    fallback = capture_rows(computed, causal.gather(1, computed))
    chosen = torch.where(row_valid[:, None], recorded, fallback)
    valid = (chosen >= 0) & (chosen < causal.shape[1])
    safe = chosen.clamp(0, causal.shape[1] - 1).long()
    valid = valid & causal.gather(1, safe)
    # Invalid sentinel slots may map to key zero too. Accumulation prevents a
    # later false slot from overwriting an earlier true selection of key zero.
    support = torch.zeros_like(causal, dtype=torch.int32).scatter_add_(1, safe, valid.int()).bool()
    return safe, valid, support
