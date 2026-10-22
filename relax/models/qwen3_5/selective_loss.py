# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reuse the existing CPU row plans with the native fused-linear head."""

import torch


def selected_linear_loss(
    pairs: list[tuple[torch.Tensor, torch.Tensor]], plans: list[tuple[int, torch.Tensor]], head: object
) -> torch.Tensor:
    assert len(pairs) == len(plans)
    counts = [rows.numel() for _, rows in plans]
    for (hidden, target), (full, rows) in zip(pairs, plans, strict=True):
        assert hidden.size(0) == target.numel() == full
    if sum(counts) == 0:
        hidden, target = next((hidden, target) for hidden, target in pairs if hidden.size(0) > 0)
        # All TP lanes still execute the same CE collectives and DDP sees a zero dW.
        dummy = head(hidden[:1], labels=target[:1]).sum() * 0
        return torch.cat([hidden.new_zeros(full, dtype=torch.float32) for full, _ in plans]) + dummy
    hidden = torch.cat([h.index_select(0, rows) for (h, _), (_, rows) in zip(pairs, plans, strict=True)])
    targets = torch.cat([t.index_select(0, rows) for (_, t), (_, rows) in zip(pairs, plans, strict=True)])
    losses = head(hidden, labels=targets)
    return torch.cat(
        [
            part.new_zeros(full).index_copy(0, rows, part)
            for part, (full, rows) in zip(losses.split(counts), plans, strict=True)
        ]
    )
