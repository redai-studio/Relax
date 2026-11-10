# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mask output rows without changing the tensor-parallel shard layout."""

import torch


def mask_policy_logits(logits: torch.Tensor, *, vocab_size: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("Policy scoring requires FP32 logits [response, vocab/TP]")
    width = logits.size(-1)
    if type(vocab_size) is not int or not 0 < vocab_size <= width * tp_size or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid policy vocabulary or tensor-parallel layout")
    first_invalid = max(0, min(width, vocab_size - tp_rank * width))
    if first_invalid == width:
        return logits
    # A finite floor keeps entropy's p * logits finite, even when this entire
    # TP shard has no legal tokens. Masked output rows receive zero gradients.
    return logits.masked_fill(torch.arange(width, device=logits.device) >= first_invalid, -1e30)
