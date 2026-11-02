# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise configurable sample limits through the real SFT oversize policy."""

from functools import partial

import pytest
import torch

from relax.engine.sft.dataset.oversize import keep_within_sequence_limit
from relax.engine.sft.dataset.streaming import apply_oversize_strategy


@pytest.mark.parametrize("boundary", ["budget", "budget_plus_one", "limit", "limit_plus_one"])
@pytest.mark.parametrize("has_multimodal", [False, True])
def test_sample_limit_preserves_tokens_and_loss_mask(boundary, has_multimodal):
    limit, capacity = 64, 32
    hook = partial(keep_within_sequence_limit, seq_length=limit)
    length = {
        "budget": capacity,
        "budget_plus_one": capacity + 1,
        "limit": limit,
        "limit_plus_one": limit + 1,
    }[boundary]
    tokens = torch.arange(length)
    loss_mask = (tokens % 3 == 0).long()
    result = apply_oversize_strategy(
        tokens=tokens,
        loss_mask=loss_mask,
        capacity=capacity,
        strategy="custom",
        custom_fn=hook,
        idx=0,
        has_multimodal=has_multimodal,
    )
    if length > limit:
        assert result is None
    else:
        assert result[0] is tokens
        assert result[1] is loss_mask


@pytest.mark.parametrize("limit,capacity", [(None, 8), (0, 8), (7, 8), (16, 0)])
def test_sample_limit_rejects_invalid_packing_budget(limit, capacity):
    tokens = torch.arange(9)
    with pytest.raises(ValueError, match="--seq-length"):
        keep_within_sequence_limit(tokens=tokens, loss_mask=tokens, capacity=capacity, idx=0, seq_length=limit)
