# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import pytest
import torch

from relax.backends.fsdp_turbo.qwen4_indexer import validate_lightning_mask


def test_prefix_validation_rechecks_mutated_mask():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    validate_lightning_mask(mask, 4, 4)
    validate_lightning_mask(mask, 4, 4)
    mask[0, 0, 0, 3] = True
    with pytest.raises(RuntimeError, match="causal-prefix"):
        validate_lightning_mask(mask, 4, 4)


def test_prefix_validation_accepts_padded_queries():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    mask[..., 3, :] = False
    validate_lightning_mask(mask, 4, 4)
