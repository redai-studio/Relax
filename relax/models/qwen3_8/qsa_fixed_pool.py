# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in fixed-order FP64 pooling for frozen R4 packed QSA indexers."""

from __future__ import annotations

from typing import Any

import torch

from relax.models.qwen3_8.ops.qsa_pool import (
    _pool_r4 as _pool_r4,
)
from relax.models.qwen3_8.ops.qsa_pool import (
    fixed_pool as fixed_pool,
)
from relax.utils.env import Envs


def pool_keys(
    self: Any, raw_keys: torch.Tensor, doc_ids: torch.Tensor, positions: torch.Tensor, max_doc_len: int
) -> tuple:
    from relax.models.qwen3_8.qsa_select import _layout

    if not Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
        raise RuntimeError("QSA fixed pooling requires RELAX_QWEN38_QSA_FUSED_SELECTOR=1")
    offsets = _layout.get()
    if (
        self.compress_ratio != 4
        or self.config.qsa_indexer_loss_coeff != 0
        or self.config.context_parallel_size != 1
        or any(p.requires_grad for p in self.parameters())
        or doc_ids.shape != (raw_keys.shape[0],)
        or positions.shape != doc_ids.shape
        or doc_ids.device != raw_keys.device
        or positions.device != raw_keys.device
    ):
        raise RuntimeError("Fixed QSA pool requires frozen packed R4 CP1 indexer without auxiliary loss")
    return fixed_pool(raw_keys, offsets, max_doc_len)
