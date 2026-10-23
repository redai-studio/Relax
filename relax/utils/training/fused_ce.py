# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in CE experiments; preserve native heads and their main_grad
protocol."""

import os
from functools import lru_cache

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def ce_implementation(branch: str) -> str:
    value = os.environ.get(f"RELAX_{branch}_CE_IMPL", "native")
    if value not in ("native", "te"):
        raise ValueError(f"Unsupported RELAX_{branch}_CE_IMPL={value!r}")
    return value


@lru_cache(maxsize=1)
def _te_ce():
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    logger.info("[fused-ce] TE Triton CE enabled; TP1, FP32 loss, native head gradient accumulation")
    return parallel_cross_entropy


def te_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, tp_group=None) -> torch.Tensor:
    """Unreduced CE with labels matching logits' leading dimensions; TP1
    only."""
    if tp_group is not None and dist.get_world_size(tp_group) != 1:
        raise ValueError("Experimental TE CE currently requires tensor parallel size 1")
    if logits.ndim != 3 or labels.shape != logits.shape[:-1]:
        raise ValueError("TE CE requires logits [S,B,V] and labels [S,B]")
    # Singleton B=1 transposes can be contiguous while retaining label stride
    # (1, full_sequence_length). TE uses target.stride(-1) to walk all rows,
    # so normalize to [1, rows, vocab] / [1, rows] explicitly.
    loss = _te_ce()(
        logits.reshape(1, -1, logits.shape[-1]).contiguous(),
        labels.contiguous().reshape(1, -1),
        reduce_loss=False,
        dist_process_group=None,
        is_cg_capturable=True,
    )
    return loss.reshape(labels.shape)
