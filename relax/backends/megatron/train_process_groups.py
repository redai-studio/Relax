# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Choose the Actor's communication-group lifetime independently of weights."""

from typing import Any

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def configure_train_process_group_retention(args: Any, offloader: Any, group: Any) -> bool:
    """Agree once across Actor ranks before keeping their NCCL resources.

    Selective offload moves model/optimizer storage explicitly. TMS pauses its
    whole allocation pool, so retaining training communicators through that
    pause is not supported by this policy.
    """
    import torch.distributed as dist

    requested = Envs.RELAX_COLOCATE_KEEP_TRAIN_PROCESS_GROUPS
    reason = None
    if requested:
        if not getattr(args, "colocate", False) or not getattr(args, "offload_train", False):
            reason = "requires colocate and train-state offload"
        elif getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
            reason = "requires the synchronous Actor lifecycle"
        elif offloader.uses_tms:
            reason = "requires --selective-offload; retaining training communicators across TMS pause is unsupported"
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (requested, reason), group=group)
    if any(record[0] != requested for record in records):
        raise ValueError("Actor ranks disagree on RELAX_COLOCATE_KEEP_TRAIN_PROCESS_GROUPS")
    reasons = sorted({record[1] for record in records if record[1] is not None})
    if reasons:
        raise ValueError("Cannot retain training process groups: " + "; ".join(reasons))
    if requested and dist.get_rank(group) == 0:
        logger.info("Keeping training process groups across per-step offload; NCCL allocations remain resident")
    return requested
