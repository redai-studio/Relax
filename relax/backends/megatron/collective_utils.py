# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import torch
import torch.distributed as dist
from megatron.core import mpu

from relax.utils import device as device_utils


def _agree_drained(local_status: bool, include_pipeline: bool) -> torch.Tensor:
    """AND-reduce a drain flag over the ranks that must leave the loop
    together.

    Non-querying ranks pass ``True``, the identity element of MIN, so the result
    is exactly the querying rank's answer -- same semantics as broadcasting from
    (TP=0, CP=0, PP=0), but order-independent and one collective cheaper: the
    tensor-and-context-parallel group covers TP x CP within a single PP stage in
    one call.

    Args:
        local_status: This rank's answer, or True if it did not query.
        include_pipeline: Also reduce across the pipeline group. Must be False on
            the 1F1B streaming path, where PP stages are not in lockstep and a
            pipeline-group collective deadlocks.
    """
    status = torch.tensor(
        [1 if local_status else 0],
        dtype=torch.int32,
        device=device_utils.make_current_torch_device(),
    )
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=mpu.get_tensor_and_context_parallel_group())
    if include_pipeline:
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=mpu.get_pipeline_model_parallel_group())
    return status[0]
