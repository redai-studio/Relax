# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Retain a deterministic subset of BF16 all-gathers until the native step
reset."""

from functools import wraps

import torch


def bind(pipeline, limit_bytes):
    config = pipeline.buffer.ddp_config
    if (
        config.fsdp_double_buffer
        or config.outer_dp_sharding_strategy != "no_shard"
        or config.data_parallel_sharding_strategy != "optim_grads_params"
    ):
        raise ValueError(
            "Partial residency requires native single-level full sharding and separately allocated buckets"
        )
    groups = pipeline.buffer.parameter_groups
    ordered = sorted(range(len(groups)), key=lambda index: (not groups[index].is_expert_param, index))
    held = {}
    total = 0
    for index in ordered:
        group = groups[index]
        buf = group.model_weight_buffer
        if (
            group.fsdp_unit_id is None
            or group.fsdp_unit_id < 0
            or buf is None
            or group.transpose_weight_buffer is not None
        ):
            continue
        if buf.dtype != torch.bfloat16 or buf.data_parallel_group.size() < 2 or not buf.is_data_distributed:
            continue
        size = buf.bucket_index.size * torch.bfloat16.itemsize
        if size > 0 and total + size <= limit_bytes:
            held[index] = size
            total += size
    if not held:
        raise ValueError("No eligible BF16 all-gather bucket fit the requested limit")
    stats = {"bytes": total, "buckets": list(held), "skips": 0, "resets": 0}
    releasing = False
    native_release = pipeline.release_bucket
    native_reset = pipeline.reset

    @wraps(native_release)
    def release(bucket_id, bwd, lazy=False):
        if bucket_id in held and not releasing:
            stats["skips"] += 1
            pipeline.bucket_can_be_released[pipeline.get_bucket_key(bucket_id, bwd)] = False
            return
        return native_release(bucket_id, bwd, lazy)

    @wraps(native_reset)
    def reset(*args, **kwargs):
        nonlocal releasing
        releasing = True
        try:
            result = native_reset(*args, **kwargs)
            if any(pipeline.bucket_status[pipeline.get_bucket_key(index, False)].name != "EMPTY" for index in held):
                raise RuntimeError("Retained full parameters survived the native optimizer boundary")
            stats["resets"] += 1
            return result
        finally:
            releasing = False

    pipeline.release_bucket = release
    pipeline.reset = reset
    return stats


def install(args, front_gib=8):
    if front_gib not in (8, 12, 16):
        raise ValueError("Unexpected front-stage parameter budget")
    from megatron.core import parallel_state

    from relax.backends.megatron import model
    from relax.utils.logging_utils import get_logger

    if (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.context_parallel_size) != (2, 2, 1):
        raise ValueError("The configured budgets apply to TP2/PP2/CP1")
    stage = parallel_state.get_pipeline_model_parallel_rank()
    native = model.get_bridge_distributed_model

    @wraps(native)
    def build(*positional, **kwargs):
        chunks = native(*positional, **kwargs)
        owners = [
            m
            for chunk in chunks
            for m in chunk.modules()
            if hasattr(m, "all_gather_pipeline") and "all_gather_pipeline" in vars(m)
        ]
        if len(owners) != 1:
            raise RuntimeError(f"Expected one MFSDP gather pipeline per pipeline rank, found {len(owners)}")
        stats = bind(owners[0].all_gather_pipeline, (front_gib if stage == 0 else 22) * 2**30)
        get_logger(__name__).info(
            "PP%d keeps %d BF16 all-gather buckets (%.3f GiB) within each optimizer update",
            stage,
            len(stats["buckets"]),
            stats["bytes"] / 2**30,
        )
        return chunks

    model.get_bridge_distributed_model = build
