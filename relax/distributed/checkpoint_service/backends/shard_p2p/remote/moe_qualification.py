# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Agree support or native fallback before entering payload collectives."""

import socket
from typing import Any

import torch.distributed as dist

from relax.utils.env import Envs

from .common import enabled, weight_update_modes


def _validate_source_configuration(backend: Any, world_size: int) -> None:
    args = backend.args
    sizes = (
        world_size,
        args.tensor_model_parallel_size,
        args.rollout_num_gpus_per_engine,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        args.num_experts,
    )
    if (
        any(type(value) is not int or value <= 0 for value in sizes)
        or world_size % (args.tensor_model_parallel_size * args.context_parallel_size)
        or world_size % args.expert_model_parallel_size
        or world_size % args.rollout_num_gpus_per_engine
        or args.expert_model_parallel_size < 2
        or args.expert_tensor_parallel_size != 1
        or args.num_experts % args.expert_model_parallel_size
        or args.pipeline_model_parallel_size != 1
        or getattr(args, "virtual_pipeline_model_parallel_size", None) is not None
        or not getattr(args, "true_on_policy_mode", False)
        or backend._lora_enabled
        or not backend._use_bridge
        or backend.quantization_config is not None
    ):
        raise NotImplementedError(
            "Remote expert shards require on-policy BF16 Bridge, complete TP/CP and EP groups, EP>=2, ETP1 and PP1"
        )


def _validate_pipeline_configuration(backend: Any) -> None:
    if backend.args.pipeline_model_parallel_size != 1:
        raise NotImplementedError("Pipeline stages use native Gather/broadcast in this integration")


def qualify_before_native_reference(backend: Any, group: Any) -> bool:
    """Unsupported layouts agree on native; inconsistent configuration
    fails."""
    world_size = dist.get_world_size(group)
    record, error = None, None
    try:
        args = backend.args
        requested = enabled(args)
        config = (
            (
                args.tensor_model_parallel_size,
                args.rollout_num_gpus_per_engine,
                args.expert_model_parallel_size,
                args.expert_tensor_parallel_size,
                getattr(args, "num_experts", None),
                args.context_parallel_size,
                args.pipeline_model_parallel_size,
                args.update_weight_buffer_size,
                getattr(args, "true_on_policy_mode", False),
                backend._lora_enabled,
                backend._use_bridge,
                repr(backend.quantization_config),
                Envs.RELAX_WEIGHT_SYNC_SHARED_DIR,
                Envs.RELAX_P2P_AUDIT_DIR,
            )
            if requested
            else ()
        )
        supported = requested
        if requested:
            if type(args.update_weight_buffer_size) is not int or args.update_weight_buffer_size <= 0:
                raise ValueError("Weight-transfer bucket budget must be a positive integer")
            try:
                _validate_source_configuration(backend, world_size)
            except NotImplementedError:
                supported = False
        record = (requested, config, supported, socket.gethostname())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    records = [None] * world_size
    dist.all_gather_object(records, (record, error), group=group)
    if any(problem for _, problem in records):
        raise RuntimeError(f"Shard transfer qualification failed: {[problem for _, problem in records]}")
    rows = [record for record, _ in records]
    if len({row[:2] for row in rows}) != 1:
        raise RuntimeError("Actor ranks disagree on the shard transfer configuration")
    if not all(row[2] for row in rows) or len({row[3] for row in rows}) != 1:
        return False
    supported = [False]
    if dist.get_rank(group) == 0:
        try:
            weight_update_modes(getattr(backend, "rollout_topology", None))
            supported[0] = True
        except NotImplementedError:
            pass
    dist.broadcast_object_list(supported, src=dist.get_global_rank(group, 0), group=group)
    return supported[0]
