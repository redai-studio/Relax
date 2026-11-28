# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backport the DP checkpoint padding fix to supported older Megatron
images."""

import ast
import hashlib
import inspect
import textwrap
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_LEGACY_SERIALIZER = "da2acf9258b00ce8793e300dd8dafbde018186ce54e1133846a6a24dbe199d86"
_FIXED_SERIALIZER = "1f5d00668260f82280db0603a3dee22138d13bc9da1c7212d0a42efe962f00ff"


def _serializer_fingerprint(method: Any) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    # Python 3.12 adds an empty type_params field to ordinary functions.
    for node in ast.walk(tree):
        if getattr(node, "type_params", None) == []:
            node._fields = tuple(field for field in node._fields if field != "type_params")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _patch_dp_reshardable_checkpoint(optimizer_cls: type) -> bool:
    original = getattr(optimizer_cls, "sharded_param_state_dp_reshardable", None)
    if original is None or original is _sharded_param_state_dp_reshardable:
        return False
    if hasattr(original, "__wrapped__"):
        logger.warning("Preserving an externally wrapped Megatron DP checkpoint serializer")
        return False
    try:
        fingerprint = _serializer_fingerprint(original)
    except (OSError, TypeError, SyntaxError):
        logger.warning("Cannot identify the Megatron DP checkpoint serializer; preserving its implementation")
        return False
    if fingerprint == _FIXED_SERIALIZER:
        return False
    if fingerprint != _LEGACY_SERIALIZER:
        logger.warning(
            "Unrecognized Megatron DP checkpoint serializer %s.%s (AST SHA256=%s); preserving its implementation",
            original.__module__,
            original.__qualname__,
            fingerprint,
        )
        return False
    optimizer_cls.sharded_param_state_dp_reshardable = _sharded_param_state_dp_reshardable
    logger.info("Applied compatibility fix for dp_reshardable checkpoint padding")
    return True


def patch_dp_reshardable_checkpoint() -> bool:
    """Install once per actor, without changing installed Megatron source
    files.

    Only the known older serializer is replaced. Images containing the 20261028
    DP checkpoint patch, newer implementations, and external wrappers keep
    their own implementation.
    """
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    return _patch_dp_reshardable_checkpoint(DistributedOptimizer)


def _sharded_param_state_dp_reshardable(
    self: Any, model_sharded_state_dict: dict, is_loading: bool = False, metadata: dict | None = None
) -> dict:
    """Keep native bucket keys and tensor aliases, excluding DP tail padding.

    This is the runtime equivalent of 20261028-dp-reshardable-checkpoint.patch.
    Only tensor dtype schemas are exchanged; parameter and optimizer tensors
    stay on their original devices and are never gathered or copied here.
    """
    import torch
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor

    group = self.data_parallel_group
    rank, world_size = group.rank(), group.size()
    state = self.get_parameter_state_dp_reshardable()
    schemas = {}
    for gbuf_idx in range(len(self.gbuf_ranges)):
        for dtype, buckets in state[gbuf_idx].items():
            for bucket_idx, bucket in enumerate(buckets):
                schemas[gbuf_idx, dtype, bucket_idx] = {
                    tuple(
                        sorted(
                            (key, value.dtype)
                            for key, value in tensors.items()
                            if isinstance(value, torch.Tensor) and key != "step"
                        )
                    )
                    for tensors in bucket
                }
    peer_schemas = [schemas]
    if world_size > 1:
        peer_schemas = [None] * world_size
        torch.distributed.all_gather_object(peer_schemas, schemas, group=group)
    if any(peer.keys() != schemas.keys() for peer in peer_schemas):
        raise ValueError("Optimizer-state bucket keys differ across DP ranks")
    padding_schemas = {}
    for bucket_key in schemas:
        candidates = set().union(*(peer[bucket_key] for peer in peer_schemas))
        if len(candidates) != 1 or not next(iter(candidates)):
            raise ValueError(f"Invalid optimizer-state dtype schemas for bucket {bucket_key}")
        padding_schemas[bucket_key] = dict(next(iter(candidates)))

    prefix = f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
    replica_id = (self.distributed_optimizer_instance_id, 0, 0)
    for key in ("per_bucket_numel", "per_bucket_numel_unpadded"):
        state[key] = ShardedObject(
            f"{prefix}.{key}",
            state[key],
            (1,),
            (0,),
            replica_id=(self.distributed_optimizer_instance_id, 0, rank),
        )
    for gbuf_idx in range(len(self.gbuf_ranges)):
        for dtype, buckets in state[gbuf_idx].items():
            for bucket_idx, bucket_state in enumerate(buckets):
                bucket = self.buffers[gbuf_idx].buckets[bucket_idx]
                unpadded = bucket.numel_unpadded
                padded = bucket.grad_data.numel()
                if unpadded > padded or padded % world_size:
                    raise ValueError("Invalid DP optimizer bucket sizes")
                local_numel = padded // world_size
                local_end = max(0, min(local_numel, unpadded - rank * local_numel))
                bucket_key = f"{prefix}.gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket_idx}"

                if not bucket_state:
                    offset = min(rank * local_numel, unpadded)
                    # Keep even a zero-length list slot: dropping it shifts later
                    # buckets when the native DCP loader reconstructs the list.
                    buckets[bucket_idx] = [
                        {
                            "padding": LocalNonpersistentObject(True),
                            **{
                                key: ShardedTensor(
                                    f"{bucket_key}.{key}",
                                    torch.zeros(local_end, dtype=state_dtype, device=bucket.grad_data.device),
                                    state_dtype,
                                    (local_end,),
                                    (unpadded,),
                                    (offset,),
                                    axis_fragmentations=None,
                                    replica_id=replica_id,
                                )
                                for key, state_dtype in padding_schemas[gbuf_idx, dtype, bucket_idx].items()
                            },
                        }
                    ]
                    continue

                padding = {}
                for index in range(-1, len(bucket_state)):
                    start = 0 if index == -1 else bucket_state[index]["gbuf_local_end"]
                    end = (
                        local_numel if index == len(bucket_state) - 1 else bucket_state[index + 1]["gbuf_local_start"]
                    )
                    end = min(end, local_end)
                    if end > start:
                        padding[index + 1] = {
                            key: torch.empty(end - start, dtype=value.dtype, device=value.device)
                            for key, value in bucket_state[index].items()
                            if isinstance(value, torch.Tensor) and key != "step"
                        }
                        padding[index + 1].update(gbuf_local_start=start, gbuf_local_end=end, padding=True)
                for index in sorted(padding, reverse=True):
                    bucket_state.insert(index, padding[index])
                for tensors in bucket_state:
                    start = tensors.pop("gbuf_local_start")
                    end = tensors.pop("gbuf_local_end")
                    tensors.setdefault("padding", False)
                    for key, tensor in list(tensors.items()):
                        if key in ("padding", "step"):
                            tensors[key] = LocalNonpersistentObject(tensor)
                            continue
                        if tensor.shape != (end - start,):
                            raise ValueError(f"Optimizer-state shape does not match bucket range: {tensor.shape}")
                        tensors[key] = ShardedTensor(
                            f"{bucket_key}.{key}",
                            tensor,
                            tensor.dtype,
                            tensor.shape,
                            (unpadded,),
                            (rank * local_numel + start,),
                            axis_fragmentations=None,
                            replica_id=replica_id,
                        )
    return state
