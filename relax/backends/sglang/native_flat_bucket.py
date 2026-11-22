# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Use SGLang's native flattened transport without duplicate receive
buffers."""

import inspect
from math import prod
from typing import Any

import torch
import torch.distributed as dist


def aligned_bucket(values: list[tuple[str, torch.Tensor]]) -> bool:
    offset = 0
    for _, value in values:
        if not value.is_contiguous() or not value.numel() or offset % value.element_size():
            return False
        offset += value.numel() * value.element_size()
    return bool(offset)


def receive_flat_bucket(self: Any, names: list[str], dtypes: list, shapes: list, group_name: str) -> tuple[bool, str]:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket, FlattenedTensorMetadata

    try:
        metadata, offset = [], 0
        for name, dtype, shape in zip(names, dtypes, shapes, strict=True):
            dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            nbytes = prod(shape) * dtype.itemsize
            if offset % dtype.itemsize:
                raise ValueError("Native flattened tensor offset is not dtype aligned")
            metadata.append(FlattenedTensorMetadata(name, torch.Size(shape), dtype, offset, offset + nbytes, nbytes))
            offset += nbytes
        flat = torch.empty(offset, dtype=torch.uint8, device=self.device)
        dist.broadcast(flat, src=0, group=self._model_update_group[group_name])
        bucket = FlattenedTensorBucket(flattened_tensor=flat, metadata=metadata)
        self.get_model().load_weights(bucket.reconstruct_tensors())
        return True, "Succeeded to update flattened model weights."
    except Exception as error:
        return False, f"Native flattened update failed; discard partially updated model weights: {error}"


def install_native_flat_bucket() -> None:
    """Patch only the recognized optional worker interface; others stay
    native."""
    try:
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket, FlattenedTensorMetadata
    except ImportError:
        return
    original = getattr(WeightUpdater, "_update_bucketed_weights_from_distributed", None)
    if getattr(original, "_relax_native_flat_bucket", False):
        return
    if (
        not callable(original)
        or original.__module__ != WeightUpdater.__module__
        or not all(hasattr(WeightUpdater, name) for name in ("get_model", "device", "_model_update_group"))
        or tuple(inspect.signature(original).parameters) != ("self", "names", "dtypes", "shapes", "group_name")
        or not getattr(FlattenedTensorBucket, "supports_multi_dtypes", False)
        or tuple(inspect.signature(FlattenedTensorBucket).parameters)
        != ("named_tensors", "flattened_tensor", "metadata")
        or tuple(inspect.signature(FlattenedTensorMetadata).parameters)
        != ("name", "shape", "dtype", "start_idx", "end_idx", "numel")
    ):
        return
    receive_flat_bucket._relax_native_flat_bucket = True
    WeightUpdater._update_bucketed_weights_from_distributed = receive_flat_bucket


def native_flat_bucket_supported(model: torch.nn.Module) -> bool:
    guard = getattr(model, "_relax_native_weight_guard", None)
    manager = getattr(guard, "manager", None)
    selector = getattr(manager, "_select_runners", None)
    runners = (
        [runner for _, runner in selector("all")]
        if callable(selector)
        else [getattr(getattr(manager, "tp_worker", None), "model_runner", None)]
    )
    for runner in runners:
        updater = getattr(runner, "weight_updater", None)
        getter = getattr(updater, "get_model", None)
        if callable(getter) and getter() is model:
            method = getattr(updater, "_update_bucketed_weights_from_distributed", None)
            return bool(getattr(method, "_relax_native_flat_bucket", False))
    return False
