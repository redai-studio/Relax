# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Use SGLang's native flattened transport without duplicate receive
buffers."""

import inspect
from math import prod
from typing import Any

import torch
import torch.distributed as dist


def aligned_bucket(values: list[tuple[str, torch.Tensor]]) -> bool:
    offset_bytes = 0
    for _, value in values:
        if not value.is_contiguous() or not value.numel() or offset_bytes % value.element_size():
            return False
        offset_bytes += value.numel() * value.element_size()
    return bool(offset_bytes)


def _receive_flat_bucket(
    self: Any, names: list[str], dtypes: list, shapes: list, group_name: str
) -> list[tuple[str, torch.Tensor]]:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket, FlattenedTensorMetadata

    metadata, offset_bytes = [], 0
    for name, dtype, shape in zip(names, dtypes, shapes, strict=True):
        dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        nbytes = prod(shape) * dtype.itemsize
        if offset_bytes % dtype.itemsize:
            raise ValueError("Native flattened tensor byte offset is not dtype aligned")
        metadata.append(
            FlattenedTensorMetadata(name, torch.Size(shape), dtype, offset_bytes, offset_bytes + nbytes, nbytes)
        )
        offset_bytes += nbytes
    flat = torch.empty(offset_bytes, dtype=torch.uint8, device=self.device)
    dist.broadcast(flat, src=0, group=self._model_update_group[group_name])
    return FlattenedTensorBucket(flattened_tensor=flat, metadata=metadata).reconstruct_tensors()


def receive_flat_bucket(self: Any, names: list[str], dtypes: list, shapes: list, group_name: str) -> tuple[bool, str]:
    try:
        weights = _receive_flat_bucket(self, names, dtypes, shapes, group_name)
        self.get_model().load_weights(weights)
        return True, "Succeeded to update flattened model weights."
    except Exception as error:
        return False, f"Native flattened update failed; discard partially updated model weights: {error}"


def _receiver_name(updater: Any) -> str:
    # SGLang now receives once and loads the returned tensors on the selected
    # runners in the scheduler. Patch only the receive side on that interface.
    if callable(getattr(updater, "receive_weights_from_distributed", None)):
        return "_receive_bucketed_weights_from_distributed"
    return "_update_bucketed_weights_from_distributed"


def install_native_flat_bucket() -> None:
    """Patch only the recognized optional worker interface; others stay
    native."""
    try:
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket, FlattenedTensorMetadata
    except ImportError:
        return
    name = _receiver_name(WeightUpdater)
    original = getattr(WeightUpdater, name, None)
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
    replacement = _receive_flat_bucket if name == "_receive_bucketed_weights_from_distributed" else receive_flat_bucket
    replacement._relax_native_flat_bucket = True
    setattr(WeightUpdater, name, replacement)


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
            method = getattr(updater, _receiver_name(updater), None)
            return bool(getattr(method, "_relax_native_flat_bucket", False))
    return False
