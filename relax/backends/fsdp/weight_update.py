# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Full-transformer weight snapshot: deterministic DTensor gather + manifest.

The FSDP2 actor streams the *complete* trainable transformer to every SGLang
engine each optimizer step (synchronous colocate, design doc 10.3). This module
owns the two backend-specific pieces the transport needs:

* :class:`FullWeightChunkIterator` — walks the trainable parameters in a
  deterministic (name-sorted) order, materializes each shard via DTensor
  ``full_tensor()`` (a collective every rank enters together — the complete
  state dict is never aggregated on rank 0, and no whole-model Ray object is
  built), casts to the wire dtype, and yields fixed-size ``(name, tensor)``
  buckets (default 512 MiB).
* :func:`build_full_weight_manifest` — produces the :class:`FullWeightManifest`
  every engine validates (tensor count / bytes / ordered name-shape hash) before
  a version is committed.

The iterator is the object injected into the existing
``UpdateWeightFromTensor`` chunk loop as ``weight_chunk_iterator`` (design doc
8.3 / update_weight_from_tensor.py), so the FSDP path reuses the proven Ray IPC
/ engine fan-out / version-check machinery unchanged.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Tuple

import torch

from relax.models.generative import FullWeightManifest, ordered_name_shape_hash
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = [
    "FullWeightChunkIterator",
    "TensorSource",
    "build_full_weight_manifest",
    "iter_full_named_tensors",
    "materialize_full_tensor",
    "strip_transport_wrappers",
]

_WIRE_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}

# Activation checkpointing wraps each block in a ``CheckpointWrapper`` whose
# child module is named ``_checkpoint_wrapped_module``, so it appears in
# ``named_parameters()`` but NOT in ``state_dict()``. It is a train-side
# artifact with no counterpart on the engine.
_CHECKPOINT_WRAPPER_SEGMENT = "._checkpoint_wrapped_module."


def strip_transport_wrappers(name: str) -> str:
    """Remove wrappers that exist only in the train-side module graph.

    ``blocks.0._checkpoint_wrapped_module.attn.to_q.weight`` ->
    ``blocks.0.attn.to_q.weight``.

    This must be applied to EVERY name that goes on the wire. SGLang's weight
    loader skips names it does not recognise (``if name not in model_params:
    continue``) and still reports success, so an unstripped name silently drops
    that tensor: with ``--fsdp-activation-checkpointing`` on -- which both
    shipped recipes enable -- that is every transformer-block parameter, i.e.
    a weight sync that reports success and transfers nothing that matters.
    Neither the sent-count/bytes check nor ``ordered_name_shape_hash`` catches
    it, because both are computed from the same unstripped names.
    """
    return name.replace(_CHECKPOINT_WRAPPER_SEGMENT, ".") if _CHECKPOINT_WRAPPER_SEGMENT in name else name


def _wire_name(name: str, name_map: Optional[Callable[[str], str]]) -> str:
    """The engine-side name for a train-side parameter name.

    Single choke point shared by the manifest and the chunk iterator so the two
    cannot disagree: transport wrappers are stripped first, then the adapter's
    family-specific rename (e.g. dropping a ``transformer.`` prefix) runs.
    """
    out = strip_transport_wrappers(name)
    return name_map(out) if name_map is not None else out


# ``(name, param, gather_device) -> full tensor``. Overrides how a parameter is
# turned into the tensor that goes on the wire; the LoRA merged-sync path uses
# it to fold ``B @ A`` into the base weight (see
# :meth:`relax.backends.fsdp.lora.LoraSyncPlan.materialize`). Whatever it
# returns must keep the parameter's logical shape and stay at master width --
# the wire cast happens after it, in :func:`iter_full_named_tensors`.
TensorSource = Callable[[str, torch.Tensor, Optional[torch.device]], torch.Tensor]


def materialize_full_tensor(param: torch.Tensor, gather_device: Optional[torch.device] = None) -> torch.Tensor:
    """Return the unsharded tensor for a (possibly DTensor) parameter.

    ``full_tensor()`` is a collective — every rank must call it on the same
    parameter in the same order, which the deterministic name sort guarantees.
    Plain tensors (single-device / unsharded) are returned as-is.

    ``gather_device`` handles ``--fsdp-cpu-offload``: FSDP2's CPUOffloadPolicy
    keeps the DTensor shards on CPU, but the all-gather has no CPU process-group
    backend (``No backend type associated with device type cpu``). Moving the
    local shard to the GPU first makes the gather run over NCCL. No-op when the
    shard is already on ``gather_device`` (non-offload full-FT).
    """
    full_tensor = getattr(param, "full_tensor", None)
    if callable(full_tensor):
        if gather_device is not None and param.device.type != gather_device.type:
            param = param.to(gather_device)
        return param.full_tensor()
    return param


def _default_tensor_source(
    _name: str,
    param: torch.Tensor,
    gather_device: Optional[torch.device] = None,
) -> torch.Tensor:
    return materialize_full_tensor(param, gather_device)


def iter_full_named_tensors(
    named_params: List[Tuple[str, torch.Tensor]],
    *,
    wire_dtype: torch.dtype,
    name_map: Optional[Callable[[str], str]] = None,
    gather_device: Optional[torch.device] = None,
    tensor_source: Optional[TensorSource] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(mapped_name, full_tensor)`` in deterministic name-sorted order.

    ``name_map`` (the adapter's ``weight_name_map``) renames FSDP parameter
    names to the engine-side (HF) names; identity when omitted. Transport
    wrappers are stripped first by :func:`strip_transport_wrappers`.
    ``gather_device`` is forwarded to :func:`materialize_full_tensor` for CPU-
    offloaded shards. ``tensor_source`` overrides the materialization step (see
    :data:`TensorSource`); the wire cast below stays this function's job so a
    source that folds a LoRA delta in cannot accidentally round it away.
    """
    source = tensor_source if tensor_source is not None else _default_tensor_source
    for name, param in sorted(named_params, key=lambda kv: kv[0]):
        full = source(name, param, gather_device).detach()
        if full.dtype != wire_dtype:
            full = full.to(wire_dtype)
        yield _wire_name(name, name_map), full


def _ordered_named_shapes(
    named_params: List[Tuple[str, torch.Tensor]],
    name_map: Optional[Callable[[str], str]],
) -> List[Tuple[str, Tuple[int, ...]]]:
    ordered: List[Tuple[str, Tuple[int, ...]]] = []
    for name, param in sorted(named_params, key=lambda kv: kv[0]):
        # DTensor exposes the logical (global) shape via ``.shape`` already.
        ordered.append((_wire_name(name, name_map), tuple(int(s) for s in param.shape)))
    return ordered


def build_full_weight_manifest(
    named_params: List[Tuple[str, torch.Tensor]],
    *,
    model_family: str,
    task: str,
    policy_version: int,
    base_model_sha256: str,
    wire_dtype: str,
    bucket_size_bytes: int,
    name_map: Optional[Callable[[str], str]] = None,
) -> FullWeightManifest:
    """Build the deterministic full-weight manifest (design doc 10.2).

    Computes tensor count, total wire-size bytes and the ordered name-shape
    hash from parameter metadata only (no full-gather) so it is cheap to call
    on every rank; the values match what :class:`FullWeightChunkIterator` will
    stream.
    """
    if wire_dtype not in _WIRE_DTYPES:
        raise ValueError(f"Unsupported wire_dtype {wire_dtype!r}; expected one of {sorted(_WIRE_DTYPES)}.")
    elem_size = torch.empty(0, dtype=_WIRE_DTYPES[wire_dtype]).element_size()
    ordered = _ordered_named_shapes(named_params, name_map)
    tensor_count = len(ordered)
    total_bytes = 0
    for _name, shape in ordered:
        numel = 1
        for s in shape:
            numel *= int(s)
        total_bytes += numel * elem_size
    return FullWeightManifest(
        schema_version=1,
        model_family=model_family,
        task=task,
        policy_version=int(policy_version),
        base_model_sha256=base_model_sha256,
        tensor_count=tensor_count,
        total_bytes=total_bytes,
        wire_dtype=wire_dtype,
        bucket_size_bytes=int(bucket_size_bytes),
        ordered_name_shape_hash=ordered_name_shape_hash(ordered),
    )


class FullWeightChunkIterator:
    """Deterministic DTensor full-gather bucketed weight iterator.

    Iterating yields ``list[(name, tensor)]`` buckets, each capped at
    ``bucket_size_bytes`` (a single oversized tensor forms its own bucket). The
    order matches :func:`build_full_weight_manifest`, so a receiver replaying
    the same order reconstructs the identical ordered-name-shape hash.
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        *,
        wire_dtype: str = "bf16",
        bucket_size_bytes: int = 512 * 1024 * 1024,
        name_map: Optional[Callable[[str], str]] = None,
        gather_device: Optional[torch.device] = None,
        tensor_source: Optional[TensorSource] = None,
    ) -> None:
        if wire_dtype not in _WIRE_DTYPES:
            raise ValueError(f"Unsupported wire_dtype {wire_dtype!r}.")
        self._named_params = named_params
        self._wire_dtype = _WIRE_DTYPES[wire_dtype]
        self._bucket_size_bytes = int(bucket_size_bytes)
        self._name_map = name_map
        self._gather_device = gather_device
        self._tensor_source = tensor_source

    def __iter__(self) -> Iterator[List[Tuple[str, torch.Tensor]]]:
        bucket: List[Tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        for name, full in iter_full_named_tensors(
            self._named_params,
            wire_dtype=self._wire_dtype,
            name_map=self._name_map,
            gather_device=self._gather_device,
            tensor_source=self._tensor_source,
        ):
            nbytes = full.numel() * full.element_size()
            if bucket and bucket_bytes + nbytes > self._bucket_size_bytes:
                yield bucket
                bucket, bucket_bytes = [], 0
            bucket.append((name, full))
            bucket_bytes += nbytes
        if bucket:
            yield bucket
