# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Synchronize FSDPTurbo weights to colocated SGLang engines.

Ordinary tensors are gathered and bucketed; replicated CPU PLE tables are
streamed as HF row shards. Both use per-engine Gloo gather and Ray IPC with
Megatron's ``flattened_bucket`` protocol. Colocate-only deployments have no
distributed-broadcast path.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor

from relax.backends.megatron.sglang import FlattenedTensorBucket, MultiprocessingSerializer
from relax.utils.device import make_current_torch_device
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# FSDPTurbo prefixes HF parameter names with ``model.``; SGLang expects plain HF names.
_TURBO_PARAM_PREFIX = "model."
_PLE_TABLE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight"

# NPU IPC handles can fault during engine weight loads. FSDPTURBO_SYNC_CPU_SHM=1
# instead sends host tensors via /dev/shm and lets the engine copy them to device.
_SYNC_CPU_SHM = bool(int(os.environ.get("FSDPTURBO_SYNC_CPU_SHM", "0") or 0))


def _strip_turbo_param_prefix(name: str) -> str:
    return name[len(_TURBO_PARAM_PREFIX) :] if name.startswith(_TURBO_PARAM_PREFIX) else name


def _serialize_payload(tensor_data: dict) -> str:
    """Serialize a bucket, using /dev/shm names for CPU shared memory.

    ``file_system`` sharing survives Gloo gather and Ray hops; raw file
    descriptors do not.
    """
    if not _SYNC_CPU_SHM:
        return MultiprocessingSerializer.serialize(tensor_data, output_str=True)
    prev_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")
    try:
        return MultiprocessingSerializer.serialize(tensor_data, output_str=True)
    finally:
        torch.multiprocessing.set_sharing_strategy(prev_strategy)


def _iter_weight_chunks(
    named_params_getter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    buffer_size_bytes: int,
    device: torch.device,
):
    """Yield full-tensor chunks bounded by ``buffer_size_bytes``.

    ``full_tensor()`` is collective, so every rank must walk parameters in the
    same deterministic order.
    """
    chunk: list[tuple[str, torch.Tensor]] = []
    chunk_bytes = 0
    for raw_name, param in named_params_getter():
        name = _strip_turbo_param_prefix(raw_name)
        if name.endswith(_PLE_TABLE_SUFFIX):
            continue
        tensor = param.detach()
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        if tensor.device != device:
            tensor = tensor.to(device)
        tensor = tensor.contiguous()
        item_bytes = tensor.numel() * tensor.element_size()
        if chunk and chunk_bytes + item_bytes > buffer_size_bytes:
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append((name, tensor))
        chunk_bytes += item_bytes
    if chunk:
        yield chunk


def _iter_ple_weight_chunks(
    named_params_getter: Callable[[], Iterable[tuple[str, torch.Tensor]]],
    buffer_size_bytes: int,
    num_shards: int,
    group_rank: int,
    sync_width: int,
):
    """Stream each HF shard to the rollout TP ranks whose vocabulary ranges
    overlap."""
    for raw_name, param in named_params_getter():
        name = _strip_turbo_param_prefix(raw_name)
        if not name.endswith(_PLE_TABLE_SUFFIX):
            continue
        if isinstance(param, DTensor) or param.device.type != "cpu":
            raise ValueError("PLE weight sync currently requires a replicated CPU table; VP tables are unsupported.")
        if num_shards < 1 or sync_width < 1:
            raise ValueError("PLE shard count and synchronization width must be positive")
        rows = param.size(0)
        shard_rows = (rows + num_shards - 1) // num_shards
        shard_bytes = min(rows, shard_rows) * param.size(1) * param.element_size()
        if shard_bytes > buffer_size_bytes:
            raise ValueError(
                f"PLE HF shard requires {shard_bytes} bytes, exceeding update_weight_buffer_size={buffer_size_bytes}"
            )
        table = param.detach()
        # SGLang 0.5.22 pads the original vocabulary to 64 rows, increasing
        # the padding multiple by TP size when needed for an even partition.
        padding = 64
        padded_rows = ((rows + padding - 1) // padding) * padding
        if padded_rows % sync_width:
            padding *= sync_width
            padded_rows = ((rows + padding - 1) // padding) * padding
        partition_rows = padded_rows // sync_width
        tp_start = min(group_rank * partition_rows, rows)
        tp_end = min((group_rank + 1) * partition_rows, rows)
        for shard in range(num_shards):
            start = shard * shard_rows
            end = min(start + shard_rows, rows)
            if group_rank < 0 or group_rank >= sync_width or max(start, tp_start) >= min(end, tp_end):
                yield []
                continue
            shard_name = f"{name.removesuffix('.weight')}.shard_{shard}.weight"
            yield [(shard_name, table[start:end])]


class FSDPTurboUpdateWeight:
    """Pushes FSDPTurbo-trained HF weights into colocated SGLang engines."""

    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0
        self._ipc_gather_group = None
        self._ipc_group_ranks: tuple[tuple[int, ...], ...] | None = None
        self._ipc_gather_src: int | None = None
        self._ipc_engine: ActorHandle | None = None
        self.rollout_engines: list[ActorHandle] = []

    # ------------------------------------------------------------------
    # Engine connection
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Map training ranks to colocated engines and create Gloo groups.

        Every engine must share a Ray node with a training actor.
        """
        self.rollout_engines = list(rollout_engines)

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for count in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += count

        engine_node_ids = [
            info.get("node_id", "")
            for info in ray.get([engine.get_pid_and_node_id.remote() for engine in rollout_engines])
        ]
        local_actor_node_id = ray.get_runtime_context().get_node_id()
        gathered_actor_node_ids: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_actor_node_ids, local_actor_node_id, group=get_gloo_group())
        actor_node_id_set = {node_id for node_id in gathered_actor_node_ids if node_id}
        for engine_node_id in engine_node_ids:
            if not engine_node_id or engine_node_id not in actor_node_id_set:
                raise RuntimeError(
                    "fsdp_turbo weight sync requires all rollout engines to be colocated with the train actors; "
                    f"found engine on node {engine_node_id!r} outside the actor node set."
                )

        group_ranks = tuple(
            tuple(range(offset, offset + count)) for offset, count in zip(engine_gpu_offsets, engine_gpu_counts)
        )
        if self._ipc_group_ranks is not None and group_ranks != self._ipc_group_ranks:
            raise RuntimeError("fsdp_turbo weight sync does not support changing IPC rank topology during recovery")
        if self._ipc_group_ranks is None:
            for ranks in group_ranks:
                new_group = dist.new_group(ranks=list(ranks), backend="gloo")
                if dist.get_rank() in ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = ranks[0]
            self._ipc_group_ranks = group_ranks

        self._ipc_engine = None
        for index, engine in enumerate(self.rollout_engines):
            start = engine_gpu_offsets[index]
            end = start + engine_gpu_counts[index]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def _check_errors(self, error: str | None) -> None:
        """Fail every training rank when an engine leader's HTTP call fails."""
        errors = [None] * dist.get_world_size(group=get_gloo_group())
        dist.all_gather_object(errors, error, group=get_gloo_group())
        failures = [message for message in errors if message]
        if failures:
            raise RuntimeError("FSDPTurbo weight sync failed: " + " | ".join(failures))

    def _call_engines(self, method: str, **kwargs) -> None:
        error = None
        if dist.get_rank() == 0:
            try:
                results = ray.get([getattr(engine, method).remote(**kwargs) for engine in self.rollout_engines])
                for result in results:
                    if isinstance(result, dict) and result.get("success") is False:
                        raise RuntimeError(str(result))
            except Exception as exc:
                error = f"{method}: {exc}; {getattr(exc, '__notes__', [])}"
        self._check_errors(error)

    @torch.no_grad()
    def update_weights(self) -> None:
        """version++, pause engines, stream full-tensor chunks over IPC."""
        self.weight_version += 1
        rank = dist.get_rank()

        self._call_engines("pause_generation")
        self._call_engines("flush_cache")
        self._call_engines("begin_weight_update")

        device = make_current_torch_device()
        buffer_size_bytes = int(getattr(self.args, "update_weight_buffer_size", 1024**3) or (1024**3))
        hf_config = getattr(getattr(self.model, "model", self.model), "config", None)
        text_config = getattr(hf_config, "text_config", hf_config)
        num_ple_shards = int(getattr(text_config, "split_ngram_parts", getattr(hf_config, "split_ngram_parts", 512)))
        group_rank = dist.get_rank(self._ipc_gather_group) if self._ipc_gather_group is not None else -1
        sync_width = dist.get_world_size(self._ipc_gather_group) if self._ipc_gather_group is not None else 1
        chunks = chain(
            _iter_weight_chunks(lambda: self.model.named_parameters(), buffer_size_bytes, device),
            _iter_ple_weight_chunks(
                lambda: self.model.named_parameters(),
                buffer_size_bytes,
                num_ple_shards,
                group_rank,
                sync_width,
            ),
        )
        num_chunks = 0
        for chunk in chunks:
            chunk_tensors = []
            error = None
            try:
                chunk_tensors = self._send_chunk_to_engine(chunk, device)
            except Exception as exc:
                error = f"chunk {num_chunks}: {exc}; {getattr(exc, '__notes__', [])}"
            self._check_errors(error)
            num_chunks += 1
            # The source waits for engine consumption in _send_chunk_to_engine;
            # fence the whole group before releasing backing tensors. Retaining
            # every chunk until the final barrier would hold the model per rank.
            if self._ipc_gather_group is not None:
                dist.barrier(group=self._ipc_gather_group)
            del chunk_tensors
            del chunk

        dist.barrier(group=get_gloo_group())
        self._call_engines("end_weight_update")
        self._call_engines("update_weight_version", weight_version=str(self.weight_version))
        self._call_engines("continue_generation")
        dist.barrier(group=get_gloo_group())
        if rank == 0:
            logger.info(f"[fsdp_turbo] weight sync v{self.weight_version} done ({num_chunks} IPC chunks)")

    def _send_chunk_to_engine(
        self,
        hf_named_tensors: list[tuple[str, torch.Tensor]],
        device: torch.device,
    ) -> list[dict]:
        """Serialize and gather one chunk, then send it to the local engine.

        Return payloads whose backing memory must stay alive until the caller's
        engine-group barrier confirms consumption.
        """
        if self._ipc_engine is None or self._ipc_gather_group is None:
            return []

        serialized_tensors = []
        long_live_tensors = []
        error = None
        try:
            hf_named_tensors = [(name, tensor.to(device).contiguous()) for name, tensor in hf_named_tensors]
            if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
                serialized_by_dtype: dict[Any, list[tuple[str, torch.Tensor]]] = (
                    {"dtype": hf_named_tensors} if hf_named_tensors else {}
                )
            else:
                serialized_by_dtype = {}
                for name, tensor in hf_named_tensors:
                    serialized_by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            # Serialized handles reference this memory; keep it alive until the
            # engine confirms the load to avoid dereferencing freed storage.
            for named_tensors in serialized_by_dtype.values():
                bucket = FlattenedTensorBucket(named_tensors=named_tensors)
                flat = bucket.get_flattened_tensor()
                if _SYNC_CPU_SHM:
                    # Keep the /dev/shm file alive until the engine confirms its H2D copy.
                    flat = flat.cpu()
                tensor_data = {"flattened_tensor": flat, "metadata": bucket.get_metadata()}
                long_live_tensors.append(tensor_data)
                serialized_tensors.append(_serialize_payload(tensor_data))

        except Exception as exc:
            error = str(exc)
        errors = [None] * dist.get_world_size(self._ipc_gather_group)
        dist.all_gather_object(errors, error, group=self._ipc_gather_group)
        if any(message is not None for message in errors):
            raise RuntimeError(f"Weight payload preparation failed: {errors}")

        gather_src = self._ipc_gather_src
        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group) if gather_src == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=gather_src,
            group=self._ipc_gather_group,
        )

        refs: list[ObjectRef] = []
        if gather_src == dist.get_rank():
            num_buckets = max((len(tensors) for tensors in serialized_named_tensors), default=0)
            empty_serialized_tensor = None
            for bucket_index in range(num_buckets):
                serialized_for_bucket = []
                for tensors in serialized_named_tensors:
                    if bucket_index < len(tensors):
                        serialized_for_bucket.append(tensors[bucket_index])
                        continue
                    if empty_serialized_tensor is None:
                        empty_data = {
                            "flattened_tensor": torch.empty(
                                0, dtype=torch.uint8, device=None if _SYNC_CPU_SHM else device
                            ),
                            "metadata": [],
                        }
                        long_live_tensors.append(empty_data)
                        empty_serialized_tensor = _serialize_payload(empty_data)
                    serialized_for_bucket.append(empty_serialized_tensor)
                refs.append(
                    self._ipc_engine.update_weights_from_tensor.remote(
                        serialized_named_tensors=serialized_for_bucket,
                        load_format="flattened_bucket",
                    )
                )
            ray.get(refs)
        # Keep only the payloads alive; the refs were consumed above.
        return long_live_tensors
