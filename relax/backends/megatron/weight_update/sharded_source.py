# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from verl, Copyright 2025 Bytedance Ltd. and/or its affiliates, Apache-2.0.

"""Bind verl's local Megatron probes and sparse gather to the selected Relax
Actor weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from .shard_snapshot import CudaShardInputs
from .verl_gather_queue import _GatherQueue
from .verl_projection import build_export_index, mcore_hf_delta_entry
from .verl_sparse_gather import shard_delta_indices


class BridgeShardSource:
    def __init__(self, iterator: Any) -> None:
        self.iterator = iterator
        self.schema: dict = {}
        self.index: list | None = None
        self.keys: list[str | None] = []
        self.dtypes: list[torch.dtype] = []
        self.output_dtypes: list[torch.dtype] = []
        self.metrics: dict = {}

    def seed(self, shards: Mapping) -> Iterator:
        schema = {}
        for chunk in self.iterator.get_hf_weight_chunks(shards):
            for name, tensor in chunk:
                signature = (tuple(tensor.shape), tensor.dtype)
                if name in schema and schema[name] != signature:
                    raise ValueError(f"Conflicting native FULL slots for {name}")
                schema[name] = signature
            yield chunk
        if self.schema != schema:
            self.index = None
        self.schema = schema
        self.metrics = {}

    def _postprocess(self, name: str, outputs: dict) -> dict:
        from relax.backends.megatron.weight_conversion.processors import remove_padding

        cropped = {key: remove_padding(name, tensor, self.iterator.args.vocab_size) for key, tensor in outputs.items()}
        return dict(self.iterator._bridge_converter.postprocess_hf_outputs(name, cropped))

    def prepare(self, metadata_group) -> None:
        if self.index is not None and self.group is metadata_group:
            return
        from megatron.bridge import AutoBridge

        from relax.utils.megatron_bridge_utils import patch_megatron_model

        from .common import named_params_and_buffers

        model, args = self.iterator.model, self.iterator.args
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.iterator._bridge_converter.init_tasks()
        with patch_megatron_model(model):
            index = build_export_index(bridge, model, {}, metadata_group=metadata_group, postprocess=self._postprocess)
        tensors = list(
            named_params_and_buffers(args, model, convert_to_global_name=False, include_persistent_buffers=True)
        )
        by_object = {id(t): key for key, t in tensors}

        def identity(t):
            return t.device, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype

        by_storage = {identity(t): key for key, t in tensors if t.numel()}
        keys, local_types = [], []
        for rec in index:
            if rec.param is None:
                keys.append(None)
                local_types.append(None)
                continue
            key = by_object.get(id(rec.param)) or by_storage.get(identity(rec.param))
            if key is None:
                raise ValueError(f"No Actor-local weight for Bridge task {rec.megatron_name}")
            keys.append(key)
            local_types.append(rec.param.dtype)
        types = [None] * dist.get_world_size(metadata_group)
        dist.all_gather_object(types, local_types, group=metadata_group)
        dtypes = []
        for column in zip(*types, strict=True):
            owned = {dt for dt in column if dt is not None}
            if len(owned) != 1:
                raise ValueError("Bridge ranks disagree on a weight storage dtype")
            dtypes.append(owned.pop())
        projected = {n: tuple(shape) for rec in index for n, shape in rec.slots}
        native = {name: shape for name, (shape, _) in self.schema.items()}
        if native != projected:
            missing = sorted(self.schema.keys() - projected.keys())
            extra = sorted(projected.keys() - self.schema.keys())
            wrong = sorted(n for n in self.schema.keys() & projected.keys() if native[n] != projected[n])
            raise ValueError(
                f"Bridge projection differs from native FULL: missing={missing[:8]} extra={extra[:8]} geometry={[(n, self.schema[n], projected[n]) for n in wrong[:4]]}"
            )
        outputs = []
        for rec in index:
            types = {self.schema[name][1] for name, _ in rec.slots}
            if len(types) != 1:
                raise ValueError(f"One verl projection row must have one native output dtype: {rec.megatron_name}")
            outputs.append(types.pop())
        self.index, self.keys, self.dtypes, self.output_dtypes, self.group = (
            index,
            keys,
            dtypes,
            outputs,
            metadata_group,
        )

    @staticmethod
    def _local(mapping: Mapping, key: str, device) -> torch.Tensor:
        if isinstance(mapping, CudaShardInputs):
            mapping = mapping.shards
        return mapping[key].detach().to(device=device, non_blocking=True).contiguous().view(-1)

    def chunks(self, current, previous, *, full_names, metadata_group, device, max_round_bytes, staging_budget_bytes):
        from .delta_export import DeltaWeight

        if self.index is None or full_names:
            raise RuntimeError("The verl sparse source needs its acknowledged native FULL seed")
        root = dist.get_rank(metadata_group) == 0
        source_rank = dist.get_process_group_ranks(metadata_group)[0]
        limit = min(max_round_bytes, staging_budget_bytes)
        projection_s = gather_s = 0.0
        group_stats = {}
        rank_participation = {}

        # Every Actor walks the same directory. Only the wire publisher retains gathered slots.
        for start in range(0, len(self.index), 32):
            chunks: list[list] = []
            used = 0

            def consume(name, dtype_str, shape, volume, idx, values):
                nonlocal used
                if not idx.numel():
                    return
                dtype = getattr(torch, dtype_str)
                width = 4 + values.element_size()
                max_count = max(1, (limit - 16) // width)
                for lo in range(0, idx.numel(), max_count):
                    indices = idx[lo : lo + max_count]
                    data = values[lo : lo + max_count]
                    size = indices.numel() * width + 16
                    if not chunks or used + size > limit:
                        chunks.append([])
                        used = 0
                    chunks[-1].append(DeltaWeight(name, shape, dtype, indices, data))
                    used += size

            queue = _GatherQueue(32, max_round_bytes, root, consume)
            started = perf_counter()
            for i in range(start, min(start + 32, len(self.index))):
                rec, key, dtype = self.index[i], self.keys[i], self.dtypes[i]
                group = rec.spec.gather_group
                if key is not None and rec.spec.contributes and (group is not None or root):
                    now = self._local(current, key, device)
                    base = self._local(previous, key, device)
                    indices, values = shard_delta_indices(now, base, 0)
                else:
                    indices = torch.empty(0, dtype=torch.int64, device=device)
                    values = torch.empty(0, dtype=dtype, device=device)
                slots, _, counts, idx, val = mcore_hf_delta_entry(rec, 0, indices, values, {})
                wire_dtype = self.output_dtypes[i]
                val = val.to(wire_dtype)
                if group not in rank_participation:
                    rank_participation[group] = tuple(dist.get_process_group_ranks(group)) if group is not None else ()
                members = rank_participation[group]
                totals = group_stats.setdefault(members, [0, 0, 0])
                totals[0] += 1
                totals[1] += idx.numel() * idx.element_size()
                totals[2] += val.numel() * val.element_size()
                projected_at = perf_counter()
                projection_s += projected_at - started
                queue.put(group, slots, str(wire_dtype).removeprefix("torch."), counts, idx, val)
                started = perf_counter()
                gather_s += started - projected_at
            queue.flush_all()
            gather_s += perf_counter() - started
            count = [len(chunks) if root else 0]
            dist.broadcast_object_list(count, src=source_rank, group=metadata_group)
            for j in range(count[0]):
                yield chunks[j] if root else []
        self.metrics = dict(
            source="verl_delta_sharded",
            directory_rows=len(self.index),
            hf_slots=len(self.schema),
            projection_host_wall_s=projection_s,
            gather_host_wall_s=gather_s,
            groups=[
                dict(ranks=list(ranks), rows=stats[0], local_index_bytes=stats[1], local_value_bytes=stats[2])
                for ranks, stats in group_stats.items()
            ],
        )
