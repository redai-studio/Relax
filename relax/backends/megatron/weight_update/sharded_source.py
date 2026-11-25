# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from verl, Copyright 2025 Bytedance Ltd. and/or its affiliates, Apache-2.0.

"""Bind verl's local Megatron probes and sparse gather to the selected Relax
Actor weights."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from itertools import islice
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from .shard_snapshot import CudaShardInputs
from .streamed_row_projection import UnsupportedStreamedLayout
from .verl_gather_queue import _GatherQueue
from .verl_projection import build_export_index, mcore_hf_delta_entry
from .verl_sparse_gather import SparseGatherTraffic, shard_delta_indices


@contextmanager
def _agree_on_projection(metadata_group, flag: torch.Tensor) -> Iterator[None]:
    """Complete all local probes before any rank enters a payload gather."""
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    world = dist.get_world_size(metadata_group)
    if world == 1:
        if error is not None:
            raise error
        return
    flag.fill_(int(error is not None))
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=metadata_group)
    if flag.item():
        errors = [None] * world
        message = f"{type(error).__name__}: {error}" if error is not None else None
        dist.all_gather_object(errors, message, group=metadata_group)
        raise RuntimeError(f"Source projection failed before sparse gather: {errors}") from error


class BridgeShardSource:
    def __init__(self, iterator: Any) -> None:
        self.iterator = iterator
        self.schema: dict = {}
        self.index: list | None = None
        self.keys: list[str | None] = []
        self.dtypes: list[torch.dtype] = []
        self.output_dtypes: list[torch.dtype] = []
        self.metrics: dict = {}
        self.native_only_reason: str | None = None
        self.group = None

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
            self.native_only_reason = None
        self.schema = schema
        self.metrics = {"native_full_reason": self.native_only_reason} if self.native_only_reason else {}

    def _postprocess(self, name: str, outputs: dict) -> dict:
        from relax.backends.megatron.weight_conversion.processors import remove_padding

        cropped = {key: remove_padding(name, tensor, self.iterator.args.vocab_size) for key, tensor in outputs.items()}
        return dict(self.iterator._bridge_converter.postprocess_hf_outputs(name, cropped))

    def prepare(self, metadata_group) -> None:
        if (self.index is not None or self.native_only_reason is not None) and self.group is metadata_group:
            return
        from megatron.bridge import AutoBridge

        from relax.utils.megatron_bridge_utils import patch_megatron_model

        from .common import named_params_and_buffers

        model, args = self.iterator.model, self.iterator.args
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.iterator._bridge_converter.init_tasks()
        self.native_only_reason = None
        try:
            with patch_megatron_model(model):
                index = build_export_index(
                    bridge,
                    model,
                    {},
                    metadata_group=metadata_group,
                    postprocess=self._postprocess,
                    postprocess_preserves_bits=self.iterator._bridge_converter.postprocess_preserves_bits,
                    native_schema=self.schema,
                )
        except UnsupportedStreamedLayout as exc:
            # The native FULL just completed. Preserve that exact stream on subsequent versions.
            self.index, self.group, self.native_only_reason = None, metadata_group, str(exc)
            self.metrics = {"native_full_reason": str(exc)}
            return
        tensors = list(
            named_params_and_buffers(args, model, convert_to_global_name=False, include_persistent_buffers=True)
        )
        by_object = {id(t): key for key, t in tensors}

        def identity(t):
            return t.device, t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype

        by_storage = {identity(t): key for key, t in tensors if t.numel()}
        keys, local_types = [], []
        lookup_error = None
        for rec in index:
            if rec.param is None:
                keys.append(None)
                local_types.append(None)
                continue
            key = by_object.get(id(rec.param)) or by_storage.get(identity(rec.param))
            if key is None and lookup_error is None:
                lookup_error = f"No Actor-local weight for Bridge task {rec.megatron_name}"
            keys.append(key)
            local_types.append(rec.param.dtype)
        types = [None] * dist.get_world_size(metadata_group)
        dist.all_gather_object(types, (lookup_error, local_types), group=metadata_group)
        errors = [message for message, _ in types if message]
        if errors:
            raise ValueError(f"Source native FULL directory has missing inputs: {errors}")
        dtypes = []
        for column in zip(*(rank_types for _, rank_types in types), strict=True):
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

    def _project_local(self, i, current, previous, device, root):
        rec, key, dtype = self.index[i], self.keys[i], self.dtypes[i]
        base = None
        if rec.param is not None and rec.spec.contributes and (rec.spec.gather_group is not None or root):
            now = self._local(current, key, device)
            base = self._local(previous, key, device)
            if now.shape != base.shape or now.dtype != base.dtype or now.numel() != rec.param.numel():
                raise ValueError(f"Local snapshot geometry changed for {rec.megatron_name}")
            indices, values = shard_delta_indices(now, base, 0)
        else:
            indices = torch.empty(0, dtype=torch.int64, device=device)
            values = torch.empty(0, dtype=dtype, device=device)
        slots, _, counts, idx, val = mcore_hf_delta_entry(
            rec, 0, indices, values, {}, previous_shard=base, wire_dtype=self.output_dtypes[i]
        )
        return slots, counts, idx, val

    def _batches(self, window_budget, current, previous, device, group, control, stats):
        start = 0
        for i, rec in enumerate(self.index):
            if rec.streamed_rows is None:
                if i + 1 - start == 32:
                    yield range(start, i + 1), None, 0
                    start = i + 1
                continue
            if start < i:
                yield range(start, i), None, 0
            table = rec.streamed_rows
            windows = table.windows(window_budget, self.dtypes[i].itemsize, dist.get_world_size(group))
            count_batch = max(1, min(128, window_budget // 32))
            while batch := list(islice(windows, count_batch)):
                started = perf_counter()
                with _agree_on_projection(group, control):
                    counts = table.change_counts(
                        batch,
                        current,
                        previous,
                        self.keys[i],
                        self.dtypes[i],
                        device,
                        rec.spec.contributes,
                        control.device,
                    )
                # Exactly one rank owns each window. All Actors can skip unchanged windows together.
                dist.all_reduce(counts, op=dist.ReduceOp.MAX, group=group)
                stats["count_input_bytes"] += counts.nbytes
                amounts = counts.cpu().tolist()
                del counts
                stats["prescan_host_wall_s"] += perf_counter() - started
                stats["windows"] += len(batch)
                for window, amount in zip(batch, amounts, strict=True):
                    if amount:
                        stats["changed_windows"] += 1
                        yield range(i, i + 1), window, amount
            start = i + 1
        if start < len(self.index):
            yield range(start, len(self.index)), None, 0

    def chunks(
        self,
        current,
        previous,
        *,
        full_names,
        metadata_group,
        device,
        max_round_bytes,
        staging_budget_bytes,
        gpu_budget_bytes=None,
    ):
        from .delta_export import DeltaWeight

        if self.index is None or full_names:
            raise RuntimeError("The verl sparse source needs its acknowledged native FULL seed")
        ranks = dist.get_process_group_ranks(metadata_group)
        rank = dist.get_rank(metadata_group)
        root = rank == 0
        control = torch.zeros(
            1, dtype=torch.int32, device=device if dist.get_backend(metadata_group) == "nccl" else "cpu"
        )
        limit = min(max_round_bytes, staging_budget_bytes)
        projection_s = gather_s = 0.0
        group_stats = {}
        traffic = defaultdict(SparseGatherTraffic)
        streamed = dict(
            windows=0, changed_windows=0, prescan_host_wall_s=0.0, count_input_bytes=0, tp_sum_input_bytes=0
        )

        # Every Actor walks the same directory. Only the wire publisher retains gathered slots.
        for batch, window, amount in self._batches(
            min(limit, gpu_budget_bytes or limit), current, previous, device, metadata_group, control, streamed
        ):
            chunks: list[list] = []
            used = 0

            def consume(name, dtype_str, shape, idx, values):
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

            queue = _GatherQueue(max_round_bytes, root, consume, traffic)
            started = perf_counter()
            if window is not None:
                i = batch.start
                rec = self.index[i]
                table = rec.streamed_rows
                with _agree_on_projection(metadata_group, control):
                    changes = table.local_changes(
                        window, current, previous, self.keys[i], self.dtypes[i], device, rec.spec.contributes, amount
                    )
            with _agree_on_projection(metadata_group, control):
                for i in batch:
                    rec = self.index[i]
                    group = rec.spec.gather_group
                    wire_dtype = self.output_dtypes[i]
                    if window is None:
                        slots, counts, idx, val = self._project_local(i, current, previous, device, root)
                    else:
                        slots, counts, idx, val, native_bytes = table.native_entry(
                            window,
                            changes,
                            rec.spec.contributes,
                            amount,
                            lambda: _agree_on_projection(metadata_group, control),
                        )
                        streamed["tp_sum_input_bytes"] += native_bytes
                        del changes
                    totals = group_stats.setdefault(group, [0, 0, 0])
                    totals[0] += 1
                    totals[1] += idx.nbytes
                    totals[2] += val.nbytes
                    queue.put(group, slots, str(wire_dtype).removeprefix("torch."), counts, idx, val)
                projection_s += perf_counter() - started
            started = perf_counter()
            queue.flush_all()
            gather_s += perf_counter() - started
            control.fill_(len(chunks) if root else 0)
            dist.broadcast(control, src=ranks[0], group=metadata_group)
            for j in range(int(control.item())):
                yield chunks[j] if root else []
            chunks.clear()
        self.metrics = dict(
            source="verl_delta_sharded",
            source_global_rank=ranks[rank],
            directory_rows=len(self.index),
            hf_slots=len(self.schema),
            projection_host_wall_s=projection_s + streamed["prescan_host_wall_s"],
            gather_host_wall_s=gather_s,
            streamed_rows=streamed,
            groups=[
                dict(
                    ranks=dist.get_process_group_ranks(group) if group is not None else [],
                    rows=stats[0],
                    local_index_bytes=stats[1],
                    local_value_bytes=stats[2],
                    **asdict(traffic[group]),
                )
                for group, stats in group_stats.items()
            ],
        )
