# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from verl (Copyright 2025 Bytedance Ltd. and/or its affiliates),
# licensed under the Apache License, Version 2.0.

"""Project acknowledged local Megatron shard changes into the native FULL HF
directory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger

from .shard_snapshot import CudaShardInputs


logger = get_logger(__name__)
Slot = tuple[str, tuple[int, ...], torch.dtype]
_WORD = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@dataclass(frozen=True)
class _Route:
    group: dist.ProcessGroup
    ranks: tuple[int, ...]
    lane: int
    lanes: int
    active: bool
    primary_replica: bool


@dataclass(frozen=True)
class _Row:
    info: Any
    slots: tuple[Slot, ...]
    expert: bool
    owners: tuple[int, ...]
    layout: Any
    fallback: str | None


def _source_tensor(shards, key, device):
    # Projection and comparison read these inputs without mutation. The FULL
    # converter retains its private copies for mappings that write in place.
    if isinstance(shards, CudaShardInputs):
        value = shards.active.get(key)
        if value is None:
            value = shards.shards[key]
    else:
        value = shards[key]
    return (
        value.detach()
        .view(_WORD[value.element_size()])
        .to(device=device, non_blocking=True)
        .contiguous()
        .view(value.dtype)
    )


def _changed_indices(current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """verl's shard_delta_indices, on same-width integer words (including NaN
    payloads)."""
    words = _WORD[current.element_size()]
    return (current.reshape(-1).view(words) != baseline.reshape(-1).view(words)).nonzero().flatten()


def _batch_slices(slices, contributions, device):
    """A single nonzero and boundary transfer per dtype for contiguous local HF
    regions."""
    for dtype, pieces in slices.items():
        words = _WORD[torch.empty((), dtype=dtype).element_size()]
        current = torch.cat([new.view(words) for _, new, _, _ in pieces])
        baseline = torch.cat([old.view(words) for _, _, old, _ in pieces])
        indices = (current != baseline).nonzero().flatten()
        values = current.index_select(0, indices).view(dtype)
        starts = [0]
        for _, new, _, _ in pieces:
            starts.append(starts[-1] + new.numel())
        boundaries = torch.tensor(starts, device=device, dtype=torch.int64)
        cuts = torch.searchsorted(indices, boundaries)
        # Translate the complete batch in one kernel sequence. Slot views below
        # share these sparse allocations until the count and payload gathers finish.
        offsets = torch.tensor([hf_start - start for start, (_, _, _, hf_start) in zip(starts, pieces)], device=device)
        offsets = torch.repeat_interleave(offsets, cuts.diff(), output_size=indices.numel())
        translated = (indices + offsets).int()
        cuts = cuts.cpu().tolist()
        for (slot, _, _, _), lo, hi in zip(pieces, cuts[:-1], cuts[1:], strict=True):
            contributions[slot] = (translated[lo:hi], values[lo:hi])


class BridgeShardSource:
    """The seed fixes loader invocation boundaries, slot order, shapes and
    storage dtypes.

    The source directory associates those actual native exports with their
    ParamInfo. No shape or expert id is inferred from the recipient TP layout.
    """

    def __init__(self, iterator: Any) -> None:
        self.iterator = iterator
        self.seed_chunks: list[tuple[Slot, ...]] = []
        self.native_sources: dict[str, tuple[Slot, ...]] = {}
        self.rows: dict[str, _Row] = {}
        self.by_slot: dict[str, _Row] = {}
        self.full_reason: str | None = None
        self.metrics: dict[str, Any] = {}

    def seed(self, shards: Mapping) -> Iterator:
        directory = {}
        chunks = []
        self.metrics = {}
        self.iterator._seed_sources = directory
        try:
            for items in self.iterator.get_hf_weight_chunks(shards):
                chunks.append(tuple((name, tuple(value.shape), value.dtype) for name, value in items))
                yield items
        finally:
            self.iterator._seed_sources = None
        directory = {name: tuple(items) for name, items in directory.items()}
        # An invalidation can replace groups as well as storage. Only Python
        # geometry is cached between publications, never an old communicator.
        self.seed_chunks = chunks
        self.native_sources = directory
        self.rows.clear()
        self.by_slot.clear()
        self.full_reason = None

    @staticmethod
    def _routes(group):
        from megatron.core import mpu

        rank = dist.get_rank(dist.group.WORLD)
        source = dist.get_process_group_ranks(group)[0]
        pp = mpu.get_pipeline_model_parallel_world_size()
        tp = mpu.get_tensor_model_parallel_group()
        etp = mpu.get_expert_tensor_parallel_group()
        expert = mpu.get_expert_tensor_and_model_parallel_group()
        routes = {}
        for is_expert, shard_group, merge_group in ((False, tp, tp), (True, etp, expert)):
            if pp > 1:
                merge_group = dist.group.WORLD
            ranks = tuple(dist.get_process_group_ranks(merge_group))
            # PP owners can be anywhere. With a single stage, select the TP or
            # EP x ETP replica containing the publication root, as in verl.
            primary = pp == 1 or (
                mpu.get_expert_data_parallel_rank() == 0
                if is_expert
                else mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            )
            routes[is_expert] = _Route(
                merge_group,
                ranks,
                dist.get_rank(shard_group),
                dist.get_world_size(shard_group),
                source in ranks,
                primary,
            )
            if source in ranks and ranks[0] != source:
                raise ValueError("The sparse gather root differs from the publication root")
        return rank, routes

    def prepare(self, metadata_group) -> None:
        """Build verl's slot-table union using the ordinary seed's actual
        conversion results."""
        from .shard_projection import ShardProjector, UnsupportedShardProjection

        rank, routes = self._routes(metadata_group)
        iterator = self.iterator
        projector = ShardProjector(iterator._bridge_converter, iterator._gdn_split_sizes)
        infos = {i.name: i for bucket in (*iterator._non_expert_buckets, *iterator._expert_buckets) for i in bucket}
        report, layouts = {}, {}
        error = None
        try:
            for name, info in infos.items():
                expert = ".experts." in name
                route = routes[expert]
                if not route.active or not route.primary_replica or info.src_rank != rank:
                    continue
                sharded = (
                    info.attrs["tensor_model_parallel"]
                    and info.attrs["parallel_mode"] != "duplicated"
                    and "expert_bias" not in name
                )
                if not sharded and route.lane != 0:
                    continue
                slots = self.native_sources[name]
                reason = None
                try:
                    layout = projector.layout(info, route.lane, route.lanes)
                    if tuple(layout.slots) != slots:
                        raise UnsupportedShardProjection("projection directory differs from the native seed")
                    layouts[name] = layout
                except UnsupportedShardProjection as exc:
                    reason = str(exc)
                report[name] = (slots, rank, route.lane, route.lanes if sharded else 1, reason)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        peers = [None] * dist.get_world_size(metadata_group)
        dist.all_gather_object(peers, (report, error), group=metadata_group)
        errors = [failure for _, failure in peers if failure]
        if errors:
            raise ValueError(f"Cannot build the native Bridge shard directory: {errors}")
        ownership = defaultdict(list)
        for part, _ in peers:
            for name, data in part.items():
                ownership[name].append(data)
        self.rows.clear()
        self.by_slot.clear()
        for name, entries in ownership.items():
            slots, _, _, lanes, _ = entries[0]
            if any(schema != slots or size != lanes for schema, _, _, size, _ in entries):
                raise ValueError(f"Native slot directory or shard geometry differs across owners for {name}")
            if sorted(lane for _, _, lane, _, _ in entries) != list(range(lanes)):
                raise ValueError(f"Missing or duplicate contributing shard lanes for {name}")
            row = _Row(
                infos[name],
                slots,
                ".experts." in name,
                tuple(r for _, r, _, _, _ in entries),
                layouts.get(name),
                next((reason for *_, reason in entries if reason is not None), None),
            )
            self.rows[name] = row
            for slot, _, _ in slots:
                if slot in self.by_slot:
                    self.full_reason = f"multiple native source rows produce {slot}"
                self.by_slot[slot] = row
            if not slots:
                self.full_reason = f"{name} has no independently replayable native outputs"
        observed = {slot for chunk in self.seed_chunks for slot in chunk}
        catalog = {slot for row in self.rows.values() for slot in row.slots}
        if observed != catalog or set(ownership) != set(infos):
            self.full_reason = "native FULL directory contains coupled or unowned conversion rows"
        reasons = [None] * dist.get_world_size(metadata_group)
        dist.all_gather_object(reasons, self.full_reason, group=metadata_group)
        self.full_reason = next((reason for reason in reasons if reason), None)
        if self.full_reason and rank == dist.get_process_group_ranks(metadata_group)[0]:
            logger.info("Sharded publication requires a native FULL anchor: %s", self.full_reason)

    def _project_bucket(self, wanted, current, previous, device):
        contributions = {}
        slices = defaultdict(list)
        for name, requested in wanted.items():
            row = self.rows[name]
            key = self.iterator._vanilla_key_map[name]
            new = _source_tensor(current, key, device)
            old = _source_tensor(previous, key, device)
            if new.shape != old.shape or new.dtype != old.dtype:
                raise ValueError(f"Acknowledged shard schema changed for {name}")
            layout = row.layout
            if layout.local_slices is not None:
                for (slot, _, dtype), (start, stop, offset) in zip(row.slots, layout.local_slices, strict=True):
                    if slot in requested:
                        slices[dtype].append((slot, new.view(-1)[start:stop], old.view(-1)[start:stop], offset))
            else:
                indices = _changed_indices(new, old)
                projected = layout.project(indices, new)
                # Pointwise numerical mappings can round distinct source words to
                # the same FULL value. Filter these on the contributing rank.
                if layout.transforms_values:
                    before = layout.project(indices, old)
                    for (slot, _, pos, values), (_, _, _, baseline) in zip(projected, before, strict=True):
                        if slot in requested:
                            words = _WORD[values.element_size()]
                            keep = values.view(words) != baseline.view(words)
                            contributions[slot] = (pos[keep], values.view(words)[keep].view(values.dtype))
                else:
                    contributions.update(
                        (slot, (pos, values)) for slot, _, pos, values in projected if slot in requested
                    )
        _batch_slices(slices, contributions, device)
        return contributions

    def chunks(self, current, previous, *, full_names, metadata_group, device, max_round_bytes, staging_budget_bytes):
        """Gather named changes and preserve the native loader's complete frame
        boundary."""
        from math import prod

        from relax.distributed.weight_sync.codec import WIRE_ALIGNMENT

        from .delta_export import DeltaWeight

        rank, routes = self._routes(metadata_group)
        root = dist.get_process_group_ranks(metadata_group)[0]
        fallback = {
            name: row.fallback or "the recipient requires complete native input"
            for name, row in self.rows.items()
            if row.fallback or any(slot in full_names for slot, _, _ in row.slots)
        }
        details = {name: {"reason": reason, "full_bytes": 0} for name, reason in fallback.items()}
        self.metrics = {
            "native_fallback_rows": len(details),
            "native_fallback_wire_bytes": 0,
            "native_fallback_details": details,
        }
        if fallback and rank == root:
            logger.info(
                "Sharded publication uses native current-weight fallback for %d rows: %s",
                len(fallback),
                list(fallback.items())[:4],
            )

        def aligned(size, alignment):
            return (size + alignment - 1) // alignment * alignment

        def fit_frame(chunk, native, received):
            # codec.encode aligns every entry to WIRE_ALIGNMENT and every value
            # to its own width. Except for the last entry the trailing gap is
            # charged to this entry, so changing one source row has additive cost.
            if type(staging_budget_bytes) is not int or staging_budget_bytes <= 0:
                raise ValueError("staging_budget_bytes must be a positive integer")
            size, savings = 0, defaultdict(int)
            for i, (slot, shape, dtype) in enumerate(chunk):
                name = self.by_slot[slot].info.name
                width = dtype.itemsize
                full_bytes = prod(shape) * width
                encoded = full_bytes
                if name not in native:
                    count = received[slot][0].numel()
                    encoded = aligned(count * 4, width) + count * width
                if i + 1 < len(chunk):
                    full_bytes = aligned(full_bytes, WIRE_ALIGNMENT)
                    encoded = aligned(encoded, WIRE_ALIGNMENT)
                size += encoded
                if name not in native:
                    savings[name] += encoded - full_bytes
            before = size
            chosen = []
            if size > staging_budget_bytes:
                # Keeping the largest savings first minimizes the number of
                # additional native rows read for this particular loader frame.
                for name in sorted(savings, key=savings.get, reverse=True):
                    if savings[name] <= 0:
                        break
                    chosen.append(name)
                    size -= savings[name]
                    if size <= staging_budget_bytes:
                        break
                if size > staging_budget_bytes:
                    raise ValueError(
                        f"Complete loader group needs at least {size} bytes, staging_budget_bytes={staging_budget_bytes}"
                    )
            return chosen, before

        local = []
        for chunk in self.seed_chunks:
            names = dict.fromkeys(self.by_slot[slot].info.name for slot, _, _ in chunk)
            local.append([self.rows[n].info for n in names if rank in self.rows[n].owners and n not in fallback])
        prefetch = None
        control_device = device if dist.get_backend(metadata_group) == "nccl" else "cpu"
        decision = torch.empty(1, dtype=torch.int32, device=control_device)
        done = object()
        for chunk in self.seed_chunks:
            wanted, batches, native = defaultdict(set), defaultdict(list), {}
            error = None
            try:
                if prefetch is None:
                    prefetch = iter(
                        previous.prefetch(local, self.iterator._vanilla_key_map, rank)
                        if isinstance(previous, CudaShardInputs)
                        else local
                    )
                if next(prefetch, done) is done:
                    raise ValueError("Native shard prefetch ended before the loader directory")
                for slot, _, dtype in chunk:
                    row = self.by_slot[slot]
                    name = row.info.name
                    if name in fallback:
                        native[name] = row
                    else:
                        batches[row.expert, dtype].append(slot)
                        if rank in row.owners:
                            wanted[name].add(slot)
                projected = self._project_bucket(wanted, current, previous, device)
            except Exception as exc:
                error = exc
            self._agree(error, metadata_group)

            received = {}
            for (expert, dtype), slots in batches.items():
                route = routes[expert]
                error = None
                try:
                    if route.active:
                        parts = [projected[s] for s in slots if s in projected]
                        lengths = [projected[s][0].numel() if s in projected else 0 for s in slots]
                        idx = (
                            torch.cat([i for i, _ in parts])
                            if parts
                            else torch.empty(0, dtype=torch.int32, device=device)
                        )
                        val = (
                            torch.cat([v.view(_WORD[v.element_size()]) for _, v in parts]).view(dtype)
                            if parts
                            else torch.empty(0, dtype=dtype, device=device)
                        )
                        counts = torch.tensor(lengths, dtype=torch.int64)
                except Exception as exc:
                    error = exc
                self._agree(error, metadata_group)
                error = None
                try:
                    if route.active:
                        result = self._gather_slots(
                            expert=expert,
                            slots=slots,
                            indices=idx,
                            values=val,
                            counts=counts,
                            group=route.group,
                            max_round_bytes=max_round_bytes,
                        )
                        if rank == root:
                            received.update(zip(slots, result, strict=True))
                        del result, idx, val, parts
                except Exception as exc:
                    error = exc
                self._agree(error, metadata_group)
            del projected

            # Only the publication root has the merged counts. One scalar tells
            # every peer whether a native read or a shared frame error follows.
            plan = [None]
            if rank == root:
                try:
                    selected, before = fit_frame(chunk, native, received)
                    plan[0] = (selected, before)
                    decision.fill_(len(selected))
                except Exception as exc:
                    plan[0] = str(exc)
                    decision.fill_(-1)
            dist.broadcast(decision, src=root, group=metadata_group)
            if int(decision):
                dist.broadcast_object_list(plan, src=root, group=metadata_group)
                if isinstance(plan[0], str):
                    raise ValueError(plan[0])
                selected, before = plan[0]
                reason = f"sparse loader frame of {before} bytes exceeds staging_budget_bytes={staging_budget_bytes}"
                for name in selected:
                    native[name] = self.rows[name]
                    details.setdefault(name, {"reason": reason, "full_bytes": 0, "supported_changed": True})
                self.metrics["native_fallback_rows"] = len(details)
                if rank == root:
                    logger.info(
                        "Native current-weight fallback for %d rows including %s: %s",
                        len(selected),
                        selected[:4],
                        reason,
                    )

            full = {}
            needed = {slot for slot, _, _ in chunk if self.by_slot[slot].info.name in native}
            for slot in needed:
                received.pop(slot, None)
            for name, row in native.items():
                # Work one row at a time. An exception in a rank-local result
                # or prefetch finalizer must be agreed before any rank advances
                # the native iterator into the next row's collectives.
                source = None
                for expected in (*row.slots, None):
                    error = None
                    try:
                        if source is None:
                            group = ([row.info],)
                            selection = (group, ()) if row.expert else ((), group)
                            source = iter(self.iterator._iter_hf_params(current, selected=selection))
                        item = next(source, done)
                        if expected is None:
                            if item is not done:
                                raise ValueError(f"Native fallback emitted an extra slot for {name}")
                        elif item is done:
                            raise ValueError(f"Native fallback omitted {expected[0]} from {name}")
                        else:
                            slot, tensor = item
                            if (slot, tuple(tensor.shape), tensor.dtype) != expected:
                                raise ValueError(f"Native fallback schema differs from the seed for {name}: {slot}")
                            if rank == root and slot in needed:
                                word = _WORD[tensor.element_size()]
                                full[slot] = tensor.detach().view(word).contiguous().view(tensor.dtype).reshape(-1)
                            del item, tensor
                    except Exception as exc:
                        error = exc
                    self._agree(error, metadata_group)

            entries, error = [], None
            try:
                if rank == root:
                    for slot, shape, dtype in chunk:
                        if slot in needed:
                            value = full[slot]
                            name = self.by_slot[slot].info.name
                            details[name]["full_bytes"] += value.nbytes
                            self.metrics["native_fallback_wire_bytes"] += value.nbytes
                            entries.append(DeltaWeight(slot, shape, dtype, None, value))
                            del value
                        else:
                            entries.append(DeltaWeight(slot, shape, dtype, *received[slot]))
            except Exception as exc:
                error = exc
            self._agree(error, metadata_group)
            yield entries
            del entries, received, full

        # Draining releases the last prefetched bucket and can surface a
        # rank-local copy/fence failure even after the final yield.
        error = None
        try:
            if prefetch is not None and next(prefetch, done) is not done:
                raise ValueError("Native shard prefetch outlasted the loader directory")
        except Exception as exc:
            error = exc
        self._agree(error, metadata_group)

    def _gather_slots(self, *, expert, slots, indices, values, counts, group, max_round_bytes):
        from .sparse_gather import gather_slot_entries_to_rank0

        if len(slots) != counts.numel():
            raise ValueError("The sparse count table must match its ordered native HF slots")
        return gather_slot_entries_to_rank0(
            indices,
            values,
            counts,
            group=group,
            max_round_bytes=max_round_bytes,
            observe=partial(self._record_round, "routed_expert" if expert else "ordinary_tp"),
        )

    def _record_round(self, kind, event):
        totals = self.metrics.setdefault(kind, {"group": event.group_ranks})
        for key in ("control_bytes", "count_bytes", "index_bytes", "value_bytes", "padding_bytes"):
            totals[key] = totals.get(key, 0) + getattr(event, key)

    @staticmethod
    def _agree(error, group):
        flag = torch.tensor(
            [int(error is not None)], dtype=torch.int32, device="cuda" if dist.get_backend(group) == "nccl" else "cpu"
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if int(flag):
            errors = [None] * dist.get_world_size(group)
            dist.all_gather_object(errors, str(error) if error is not None else None, group=group)
            raise RuntimeError(f"Shard export failed on source ranks: {errors}") from error
