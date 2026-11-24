# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Partition verl HF coordinates using the native FULL layout acknowledged by
each scheduler."""

from __future__ import annotations

from itertools import groupby, pairwise

import torch

from .codec import WireBucket, encode, encode_flushes, frame_flushes


def _expert_name(name: str) -> tuple[str, int, int] | None:
    parts = name.rsplit(".", 3)
    if (
        len(parts) == 4
        and parts[0].endswith(".experts")
        and parts[1].isascii()
        and parts[1].isdecimal()
        and parts[2] in ("gate_proj", "up_proj", "down_proj")
        and parts[3] == "weight"
    ):
        return parts[0], int(parts[1]), int(parts[2] == "down_proj")
    return None


class ExpertLanePlan:
    """An entire TP/EP replica must prove exactly one owner for every
    intermediate coordinate."""

    def __init__(self, engines: list[list[dict]]) -> None:
        self.members = sum(len(engine) for engine in engines)
        self.layers = {}
        self.rejected = {}
        if not engines or not all(engines):
            return
        ranks = [[{r["prefix"]: r for r in worker["routes"]} for worker in engine] for engine in engines]
        prefixes = {prefix for engine in ranks for worker in engine for prefix in worker}
        for prefix in prefixes:
            if any(prefix not in worker for engine in ranks for worker in engine):
                self.rejected[prefix] = "incomplete_native_rank_directory"
                continue
            groups = [[worker[prefix] for worker in engine] for engine in ranks]
            records = [r for engine in groups for r in engine]
            schemas = {(r["intermediate_size"], r["hidden_size"], r["dtype"]) for r in records}
            if len(schemas) != 1:
                self.rejected[prefix] = "native_rank_schema_disagreement"
                continue
            width, hidden, dtype = schemas.pop()
            bounds = sorted({v for r in records for v in (r["expert_start"], r["expert_stop"])})
            intervals = []
            for first, stop in pairwise(bounds):
                owners = []
                for engine in groups:
                    slices = [
                        (r["intermediate_start"], r["intermediate_stop"])
                        if r["expert_start"] <= first < r["expert_stop"]
                        else None
                        for r in engine
                    ]
                    spans = sorted(s for s in slices if s is not None)
                    if (
                        not spans
                        or spans[0][0] != 0
                        or spans[-1][1] != width
                        or any(lo >= hi for lo, hi in spans)
                        or any(left[1] != right[0] for left, right in pairwise(spans))
                    ):
                        break
                    owners.extend(slices)
                if len(owners) != self.members:
                    break
                intervals.append((first, stop, tuple(owners)))
            else:
                self.layers[prefix] = (width, hidden, dtype, intervals)
                continue
            self.rejected[prefix] = "overlapping_or_missing_native_ownership"

    def _placement(self, slot: dict) -> tuple[tuple | None, str | None]:
        parsed = _expert_name(slot["name"])
        if parsed is None:
            return None, "unsupported_native_expert_input" if ".experts." in slot["name"] else None
        prefix, expert, axis = parsed
        if prefix not in self.layers:
            return None, self.rejected.get(prefix, "native_layout_unavailable")
        width, hidden, dtype, intervals = self.layers[prefix]
        if slot["full"] or slot["dtype"] != dtype or slot["shape"] != ([hidden, width] if axis else [width, hidden]):
            return None, "native_expert_schema_mismatch"
        for first, stop, spans in intervals:
            if first <= expert < stop:
                return (axis, spans), None
        return None, "unobserved_native_expert"

    def partition(self, directory: dict, raw: torch.Tensor, limit: int, metrics: dict) -> tuple:
        """Return a shared broadcast or one bounded directory/slab per delivery
        rank in order."""
        placement, fallback = [], {}
        for slot in directory["tensors"]:
            local, reason = self._placement(slot)
            placement.append(local)
            if reason is not None:
                size = slot["count"] * (getattr(torch, slot["dtype"]).itemsize + (0 if slot["full"] else 4))
                fallback[reason] = fallback.get(reason, 0) + size * self.members
        reasons = metrics.setdefault("expert_fallback_delivery_bytes", {})
        if not any(p is not None for p in placement):
            for reason, amount in fallback.items():
                reasons[reason] = reasons.get(reason, 0) + amount
            return directory, raw

        # Replicas with identical native slices share a source slab. Each proxy
        # still receives and acknowledges its own physical delivery.
        cache = {}
        lanes = []
        for rank in range(self.members):
            key = tuple(None if p is None else (p[0], p[1][rank]) for p in placement)
            if key not in cache:
                cache[key] = _encode_lane(directory, raw, key, limit)
            if cache[key] is None:
                # Additional alignment at direct/shared boundaries can exceed
                # the original admission. Broadcast the original complete frame.
                for slot, owner in zip(directory["tensors"], placement, strict=True):
                    if owner is not None:
                        size = slot["count"] * (4 + getattr(torch, slot["dtype"]).itemsize) * self.members
                        fallback["lane_staging_budget"] = fallback.get("lane_staging_budget", 0) + size
                for reason, amount in fallback.items():
                    reasons[reason] = reasons.get(reason, 0) + amount
                return directory, raw
            lanes.append(cache[key])
        for reason, amount in fallback.items():
            reasons[reason] = reasons.get(reason, 0) + amount
        routed = sum(
            s["count"] * (4 + getattr(torch, s["dtype"]).itemsize)
            for s, p in zip(directory["tensors"], placement, strict=True)
            if p is not None
        )
        metrics["direct_expert_input_bytes"] = metrics.get("direct_expert_input_bytes", 0) + routed
        metrics["direct_expert_delivery_bytes"] = metrics.get("direct_expert_delivery_bytes", 0) + sum(
            slot["count"] * (4 + getattr(torch, slot["dtype"]).itemsize)
            for meta, _ in lanes
            for slot in meta["tensors"]
            if slot.get("expert_direct")
        )
        return [meta for meta, _ in lanes], [slab for _, slab in lanes]


def _local_flush(
    rows: list, pos: torch.Tensor, values: torch.Tensor
) -> tuple[list[dict], torch.Tensor, torch.Tensor] | None:
    """Reindex one contiguous run of HF slots, exchanging counts with the host
    once per run."""
    first, last = rows[0][0], rows[-1][0]
    indices = pos[first["pos_start"] : last["pos_end"]].view(torch.int32)
    vals = values[first["val_start"] : last["val_end"]]
    if not indices.numel() or all(span is None for _, (_, span) in rows):
        return None
    counts = [slot["count"] for slot, _ in rows]
    descriptors = [(axis, (span or (0, 0))[0], (span or (0, 0))[1], slot["shape"][1]) for slot, (axis, span) in rows]
    if len(set(descriptors)) == 1:
        axis, start, stop, columns = descriptors[0]
        if axis:
            coordinate = indices % columns
            local = (indices // columns) * (stop - start) + coordinate - start
        else:
            coordinate = indices // columns
            local = indices - start * columns
    else:
        device = indices.device
        descriptors = torch.tensor(descriptors, device=device, dtype=torch.int32)
        repeats = torch.tensor(counts, device=device, dtype=torch.int32)
        axis, start, stop, columns = descriptors.repeat_interleave(repeats, dim=0, output_size=indices.numel()).unbind(
            1
        )
        row, col = indices // columns, indices % columns
        coordinate = torch.where(axis == 1, col, row)
        local = torch.where(axis == 1, row * (stop - start) + col - start, indices - start * columns)
    selected = torch.nonzero((coordinate >= start) & (coordinate < stop), as_tuple=True)[0]
    positions = local.index_select(0, selected)
    vals = vals.index_select(0, selected)
    ends = torch.tensor(counts, device=selected.device).cumsum(0)
    output_ends = torch.searchsorted(selected, ends).tolist()
    entries = []
    begin = 0
    for (slot, (axis, span)), end in zip(rows, output_ends, strict=True):
        if end > begin:
            shape = list(slot["shape"])
            shape[axis] = span[1] - span[0]
            entries.append(
                dict(
                    slot,
                    shape=shape,
                    pos_start=begin * 4,
                    pos_end=end * 4,
                    val_start=begin,
                    val_end=end,
                    count=end - begin,
                    expert_direct=True,
                )
            )
        begin = end
    # index_select has already packed the lane in slot order. Keep the two flat
    # blobs instead of slicing them into thousands of views and concatenating them again.
    return (entries, positions.view(torch.uint8), vals) if entries else None


def _encode_lane(directory: dict, raw: torch.Tensor, placements: tuple, limit: int) -> tuple | None:
    bucket = WireBucket(limit)
    cursor = 0
    for spec, positions, values in frame_flushes(directory, raw):
        slots = spec["params"]
        rows = zip(slots, placements[cursor : cursor + len(slots)], strict=True)
        cursor += len(slots)
        for direct, group in groupby(rows, key=lambda row: row[1] is not None):
            if direct:
                flush = _local_flush(list(group), positions, values)
                if flush is None:
                    continue
                manifest, slab = encode_flushes((flush,), max_bytes=limit)
                del flush  # The wire slab now owns the data. Release the index_select outputs before the next group.
            else:
                items = (
                    (
                        s["name"],
                        s["shape"],
                        values.dtype,
                        None if s["full"] else positions[s["pos_start"] : s["pos_end"]].view(torch.int32),
                        values[s["val_start"] : s["val_end"]],
                    )
                    for s, _ in group
                )
                manifest, slab = encode(items, max_bytes=limit)
            if bucket.offer(manifest, slab) is not None:
                return None
    result = bucket.finish()
    if result is None:
        return dict(format=2, bytes=0, tensors=[], flushes=[], expert_direct=True), raw.new_empty(0)
    result[0]["expert_direct"] = True
    return result
