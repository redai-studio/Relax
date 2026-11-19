# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build and validate weight-transfer plans without sending payloads or
creating process groups."""

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .common import validate_worker_identity
from .layout import MatrixShard, TensorRegion


# Shard plan.


def intersection(source: MatrixShard, target: MatrixShard) -> dict[str, int] | None:
    """Map a global matrix intersection to local source/target coordinates."""
    start = tuple(max(a, b) for a, b in zip(source.offset, target.offset, strict=True))
    end = tuple(
        min(a + n, b + m) for a, n, b, m in zip(source.offset, source.shape, target.offset, target.shape, strict=True)
    )
    if any(b <= a for a, b in zip(start, end, strict=True)):
        return None
    return dict(
        src_row=start[0] - source.offset[0] + source.storage_offset[0],
        src_col=start[1] - source.offset[1] + source.storage_offset[1],
        dst_row=start[0] - target.offset[0] + target.storage_offset[0],
        dst_col=start[1] - target.offset[1] + target.storage_offset[1],
        rows=end[0] - start[0],
        cols=end[1] - start[1],
    )


def region_write(source_region: TensorRegion, target_region: TensorRegion, **identity: Any) -> dict | None:
    """Lower one logical intersection to the common byte-transfer
    descriptor."""
    source, target = source_region, target_region
    if source.itemsize != target.itemsize:
        raise ValueError("Transfer regions require the same element size")
    part = intersection(source.shard, target.shard)
    if part is None:
        return None
    size = source.itemsize
    write = dict(
        identity,
        src_offset=(source.storage_offset + part["src_row"] * source.row_stride + part["src_col"]) * size,
        src_stride=source.row_stride * size,
        dst_offset=(target.storage_offset + part["dst_row"] * target.row_stride + part["dst_col"]) * size,
        dst_stride=target.row_stride * size,
        rows=part["rows"],
        row_bytes=part["cols"] * size,
    )
    return _contiguous_write(write)


def _contiguous_write(write: dict) -> dict:
    if write["src_stride"] == write["dst_stride"] == write["row_bytes"]:
        write = dict(write)
        write["row_bytes"] *= write["rows"]
        write["rows"] = 1
        write["src_stride"] = write["dst_stride"] = write["row_bytes"]
    return write


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_coverage(writes: list[dict], layouts: dict, *, source_world_size: int = 4) -> set[str]:
    """Reject duplicate IDs, holes, overlaps, invalid owners and out-of-bounds
    writes."""
    if type(source_world_size) is not int or source_world_size <= 0:
        raise ValueError("Invalid DSR source world size")
    intervals = defaultdict(list)
    ids = set()
    for write in writes:
        identity, name = write["id"], write["param"]
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(name, str)
            or identity in ids
            or name not in layouts
            or type(write["source_rank"]) is not int
            or write["source_rank"] not in range(source_world_size)
        ):
            raise ValueError("Invalid DSR write identity, parameter or owner")
        ids.add(identity)
        offset, rows, width, stride = (write[k] for k in ("dst_offset", "rows", "row_bytes", "dst_stride"))
        capacity = layouts[name]["nbytes"]
        if (
            any(type(value) is not int for value in (offset, rows, width, stride, capacity))
            or offset < 0
            or rows <= 0
            or width <= 0
            or stride < width
            or capacity <= 0
            or offset + (rows - 1) * stride + width > capacity
        ):
            raise ValueError("Invalid DSR rectangle")
        if stride == width:
            intervals[name].append((offset, offset + rows * width))
            continue
        for row in range(rows):
            begin = offset + row * stride
            intervals[name].append((begin, begin + width))
    for name, ranges in intervals.items():
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor or end > layouts[name]["nbytes"]:
                raise ValueError(f"DSR coverage hole, overlap or overflow: {name}")
            cursor = end
        if cursor != layouts[name]["nbytes"]:
            raise ValueError(f"DSR incomplete coverage: {name}")
    return set(intervals)


def _source_fragment(write: dict) -> tuple:
    return tuple(write[key] for key in ("source_rank", "source", "src_offset", "rows", "row_bytes", "src_stride")) + (
        write.get("tp_rank"),
    )


def coalesce_writes(
    writes: Sequence[dict],
    *,
    max_bytes: int | None = None,
    protected_sources: frozenset = frozenset(),
    allow_strided: bool = False,
) -> list[dict]:
    """Merge adjacent source AND destination ranges during cold planning.

    All other fields are semantic boundaries, including owner, target, source
    producer and any adapter fields (e.g. expert TP slice). Never merge across
    them, packed-wire mappings or separate source allocations. Input IDs must
    be unique; merged IDs bind every original piece before receipts are built.
    Direct-view transports may opt into affine row merging without copying the
    intervening gaps. Packed transports keep their existing default.
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("Invalid transfer merge byte budget")
    geometry = {"id", "src_offset", "dst_offset", "rows", "row_bytes", "src_stride", "dst_stride"}
    entries, ids = [], set()
    for index, original in enumerate(writes):
        if original["id"] in ids:
            raise ValueError("Duplicate transfer identity before coalescing")
        ids.add(original["id"])
        if (
            any(type(original[k]) is not int or original[k] < 0 for k in ("src_offset", "dst_offset"))
            or any(type(original[k]) is not int or original[k] <= 0 for k in ("rows", "row_bytes"))
            or any(
                type(original[k]) is not int or original[k] < original["row_bytes"]
                for k in ("src_stride", "dst_stride")
            )
        ):
            raise ValueError("Invalid transfer source or destination rectangle")
        write = _contiguous_write(dict(original))
        boundary = fingerprint({key: value for key, value in write.items() if key not in geometry})
        if _source_fragment(write) in protected_sources:
            # A shared pack must stay identical for every consumer. Independent
            # per-target merging can otherwise turn A+B into AB plus another A.
            boundary = fingerprint((boundary, original["id"]))
        entries.append((boundary, index, (write["id"],), write))

    # Join columns first, then rows. This also joins rectangular tiles without
    # expanding each strided row into thousands of individual descriptors.
    for axis in ("columns", "rows", "affine_rows") if allow_strided else ("columns", "rows"):
        groups = defaultdict(list)
        for boundary, index, members, write in entries:
            if axis == "columns":
                shape = (
                    (write["rows"],)
                    if write["rows"] == 1
                    else (write["rows"], write["src_stride"], write["dst_stride"])
                )
            elif axis == "rows":
                shape = (write["row_bytes"], write["src_stride"], write["dst_stride"])
            else:
                shape = (write["row_bytes"],)
            groups[boundary, shape].append((index, members, write))
        entries = []
        for (boundary, _), values in groups.items():
            joined = []
            for index, members, write in sorted(
                values, key=lambda item: (item[2]["src_offset"], item[2]["dst_offset"])
            ):
                if joined:
                    old_index, old_members, previous = joined[-1]
                    if axis == "columns":
                        adjacent = all(
                            write[k] == previous[k] + previous["row_bytes"] for k in ("src_offset", "dst_offset")
                        )
                        width = previous["row_bytes"] + write["row_bytes"]
                        adjacent &= previous["rows"] == 1 or width <= min(
                            previous["src_stride"], previous["dst_stride"]
                        )
                        merged = dict(previous, row_bytes=width)
                        if previous["rows"] == 1:
                            merged.update(src_stride=width, dst_stride=width)
                    elif axis == "rows":
                        adjacent = all(
                            write[f"{side}_offset"]
                            == previous[f"{side}_offset"] + previous["rows"] * previous[f"{side}_stride"]
                            for side in ("src", "dst")
                        )
                        merged = dict(previous, rows=previous["rows"] + write["rows"])
                    else:
                        # Direct-view executors can read equal-width rows
                        # separated by gaps. Infer a positive stride from two
                        # single rows; never include the bytes in those gaps.
                        strides = {
                            side: (
                                write[f"{side}_offset"] - previous[f"{side}_offset"]
                                if previous["rows"] == 1
                                else previous[f"{side}_stride"]
                            )
                            for side in ("src", "dst")
                        }
                        adjacent = all(
                            stride >= previous["row_bytes"]
                            and write[f"{side}_offset"] == previous[f"{side}_offset"] + previous["rows"] * stride
                            and (write["rows"] == 1 or write[f"{side}_stride"] == stride)
                            for side, stride in strides.items()
                        )
                        merged = dict(
                            previous,
                            rows=previous["rows"] + write["rows"],
                            src_stride=strides["src"],
                            dst_stride=strides["dst"],
                        )
                    if adjacent and (max_bytes is None or merged["rows"] * merged["row_bytes"] <= max_bytes):
                        joined[-1] = (min(index, old_index), old_members + members, merged)
                        continue
                joined.append((index, members, write))
            entries.extend((boundary, index, members, _contiguous_write(write)) for index, members, write in joined)
    result = []
    for _, _, members, write in sorted(entries, key=lambda item: item[1]):
        if len(members) > 1:
            write = dict(write, id="merged:" + fingerprint(sorted(members)))
        result.append(write)
    return result


@dataclass(frozen=True)
class TransferPlan:
    """Common geometry and coverage contract; adapters add delivery
    metadata."""

    target: str
    source_world_size: int
    writes: tuple[dict, ...]

    @classmethod
    def create(
        cls,
        target: str,
        writes: Sequence[dict],
        layouts: dict,
        *,
        source_world_size: int,
        parameters: set[str],
        max_bytes: int | None = None,
        protected_sources: frozenset = frozenset(),
    ) -> "TransferPlan":
        if not target or not parameters or any(write["target"] != target for write in writes):
            raise ValueError("Transfer plan has missing parameters or a misrouted target")
        # Validate before rewriting: merging must never conceal malformed IDs,
        # overlaps, missing parameters or an unauthorized owner.
        if validate_coverage(writes, layouts, source_world_size=source_world_size) != parameters:
            raise ValueError("Transfer plan must cover every selected native parameter")
        merged = coalesce_writes(writes, max_bytes=max_bytes, protected_sources=protected_sources)
        if validate_coverage(merged, layouts, source_world_size=source_world_size) != parameters:
            raise ValueError("Merged transfer coverage differs from the native parameters")
        return cls(target, source_world_size, tuple(merged))

    @classmethod
    def create_all(
        cls,
        plans: dict,
        layouts: dict,
        parameters: dict,
        *,
        max_bytes: int | None = None,
    ) -> dict:
        """Finalize a plane together, preserving existing multi-consumer packs.

        Only unshared fragments are merged. Shared fragments need coordinated
        merge decisions across ALL consumers, not just matching target sets.
        """
        uses = defaultdict(int)
        for plan in plans.values():
            for write in plan["writes"]:
                uses[_source_fragment(_contiguous_write(write))] += 1
        protected = frozenset(key for key, count in uses.items() if count > 1)
        return {
            key: cls.create(
                key,
                plan["writes"],
                layouts[key],
                source_world_size=plan["source_world_size"],
                parameters=parameters[key],
                max_bytes=max_bytes,
                protected_sources=protected,
            ).as_dict(
                **{
                    name: value
                    for name, value in plan.items()
                    if name not in {"target", "source_world_size", "writes"}
                }
            )
            for key, plan in plans.items()
        }

    def as_dict(self, **metadata: Any) -> dict:
        if {"target", "source_world_size", "writes"} & metadata.keys():
            raise ValueError("Adapter metadata cannot replace common transfer geometry")
        return dict(metadata, target=self.target, source_world_size=self.source_world_size, writes=list(self.writes))

    def delivery(self, sources: Mapping[int, dict], receiver: dict) -> dict:
        """Partition an already validated union without weakening coverage.

        One logical source can use IPC for one receiver and NIXL for another.
        Both subsets retain the original write IDs; neither is a replacement
        for validating the complete native parameter coverage above.
        """
        return self.bind_sources(sources)(receiver)

    def bind_sources(self, sources: Mapping[int, dict]) -> Callable[[dict], dict]:
        """Own one logical plan and source snapshot for multiple receivers.

        Each call still validates the actual receiver and produces its own
        routes and physical hash. No caller-owned dictionary is cached.
        """
        description = copy.deepcopy(self.as_dict())
        endpoints = copy.deepcopy(dict(sources))
        for rank, endpoint in endpoints.items():
            if type(rank) is not int or not 0 <= rank < self.source_world_size:
                raise ValueError("Delivery endpoint is outside the source world")
            validate_worker_identity(endpoint)
        owners = tuple((write["id"], write["source_rank"]) for write in description["writes"])
        if any(rank not in endpoints for _, rank in owners):
            raise ValueError("Delivery plan is missing a physical source endpoint")
        logical_hash = fingerprint(description)
        source_records = [dict(rank=rank, **endpoint) for rank, endpoint in sorted(endpoints.items())]

        def bind(receiver: dict) -> dict:
            validate_worker_identity(receiver)
            routes: dict[str, list[str]] = {"cuda_ipc": [], "nixl": []}
            for identity, rank in owners:
                # Same-host IPC can read a different physical GPU.
                local = endpoints[rank]["boot_id"] == receiver["boot_id"]
                routes["cuda_ipc" if local else "nixl"].append(identity)
            result = copy.deepcopy(
                dict(logical_hash=logical_hash, receiver=receiver, sources=source_records, routes=routes)
            )
            return dict(result, hash=fingerprint(result))

        return bind


def bind_delivery(writes: list[dict], geometry: dict, sources: list[dict], receiver: dict, *, layouts: dict) -> dict:
    """Bind complete expert coverage to one actual receiving CUDA worker.

    The logical plan may be shared by equivalent TP coordinates; physical
    delivery cannot, because those coordinates can live on different hosts.
    """
    endpoints = {item["rank"]: item["endpoint"] for item in sources}
    if len(endpoints) != len(sources):
        raise ValueError("Duplicate expert source endpoint")
    if not writes:
        raise ValueError("Expert delivery requires complete logical coverage")
    target = writes[0]["target"]
    if not target or any(write["target"] != target for write in writes):
        raise ValueError("Expert delivery has a misrouted logical target")
    if validate_coverage(writes, layouts, source_world_size=geometry["source_world_size"]) != set(layouts):
        raise ValueError("Expert delivery requires complete logical coverage")
    # Preserve final IDs. Coalescing again can discover new adjacent spans and
    # would invalidate the partition the source and receiver are binding.
    logical = TransferPlan(target, geometry["source_world_size"], tuple(writes))
    delivery = logical.delivery(endpoints, receiver)
    logical_hash = fingerprint(dict(geometry=geometry, writes=writes))
    result = dict(logical_plan_hash=logical_hash, delivery=delivery)
    return dict(result, plan_hash=fingerprint(result))
