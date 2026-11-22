# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validated physical destinations for target-local BF16 expert coordinates."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from relax.utils.sparse_tensor import storage_identity


if TYPE_CHECKING:
    from .delta_target_lane import ExpertTargets


def marked_ordinals(marker: dict, rank: int, count: int) -> list[int]:
    if (
        not isinstance(marker, dict)
        or set(marker) != {"format", "tp_size", "tp_rank", "entries"}
        or any(type(marker[k]) is not int for k in ("format", "tp_size", "tp_rank"))
        or (marker["format"], marker["tp_size"], marker["tp_rank"]) != (1, 4, rank)
    ):
        raise ValueError("Invalid target shard lane binding")
    ordinals = marker["entries"]
    if (
        not isinstance(ordinals, list)
        or not 0 < len(ordinals) <= count
        or any(
            type(n) is not int or not 0 <= n < count or (i and n <= ordinals[i - 1]) for i, n in enumerate(ordinals)
        )
    ):
        raise ValueError("Invalid target shard entry ordinals")
    return ordinals


def _byte_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Enclose all addressed bytes of a strided view on its own device."""
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError("Cannot establish target tensor address range")
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided:
        # Ordinary FULL may own SGLang Parameter subclasses. Their entire
        # allocation is a conservative alias bound independent of shard layout.
        storage = tensor.untyped_storage()
        size, base = storage.nbytes(), storage.data_ptr()
        if size and not base:
            raise ValueError("Cannot establish live allocation address range")
        return (base, base + size) if size else None
    if not tensor.numel():
        return None
    strides = tensor.stride()
    if min(strides, default=0) < 0:
        raise ValueError("Cannot establish target tensor address range for negative strides")
    width = tensor.element_size()
    storage = tensor.untyped_storage()
    start = tensor.data_ptr()
    base = storage.data_ptr()
    # Holes in a strided static buffer are conservatively considered occupied.
    size = width * (1 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, strides, strict=True)))
    if (
        not base
        or not base <= start <= base + storage.nbytes() - size
        or start != base + tensor.storage_offset() * width
    ):
        raise ValueError("Ordered tensor address range exceeds its storage")
    return start, start + size


def _live_tensors(model: Any) -> list[tuple[str, torch.Tensor]]:
    return [
        *(("buffer", b) for b in model.buffers()),
        *(("param", p) for _, p in model.named_parameters(remove_duplicate=False)),
    ]


def _tensor_attributes(model: Any) -> list[tuple[str, torch.Tensor]]:
    # Native model loaders also store derived weights directly on a module,
    # e.g. the MLA K/V projections. These are live bytes, but are not writable
    # ordinary parameter destinations.
    return [
        ("buffer", value)
        for module in model.modules()
        for value in vars(module).values()
        if isinstance(value, torch.Tensor)
    ]


def _check_address_ranges(live: list[tuple[str, torch.Tensor]], io: tuple[torch.Tensor, ...]) -> bool:
    """Reject transport collisions and report whether live parameters are
    disjoint."""
    by_device = {}
    for kind, tensor in chain(live, (("io", t) for t in io)):
        span = _byte_span(tensor)
        if span is not None:
            by_device.setdefault(tensor.device, []).append((*span, kind))
    eligible = True
    for ranges in by_device.values():
        any_end = io_end = param_end = 0
        for start, end, kind in sorted(ranges):
            if start < io_end or (kind == "io" and start < any_end):
                raise ValueError("Target transport byte ranges overlap another input, status or live model tensor")
            if kind == "io":
                io_end = end
            if start < param_end or (kind == "param" and start < any_end):
                eligible = False
            if kind == "param":
                param_end = max(param_end, end)
            any_end = max(any_end, end)
    return eligible


def validate_transport_storage(model: Any, io: tuple[torch.Tensor, ...]) -> None:
    # Final ACK only needs safe I/O, even when the ordinary layout cannot shard.
    devices = {tensor.device for tensor in io}
    _check_address_ranges(
        [(kind, t) for kind, t in (*_live_tensors(model), *_tensor_attributes(model)) if t.device in devices], io
    )


def validate_shard_storage(
    model: Any, destinations: frozenset, io: tuple[torch.Tensor, ...], direct: frozenset | None = None
) -> bool:
    live = _live_tensors(model)
    if direct is not None:
        # Only the direct expert parameters must be disjoint. Ordinary dense
        # parameters can be tied, padded, FP32, or SGLang Parameter subclasses.
        ranges = [(kind if id(tensor) in direct else "buffer", tensor) for kind, tensor in live]
        eligible = _check_address_ranges(ranges + _tensor_attributes(model), io)
        return (
            eligible
            and all(type(t) is torch.Tensor for t in io)
            and {storage_identity(t) for _, t in live} == destinations
        )
    if not _check_address_ranges(live + _tensor_attributes(model), io) or any(
        type(t) is not torch.Tensor or t.layout != torch.strided for t in io
    ):
        return False
    storage = set()
    for kind, tensor in live:
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided:
            return False
        identity = storage_identity(tensor)
        if kind == "param" and (
            tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            or identity in storage
        ):
            return False
        storage.add(identity)
    return storage == destinations


@dataclass(frozen=True)
class CopyRegion:
    """A contiguous destination receives rows of width from a strided HF
    view."""

    weight: torch.Tensor
    base: int
    offset: int
    rows: int
    stride: int
    width: int


class BroadcastTargets(TorchDispatchMode):
    """Observe real seed copies without replacing inputs or repeating a loader.

    Only a dispatcher with an independent copy-only input contract may enter
    this mode. ATen cannot observe arbitrary Python cache or lifecycle effects.
    The mode runs while the dispatcher consumes one complete, ordinary input.
    """

    broadcast = True

    def __init__(self, model: Any):
        super().__init__()
        self.owners = {
            storage_identity(p): p
            for p in model.parameters()
            if isinstance(p, torch.nn.Parameter)
            and p.layout == torch.strided
            and p.device.type in ("cpu", "cuda")
            and p.is_contiguous()
            and not p.is_neg()
            and not p.is_conj()
            and type(p).__torch_dispatch__ is torch.nn.Parameter.__torch_dispatch__
            and type(p).__torch_function__ is torch.nn.Parameter.__torch_function__
        }
        # A transient DLPack view can have another StorageImpl for live bytes.
        # Merge physical intervals to recognize every write to model storage.
        ranges = {}
        for _, tensor in _live_tensors(model):
            if tensor.device.type in ("cpu", "cuda"):
                span = _byte_span(tensor)
                if span is not None:
                    ranges.setdefault(tensor.device, []).append(span)
        self.ends = {}
        self.starts = {}
        for device, intervals in ranges.items():
            starts, ends = [], []
            for lo, hi in sorted(intervals):
                if ends and lo <= ends[-1]:
                    ends[-1] = max(ends[-1], hi)
                else:
                    starts.append(lo)
                    ends.append(hi)
            self.starts[device], self.ends[device] = starts, ends
        self.targets: dict[str, CopyRegion | None] = {}
        self.direct: frozenset[int] = frozenset()
        self.writes: dict[tuple, list] = {}
        self.conflicts: set[str] = set()
        self.name = ""
        self.source = None

    def observe(self, name: str, value: torch.Tensor):
        self.name, self.source = name, value
        self.source_storage = storage_identity(value)
        self.region = None
        self.supported = value.dtype in (torch.float16, torch.bfloat16, torch.float32) and value.numel() > 0
        try:
            yield name, value
            if self.supported:
                self.targets[name] = self.region
        finally:
            self.name = ""
            self.source = None

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        for i, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = args[i] if i < len(args) else kwargs.get(argument.name)
            if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                span = _byte_span(value)
                if span is not None and value.device in self.starts:
                    idx = bisect_right(self.ends[value.device], span[0])
                    starts = self.starts[value.device]
                    if idx < len(starts) and starts[idx] < span[1]:
                        self.writes.setdefault(value.device, []).append((*span, self.name))
        if self.source is not None and self.supported:
            if func == torch.ops.aten.copy_.default:
                dest, source = args[:2]
                region = self._copy(dest, source)
                if region is None or self.region is not None:
                    self.supported = False
                else:
                    self.region = region
            elif func._schema.name not in {
                "aten::alias",
                "aten::as_strided",
                "aten::detach",
                "aten::select",
                "aten::slice",
                "aten::view",
                "aten::squeeze",
                "aten::unsqueeze",
                "aten::transpose",
                "aten::t",
                "aten::permute",
            }:
                # Reject arithmetic, additional stores and value-dependent decisions.
                # Geometry work on unrelated runtime tensors can conservatively decline.
                touches_input = any(
                    isinstance(t, torch.Tensor)
                    and t.layout == torch.strided
                    and storage_identity(t) == self.source_storage
                    for t in tree_leaves((args, kwargs))
                )
                if touches_input or any(
                    a.alias_info is not None and a.alias_info.is_write for a in func._schema.arguments
                ):
                    self.supported = False
        return func(*args, **kwargs)

    def _copy(self, dest: torch.Tensor, source: torch.Tensor) -> CopyRegion | None:
        weight = self.owners.get(storage_identity(dest))
        if (
            weight is None
            or storage_identity(source) != storage_identity(self.source)
            or dest.dtype != source.dtype
            or weight.dtype != source.dtype
            or source.dtype != self.source.dtype
            or source.shape != dest.shape
            or not dest.is_contiguous()
            or source.is_conj()
            or source.is_neg()
            or dest.is_neg()
            or dest.is_conj()
            or not source.numel()
        ):
            return None
        base = dest.storage_offset() - weight.storage_offset()
        offset = source.storage_offset() - self.source.storage_offset()
        if source.is_contiguous():
            rows, width, stride = 1, source.numel(), source.numel()
        elif source.ndim == 2 and source.stride(1) == 1 and source.stride(0) >= source.shape[1]:
            (rows, width), stride = source.shape, source.stride(0)
        else:
            return None
        if not (
            0 <= base <= weight.numel() - dest.numel()
            and 0 <= offset <= self.source.numel() - (rows - 1) * stride - width
        ):
            return None
        return CopyRegion(weight, base, offset, rows, stride, width)

    def seal(self, receiver: Any, *, final_mask: bool) -> bool:
        # Sweep connected components of overlapping physical intervals. If a
        # component has multiple writers, every name in it must replay FULL
        # in native order, even when that writer's source bits did not change.
        for writes in self.writes.values():
            end, first, mixed = 0, None, False
            for lo, hi, name in sorted(writes):
                if lo >= end:
                    first, mixed = name, False
                elif name != first:
                    self.conflicts.add(first)
                    mixed = True
                if mixed:
                    self.conflicts.add(name)
                end = max(end, hi)
        if "" in self.conflicts:
            # A buffered loader wrote live bytes outside the active input. We
            # cannot name every writer of that interval in a per-input ACK.
            return False
        if final_mask:
            receiver.full_only.update(self.conflicts)
            for name in self.conflicts:
                self.targets.pop(name, None)
        self.writes.clear()
        self.starts.clear()
        self.ends.clear()
        self.direct = frozenset(id(r.weight) for r in self.targets.values() if r is not None)
        self.signature = self._bindings(receiver.model)
        self.by_owner = {binding[0]: (name, binding) for name, binding in self.signature}
        self.owners.clear()
        return (
            (bool(self.targets) or (final_mask and receiver.full_only.issuperset(receiver.schema)))
            and not (self.conflicts - receiver.full_only)
            and validate_shard_storage(receiver.model, receiver.destinations, (), self.direct)
        )

    @staticmethod
    def _binding(t: torch.Tensor) -> tuple:
        return id(t), storage_identity(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype

    def _bindings(self, model: Any) -> tuple:
        return tuple(
            (name, self._binding(t))
            for name, t in model.named_parameters(remove_duplicate=False)
            if id(t) in self.direct
        )

    def check_run(self, model: Any, batch: list) -> None:
        # The dispatcher has executed its per-invocation setup since frame
        # admission. Verify the actual parameters touched by this run before
        # using the captured references, including .data storage replacement.
        for (weight, _layout), _, _, _ in batch:
            if weight is not None:
                name, binding = self.by_owner[id(weight)]
                current = model.get_parameter(name)
                if current is not weight or self._binding(current) != binding:
                    raise ValueError("Ordinary loader setup changed an admitted destination")

    def current(self, receiver: Any, raw: torch.Tensor, status: torch.Tensor) -> bool:
        return self.signature == self._bindings(receiver.model) and validate_shard_storage(
            receiver.model, receiver.destinations, (raw, status), self.direct
        )


class ShardFrame:
    """One bounded tile directory and ordered runs sharing the original consume
    fence."""

    BLOCK = 512
    # Limit each vector of temporary int64 tile indices to 256 KiB.
    _HOST_BATCH_TILES = 32 * 1024

    @classmethod
    def ordered_workspace(cls, codes: list) -> int:
        # Six int64s per tile, validation flags and the group reduction. Include
        # CUDA's minimum allocation granularity for each retained frame.
        tiles = sum((count + cls.BLOCK - 1) // cls.BLOCK for _, count in codes)
        return ((tiles * 48 + 511) // 512 * 512 + 3 * 512) if tiles else 0

    def __init__(
        self,
        plan: ExpertTargets | BroadcastTargets,
        ordinals: frozenset,
        entries: list,
        raw: torch.Tensor,
        ends: list[int] | None = None,
        *,
        lane_ordinals: frozenset | None = None,
    ):
        self.plan = plan
        self.raw = raw
        self.runs = []
        self.local_words = (
            torch.zeros((), dtype=torch.int64, device=raw.device)
            if (plan.broadcast or (lane_ordinals is not None and ordinals != lane_ordinals)) and raw.is_cuda
            else 0
        )
        descriptors = np.empty((len(ordinals), 7), dtype=np.int64)
        tile_count = descriptor_count = 0
        self.local_checks = []
        selected = set(ordinals)
        self.group_runs = []
        ends = iter(ends or [len(entries)])
        end, group_start = next(ends), 0
        start = 0
        while start < len(entries):
            direct = start in selected
            stop = start + 1
            while stop < end and (stop in selected) == direct:
                stop += 1
            if not direct:
                self.runs.append((False, entries[start:stop]))
            else:
                groups = {}
                for index, (entry, positions, values) in enumerate(entries[start:stop], start):
                    name = entry["name"]
                    global_coords = plan.broadcast or (lane_ordinals is not None and index not in lane_ordinals)
                    targets = plan.bindings.targets if global_coords and not plan.broadcast else plan.targets
                    if name not in targets or positions is None:
                        raise ValueError("A direct entry needs an acknowledged sparse input")
                    if global_coords:
                        region = targets[name]
                        weight = region.weight if region is not None else None
                        base = region.base if region is not None else 0
                        layout = (region.offset, region.rows, region.stride, region.width) if region else None
                        limit = math.prod(entry["shape"])
                    else:
                        weight, base, limit = plan.targets[name]
                        layout = None
                    if (weight is not None and (weight.device != raw.device or weight.dtype != values.dtype)) or entry[
                        "count"
                    ] > limit:
                        raise ValueError("Target shard has an invalid physical device, dtype or coordinate count")
                    groups.setdefault((weight, layout), []).append((entry, positions, values, base, limit))
                batch = []
                for (weight, layout), parts in groups.items():
                    first = tile_count
                    for entry, positions, values, base, limit in parts:
                        count = entry["count"]
                        if raw.device.type == "cpu":
                            self.local_checks.append((positions, limit))
                        if count:
                            descriptors[descriptor_count] = (
                                tile_count,
                                entry["offset"],
                                entry["value_offset"],
                                count,
                                base,
                                limit,
                                values.element_size(),
                            )
                            descriptor_count += 1
                            tile_count += (count + self.BLOCK - 1) // self.BLOCK
                    batch.append(
                        (
                            (weight, layout),
                            first,
                            tile_count,
                            parts if raw.device.type == "cpu" else None,
                        )
                    )
                self.runs.append((True, batch))
            if stop == end:
                self.group_runs.append(self.runs[group_start:])
                group_start = len(self.runs)
                end = next(ends, len(entries))
            start = stop
        self.table = self._tile_table(descriptors[:descriptor_count], tile_count, raw) if tile_count else None

    @classmethod
    def _tile_table(cls, descriptors: np.ndarray, tile_count: int, raw: torch.Tensor) -> torch.Tensor:
        host = torch.empty((tile_count, 6), dtype=torch.int64, device="cpu", pin_memory=raw.is_cuda)
        table = host.numpy()
        first, position_offset, value_offset, count, base, limit, word_size = descriptors.T
        for start in range(0, tile_count, cls._HOST_BATCH_TILES):
            stop = min(start + cls._HOST_BATCH_TILES, tile_count)
            rows = table[start:stop]
            tiles = np.arange(start, stop, dtype=np.int64)
            owner = np.searchsorted(first, tiles, side="right") - 1
            offset = (tiles - first[owner]) * cls.BLOCK
            rows[:, 0] = position_offset[owner] + 4 * offset
            rows[:, 1] = value_offset[owner] + word_size[owner] * offset
            rows[:, 2] = np.minimum(count[owner] - offset, cls.BLOCK)
            rows[:, 3] = offset != 0
            rows[:, 4] = base[owner]
            rows[:, 5] = limit[owner]
        return host.to(raw.device, non_blocking=True)

    def invalid_positions(self) -> torch.Tensor | None:
        if self.table is None:
            return None
        if self.raw.device.type == "cpu":
            return torch.stack(
                [((pos < 0) | (pos >= limit)).any() | (pos[1:] <= pos[:-1]).any() for pos, limit in self.local_checks]
            ).any()
        from .delta_shard_kernel import shard_tiles

        self.invalid = torch.zeros(1, dtype=torch.int32, device=self.raw.device)
        shard_tiles[(self.table.shape[0],)](self.raw, self.table, self.raw, self.invalid, False, self.BLOCK)
        return self.invalid[0].bool()

    def scatter(self, batch: list) -> None:
        for (weight, layout), first, end, parts in batch:
            if end == first or weight is None:
                continue
            word_type = torch.int16 if weight.element_size() == 2 else torch.int32
            target = weight.detach().view(word_type).view(-1)
            if self.raw.device.type == "cpu":
                for _, pos, values, base, _ in parts:
                    words, pos = values.view(word_type), pos.long()
                    if layout is not None:
                        offset, rows, stride, width = layout
                        rel = pos - offset
                        keep = (rel >= 0) & (rel < rows * stride) & (rel % stride < width)
                        rel, words = rel[keep], words[keep]
                        pos = (rel // stride) * width + rel % stride
                        self.local_words += pos.numel()
                    target.index_copy_(0, pos + base, words)
            else:
                from .delta_shard_kernel import shard_tiles

                shard_tiles[(end - first,)](
                    self.raw,
                    self.table[first:end],
                    target,
                    self.invalid,
                    True,
                    self.BLOCK,
                    REGION=layout is not None,
                    OFFSET=layout[0] if layout else 0,
                    ROWS=layout[1] if layout else 1,
                    STRIDE=layout[2] if layout else 1,
                    WIDTH=layout[3] if layout else 1,
                    WORD_BYTES=weight.element_size(),
                    written=self.local_words if layout else self.invalid,
                )
