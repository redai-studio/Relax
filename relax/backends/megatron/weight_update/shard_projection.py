# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from verl (Copyright 2025 Bytedance Ltd. and/or its affiliates),
# licensed under the Apache License, Version 2.0.

"""Communication-free local shard projection for the Bridge FULL export.

Adapted from verl 78105206, workers/engine/megatron/delta_export.py. Relax
reassembles TP before invoking Bridge. Apply that placement to explicit
indices, then apply the same Bridge mapping to determine HF slots and rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass, field
from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.utils.types import ParamInfo


Slot = tuple[str, tuple[int, ...], torch.dtype]
Entry = tuple[str, tuple[int, ...], torch.Tensor, torch.Tensor]
Slice = tuple[int, int, int]  # source start, source stop, destination start


class UnsupportedShardProjection(ValueError):
    """FULL conversion has no supported local, element-wise projection."""


# Exact Bridge implementation classes. An arbitrary subclass can couple
# values, read another live tensor, or invoke its own collective.
_DIRECT = {"DirectMapping", "AutoMapping", "ColumnParallelMapping", "RowParallelMapping", "ReplicatedMapping"}
_EXPERT = {"GatedMLPMapping", "FusedExpertMapping", "FusedGatedExpertMapping"}
_QKV = {"QKVMapping", "ConcatenatedQKVMapping"}
_ROWS = _QKV | {"GDNConv1dMapping", "GDNLinearMapping", "GDNLinearMappingSeparate"}
_NORM = "RMSNorm2ZeroCenteredRMSNormMapping"
_LEGACY_EXPERT = {"ExpertMLPDownProjMapping", "ExpertMLPGateUpProjMapping"}
_WORDS = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@dataclass
class _RowIndex:
    """Constant-offset runs of Bridge's row permutation, shared across
    layers."""

    cpu: torch.Tensor
    cache_on_device: bool
    devices: dict[torch.device, torch.Tensor] = field(default_factory=dict)

    def take(self, rows: torch.Tensor) -> torch.Tensor:
        table = self.devices.get(rows.device)
        if table is None:
            table = self.cpu.to(rows.device)
            if self.cache_on_device:
                self.devices[rows.device] = table
        run = torch.searchsorted(table[0, 1:], rows, right=True)
        return rows + table[1].index_select(0, run)


@dataclass(frozen=True)
class _Placement:
    """Local Megatron shard to the contiguous input of
    BridgeConverter.convert."""

    shape: tuple[int, ...]
    full_shape: tuple[int, ...]
    block: int
    rank: int
    size: int
    components: tuple[int, ...]  # flattened lengths per local TP component

    @classmethod
    def build(cls, info: ParamInfo, rank: int, size: int, split_sizes: tuple[int, ...] | None) -> _Placement:
        name = info.name.replace(".to_wrap.", ".")
        shape = tuple(info.shape)
        if size < 1 or not 0 <= rank < size or any(n < 1 for n in shape):
            raise UnsupportedShardProjection(f"{name}: invalid shard placement {shape} rank={rank}/{size}")
        attrs = info.attrs
        if (
            size == 1
            or not attrs.get("tensor_model_parallel")
            or attrs.get("parallel_mode") == "duplicated"
            or "expert_bias" in name
        ):
            return cls(shape, shape, prod(shape), 0, 1, ())
        axis = attrs.get("partition_dim", -1)
        if axis not in range(-len(shape), len(shape)):
            raise UnsupportedShardProjection(f"{name}: invalid TP partition dimension {axis}")
        if "linear_fc1" not in name and attrs.get("partition_stride", 1) != 1:
            raise UnsupportedShardProjection(f"{name}: interleaved TP stride")
        # Same component order as common._restore_tp_weight, including the
        # grouped FC2 metadata which advertises axis zero for a column shard.
        gated = attrs.get("gated_fc1")
        if gated is None:
            gated = "linear_fc1.weight" in name
        if "vision_model" not in name:
            if gated and re.search(r"(?:^|\.)linear_fc1\.(?:weight|bias)\d*$", name):
                if shape[0] % 2:
                    raise UnsupportedShardProjection(f"{name}: uneven gated FC1 rows")
                split_sizes = (shape[0] // 2,) * 2
            if "linear_fc2.weight" in name and axis == 0:
                axis = 1
        if axis >= len(shape):
            raise UnsupportedShardProjection(f"{name}: missing TP axis {axis}")
        if split_sizes is not None and (axis != 0 or sum(split_sizes) != shape[0] or min(split_sizes) < 1):
            raise UnsupportedShardProjection(f"{name}: incompatible TP components {split_sizes} for {shape}")
        axis %= len(shape)
        full_shape = list(shape)
        full_shape[axis] *= size
        row_width = prod(shape[1:])
        components = tuple(rows * row_width for rows in split_sizes or ())
        return cls(shape, tuple(full_shape), prod(shape[axis:]), rank, size, components)

    def flat_slices(self) -> tuple[Slice, ...] | None:
        if self.block != prod(self.shape):
            return None
        start = 0
        slices = []
        for count in self.components or (self.block,):
            slices.append((start, start + count, start * self.size + self.rank * count))
            start += count
        return tuple(slices)

    def full_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return indices
        if self.components:
            # Component boundaries are deterministic Python integers; the
            # operations below scale only with this rank's changed entries.
            offset = self.rank * self.components[0]
            result = indices + offset
            start = 0
            for previous, current in zip(self.components, self.components[1:]):
                start += previous
                new_offset = start * (self.size - 1) + self.rank * current
                result = torch.where(indices >= start, indices + new_offset, result)
            return result
        return indices + (indices // self.block * (self.size - 1) + self.rank) * self.block


@dataclass
class ShardProjection:
    slots: tuple[Slot, ...]
    local_slices: tuple[Slice, ...] | None
    _placement: _Placement
    _dtype: torch.dtype
    _transposes: tuple[bool, ...]
    _rows: _RowIndex | None = None
    _values: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def transforms_values(self) -> bool:
        """Whether distinct source words can round to the same emitted
        value."""
        return self._values is not None

    def project(self, changed_local_idx: torch.Tensor, current_shard: torch.Tensor) -> list[Entry]:
        """Project sorted unique raw-bit change indices, retaining all empty HF
        slots."""
        if tuple(current_shard.shape) != self._placement.shape or current_shard.dtype != self._dtype:
            raise ValueError("Shard geometry or dtype changed after projection layout was built")
        if (
            changed_local_idx.ndim != 1
            or changed_local_idx.dtype not in (torch.int32, torch.int64)
            or changed_local_idx.device != current_shard.device
        ):
            raise ValueError("Expected a one-dimensional integral index on the shard device")
        if changed_local_idx.numel() == 0:
            empty_idx = changed_local_idx.to(torch.int32)
            return [
                (name, shape, empty_idx, current_shard.new_empty(0, dtype=dtype)) for name, shape, dtype in self.slots
            ]

        # Index raw words. This also covers dtypes without take/index_select
        # kernels (e.g. uint32). torch.take respects strided native parameters.
        word_dtype = _WORDS[current_shard.element_size()]
        local = changed_local_idx.long()
        values = torch.take(current_shard.detach().view(word_dtype), local)
        if self._values is not None:
            values = self._values(values.view(self._dtype)).view(word_dtype)
        if self.local_slices is not None and len(self.slots) == 1:
            lo, hi, offset = self.local_slices[0]
            if lo == 0 and hi == current_shard.numel():
                name, shape, dtype = self.slots[0]
                positions = local + offset if offset else local
                return [(name, shape, positions.to(torch.int32), values.view(dtype))]
        # An individual HF slot is int32-addressable. The concatenated source
        # and row arithmetic may be larger, so keep intermediates in int64.
        indices = self._placement.full_indices(local)
        if self._rows is not None:
            width = prod(self._placement.full_shape[1:])
            indices = self._rows.take(indices // width).long() * width + indices % width

        result = []
        start = 0
        for (name, shape, dtype), transpose in zip(self.slots, self._transposes, strict=True):
            count = prod(shape)
            if count == prod(self._placement.full_shape):
                positions, selected = indices, values
            else:
                valid = ((indices >= start) & (indices < start + count)).nonzero().flatten()
                positions, selected = indices.index_select(0, valid) - start, values.index_select(0, valid)
            if transpose:
                # The untransposed source segment has shape (shape[1], shape[0]).
                positions = positions % shape[0] * shape[1] + positions // shape[0]
            if transpose or self._rows is not None:
                positions, order = positions.sort()
                selected = selected.index_select(0, order)
            result.append((name, shape, positions.to(torch.int32), selected.view(dtype)))
            start += count
        return result


def _local_slices(placement: _Placement, slots: tuple[Slot, ...]) -> tuple[Slice, ...] | None:
    source = placement.flat_slices()
    if source is None:
        return None
    result = []
    start = 0
    for _, shape, _ in slots:
        stop = start + prod(shape)
        overlaps = []
        for local_start, local_stop, full_start in source:
            lo, hi = max(start, full_start), min(stop, full_start + local_stop - local_start)
            if lo < hi:
                overlaps.append((local_start + lo - full_start, local_start + hi - full_start, lo - start))
        if len(overlaps) > 1:
            return None
        result.append(overlaps[0] if overlaps else (0, 0, 0))
        start = stop
    return tuple(result)


def _mapping_kind(mapping: MegatronParamMapping) -> str:
    cls = type(mapping)
    if cls.__module__ == "megatron.bridge.models.conversion.param_mapping":
        if cls.__name__ in _DIRECT | _EXPERT | _ROWS | {_NORM}:
            return cls.__name__
    elif cls.__module__ == "megatron.bridge.models.qwen_vl.qwen3_vl_bridge" and cls.__name__ in _LEGACY_EXPERT:
        return cls.__name__
    raise UnsupportedShardProjection(f"{cls.__module__}.{cls.__name__}: no independent coordinate transform")


class ShardProjector:
    def __init__(
        self, converter: BridgeConverter, gdn_split_sizes: Mapping[str, tuple[int, ...] | None] | None = None
    ) -> None:
        self.converter = converter
        self.gdn_split_sizes = gdn_split_sizes or {}
        self._row_maps: dict[tuple, _RowIndex | None] = {}
        self._matrix_transposes: dict[tuple, tuple[bool, ...]] = {}

    def layout(self, info: ParamInfo, tp_rank: int, tp_size: int) -> ShardProjection:
        """Compile a reusable owner layout from the initialized FULL Bridge
        directory."""
        task = self.converter.local_projection_task(info.name)
        if task is None:
            raise UnsupportedShardProjection(f"{info.name}: missing local Bridge task, stateful or quantized export")
        if info.dtype.itemsize not in _WORDS:
            raise UnsupportedShardProjection(f"{info.name}: no lossless integer word for {info.dtype}")
        mapping = task.mapping
        kind = _mapping_kind(mapping)
        for child in self.converter.collect_all_mappings(mapping):
            if "megatron_to_hf" in vars(child):
                raise UnsupportedShardProjection(f"{info.name}: custom instance conversion")
            if child is not mapping:
                cls = type(child)
                if cls.__module__ != "megatron.bridge.models.conversion.param_mapping" or cls.__name__ not in (
                    _DIRECT | {"GatedMLPMapping", "_LooseGatedMLPMapping"}
                ):
                    raise UnsupportedShardProjection(f"{info.name}: unsupported nested mapping {cls.__name__}")
            if getattr(child, "permute_dims", None) is not None:
                raise UnsupportedShardProjection(f"{info.name}: permuted mapping {type(child).__name__}")
            if hasattr(child, "_detect_parallelism_type") and child._mapping is None:
                raise UnsupportedShardProjection(f"{info.name}: uninitialized AutoMapping")
        name = info.name.replace(".to_wrap.", ".")
        try:
            splits = self.gdn_split_sizes.get(name)
            if kind.startswith("GDN") and name not in self.gdn_split_sizes and tp_size > 1:
                from relax.backends.megatron.weight_update.common import gdn_component_sizes

                config = mapping._get_config(task.megatron_module)
                splits = gdn_component_sizes(config, tp_size, name)
            placement = _Placement.build(info, tp_rank, tp_size, splits)
            # Bridge's QKV helpers also construct small arange indices. Keep
            # these on CPU even if the actor changed the default device.
            with torch.device("cpu"):
                meta = torch.empty(placement.full_shape, device="meta", dtype=info.dtype)
                slots = tuple((key, tuple(t.shape), t.dtype) for key, t in self.converter.convert(info.name, meta))
                if not slots or any(prod(shape) > 2**31 or dtype != info.dtype for _, shape, dtype in slots):
                    raise UnsupportedShardProjection(f"{name}: empty, oversized or dtype-changing HF output")
                if len({key for key, _, _ in slots}) != len(slots) or any(
                    key.endswith((".experts.gate_up_proj", ".experts.down_proj")) for key, _, _ in slots
                ):
                    raise UnsupportedShardProjection(f"{name}: export combines multiple experts in one HF slot")
                return self._layout(info, task, kind, placement, slots)
        except (
            AssertionError,
            AttributeError,
            IndexError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            if isinstance(e, UnsupportedShardProjection):
                raise
            raise UnsupportedShardProjection(f"{name}: Bridge layout probe failed: {e}") from e

    def _layout(
        self, info: ParamInfo, task: WeightConversionTask, kind: str, placement: _Placement, slots: tuple[Slot, ...]
    ) -> ShardProjection:
        shape = placement.full_shape
        convert = self.converter.convert
        rows = None
        values = None
        transposes = (False,) * len(slots)
        if kind in _ROWS:
            rows = self._row_index(task, shape, slots)
        elif kind == _NORM:
            if len(shape) != 1 or len(slots) != 1 or slots[0][1] != shape:
                raise UnsupportedShardProjection(f"{info.name}: unexpected RMSNorm geometry")

            # Use the actual Bridge elementwise arithmetic in the FULL dtype.
            def values(selected: torch.Tensor) -> torch.Tensor:
                return convert(info.name, selected)[0][1]

        elif kind in _EXPERT | _LEGACY_EXPERT or (len(shape) == 2 and ".experts." in info.name):
            targets = task.mapping.hf_param
            targets = targets.values() if isinstance(targets, dict) else (targets,)
            signature = (
                type(task.mapping),
                getattr(task.mapping, "transpose_on_export", None),
                tuple(str(n).endswith((".experts.down_proj", ".experts.gate_up_proj")) for n in targets),
                bool(re.search(r"weight\d+$", info.name)),
                len(shape),
                len(slots),
            )
            if signature not in self._matrix_transposes:
                self._matrix_transposes[signature] = self._matrix_probe(info.name, len(shape), len(slots))
            transposes = self._matrix_transposes[signature]
            if not shape or shape[0] % len(slots):
                raise UnsupportedShardProjection(f"{info.name}: uneven HF segments")
            original = (shape[0] // len(slots), *shape[1:])
            for (_, output, _), transpose in zip(slots, transposes, strict=True):
                if output != (original[::-1] if transpose else original):
                    raise UnsupportedShardProjection(f"{info.name}: unsupported expert/MLP output {output}")
        elif len(slots) != 1 or not (
            slots[0][1] == shape or (shape and slots[0][1][1:] == shape[1:] and slots[0][1][0] <= shape[0])
        ):
            raise UnsupportedShardProjection(f"{info.name}: unexpected direct output geometry")
        slices = _local_slices(placement, slots) if rows is None and values is None and not any(transposes) else None
        return ShardProjection(slots, slices, placement, info.dtype, transposes, rows, values)

    def _matrix_probe(self, name: str, ndim: int, num_slots: int) -> tuple[bool, ...]:
        # Older expert mappings return transposed matrices; FULL also splits
        # fused gate/up and undoes certain transposes. Test the composition,
        # including square production weights, on a small, nonsquare matrix.
        if ndim not in (1, 2) or num_slots not in (1, 2):
            raise UnsupportedShardProjection(f"{name}: unsupported projection shape or multiplicity")
        tags = torch.arange(6 if ndim == 1 else 30, dtype=torch.int32, device="cpu")
        if ndim == 2:
            tags = tags.view(6, 5)
        chunks = tags.chunk(num_slots, 0)
        transposes = []
        for (_, out), chunk in zip(self.converter.convert(name, tags), chunks, strict=True):
            if torch.equal(out, chunk):
                transposes.append(False)
            elif ndim == 2 and torch.equal(out, chunk.T):
                transposes.append(True)
            else:
                raise UnsupportedShardProjection(f"{name}: unsupported packed expert/MLP transform")
        return tuple(transposes)

    def _row_index(
        self, task: WeightConversionTask, shape: tuple[int, ...], slots: tuple[Slot, ...]
    ) -> _RowIndex | None:
        # These mappings permute/split leading rows and retain every feature
        # column in order. One labelled column runs their actual Bridge code.
        # Keeping the real head dimension explicit avoids changing QKV grouping
        # when hidden_size is reduced to one in the private probe config.
        mapping = task.mapping
        config = mapping._get_config(task.megatron_module)
        key = (type(mapping), id(config), shape, tuple(s[1] for s in slots))
        if key not in self._row_maps:
            reduced = copy(config)
            if type(mapping).__name__ in _QKV:
                reduced.kv_channels = config.kv_channels or config.hidden_size // config.num_attention_heads
            reduced.hidden_size = 1
            tags = torch.arange(shape[0], dtype=torch.int32, device="cpu").view(shape[0], *(1 for _ in shape[1:]))
            with self.converter._disable_mapping_collectives(mapping):
                outputs = mapping.megatron_to_hf(tags, SimpleNamespace(config=reduced))
            row_ids = []
            for out, (_, actual, _) in zip(outputs.values(), slots, strict=True):
                if not actual or prod(actual[1:]) != prod(shape[1:]) or out.numel() != actual[0]:
                    raise UnsupportedShardProjection("Bridge compact row probe disagrees with FULL geometry")
                row_ids.append(out.reshape(-1))
            order = torch.cat(row_ids)
            expected = torch.arange(shape[0], dtype=torch.int32, device="cpu")
            if not torch.equal(order.sort().values, expected):
                raise UnsupportedShardProjection("Bridge row mapping is not a permutation")
            row_index = None
            if not torch.equal(order, expected):
                inverse = torch.empty_like(order)
                inverse[order.long()] = expected
                offsets = inverse - expected
                changes = (offsets[1:] != offsets[:-1]).nonzero().flatten() + 1
                starts = torch.cat((changes.new_zeros(1), changes))
                runs = torch.stack((starts.int(), offsets.index_select(0, starts)))
                # Very fragmented layouts (including one run per scalar of a
                # bias) can still project, without keeping a dense index on GPU.
                cache = runs.nbytes < min(inverse.nbytes, 256 * 1024)
                row_index = _RowIndex(runs, cache)
            self._row_maps[key] = row_index
        return self._row_maps[key]
