# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual source coordinates and native weight layouts for expert, vocabulary
and TP shards."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixShard:
    """Logical matrix coordinates, independent of rank and transport."""

    offset: tuple[int, int]
    shape: tuple[int, int]
    storage_offset: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class TensorRegion:
    """A logical shard mapped to a row-strided local view, in elements.

    Adapters supply this view after any model-specific conversion. It contains
    no tensor, address or registration, so a cached plan can bind fresh
    storage.
    """

    shard: MatrixShard
    row_stride: int
    itemsize: int
    storage_offset: int = 0

    def __post_init__(self) -> None:
        shard = self.shard
        if (
            any(len(values) != 2 for values in (shard.offset, shard.shape, shard.storage_offset))
            or any(type(n) is not int or n < 0 for n in (*shard.offset, *shard.storage_offset, self.storage_offset))
            or any(type(n) is not int or n <= 0 for n in (*shard.shape, self.row_stride, self.itemsize))
            or shard.storage_offset[1] + shard.shape[1] > self.row_stride
        ):
            raise ValueError("Invalid logical shard or local storage geometry")

    @classmethod
    def from_descriptor(cls, region: Mapping, *, itemsize: int) -> "TensorRegion":
        return cls(
            MatrixShard(tuple(region["offset"]), tuple(region["shape"])),
            region["stride"],
            itemsize,
            region["storage_offset"],
        )


@dataclass(frozen=True)
class ExpertGeometry:
    """Canonical expert matrices and native TP storage, without live bindings.

    Raw IPC views and packed network blocks use the same geometry. Choosing
    owners, mapping native names, and qualifying a loader remain adapter work.
    """

    num_experts: int
    hidden: int
    intermediate: int
    target_tp: int
    up_proj_first: bool = False

    def __post_init__(self) -> None:
        if (
            any(
                type(n) is not int or n <= 0
                for n in (self.num_experts, self.hidden, self.intermediate, self.target_tp)
            )
            or self.intermediate % self.target_tp
            or type(self.up_proj_first) is not bool
        ):
            raise ValueError("Unsupported expert geometry")

    def component_shape(self, component: str, source_etp: int = 1) -> tuple[int, int]:
        if type(source_etp) is not int or source_etp <= 0 or self.intermediate % source_etp:
            raise ValueError("Invalid source expert tensor-parallel coordinate")
        width = self.intermediate // source_etp
        if component == "down":
            return self.hidden, width
        if component not in ("gate", "up"):
            raise ValueError("Unknown expert component")
        return width, self.hidden

    def native_shape(self, projection: str, count: int | None = None) -> tuple[int, int, int]:
        count = self.num_experts if count is None else count
        if type(count) is not int or not 0 < count <= self.num_experts:
            raise ValueError("Invalid expert block count")
        width = self.intermediate // self.target_tp
        if projection == "gate_up":
            return count, 2 * width, self.hidden
        if projection != "down":
            raise ValueError("Unknown native expert projection")
        return count, self.hidden, width

    def canonical_shape(self, projection: str, count: int) -> tuple[int, int, int]:
        self.native_shape(projection, count)
        return (
            (count, 2 * self.intermediate, self.hidden)
            if projection == "gate_up"
            else (count, self.hidden, self.intermediate)
        )

    def component_regions(
        self,
        expert: int,
        component: str,
        target_rank: int,
        *,
        source_offset: int = 0,
        source_etp: int = 1,
        source_etp_rank: int = 0,
    ) -> tuple[TensorRegion, TensorRegion]:
        shape = self.component_shape(component, source_etp)
        if (
            type(expert) is not int
            or not 0 <= expert < self.num_experts
            or type(target_rank) is not int
            or not 0 <= target_rank < self.target_tp
            or type(source_etp_rank) is not int
            or not 0 <= source_etp_rank < source_etp
        ):
            raise ValueError("Invalid expert source or target coordinate")
        width = self.intermediate // self.target_tp
        start = source_etp_rank * (self.intermediate // source_etp)
        if component == "down":
            source = TensorRegion(MatrixShard((0, start), shape), shape[1], 2, source_offset)
            target = TensorRegion(
                MatrixShard((0, target_rank * width), (self.hidden, width)),
                width,
                2,
                expert * self.hidden * width,
            )
        else:
            slot = int((component == "up") != self.up_proj_first)
            source = TensorRegion(MatrixShard((start, 0), shape), self.hidden, 2, source_offset)
            target = TensorRegion(
                MatrixShard((target_rank * width, 0), (width, self.hidden)),
                self.hidden,
                2,
                (expert * 2 * width + slot * width) * self.hidden,
            )
        return source, target

    def native_block_regions(self, projection: str, begin: int, count: int) -> tuple[TensorRegion, TensorRegion]:
        tail = self.native_shape(projection, count)[1:]
        if type(begin) is not int or begin < 0 or begin + count > self.num_experts:
            raise ValueError("Native expert block exceeds expert ownership")
        width = math.prod(tail)
        return (
            TensorRegion(MatrixShard((begin, 0), (count, width)), width, 2),
            TensorRegion(MatrixShard((0, 0), (self.num_experts, width)), width, 2),
        )

    def block_copy_slices(self, projection: str, count: int, target_rank: int) -> tuple:
        self.native_shape(projection, count)
        if type(target_rank) is not int or not 0 <= target_rank < self.target_tp:
            raise ValueError("Invalid target TP rank")
        width = self.intermediate // self.target_tp
        begin = target_rank * width
        all_experts = slice(None)
        if projection == "down":
            return (((all_experts, all_experts, slice(begin, begin + width)), (...,)),)
        return tuple(
            (
                (all_experts, slice(part * self.intermediate + begin, part * self.intermediate + begin + width)),
                (all_experts, slice(slot * width, (slot + 1) * width)),
            )
            for part, slot in ((0, int(self.up_proj_first)), (1, int(not self.up_proj_first)))
        )
