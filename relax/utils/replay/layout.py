# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed layout primitives for replay.

Pure CPU: topology, shard offsets, DP completeness, and canonical-tensor
reconstruction (or explicit rejection). No Ray/Megatron/torch.distributed. CP>1
stays unsupported in the frozen V1 matrix until parity evidence exists.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from relax.utils.replay.schema import StageCapability, StageId


class LayoutError(ValueError):
    """Raised when shard metadata is missing, incomplete, or inconsistent."""


@dataclass(frozen=True)
class RankTopology:
    """The parallel topology of a captured bundle."""

    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1

    @classmethod
    def from_rank_dict(cls, rank: Mapping[str, int]) -> "RankTopology":
        return cls(
            dp=int(rank.get("dp", 1)),
            tp=int(rank.get("tp", 1)),
            pp=int(rank.get("pp", 1)),
            cp=int(rank.get("cp", 1)),
        )

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


@dataclass(frozen=True)
class ShardSpec:
    """One rank-local shard of a logical tensor.

    offsets are the global start coordinates of the shard (None means "not
    recorded", which reconstruction rejects). unpadded_length is the number of
    real elements along the split dimension; the shard's shape may be larger
    when capture padded it.
    """

    rank: int
    offsets: tuple[int, ...] | None
    shape: tuple[int, ...]
    unpadded_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "offsets": list(self.offsets) if self.offsets is not None else None,
            "shape": list(self.shape),
            "unpadded_length": self.unpadded_length,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShardSpec":
        offsets = data.get("offsets")
        return cls(
            rank=int(data["rank"]),
            offsets=tuple(int(v) for v in offsets) if offsets is not None else None,
            shape=tuple(int(v) for v in data["shape"]),
            unpadded_length=data.get("unpadded_length"),
        )


@dataclass(frozen=True)
class TensorLayout:
    """Logical tensor shape plus its per-rank shard specs."""

    name: str
    dims: tuple[int, ...]
    split_dim: int = 0
    shards: dict[int, ShardSpec] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dims": list(self.dims),
            "split_dim": self.split_dim,
            "shards": {str(rank): spec.to_dict() for rank, spec in self.shards.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorLayout":
        return cls(
            name=str(data["name"]),
            dims=tuple(int(v) for v in data["dims"]),
            split_dim=int(data.get("split_dim", 0)),
            shards={int(rank): ShardSpec.from_dict(spec) for rank, spec in data.get("shards", {}).items()},
        )


def shard_order(ranks: Iterable[int]) -> list[int]:
    """Return ranks in a deterministic, sorted order."""
    return sorted(ranks)


def compute_offsets(
    name: str,
    lengths: Sequence[int],
    dims: Sequence[int],
    *,
    split_dim: int = 0,
    ranks: Sequence[int] | None = None,
    pad_to: int | None = None,
) -> TensorLayout:
    """Build a TensorLayout from per-shard unpadded lengths.

    Shards are placed contiguously along split_dim; every other dimension is
    assumed full. When pad_to is set, each shard's split dimension is padded up
    to a multiple of pad_to (the padding is recorded via unpadded_length).
    """
    dims = tuple(int(v) for v in dims)
    ranks = shard_order(ranks) if ranks is not None else list(range(len(lengths)))
    if len(ranks) != len(lengths):
        raise LayoutError(f"{name!r}: {len(ranks)} ranks but {len(lengths)} lengths")

    shards: dict[int, ShardSpec] = {}
    offset = 0
    for rank, length in zip(ranks, lengths, strict=False):
        length = int(length)
        shard_len = length
        if pad_to is not None:
            shard_len = ((length + pad_to - 1) // pad_to) * pad_to
        shape = list(dims)
        shape[split_dim] = shard_len
        offsets = [0] * len(dims)
        offsets[split_dim] = offset
        shards[rank] = ShardSpec(
            rank=rank,
            offsets=tuple(offsets),
            shape=tuple(shape),
            unpadded_length=length,
        )
        # Offsets are canonical (unpadded) logical coordinates; padding is a
        # storage detail carried by shape vs unpadded_length.
        offset += length
    return TensorLayout(name=name, dims=dims, split_dim=split_dim, shards=shards)


def validate_dp_completeness(stage: StageId, required_ranks: Iterable[int], present_ranks: Collection[int]) -> None:
    """Reject a DP-normalized stage unless every required rank shard is
    present.

    A stage whose reduction spans the DP dimension (group reward normalization,
    group advantages) cannot be recomputed from a partial rank set — replay
    must fail closed rather than guess a denominator from the physical batch.
    """
    present = set(present_ranks)
    missing = [rank for rank in shard_order(required_ranks) if rank not in present]
    if missing:
        raise LayoutError(
            f"{stage.value}: missing DP rank shard(s) {missing}; a DP-normalized stage requires the full rank set"
        )


def reconstruct_shards(layout: TensorLayout, shards: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """Assemble the canonical logical tensor from rank-local shards.

    Fails closed on any inconsistency: a missing shard, a shard without
    recorded offsets, an overlap or gap along the split dimension, or a shape
    mismatch on any non-split dimension.
    """
    present = set(shards)
    missing = [rank for rank in layout.shards if rank not in present]
    if missing:
        raise LayoutError(f"{layout.name!r}: missing shard(s) for rank {missing}")

    rank_order = shard_order(layout.shards)
    pieces: list[torch.Tensor] = []
    occupied: list[tuple[int, int]] = []  # (start, end) along split_dim, unpadded
    for rank in rank_order:
        spec = layout.shards[rank]
        tensor = shards[rank]
        if tuple(tensor.shape) != spec.shape:
            raise LayoutError(f"{layout.name!r} rank {rank}: shard shape {tuple(tensor.shape)} != spec {spec.shape}")
        if spec.offsets is None:
            raise LayoutError(f"{layout.name!r} rank {rank}: offsets not recorded; cannot reconstruct")

        for dim, offset in enumerate(spec.offsets):
            if dim == layout.split_dim:
                continue
            if offset != 0:
                raise LayoutError(f"{layout.name!r} rank {rank}: non-split dim {dim} offset {offset} != 0")
            if spec.shape[dim] != layout.dims[dim]:
                raise LayoutError(
                    f"{layout.name!r} rank {rank}: non-split dim {dim} shape {spec.shape[dim]} != canonical {layout.dims[dim]}"
                )

        start = spec.offsets[layout.split_dim]
        unpadded = spec.unpadded_length if spec.unpadded_length is not None else spec.shape[layout.split_dim]
        if unpadded > spec.shape[layout.split_dim]:
            raise LayoutError(
                f"{layout.name!r} rank {rank}: unpadded_length {unpadded} > shard dim {spec.shape[layout.split_dim]}"
            )
        end = start + unpadded
        occupied.append((start, end))

        if unpadded == spec.shape[layout.split_dim]:
            pieces.append(tensor)
        else:
            slices = [slice(None)] * tensor.dim()
            slices[layout.split_dim] = slice(0, unpadded)
            pieces.append(tensor[tuple(slices)])

    occupied.sort()
    expected = 0
    for start, end in occupied:
        if start != expected:
            raise LayoutError(
                f"{layout.name!r}: gap or overlap in split dim {layout.split_dim} (expected offset {expected}, got {start})"
            )
        expected = end
    if expected != layout.dims[layout.split_dim]:
        raise LayoutError(
            f"{layout.name!r}: shards cover {expected} elements but canonical dim is {layout.dims[layout.split_dim]}"
        )

    return torch.cat(pieces, dim=layout.split_dim)


def cp_capability(stage: StageId, cp: int, *, reconstructable: bool) -> StageCapability:
    """Resolve the replay capability of stage for a context-parallel topology.

    CP=1 is unsharded. CP>1 requires reconstructable shard offsets; the frozen
    V1 capability matrix does not yet endorse a recompute verdict for CP>1, so
    this returns unsupported even when the layout is reconstructable — the
    mechanism exists, but the evidence does not.
    """
    if cp == 1:
        return StageCapability.RECOMPUTE if reconstructable else StageCapability.UNSUPPORTED
    return StageCapability.UNSUPPORTED  # TODO(agent): flip to RECOMPUTE once CP adapters land with parity evidence
