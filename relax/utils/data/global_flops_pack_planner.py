# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Global, capacity-safe packing followed by indivisible microbatch
placement."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Sequence

from relax.utils.data.flops_scheduling import _weights
from relax.utils.data.packing import get_binpack_partitions
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


SCHEMA = 1
ALGORITHM = "no16-global-capacity-kk-v1"
COST_MODEL = "relax-p2-non-mtp-main-training-v2"
TRAIN_TASKS = frozenset(("train", "sft_train", "train_actor", "actor_train"))


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def scheduling_order(args: Any) -> str:
    order = getattr(args, "flops_scheduling_order", "dp-first")
    if order not in ("dp-first", "pack-first"):
        raise ValueError(f"Invalid FLOPs scheduling order: {order!r}")
    if order == "pack-first" and getattr(args, "flops_scheduling", "off") != "both":
        raise ValueError("pack-first requires --flops-scheduling=both")
    return order


@dataclass(frozen=True)
class ExecutionContract:
    """Actual Actor topology and the existing physical-token capacity rules."""

    dp: int
    tp: int
    pp: int
    cp: int
    vpp: int
    mb_group: int
    max_tokens: int
    pad_multiplier: int = 1
    unsplit: bool = False
    allgather_cp: bool = False
    checkpoint_signature: str = ""

    def __post_init__(self) -> None:
        for name in ("dp", "tp", "pp", "cp", "vpp", "mb_group", "max_tokens", "pad_multiplier"):
            positive(getattr(self, name), name)
        if type(self.unsplit) is not bool or type(self.allgather_cp) is not bool:
            raise ValueError("Physical layout switches must be booleans")
        if self.capacity <= 0:
            raise ValueError("Effective physical token capacity must be positive")
        if not isinstance(self.checkpoint_signature, str):
            raise ValueError("Invalid execution configuration signature")

    @property
    def capacity(self) -> int:
        alignment = self.tp * self.pad_multiplier
        return self.cp * (self.max_tokens // alignment * alignment if self.allgather_cp else self.max_tokens)

    @property
    def group(self) -> int:
        return self.mb_group if self.vpp > 1 else 1

    @property
    def signature(self) -> str:
        return digest(asdict(self))

    def physical(self, logical: Sequence[int]) -> tuple[int, ...]:
        lengths = tuple(positive(length, "logical length") for length in logical)
        if self.cp > 1 and self.unsplit:
            alignment = self.tp * self.cp * 2
            return tuple((length + alignment - 1) // alignment * alignment for length in lengths)
        return lengths

    @classmethod
    def from_dict(cls, value: Any) -> ExecutionContract:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Incomplete pack-first execution contract")
        return cls(**value)


@dataclass(frozen=True)
class Microbatch:
    mb_id: str
    rows: tuple[int, ...]
    physical_tokens: int
    estimated_flops: float
    oversized_singleton: bool


@dataclass(frozen=True)
class GlobalPackPlan:
    identity: tuple[str, int, str, int]
    cohort_signature: str
    model_signature: str
    cost_signature: str
    checkpoint_signature: str
    contract: ExecutionContract
    logical_count: int
    rows: tuple[int, ...]
    lengths: tuple[int, ...]
    physical_lengths: tuple[int, ...]
    weights: tuple[float, ...]
    microbatches: tuple[Microbatch, ...]
    rank_mbs: tuple[tuple[int, ...], ...]
    K: int
    fallback_reason: str | None
    cpu_seconds: float

    @property
    def plan_digest(self) -> str:
        fields = asdict(self)
        fields.pop("cpu_seconds")
        return digest({"schema": SCHEMA, "algorithm": ALGORITHM, "cost_model": COST_MODEL, **fields})

    def rank_slots(self, rank: int) -> tuple[tuple[int, bool], ...]:
        real = self.rank_mbs[rank]
        source = min(real, key=lambda index: (self.microbatches[index].physical_tokens, index))
        return tuple((index, False) for index in real) + tuple((source, True) for _ in range(self.K - len(real)))

    def rank_rows(self, rank: int) -> tuple[int, ...]:
        return tuple(row for index in self.rank_mbs[rank] for row in self.microbatches[index].rows)

    def diagnostics(self) -> dict[str, Any]:
        real_flops = [math.fsum(self.microbatches[i].estimated_flops for i in group) for group in self.rank_mbs]
        dummy_flops = [
            math.fsum(self.microbatches[i].estimated_flops for i, dummy in self.rank_slots(rank) if dummy)
            for rank in range(self.contract.dp)
        ]
        return dict(
            cohort_id=self.cohort_signature,
            mini_id=self.identity[3],
            task=self.identity[2],
            cost_model_version=COST_MODEL,
            scheduler_order="pack-first",
            global_real_sample_count=self.logical_count,
            global_physical_row_count=len(self.rows),
            global_real_mb_count=len(self.microbatches),
            per_rank_real_mb_count=[len(group) for group in self.rank_mbs],
            per_rank_K=[self.K] * self.contract.dp,
            per_rank_dummy_count=[self.K - len(group) for group in self.rank_mbs],
            per_rank_estimated_real_flops=real_flops,
            per_rank_estimated_dummy_flops=dummy_flops,
            per_rank_physical_token_count=[
                sum(self.microbatches[i].physical_tokens for i in group) for group in self.rank_mbs
            ],
            MB_capacity=self.contract.capacity,
            oversized_singletons=[mb.mb_id for mb in self.microbatches if mb.oversized_singleton],
            fallback_reasons=[] if self.fallback_reason is None else [self.fallback_reason],
            scheduler_cpu_time=self.cpu_seconds,
            global_plan_digest=self.plan_digest,
        )


def build_global_plan(
    rows: Sequence[int],
    logical: Sequence[int],
    physical: Sequence[int],
    weights: Sequence[float],
    contract: ExecutionContract,
    *,
    identity: tuple[str, int, str, int],
    logical_count: int,
    model_signature: str,
    cost_signature: str,
    checkpoint_signature: str,
    identities: Sequence[int] | None = None,
) -> GlobalPackPlan:
    """Freeze global real MBs before deciding any effective DP ownership."""
    started = time.perf_counter()
    row_ids = tuple(rows)
    lengths = tuple(positive(v, "logical length") for v in logical)
    physical_lengths = tuple(positive(v, "physical length") for v in physical)
    costs = tuple(_weights(weights, len(row_ids)))
    positive(logical_count, "global logical count")
    if (
        len(row_ids) < contract.dp
        or len(set(row_ids)) != len(row_ids)
        or any(type(row) is not int or row < 0 for row in row_ids)
        or len(lengths) != len(row_ids)
        or len(physical_lengths) != len(row_ids)
        or physical_lengths != contract.physical(lengths)
    ):
        raise ValueError("Invalid global cohort, physical layout, or insufficient real rows per DP")
    if (
        len(identity) != 4
        or not isinstance(identity[0], str)
        or not identity[0]
        or type(identity[1]) is not int
        or identity[1] < 0
        or identity[2] not in TRAIN_TASKS
        or type(identity[3]) is not int
        or identity[3] < 0
    ):
        raise ValueError("Invalid partition generation/task/mini identity")
    if identities is not None and (len(identities) != len(rows) or len(set(identities)) != logical_count):
        raise ValueError("Global logical quota must count identities, not physical rows")
    if identities is None and logical_count != len(row_ids):
        raise ValueError("SFT n=1 requires a complete logical batch of physical rows")
    cohort = digest([identity, list(zip(row_ids, lengths, list(identities) if identities is not None else row_ids))])
    baseline = get_binpack_partitions(list(physical_lengths), contract.capacity)
    parts = get_seqlen_balanced_partitions(list(costs), len(baseline), equal_size=False)

    def fits(groups: Sequence[Sequence[int]]) -> bool:
        return all(len(group) == 1 or sum(physical_lengths[i] for i in group) <= contract.capacity for group in groups)

    fallback = None
    if not fits(parts):
        parts = baseline
        fallback = "weighted_kk_physical_capacity"
    # Splitting happens before MB identities/DP assignment are frozen.
    while len(parts) < contract.dp:
        candidates = [i for i, group in enumerate(parts) if len(group) > 1]
        if not candidates:
            raise ValueError("Cannot provide a real microbatch to every DP rank")
        index = max(candidates, key=lambda i: (sum(physical_lengths[j] for j in parts[i]), -i))
        parts.append([parts[index].pop()])
    if not fits(parts) or sorted(i for group in parts for i in group) != list(range(len(rows))):
        raise RuntimeError("Global packing violates capacity or real sample coverage")
    microbatches = tuple(
        Microbatch(
            mb_id=digest([cohort, tuple(row_ids[i] for i in group)]),
            rows=tuple(row_ids[i] for i in group),
            physical_tokens=sum(physical_lengths[i] for i in group),
            estimated_flops=math.fsum(costs[i] for i in group),
            oversized_singleton=sum(physical_lengths[i] for i in group) > contract.capacity,
        )
        for group in parts
    )
    # Only complete, already frozen Microbatch objects are assigned to ranks.
    assignments = get_seqlen_balanced_partitions(
        [mb.estimated_flops for mb in microbatches],
        contract.dp,
        equal_size=len(microbatches) % contract.dp == 0,
    )
    K = (max(map(len, assignments)) + contract.group - 1) // contract.group * contract.group
    return GlobalPackPlan(
        identity,
        cohort,
        model_signature,
        cost_signature,
        checkpoint_signature,
        contract,
        int(logical_count),
        row_ids,
        lengths,
        physical_lengths,
        costs,
        microbatches,
        tuple(tuple(group) for group in assignments),
        K,
        fallback,
        time.perf_counter() - started,
    )
