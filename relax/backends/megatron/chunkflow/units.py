# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from relax.backends.megatron.chunkflow.scheduler import Chunk, group_of


POLICIES = ("staggered", "lockstep")


@dataclass
class Unit:
    unit_id: int
    chunks: list[Chunk]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("a Unit must hold at least one chunk")
        groups = [c.group for c in self.chunks if c.is_dependent]
        if len(groups) != len(set(groups)):
            raise ValueError(
                f"unit {self.unit_id} holds two chunks of the same dependency group {groups} -- "
                "they are causally ordered and must not share a unit"
            )

    @property
    def groups(self) -> frozenset[int]:
        return frozenset(c.group for c in self.chunks if c.is_dependent)

    @property
    def round(self) -> int:
        return max((c.group_idx for c in self.chunks if c.is_dependent), default=-1)

    @property
    def num_tokens(self) -> int:
        return sum(c.num_tokens for c in self.chunks)

    @property
    def is_dependent(self) -> bool:
        return bool(self.groups)

    @property
    def chunk_ids(self) -> list[int]:
        return [c.chunk_id for c in self.chunks]


def build_micro_batches(chunks: Sequence[Chunk], policy: str = "staggered") -> list[Unit]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")

    if policy == "staggered":
        return [Unit(unit_id=i, chunks=[c]) for i, c in enumerate(chunks)]

    units: list[Unit] = []
    for c in chunks:
        if not c.is_dependent:
            units.append(Unit(unit_id=len(units), chunks=[c]))

    by_round: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.is_dependent:
            by_round.setdefault(c.group_idx, []).append(c)
    for round_idx in sorted(by_round):
        units.append(Unit(unit_id=len(units), chunks=by_round[round_idx]))

    return units


def classify_units(units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int) -> tuple[set[int], set[int]]:
    chunk_need_fr = {
        c.chunk_id for members in group_of(chunks).values() for c in members[: max(0, len(members) - keep_last_k)]
    }

    keep: set[int] = set()
    need_fr: set[int] = set()
    for u in units:
        if set(u.chunk_ids) <= chunk_need_fr:
            need_fr.add(u.unit_id)
        else:
            keep.add(u.unit_id)
    return keep, need_fr


def make_unit_bwd_order(units: Sequence[Unit], fwd_order: Sequence[int]) -> list[int]:
    by_id = {u.unit_id: u for u in units}
    units_of_group: dict[int, list[int]] = {}
    for u in units:
        for g in u.groups:
            units_of_group.setdefault(g, []).append(u.unit_id)

    seen: set[int] = set()
    out: list[int] = []
    emitted: set[int] = set()
    for uid in fwd_order:
        seen.add(uid)
        u = by_id[uid]
        if not u.is_dependent:
            out.append(uid)
            emitted.add(uid)
            continue
        if not all(all(m in seen for m in units_of_group[g]) for g in u.groups):
            continue
        for g in sorted(u.groups):
            for m in sorted(units_of_group[g], key=lambda x: -by_id[x].round):
                if m not in emitted:
                    out.append(m)
                    emitted.add(m)
    return out


def completed_groups(unit: Unit) -> frozenset[int]:
    return frozenset(c.group for c in unit.chunks if c.is_dependent and c.group_idx == 0)


def gen_unit_ops(
    units: Sequence[Unit],
    chunks: Sequence[Chunk],
    keep_last_k: int,
) -> list[tuple[str, int]]:
    """Single-stage order; recompute immediately before each dependent
    backward."""
    keep, need_fr = classify_units(units, chunks, keep_last_k)
    fwd_order = [u.unit_id for u in units]
    bwd_order = make_unit_bwd_order(units, fwd_order)
    ops: list[tuple[str, int]] = []
    seen: set[int] = set()
    bi = 0

    def backward(uid: int) -> None:
        if uid in need_fr:
            ops.append(("Fr", uid))
        ops.append(("B", uid))

    for uid in fwd_order:
        ops.append(("F" if uid in keep else "Fp", uid))
        seen.add(uid)
        if bi < len(bwd_order) and bwd_order[bi] in seen:
            backward(bwd_order[bi])
            bi += 1
    for uid in bwd_order[bi:]:
        backward(uid)
    return ops


def pipeline_plan(
    units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int
) -> tuple[list[int], list[int], set[int], int]:
    standalone = [u.unit_id for u in units if not u.is_dependent]
    dependent = sorted((u for u in units if u.is_dependent), key=lambda u: u.round)
    fwd_seq = standalone + [u.unit_id for u in dependent]
    bwd_seq = make_unit_bwd_order(units, fwd_seq)
    _, need_fr = classify_units(units, chunks, keep_last_k)
    by_id = {u.unit_id: u for u in units}
    positions = {u: i for i, u in enumerate(fwd_seq)}
    done_at: dict[int, int] = {}
    for unit in units:
        for group in unit.groups:
            done_at[group] = max(done_at.get(group, -1), positions[unit.unit_id])
    delay = max(
        (max((done_at[g] for g in by_id[u].groups), default=positions[u]) - i for i, u in enumerate(bwd_seq)),
        default=0,
    )
    return fwd_seq, bwd_seq, need_fr, delay


@dataclass(frozen=True)
class ModelPlan:
    unit_ids: tuple[int, ...]
    forward: tuple[int, ...]
    backward: tuple[int, ...]
    recompute: frozenset[int]
    warmups: tuple[int, ...]


def make_model_plan(units: Sequence[Unit], chunks: Sequence[Chunk], keep_last_k: int, pp_size: int) -> ModelPlan:
    fwd, bwd, recompute, delay = pipeline_plan(units, chunks, keep_last_k)
    return ModelPlan(
        tuple(u.unit_id for u in units),
        tuple(fwd),
        tuple(bwd),
        frozenset(recompute),
        tuple(min(delay + pp_size - s - 1, len(fwd)) for s in range(pp_size)),
    )


def merge_model_plans(plans: Sequence[ModelPlan], pp_size: int) -> ModelPlan:
    """Join parent orders without interleaving their dependency groups."""
    unit_ids, forward, backward, recompute = [], [], [], set()
    delay = 0
    for plan in plans:
        ids = {uid: len(unit_ids) + i for i, uid in enumerate(plan.unit_ids)}
        unit_ids.extend(ids.values())
        forward.extend(ids[uid] for uid in plan.forward)
        backward.extend(ids[uid] for uid in plan.backward)
        recompute.update(ids[uid] for uid in plan.recompute)
        # Equal F/B offsets preserve each parent's dependency delay.
        delay = max(delay, plan.warmups[-1])
    return ModelPlan(
        tuple(unit_ids),
        tuple(forward),
        tuple(backward),
        frozenset(recompute),
        tuple(min(delay + pp_size - s - 1, len(forward)) for s in range(pp_size)),
    )
