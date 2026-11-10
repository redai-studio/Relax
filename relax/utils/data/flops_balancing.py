# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deterministic CPU workload balancing with a separate token capacity."""

import math
from collections.abc import Sequence

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


Workload = int | float


def _validate_inputs(capacity_lengths: Sequence[int], workloads: Sequence[Workload]) -> None:
    if len(capacity_lengths) != len(workloads):
        raise ValueError("capacity_lengths and workloads must have the same length")
    if any(length < 0 for length in capacity_lengths):
        raise ValueError("capacity lengths must be non-negative")
    if any(not math.isfinite(cost) or cost < 0 for cost in workloads):
        raise ValueError("workloads must be finite and non-negative")


def balance_workloads(workloads: Sequence[Workload], num_partitions: int, equal_size: bool = True) -> list[list[int]]:
    """Balance the already selected batch, keeping equal DP sample counts.

    Reuse Relax's deterministic Karmarkar-Karp helper; no selection, cache or
    optimizer-step boundaries are changed here.
    """
    _validate_inputs([0] * len(workloads), workloads)
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive")
    if not workloads:
        return [[] for _ in range(num_partitions)]
    if len(workloads) < num_partitions or (equal_size and len(workloads) % num_partitions):
        raise ValueError("selected sample count must accommodate the requested DP partitions")
    return get_seqlen_balanced_partitions(list(workloads), num_partitions, equal_size=equal_size)


def partition_objective(workloads: Sequence[Workload]) -> tuple[Workload, Workload, Workload]:
    """Lexicographic workload objective; fixed total/K makes sum squares
    equivalent to variance."""
    if not workloads:
        return (0, 0, 0)
    return max(workloads), max(workloads) - min(workloads), sum(cost * cost for cost in workloads)


def partition_by_flops(
    capacity_lengths: Sequence[int],
    workloads: Sequence[Workload],
    capacity: int,
    baseline_partitions: Sequence[Sequence[int]],
    *,
    stats: dict[str, int] | None = None,
) -> list[list[int]]:
    """Strictly improve a feasible baseline using bounded, fixed-K moves/swaps.

    At most 32 accepted operations. Each iteration examines two heaviest and
    two lightest ordinary bins, and at most eight samples per bin (four at
    either workload extreme). Moves precede swaps. No random/time budget and no
    feasibility solver: an unsuccessful search returns the exact baseline.
    """
    _validate_inputs(capacity_lengths, workloads)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    bins = [list(part) for part in baseline_partitions]
    if any(not part for part in bins) or sorted(i for part in bins for i in part) != list(range(len(workloads))):
        raise ValueError("baseline partitions must cover each real sample exactly once")
    tokens = [sum(capacity_lengths[i] for i in part) for part in bins]
    costs = [sum(workloads[i] for i in part) for part in bins]
    if any(
        total > capacity and not (len(part) == 1 and capacity_lengths[part[0]] > capacity)
        for total, part in zip(tokens, bins)
    ):
        raise ValueError("baseline partitions must satisfy token capacity or be oversized singletons")
    ordinary = [b for b, total in enumerate(tokens) if total <= capacity]
    counters = dict(moves=0, swaps=0, iterations=0, candidate_checks=0, improved=0)
    current = partition_objective(costs)
    budget = min(32, 2 * len(bins))
    for _ in range(budget):
        heavy = sorted(ordinary, key=lambda b: (-costs[b], b))[:2]
        light = sorted(ordinary, key=lambda b: (costs[b], b))[:2]
        descending = sorted(range(len(bins)), key=lambda b: (-costs[b], b))[:3]
        ascending = sorted(range(len(bins)), key=lambda b: (costs[b], b))[:3]
        square_sum = current[2]

        def objective(a: int, b: int, new_a: Workload, new_b: Workload) -> tuple[Workload, Workload, Workload]:
            # Two bins change; the top/bottom three suffice for the unchanged
            # extrema. No O(K) scan for every candidate pair.
            other_max = next((costs[i] for i in descending if i != a and i != b), 0)
            other_min = next((costs[i] for i in ascending if i != a and i != b), min(new_a, new_b))
            maximum = max(other_max, new_a, new_b)
            minimum = min(other_min, new_a, new_b)
            return maximum, maximum - minimum, square_sum - costs[a] ** 2 - costs[b] ** 2 + new_a**2 + new_b**2

        def candidates(b: int) -> list[int]:
            ordered = sorted(bins[b], key=lambda i: (-workloads[i], -capacity_lengths[i], i))
            return sorted(set(ordered[:4] + ordered[-4:]), key=lambda i: (-workloads[i], -capacity_lengths[i], i))

        samples = {b: candidates(b) for b in sorted(set(heavy + light))}
        best = None
        for a in heavy:
            if len(bins[a]) == 1:
                continue
            for b in light:
                if a == b or costs[a] <= costs[b]:
                    continue
                for i in samples[a]:
                    counters["candidate_checks"] += 1
                    if tokens[b] + capacity_lengths[i] > capacity:
                        continue
                    candidate = objective(a, b, costs[a] - workloads[i], costs[b] + workloads[i])
                    choice = (candidate, a, b, i, -1)
                    if candidate < current and (best is None or choice < best):
                        best = choice
        if best is None:
            for a in heavy:
                for b in light:
                    if a == b or costs[a] <= costs[b]:
                        continue
                    for i in samples[a]:
                        for j in samples[b]:
                            counters["candidate_checks"] += 1
                            if (
                                tokens[a] - capacity_lengths[i] + capacity_lengths[j] > capacity
                                or tokens[b] - capacity_lengths[j] + capacity_lengths[i] > capacity
                            ):
                                continue
                            candidate = objective(
                                a, b, costs[a] - workloads[i] + workloads[j], costs[b] - workloads[j] + workloads[i]
                            )
                            choice = (candidate, a, b, i, j)
                            if candidate < current and (best is None or choice < best):
                                best = choice
        counters["iterations"] += 1
        if best is None:
            break
        _, a, b, i, j = best
        updated = list(costs)
        updated[a] -= workloads[i]
        updated[b] += workloads[i]
        if j >= 0:
            updated[a] += workloads[j]
            updated[b] -= workloads[j]
        verified = partition_objective(updated)
        if verified >= current:
            break
        current = verified
        bins[a].remove(i)
        bins[b].append(i)
        tokens[a] -= capacity_lengths[i]
        tokens[b] += capacity_lengths[i]
        costs[a] -= workloads[i]
        costs[b] += workloads[i]
        if j < 0:
            counters["moves"] += 1
        else:
            bins[b].remove(j)
            bins[a].append(j)
            tokens[b] -= capacity_lengths[j]
            tokens[a] += capacity_lengths[j]
            costs[b] -= workloads[j]
            costs[a] += workloads[j]
            counters["swaps"] += 1
        counters["improved"] = 1
    if stats is not None:
        stats.update(counters)
    return [sorted(part) for part in bins] if counters["improved"] else bins


def expand_partitions(
    partitions: Sequence[Sequence[int]],
    lengths: Sequence[int],
    workloads: Sequence[Workload],
    target_count: int,
) -> tuple[list[list[int]], set[int]]:
    """Safely split a feasible plan to the synchronized count, then pad
    dummies.

    Splitting never increases any bin's token total. Dummy bins duplicate the
    smallest real bin and must be excluded by the caller's existing loss mask.
    The returned set identifies those dummy bins, not real sample identities.
    """
    _validate_inputs(lengths, workloads)
    bins = [sorted(part) for part in partitions]
    if target_count < len(bins) or target_count < 0:
        raise ValueError("target_count cannot shrink an existing feasible plan")
    if any(not part for part in bins) or sorted(i for part in bins for i in part) != list(range(len(lengths))):
        raise ValueError("seed partitions must cover each real sample exactly once")
    if not bins:
        if target_count:
            raise ValueError("cannot create dummy batches without a real sample")
        return [], set()
    while len(bins) < min(target_count, len(lengths)):
        source = max(
            (b for b, part in enumerate(bins) if len(part) > 1),
            key=lambda b: (sum(workloads[i] for i in bins[b]), sum(lengths[i] for i in bins[b]), -b),
        )
        index = max(bins[source], key=lambda i: (workloads[i], lengths[i], -i))
        bins[source].remove(index)
        bins.append([index])
    dummy_indices: set[int] = set()
    seed = min(range(len(bins)), key=lambda b: (sum(lengths[i] for i in bins[b]), b))
    while len(bins) < target_count:
        dummy_indices.add(len(bins))
        bins.append(list(bins[seed]))
    return bins, dummy_indices


def summarize_partitions(
    lengths: Sequence[int], workloads: Sequence[Workload], partitions: Sequence[Sequence[int]]
) -> dict[str, object]:
    """Serializable real-bin diagnostics; imbalance is max FLOPs / mean FLOPs.

    Pass real partitions only: dummy forward/backward participation represents
    no effective training work and should not be counted in these statistics.
    """
    _validate_inputs(lengths, workloads)
    tokens = [sum(lengths[i] for i in part) for part in partitions]
    costs = [sum(workloads[i] for i in part) for part in partitions]
    total = sum(costs)
    maximum = max(costs, default=0)
    return {
        "capacity_tokens": tokens,
        "estimated_flops": costs,
        "min_flops": min(costs, default=0),
        "max_flops": maximum,
        "total_flops": total,
        "imbalance_ratio": maximum * len(costs) / total if total else 1.0,
    }


def guarded_dp_assignment(
    lengths: Sequence[int],
    workloads: Sequence[Workload],
    baseline: Sequence[Sequence[int]],
    capacity: int,
    *,
    packing_strategy: str = "balanced",
    capacity_alignment: int = 1,
) -> tuple[list[list[int]], dict[str, object]]:
    """Accept a FLOPs KK candidate only if workload improves without raising DP
    MAX K.

    Controller cannot know whether the worker's eventual text model uses
    unsplit THD input. For CP, check both raw and aligned document layouts. VPP
    rounding is monotone, so guarding pre-rounding MAX also guards the rounded
    count.
    """
    from relax.utils.data.packing import get_microbatch_count

    _validate_inputs(lengths, workloads)
    if capacity_alignment < 1:
        raise ValueError("capacity_alignment must be positive")
    before = [list(part) for part in baseline]
    if not before or sorted(i for part in before for i in part) != list(range(len(lengths))):
        raise ValueError("baseline DP assignment must cover selected samples exactly once")
    candidate = balance_workloads(workloads, len(before))
    layouts = {"raw": list(lengths)}
    if capacity_alignment > 1:
        layouts["padded"] = [
            (length + capacity_alignment - 1) // capacity_alignment * capacity_alignment for length in lengths
        ]
    counts = {}
    for name, physical in layouts.items():
        counts[name] = {
            label: [get_microbatch_count([physical[i] for i in rank], capacity, packing_strategy) for rank in ranks]
            for label, ranks in (("baseline", before), ("candidate", candidate))
        }
    before_summary = summarize_partitions(lengths, workloads, before)
    candidate_summary = summarize_partitions(lengths, workloads, candidate)
    if any(max(values["candidate"]) > max(values["baseline"]) for values in counts.values()):
        decision = "rejected_by_k"
    elif candidate_summary["max_flops"] >= before_summary["max_flops"]:
        decision = "rejected_no_improvement"
    else:
        decision = "accepted"
    result = candidate if decision == "accepted" else before
    return result, {
        "decision": decision,
        "counts": counts,
        "before": before_summary,
        "candidate": candidate_summary,
        "after": candidate_summary if decision == "accepted" else before_summary,
    }
