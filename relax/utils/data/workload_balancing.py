# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only workload balancing with an additive capacity constraint."""

from collections.abc import Sequence
from numbers import Integral


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _first_fit_decreasing(lengths: list[int], capacity: int) -> list[list[int]]:
    partitions: list[list[int]] = []
    used: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        for group, occupied in enumerate(used):
            if occupied + lengths[index] <= capacity:
                partitions[group].append(index)
                used[group] += lengths[index]
                break
        else:
            partitions.append([index])
            used.append(lengths[index])
    return partitions


def _workload_first(lengths: list[int], workloads: list[int], capacity: int, count: int) -> list[list[int]] | None:
    partitions: list[list[int]] = [[] for _ in range(count)]
    used = [0] * count
    loads = [0] * count
    for index in sorted(range(len(lengths)), key=lambda i: (-workloads[i], -lengths[i], i)):
        candidates = [group for group in range(count) if used[group] + lengths[index] <= capacity]
        if not candidates:
            return None
        group = min(candidates, key=lambda group: (loads[group], used[group], group))
        partitions[group].append(index)
        used[group] += lengths[index]
        loads[group] += workloads[index]
    return partitions


def _load_score(partitions: list[list[int]], workloads: list[int]) -> tuple[int, int]:
    loads = [sum(workloads[index] for index in partition) for partition in partitions]
    return max(loads), sum(load * load for load in loads)


def pack_by_workload(
    capacity_lengths: Sequence[int],
    workloads: Sequence[int],
    max_capacity: int,
    *,
    num_partitions: int | None = None,
) -> list[list[int]]:
    """Return nonempty sample-index groups within an additive capacity budget.

    Lengths and workloads must be positive CPU integers. Capacity lengths must
    already include any caller-specific alignment or padding; a token budget
    alone is not a guarantee of GPU memory safety. Workloads may be raw FLOPs.

    First-fit decreasing (FFD) supplies a feasible fallback and, by default,
    the partition count. Workload-first placement is kept only when its maximum
    workload / squared-load sum score improves. Neither placement is optimal.

    An explicit count is honored exactly or raises ValueError. If it is smaller
    than FFD's count, the workload heuristic may fail even when a feasible
    packing exists. Requesting more groups than samples is rejected; execution
    slots and zero-gradient dummy batches belong to the caller's schedule.

    An empty input returns [] unless an explicit nonzero count is requested.
    Inputs are never modified. Complexity is O(N log N + N*K), where K is the
    number of groups, plus bounded splitting for a larger explicit count.
    """
    capacity = _positive_integer(max_capacity, "max_capacity")
    count = None if num_partitions is None else _positive_integer(num_partitions, "num_partitions")
    if len(capacity_lengths) != len(workloads):
        raise ValueError("capacity_lengths and workloads must have the same length")
    lengths = [_positive_integer(value, "capacity length") for value in capacity_lengths]
    costs = [_positive_integer(value, "workload") for value in workloads]
    if not lengths:
        if count is not None:
            raise ValueError("Cannot create nonempty partitions from an empty batch")
        return []
    if any(length > capacity for length in lengths):
        raise ValueError("A sample exceeds max_capacity; it cannot be fixed by opening a new partition")
    if count is not None and count > len(lengths):
        raise ValueError("num_partitions cannot exceed the number of real samples")

    baseline = _first_fit_decreasing(lengths, capacity)
    if count is None:
        count = len(baseline)
    if sum(lengths) > capacity * count:
        raise ValueError("Total capacity is insufficient for the requested num_partitions")

    if len(baseline) <= count:
        while len(baseline) < count:
            group = max(
                (group for group in range(len(baseline)) if len(baseline[group]) > 1),
                key=lambda group: (sum(costs[i] for i in baseline[group]), -group),
            )
            index = max(baseline[group], key=lambda index: (costs[index], -index))
            baseline[group].remove(index)
            baseline.append([index])
        candidate = _workload_first(lengths, costs, capacity, count)
        selected = (
            candidate
            if candidate is not None and _load_score(candidate, costs) < _load_score(baseline, costs)
            else baseline
        )
    else:
        selected = _workload_first(lengths, costs, capacity, count)
        if selected is None:
            raise ValueError("Heuristic could not pack the requested num_partitions; feasibility is not proven")

    flat = [index for partition in selected for index in partition]
    if len(selected) != count or any(not partition for partition in selected):
        raise RuntimeError("Packing produced an invalid partition count or an empty partition")
    if sorted(flat) != list(range(len(lengths))):
        raise RuntimeError("Packing must preserve each real sample exactly once")
    if any(sum(lengths[index] for index in partition) > capacity for partition in selected):
        raise RuntimeError("Packing exceeded max_capacity")
    return [sorted(partition) for partition in selected]
