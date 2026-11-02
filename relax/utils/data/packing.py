# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only bin packing shared by local and wholepack scheduling."""

from collections.abc import Sequence


def get_flops_packing_seed(total_lengths: Sequence[int], costs: Sequence[float], capacity: int) -> list[list[int]]:
    """Choose a capacity-safe seed before synchronizing the microbatch count.

    Retain input-order first fit as a candidate, so enabling FLOPs packing
    never requires more local microbatches. Descending first/best fit can
    reduce that count; estimated cost breaks ties between feasible plans.
    Indivisible oversized samples retain the established singleton policy.
    """
    import math

    if capacity <= 0 or not total_lengths or len(total_lengths) != len(costs):
        raise ValueError("FLOPs packing requires positive capacity and matching nonempty lengths/costs")
    if any(length <= 0 for length in total_lengths) or any(not math.isfinite(cost) or cost <= 0 for cost in costs):
        raise ValueError("FLOPs packing requires positive lengths and finite positive costs")

    def pack(order: Sequence[int], *, best_fit: bool) -> list[list[int]]:
        bins: list[list[int]] = []
        loads: list[int] = []
        for index in order:
            fits = [i for i, load in enumerate(loads) if load + total_lengths[index] <= capacity]
            target = (
                min(fits, key=lambda i: (capacity - loads[i] - total_lengths[index], i))
                if best_fit and fits
                else (fits[0] if fits else len(bins))
            )
            if target == len(bins):
                bins.append([])
                loads.append(0)
            bins[target].append(index)
            loads[target] += int(total_lengths[index])
        return [sorted(group) for group in bins]

    descending = sorted(range(len(total_lengths)), key=lambda i: (-total_lengths[i], -costs[i], i))
    candidates = [
        pack(list(range(len(total_lengths))), best_fit=False),
        pack(descending, best_fit=False),
        pack(descending, best_fit=True),
    ]

    def objective(groups: list[list[int]]) -> tuple:
        work = [sum(costs[i] for i in group) for group in groups]
        return len(groups), max(work), sum(value * value for value in work), groups

    return min(candidates, key=objective)


def get_binpack_partitions(total_lengths: list[int], max_tokens_per_gpu: int) -> list[list[int]]:
    """Worst-fit-decreasing bin packing, matching ms-swift's
    ``packing_strategy="binpack"``.

    ms-swift delegates packing to ``binpacking.to_constant_volume``, which sorts
    items by descending length and drops each into the *least loaded* bin it still
    fits in, opening a new bin only when none has room.

    Differs from the two existing helpers in ways that matter for cross-framework
    comparison: unlike :func:`relax.utils.data.seqlen_balancing.get_seqlen_balanced_partitions`
    (Karmarkar-Karp) every bin is guaranteed to respect ``max_tokens_per_gpu``,
    and unlike :func:`get_minimum_num_micro_batch_size` (plain first-fit, input
    order) it returns the assignment rather than just the count.

    ms-swift additionally packs over a rolling 1000-sample window; a Relax step
    carries at most ``global_batch_size // dp_size`` samples, so that window never
    closes and is not replicated here.

    Returns a list of bins, each a sorted list of indices into ``total_lengths``.
    """
    bin_sums: list[int] = []
    bins: list[list[int]] = []
    # Stable descending sort, matching binpacking's revargsort tie-breaking.
    for index in sorted(range(len(total_lengths)), key=lambda i: -total_lengths[i]):
        length = total_lengths[index]
        candidates = [b for b in range(len(bin_sums)) if bin_sums[b] + length <= max_tokens_per_gpu]
        if candidates:
            # Least-loaded fit. min() keeps the lowest index on ties, like argmin.
            target = min(candidates, key=lambda b: bin_sums[b])
        else:
            bin_sums.append(0)
            bins.append([])
            target = len(bins) - 1
        bin_sums[target] += length
        bins[target].append(index)
    return [sorted(current_bin) for current_bin in bins]
