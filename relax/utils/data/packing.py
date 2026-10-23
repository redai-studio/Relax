# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only bin packing shared by local and wholepack scheduling."""


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
