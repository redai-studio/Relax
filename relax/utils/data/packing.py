# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only bin packing shared by local and wholepack scheduling."""

from collections.abc import Sequence

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


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


def get_first_fit_partitions(total_lengths: list[int], max_tokens_per_gpu: int) -> list[list[int]]:
    """Pack indices in input order; oversized samples get their own bin."""
    batches: list[int] = []
    partitions: list[list[int]] = []
    for index, length in enumerate(total_lengths):
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                partitions[i].append(index)
                break
        else:
            batches.append(length)
            partitions.append([index])

    return partitions


def get_microbatch_count(lengths: list[int], capacity: int, packing_strategy: str = "balanced") -> int:
    """Predict the existing local count, preserving input order and
    strategy."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if packing_strategy == "binpack":
        return len(get_binpack_partitions(lengths, capacity))
    if packing_strategy != "balanced":
        raise ValueError(f"Unsupported packing strategy: {packing_strategy}")
    return len(get_first_fit_partitions(lengths, capacity))


def get_microbatch_token_capacity(
    max_tokens: int,
    cp_size: int,
    *,
    tp_size: int = 1,
    allgather_cp: bool = False,
    data_pad_size_multiplier: int = 1,
) -> int:
    """Existing CP/allgather token capacity, without device or process
    groups."""
    if allgather_cp:
        pad_size = tp_size * data_pad_size_multiplier
        if pad_size <= 0 or max_tokens < pad_size:
            raise ValueError(
                "allgather CP padding must fit within max_tokens_per_gpu: "
                f"max_tokens_per_gpu={max_tokens}, local_pad_size={pad_size}"
            )
        return cp_size * (max_tokens // pad_size) * pad_size
    return max_tokens * cp_size


def _get_first_fit_partitions(total_lengths: Sequence[int], capacity: int) -> list[list[int]]:
    partitions: list[list[int]] = []
    partition_token_counts: list[int] = []
    for sample_idx, length in enumerate(total_lengths):
        length = int(length)
        for partition_idx, token_count in enumerate(partition_token_counts):
            if token_count + length <= capacity:
                partitions[partition_idx].append(sample_idx)
                partition_token_counts[partition_idx] += length
                break
        else:
            partitions.append([sample_idx])
            partition_token_counts.append(length)
    return partitions


def _partitions_fit_capacity_or_singleton_oversize(
    partitions: Sequence[Sequence[int]],
    partition_token_counts: Sequence[int],
    total_lengths: Sequence[int],
    capacity: int,
) -> bool:
    return all(
        token_count <= capacity or (len(partition) == 1 and int(total_lengths[partition[0]]) > capacity)
        for partition, token_count in zip(partitions, partition_token_counts)
    )


def get_capacity_safe_balanced_partitions(
    total_lengths: Sequence[int], num_partitions: int, capacity: int
) -> list[list[int]]:
    partitions = get_seqlen_balanced_partitions(total_lengths, num_partitions, equal_size=False)
    partition_token_counts = [sum(int(total_lengths[idx]) for idx in partition) for partition in partitions]
    if _partitions_fit_capacity_or_singleton_oversize(partitions, partition_token_counts, total_lengths, capacity):
        return partitions

    balanced_max_tokens = max(partition_token_counts)
    partitions = _get_first_fit_partitions(total_lengths, capacity)
    partition_token_counts = [sum(int(total_lengths[idx]) for idx in partition) for partition in partitions]
    if len(partitions) > num_partitions or not _partitions_fit_capacity_or_singleton_oversize(
        partitions, partition_token_counts, total_lengths, capacity
    ):
        raise RuntimeError(
            "first-fit dynamic microbatch partitions cannot satisfy the synchronised schedule: "
            f"first_fit_partitions={len(partitions)}, num_partitions={num_partitions}, capacity={capacity}"
        )

    while len(partitions) < num_partitions:
        splittable = [idx for idx, partition in enumerate(partitions) if len(partition) > 1]
        if not splittable:
            raise RuntimeError(
                "dynamic microbatch partitions cannot be split into the synchronised number of non-empty batches: "
                f"current={len(partitions)}, requested={num_partitions}"
            )
        partition_idx = max(splittable, key=lambda idx: (partition_token_counts[idx], -idx))
        partition = partitions[partition_idx]
        sample_idx = max(partition, key=lambda idx: (int(total_lengths[idx]), -idx))
        partition.remove(sample_idx)
        sample_length = int(total_lengths[sample_idx])
        partition_token_counts[partition_idx] -= sample_length
        partitions.append([sample_idx])
        partition_token_counts.append(sample_length)

    flat_indices = [idx for partition in partitions for idx in partition]
    if sorted(flat_indices) != list(range(len(total_lengths))) or not _partitions_fit_capacity_or_singleton_oversize(
        partitions, partition_token_counts, total_lengths, capacity
    ):
        raise RuntimeError("capacity-safe dynamic microbatch partitioning produced an invalid partition")

    logger.warning(
        "Balanced dynamic microbatches violated token capacity; using first-fit partitions with any oversized "
        "samples isolated: "
        f"balanced_max_tokens={balanced_max_tokens}, capacity={capacity}, num_partitions={num_partitions}"
    )
    return partitions
