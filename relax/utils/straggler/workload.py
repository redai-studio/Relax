# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Pure derivation of the per-optimizer-step workload payload.

This module is deliberately free of Megatron, Ray and transfer_queue imports so
the arithmetic that the detector depends on can be tested in a CPU-only
environment. It holds no state and performs no I/O.
"""

from typing import List, Sequence, Tuple


__all__ = ["step_workloads"]


def step_workloads(
    sample_lengths: Sequence[int],
    k_partitions: int,
    num_steps_per_rollout: int,
) -> List[Tuple[int, int, int]]:
    """Return one ``(tokens, sequences, microbatches)`` entry per optimizer
    step.

    The length of the returned list is exactly ``num_steps_per_rollout``, because
    the consumer reads it by step index: ``context._step_workload`` returns the
    entry for ``(rollout_id, step_id)``, so publishing one entry per MICRO-BATCH
    made the detector read only the first group of a multi-micro-batch step. It
    under-reported the step (on the red team's example by -50% tokens, -83%
    sequences and -75% microbatches) and, because a first group's size depends on
    how the samples happen to be distributed, two ranks with identical totals
    reported different tokens and the comparability gate suppressed verdicts for
    ranks doing equal work.

    SFT prepack is the shipped case: the caller hands the model a single-element
    ``prepared_num_microbatches``, so ``num_steps_per_rollout == 1`` and that one
    step consumes every micro-batch of the window. The single entry is therefore
    the whole step total, and ``k_partitions`` (the executed, DP-wide K) becomes
    the microbatch count.

    For a hypothetical multi-step window the samples are partitioned with the
    same deterministic helper the training path uses, and the micro-batch groups
    are split across the steps in order, so each entry describes the work that
    step actually consumes rather than assuming one group per step.

    Args:
        sample_lengths (Sequence[int]): Per-sample sequence lengths for the window.
        k_partitions (int): Number of micro-batches the step(s) consume.
        num_steps_per_rollout (int): Optimizer steps that consume this window.

    Returns:
        List[Tuple[int, int, int]]: One ``(tokens, sequences, microbatches)`` entry
        per optimizer step.

    Raises:
        ValueError: If the requested shape cannot describe real work.
    """
    samples = [int(length) for length in sample_lengths]
    k = int(k_partitions)
    steps = int(num_steps_per_rollout)

    if steps < 1:
        raise ValueError(f"num_steps_per_rollout must be >= 1, got {steps}")
    if k < 1:
        raise ValueError(f"k_partitions must be >= 1, got {k}")
    # A step cannot consume more micro-batches than there are samples to fill;
    # the balancing helper asserts this too, so fail here with a clear message.
    if len(samples) < k:
        raise ValueError(f"cannot split {len(samples)} samples into {k} micro-batches")

    if steps == 1:
        return [(sum(samples), len(samples), k)]

    # Lazy import: keeps this module importable without the data/torch stack for
    # the single-step case that every shipped path uses.
    from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions

    groups = get_seqlen_balanced_partitions(samples, k, equal_size=False)
    per_step = _split_evenly(len(groups), steps)
    workloads: List[Tuple[int, int, int]] = []
    cursor = 0
    for count in per_step:
        chunk = groups[cursor : cursor + count]
        cursor += count
        indices = [index for group in chunk for index in group]
        workloads.append((sum(samples[index] for index in indices), len(indices), len(chunk)))
    return workloads


def _split_evenly(total: int, parts: int) -> List[int]:
    """Split ``total`` items into ``parts`` contiguous counts differing by <=
    1."""
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]
