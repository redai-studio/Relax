# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep FSDP/EP collective schedules aligned for distinct input batches."""

import time
from collections.abc import Callable, Sequence
from typing import Any


def build_microbatches(
    total_lengths: Sequence[int],
    max_tokens_per_gpu: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:
    """Greedy-pack sample indices under padded-token and sample-count budgets.

    Padded positions are ``rows * longest_sample``, not the sum of lengths.
    An over-budget sample still forms a one-sample microbatch.
    """
    if not total_lengths:
        return []
    budgets = [int(t) for t in total_lengths]
    microbatches: list[list[int]] = []
    current: list[int] = []
    current_max_len = 0
    for index, length in enumerate(budgets):
        over_budget = False
        if max_tokens_per_gpu and current:
            padded_positions = (len(current) + 1) * max(current_max_len, length)
            if padded_positions > max_tokens_per_gpu:
                over_budget = True
        if micro_batch_size and len(current) >= micro_batch_size:
            over_budget = True
        if over_budget:
            microbatches.append(current)
            current = []
            current_max_len = 0
        current.append(index)
        current_max_len = max(current_max_len, length)
    if current:
        microbatches.append(current)
    return microbatches


def fetch_with_agreement(
    fetch: Callable[[], Any],
    readiness: Callable[[bool, str | None], tuple[int, int]],
    *,
    attempts: int,
    pause: Callable[[float], None] = time.sleep,
    context: str = "",
) -> Any:
    """Keep ready data while partners retry; readiness propagates fetch
    errors."""
    data = None
    for _ in range(attempts):
        error = None
        if data is None:
            try:
                data = fetch()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        ready, missing = readiness(data is not None, error)
        if ready and missing:
            pause(0.25)
            continue
        return data
    raise RuntimeError(f"Data fetch {context} did not converge after {attempts} attempts")


def vision_tensor_parallel_group(parallel_state):
    """Only true TP partners share the same visual input after data
    sharding."""
    if parallel_state.get_group_size("tp") > 1:
        return parallel_state.get_group("tp")
    return None


def align_microbatches(microbatches: list[list[int]], counts: list[tuple[int, int]]) -> list[list[int]]:
    """Split existing batches without increasing their padded-token budget.

    All ranks must enter the same number of forwards/backwards. Empty slots
    cannot safely participate without an explicit dummy-batch implementation.
    """
    target = max(count for count, _ in counts)
    if min(samples for _, samples in counts) < target:
        raise ValueError("Turbo data slots need enough samples to align microbatches across FSDP/EP ranks")
    batches = [list(batch) for batch in microbatches]
    while len(batches) < target:
        index = max(range(len(batches)), key=lambda index: len(batches[index]))
        batch = batches[index]
        midpoint = len(batch) // 2
        batches[index : index + 1] = [batch[:midpoint], batch[midpoint:]]
    return batches
