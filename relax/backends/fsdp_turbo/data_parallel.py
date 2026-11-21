# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep FSDP/EP collective schedules aligned for distinct input batches."""

import time
from collections.abc import Callable
from typing import Any


def fetch_with_agreement(
    fetch: Callable[[], Any],
    readiness: Callable[[bool], tuple[int, int]],
    *,
    attempts: int,
    pause: Callable[[float], None] = time.sleep,
    context: str = "",
) -> Any:
    """Keep ready data while empty partners retry the same logical batch."""
    data = None
    for _ in range(attempts):
        if data is None:
            data = fetch()
        ready, missing = readiness(data is not None)
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
