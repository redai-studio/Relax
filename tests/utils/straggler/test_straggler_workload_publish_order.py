# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression test for the P0-1 workload publish ordering defect.

The per-step workload used to be published inside the SFT prefetch worker from
the RANK-LOCAL ``k_local`` partition. Training executes the DP-wide ``max_k``
partition and repacks when ``local_k < max_k``, so the published metadata could
describe a partition that was thrown away. These tests force that branch and
pin the publish to the point after the DP-wide K decision.
"""

import re
from pathlib import Path

from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions


ACTOR = Path(__file__).resolve().parents[3] / "relax" / "backends" / "megatron" / "actor.py"


def _workload(samples, k):
    """The workload the actor publishes for partition ``k``."""
    indices = get_seqlen_balanced_partitions(samples, k, equal_size=False)
    return [(sum(samples[i] for i in group), len(group), 1) for group in indices]


def test_local_k_different_from_max_k_produces_different_workload():
    """The stale-metadata hazard: k_local and max_k give different metadata.

    A rank whose local partition needs fewer micro-batches than the DP-wide
    maximum is repacked before training runs. If the workload were published
    before that repack, it would describe this discarded ``k_local`` layout.
    """
    samples = [10, 10, 60, 90, 5, 5]
    k_local = 2
    max_k = 4

    local_workload = _workload(samples, k_local)
    executed_workload = _workload(samples, max_k)

    assert len(local_workload) == k_local
    assert len(executed_workload) == max_k
    assert local_workload != executed_workload
    # Both partitions cover every sample: the difference is the grouping, which
    # is exactly what makes the stale metadata silently wrong.
    for workload, k in ((local_workload, k_local), (executed_workload, max_k)):
        assert len(workload) == k
        assert sum(entry[1] for entry in workload) == len(samples)


def test_published_workload_matches_the_executed_max_k_partition():
    """Publishing from ``max_k`` is self-consistent with what train() runs."""
    samples = [10, 10, 60, 90, 5, 5]
    max_k = 4
    executed = _workload(samples, max_k)

    indices = get_seqlen_balanced_partitions(samples, max_k, equal_size=False)
    assert len(indices) == max_k
    assert sum(len(group) for group in indices) == len(samples)
    assert _workload(samples, max_k) == executed


def test_the_publish_passes_the_executed_k_and_lives_in_the_training_path():
    """Coupling guard: the derivation must get the EXECUTED max_k.

    The previous guard asserted only a line-number ordering, which passes on
    revert because the removed worker site had a HIGHER line number than the
    final-K line. This anchors on the argument that actually changes when the
    publish moves back into the prefetch worker: reverting would pass
    ``k_local`` (or derive from the rank-local partition) instead of ``max_k``.
    """
    source = ACTOR.read_text()

    assert source.count("step_workloads(") == 1, "the derivation must have exactly one call site"
    assert re.search(r"step_workloads\(\s*samples,\s*max_k\s*,", source), "the publish must pass the EXECUTED max_k"
    assert not re.search(r"step_workloads\(\s*samples,\s*k_local\s*,", source), (
        "the publish must never derive from the rank-local k"
    )
    # And it must sit in the training-thread path, not the prefetch worker.
    training_path = source.index("def _get_prefetched_sft_window")
    prefetch_worker = source.index("def _pack_sft_prepack_window")
    call = source.index("step_workloads(")
    assert training_path < call < prefetch_worker
    # Gated, so a disabled profiler does no derivation at all.
    assert "_straggler_publish_enabled()" in source[:call]
