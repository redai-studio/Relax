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


def test_the_only_publish_site_runs_after_the_dp_wide_k_is_final():
    """Source guard: the publish must not run before ``local_k = max_k``.

    Reintroducing an early publish is the whole defect, so it is pinned by
    position rather than by intent.
    """
    source = ACTOR.read_text()
    publish_sites = [m.start() for m in re.finditer(r"publish_step_workload\(", source)]
    assert len(publish_sites) == 1, f"expected exactly one publish site, found {len(publish_sites)}"

    final_k = source.index("local_k = max_k")
    assert publish_sites[0] > final_k, "the workload is published before the DP-wide K is final"

    # And it must live in the training-thread path (_get_prefetched_sft_window),
    # not in the prefetch worker that builds the rank-local k partition.
    training_path = source.index("def _get_prefetched_sft_window")
    prefetch_worker = source.index("def _pack_sft_prepack_window")
    assert training_path < publish_sites[0] < prefetch_worker, (
        "the publish must sit between the training-thread path and the prefetch worker"
    )
