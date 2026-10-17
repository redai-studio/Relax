# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only proof of the per-step workload arithmetic (P0-1).

Runs in the CPU venv because ``relax.utils.straggler.workload`` imports no
Megatron, Ray or transfer_queue. This is the arithmetic half of the publish
proof; the coupling half (that ``actor.py`` calls it with the executed K) is a
source check in ``test_straggler_workload_publish_order.py``.
"""

import pytest

from relax.utils.straggler.workload import step_workloads


SAMPLES = [10, 10, 60, 90, 5, 5]
K_LOCAL = 1
MAX_K = 3


def test_sft_prepack_case_publishes_the_step_total_not_the_first_group():
    """One optimizer step consumes the window: the entry is the STEP total."""
    published = step_workloads(SAMPLES, MAX_K, 1)

    assert len(published) == 1, "one optimizer step must yield exactly one entry"
    assert published[0] == (sum(SAMPLES), len(SAMPLES), MAX_K)
    # The bug this replaces: one entry per micro-batch, of which the detector
    # read only the first group.
    assert published[0] != (90, 1, 1)
    assert published[0][2] != K_LOCAL


def test_entry_count_equals_num_steps_per_rollout():
    """The list length matches the number of steps that consume the window."""
    for steps in (1, 2, 3):
        published = step_workloads(SAMPLES, MAX_K, steps)
        assert len(published) == steps


def test_multi_step_window_splits_the_real_partition_across_steps():
    """A multi-step window reports each step's own micro-batches and tokens."""
    published = step_workloads(SAMPLES, MAX_K, 3)

    assert len(published) == 3
    assert sum(entry[2] for entry in published) == MAX_K
    assert sum(entry[1] for entry in published) == len(SAMPLES)
    assert sum(entry[0] for entry in published) == sum(SAMPLES)
    assert all(entry[2] >= 1 for entry in published)


def test_equal_totals_report_equal_tokens_regardless_of_grouping():
    """The false-suppression mechanism: a first group is not a rank's workload.

    ``[97,1,1,1]`` and ``[25,25,25,25]`` both total 100; publishing per-group
    made their first groups 97 and 25, a gap far beyond the 5% tolerance, so
    the gate suppressed verdicts for equally-loaded ranks. The step total makes
    them equal.
    """
    flat = step_workloads([97, 1, 1, 1], MAX_K, 1)
    even = step_workloads([25, 25, 25, 25], MAX_K, 1)

    assert flat == even == [(100, 4, MAX_K)]


def test_the_executed_k_is_what_becomes_the_microbatch_count():
    """k_partitions is the executed DP-wide K, not the rank-local k."""
    published = step_workloads(SAMPLES, MAX_K, 1)
    assert published[0][2] == MAX_K


def test_impossible_shapes_raise_instead_of_inventing_a_payload():
    """Bad shapes fail loudly here rather than silently publishing nothing."""
    with pytest.raises(ValueError):
        step_workloads(SAMPLES, MAX_K, 0)
    with pytest.raises(ValueError):
        step_workloads(SAMPLES, 0, 1)
    with pytest.raises(ValueError):
        step_workloads([1, 2], 3, 1)
