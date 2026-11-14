# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packing invariants; no torch, CUDA or distributed dependencies."""

import random

import pytest

from relax.utils.data.workload_balancing import pack_by_workload


def _check(partitions, lengths, capacity, count=None):
    assert sorted(i for group in partitions for i in group) == list(range(len(lengths)))
    assert all(group and group == sorted(group) for group in partitions)
    assert all(sum(lengths[i] for i in group) <= capacity for group in partitions)
    if count is not None:
        assert len(partitions) == count


def test_workload_packing_improves_long_short_example_without_exceeding_capacity():
    lengths = [1024] * 5 + [4096]
    costs = [24576 * length + length**2 for length in lengths]
    partitions = pack_by_workload(lengths, costs, 5120)
    _check(partitions, lengths, 5120, 2)
    assert sorted(sum(costs[i] for i in group) for group in partitions) == [117440512, 131072000]
    assert max(sum(costs[i] for i in group) for group in partitions) < 143654912


def test_workload_packing_uses_capacity_lengths_separately_from_costs():
    lengths = [6, 6, 4, 4]
    costs = [1000, 1000, 1, 1]
    partitions = pack_by_workload(lengths, costs, 10, num_partitions=2)
    _check(partitions, lengths, 10, 2)
    assert all(sum(costs[i] for i in group) == 1001 for group in partitions)


def test_workload_packing_keeps_legal_fallback_when_workload_first_cannot_place():
    lengths = [8, 7, 3, 2]
    partitions = pack_by_workload(lengths, [1, 2, 10, 9], 10)
    _check(partitions, lengths, 10, 2)


def test_workload_packing_can_honor_a_count_smaller_than_first_fit_decreasing():
    lengths = [6, 5, 3, 2, 2, 2]
    partitions = pack_by_workload(lengths, [100, 90, 80, 70, 60, 50], 10, num_partitions=2)
    _check(partitions, lengths, 10, 2)


def test_workload_packing_rejects_a_feasible_candidate_with_worse_maximum_work():
    lengths = [1, 9, 5, 3]
    costs = [21, 28, 25, 16]
    partitions = pack_by_workload(lengths, costs, 10)
    _check(partitions, lengths, 10, 2)
    assert sorted(sum(costs[i] for i in group) for group in partitions) == [41, 49]


def test_workload_packing_honors_larger_explicit_count_without_dummy_samples():
    lengths = [3, 3, 2, 2]
    partitions = pack_by_workload(lengths, [9, 9, 4, 4], 10, num_partitions=4)
    _check(partitions, lengths, 10, 4)


def test_workload_packing_boundary_and_empty_batch():
    assert pack_by_workload([], [], 10) == []
    assert pack_by_workload([10], [123], 10) == [[0]]


def test_reference_average_token_lower_bound_is_not_a_feasible_bin_count():
    partitions = pack_by_workload([6, 6, 6], [24576 * 6 + 6**2] * 3, 10)
    _check(partitions, [6, 6, 6], 10, 3)


def test_workload_assignment_preserves_reference_positive_scale_invariance():
    lengths = [1024] * 5 + [4096]
    normalized = [24576 * length + length**2 for length in lengths]
    assert pack_by_workload(lengths, normalized, 5120) == pack_by_workload(
        lengths, [196608 * cost for cost in normalized], 5120
    )


@pytest.mark.parametrize(
    "lengths,costs,capacity,count,error",
    [
        ([1], [], 10, None, ValueError),
        ([11], [1], 10, None, ValueError),
        ([1], [1], 10, 2, ValueError),
        ([], [], 10, 1, ValueError),
        ([6, 6], [1, 1], 10, 1, ValueError),
        ([6, 6, 6], [1, 1, 1], 10, 2, ValueError),
        ([0], [1], 10, None, ValueError),
        ([1], [0], 10, None, ValueError),
        ([-1], [1], 10, None, ValueError),
        ([1], [-1], 10, None, ValueError),
        ([True], [1], 10, None, TypeError),
        ([1.5], [1], 10, None, TypeError),
        ([1], [float("nan")], 10, None, TypeError),
        ([1], [float("inf")], 10, None, TypeError),
        ([1], [True], 10, None, TypeError),
        ([1], [1], 0, None, ValueError),
        ([1], [1], True, None, TypeError),
        ([1], [1], 10, 0, ValueError),
        ([1], [1], 10, 1.5, TypeError),
    ],
)
def test_workload_packing_rejects_invalid_requests(lengths, costs, capacity, count, error):
    with pytest.raises(error):
        pack_by_workload(lengths, costs, capacity, num_partitions=count)


def test_workload_packing_is_deterministic_and_preserves_input_lists():
    lengths, costs = [8, 2, 6, 4, 1], [20, 100, 4, 90, 6]
    original = (lengths[:], costs[:])
    assert pack_by_workload(lengths, costs, 10) == pack_by_workload(lengths, costs, 10)
    assert (lengths, costs) == original


def test_workload_packing_invariants_on_randomized_batches():
    rng = random.Random(16)
    for _ in range(250):
        lengths = [rng.randint(1, 64) for _ in range(rng.randint(1, 40))]
        costs = [rng.randint(1, 10**9) for _ in lengths]
        partitions = pack_by_workload(lengths, costs, 64)
        _check(partitions, lengths, 64)
        count = rng.randint(len(partitions), len(lengths))
        _check(pack_by_workload(lengths, costs, 64, num_partitions=count), lengths, 64, count)
