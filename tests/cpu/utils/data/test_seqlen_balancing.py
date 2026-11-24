# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for relax.utils.data.seqlen_balancing.

The module under test is pure Python (only ``copy`` and ``heapq``), so it is
imported directly at module scope and pytest can run on the host with no
numpy / torch / GPU dependency.

The Karmarkar-Karp partitioning does not produce a stable arrangement of
elements across partitions, so these tests only assert stable structural
properties: partition count, full coverage, absence of duplicates, per-partition
sizes for ``equal_size=True``, illegal-input assertions, and the
inverse-permutation contract of ``get_reverse_idx``.
"""

import pytest

from relax.utils.data.seqlen_balancing import get_reverse_idx, get_seqlen_balanced_partitions


def _assert_full_coverage_no_duplicates(partitions, n_items, k_partitions):
    """Every index in range(n_items) appears exactly once across all
    partitions."""
    assert len(partitions) == k_partitions, f"expected {k_partitions} partitions, got {len(partitions)}"
    flat = [idx for partition in partitions for idx in partition]
    assert len(flat) == n_items, f"expected {n_items} total indices, got {len(flat)}"
    assert sorted(flat) == list(range(n_items)), "partitions must cover exactly range(n_items) with no duplicates"
    for partition in partitions:
        assert len(partition) > 0, "no partition may be empty"


def test_variable_size_coverage_and_no_duplicates():
    """equal_size=False: covers every index exactly once across k partitions."""
    seqlen_list = [3, 1, 4, 1, 5, 9, 2, 6]
    k_partitions = 3
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=False)
    _assert_full_coverage_no_duplicates(partitions, n_items=len(seqlen_list), k_partitions=k_partitions)


def test_equal_size_partitions_have_equal_length():
    """equal_size=True: each of the k partitions holds exactly len/k indices."""
    seqlen_list = [10, 20, 30, 40, 50, 60, 70, 80]
    k_partitions = 4
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=True)
    _assert_full_coverage_no_duplicates(partitions, n_items=len(seqlen_list), k_partitions=k_partitions)
    expected_size = len(seqlen_list) // k_partitions
    for partition in partitions:
        assert len(partition) == expected_size, f"equal_size partition must have {expected_size} indices"


def test_partition_indices_are_sorted_ascending():
    """Returned partitions are individually sorted ascending (per
    _check_and_sort_partitions)."""
    seqlen_list = [7, 2, 9, 4, 1, 8, 3, 6, 5, 10]
    k_partitions = 2
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=False)
    _assert_full_coverage_no_duplicates(partitions, n_items=len(seqlen_list), k_partitions=k_partitions)
    for partition in partitions:
        assert partition == sorted(partition), "each partition must be sorted ascending"


def test_partition_sums_are_reasonably_balanced():
    """equal_size=False: the max-min spread of partition sums stays within the largest single item."""
    seqlen_list = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    k_partitions = 3
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size=False)
    _assert_full_coverage_no_duplicates(partitions, n_items=len(seqlen_list), k_partitions=k_partitions)
    sums = [sum(seqlen_list[idx] for idx in partition) for partition in partitions]
    # nine equal items over three partitions -> perfectly balanced sums of 15 each.
    assert sums == [15, 15, 15], f"expected balanced sums [15, 15, 15], got {sums}"


def test_single_partition_returns_all_indices():
    """k_partitions=1: the sole partition is exactly range(n_items) sorted ascending."""
    seqlen_list = [4, 8, 15, 16, 23, 42]
    partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=1, equal_size=False)
    assert partitions == [[0, 1, 2, 3, 4, 5]], "single partition must contain all indices in ascending order"


def test_raises_when_items_fewer_than_partitions():
    """Illegal input: len(seqlen_list) < k_partitions raises AssertionError."""
    with pytest.raises(AssertionError, match="number of items"):
        get_seqlen_balanced_partitions([1, 2], k_partitions=3, equal_size=False)


def test_raises_when_equal_size_length_not_divisible():
    """Illegal input: equal_size=True with len not divisible by k raises
    AssertionError."""
    with pytest.raises(AssertionError, match=r"5 % 2 != 0"):
        get_seqlen_balanced_partitions([1, 2, 3, 4, 5], k_partitions=2, equal_size=True)


def test_get_reverse_idx_is_inverse_permutation():
    """get_reverse_idx returns the inverse permutation: reverse[idx_map[i]] ==
    i."""
    idx_map = [2, 0, 3, 1]
    reverse_idx_map = get_reverse_idx(idx_map)
    assert reverse_idx_map == [1, 3, 0, 2], "inverse of [2,0,3,1] must be [1,3,0,2]"
    for i, idx in enumerate(idx_map):
        assert reverse_idx_map[idx] == i, "reverse_idx_map[idx_map[i]] must equal i"
    assert idx_map == [2, 0, 3, 1], "get_reverse_idx must not mutate its input"


def test_get_reverse_idx_round_trip_restores_identity():
    """Applying get_reverse_idx twice restores the original permutation."""
    idx_map = [4, 3, 0, 1, 2]
    round_trip = get_reverse_idx(get_reverse_idx(idx_map))
    assert round_trip == idx_map, "reversing an inverse permutation must restore the original"


def test_get_reverse_idx_identity_permutation():
    """The inverse of the identity permutation is the identity itself."""
    idx_map = [0, 1, 2, 3]
    assert get_reverse_idx(idx_map) == [0, 1, 2, 3], "inverse of identity permutation is the identity"
