# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.data.flops_balancing import build_flops_schedule


def _assert_schedule_contract(
    schedule: list[list[list[int]]],
    dummy: list[set[int]],
    lengths: list[int],
    capacity: int,
    num_microbatches: int,
) -> None:
    assert len(schedule) == 4
    assert {len(rank) for rank in schedule} == {num_microbatches}
    consumed = []
    for rank, dummy_indices in zip(schedule, dummy, strict=True):
        assert dummy_indices <= set(range(len(rank)))
        real = [pack for offset, pack in enumerate(rank) if offset not in dummy_indices]
        assert real
        assert all(rank[offset] in real for offset in dummy_indices)
        for pack in rank:
            assert pack
            assert sum(lengths[index] for index in pack) <= capacity or len(pack) == 1
        consumed.extend(index for pack in real for index in pack)
    assert sorted(consumed) == list(range(len(lengths)))


@pytest.mark.parametrize(
    "num_samples,microbatch_group_size,num_microbatches,expected_load",
    [(20, 1, 2, 5.0), (36, 4, 4, 9.0)],
    ids=["dp-alignment", "vpp-alignment"],
)
def test_flops_schedule_alignment_balances_equal_cost_samples(
    num_samples: int, microbatch_group_size: int, num_microbatches: int, expected_load: float
) -> None:
    lengths = [1024] * num_samples
    costs = [1.0] * num_samples
    schedule, dummy = build_flops_schedule(lengths, costs, 4096, 4, microbatch_group_size)

    _assert_schedule_contract(schedule, dummy, lengths, 4096, num_microbatches)
    assert not any(dummy)
    loads = [sum(costs[index] for pack in rank for index in pack) for rank in schedule]
    assert loads == [expected_load] * 4


@pytest.mark.parametrize(
    "lengths,capacity,microbatch_group_size,num_microbatches,num_dummy",
    [
        ([1, 4, 4, 6, 3, 6, 6], 10, 1, 1, 0),
        ([9] + [4] * 8, 8, 1, 2, 0),
        ([8, 4, 2, 1], 8, 2, 2, 4),
    ],
    ids=["capacity-fallback", "oversized-singleton", "dummy-padding"],
)
def test_flops_schedule_alignment_preserves_capacity_and_padding(
    lengths: list[int], capacity: int, microbatch_group_size: int, num_microbatches: int, num_dummy: int
) -> None:
    costs = [float(length * length + length) for length in lengths]
    schedule, dummy = build_flops_schedule(lengths, costs, capacity, 4, microbatch_group_size)

    _assert_schedule_contract(schedule, dummy, lengths, capacity, num_microbatches)
    assert sum(len(indices) for indices in dummy) == num_dummy


def test_flops_schedule_rebalances_aligned_binpack_fallback() -> None:
    # Three balanced packs overflow the capacity, and bin-packing already
    # returns a DP-aligned count of four: [8], [8], [3, 6], [3]. Keeping those
    # packs gives the ranks 16 and 12; rebalancing at four packs gives 14 each.
    lengths = [3, 8, 3, 8, 6]
    costs = [float(length) for length in lengths]
    schedule, dummy = build_flops_schedule(lengths, costs, 10, 2)

    assert not any(dummy)
    assert all(sum(lengths[index] for index in pack) <= 10 for rank in schedule for pack in rank)
    assert [sum(costs[index] for pack in rank for index in pack) for rank in schedule] == [14.0, 14.0]
