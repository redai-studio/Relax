# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.data.flops_balancing import (
    PLAN_METADATA_FIELD,
    STEP_PLANS_FIELD,
    build_flops_schedule,
    encode_schedule,
    get_dp_packing_geometry,
    local_schedule,
    restore_schedule,
)


@pytest.mark.parametrize(
    "lengths,costs,capacity,dp,group,expected,paired_order",
    [
        ([1] * 6, [1, 1, 1, 1, 1, 9], 5, 2, 1, [[[5]], [[0, 1, 2, 3, 4]]], [[0], [0]]),
        ([2] * 6, [1, 2, 3, 4, 5, 6], 12, 2, 2, [[[1, 5], [2]], [[0, 4], [3]]], [[1, 0], [1, 0]]),
        ([9, 9, 1, 1], [1, 1, 1, 9], 10, 1, 1, [[[3], [0, 1, 2]]], [[0, 1]]),
        ([9, 6, 6, 3, 3, 8], [90, 42, 42, 12, 12, 72], 12, 2, 1, [[[1, 3], [4, 5]], [[0], [2]]], [[0, 1], [1, 0]]),
        ([1] * 8, [1] * 8, 2, 2, 1, [[[0, 7], [3, 4]], [[1, 6], [2, 5]]], [[0, 1], [0, 1]]),
    ],
)
def test_global_schedule(lengths, costs, capacity, dp, group, expected, paired_order):
    # Small inline cases, as in upstream tests/test_dp_schedule.py. Ordered
    # expectations were checked with Slime 053bdfc using identical costs.
    plan = build_flops_schedule(lengths, costs, capacity, dp, group)
    assert plan == expected
    assert len({len(rank) for rank in plan}) == 1
    assert sorted(i for rank in plan for pack in rank for i in pack) == list(range(len(lengths)))
    metadata = {i: dict(total_lengths=length, flops_cost=cost) for i, (length, cost) in enumerate(zip(lengths, costs))}
    paired = [[rank[i] for i in order] for rank, order in zip(expected, paired_order, strict=True)]
    for pp in (1, 2):
        packing = {**geometry(capacity), "microbatch_group_size": group, "pipeline_stages": pp}
        rows, updates = encode_schedule(list(metadata), metadata, packing, dp, len(lengths), ("p", "train", 0))
        actual = [updates[rank[0]][PLAN_METADATA_FIELD]["packs"] for rank in rows]
        assert actual == (paired if pp == 1 else expected)


def test_impossible_alignment_is_rejected():
    with pytest.raises(ValueError, match="Cannot split"):
        build_flops_schedule([10] * 5, [1] * 5, 10, 2, 2)


def geometry(capacity=5):
    return dict(version=4, capacity=capacity, microbatch_group_size=1, context_parallel_size=1, pipeline_stages=1)


class Partition:
    def __init__(self):
        self.global_indexes = set(range(6))
        self.metadata = {i: dict(sample_index=i, total_lengths=1, flops_cost=9 if i == 5 else 1) for i in range(6)}

    def get_custom_meta(self, indexes):
        return {i: self.metadata[i] for i in indexes}

    def set_custom_meta(self, updates):
        self.metadata.update(updates)


def collect(sampler, partition, capacity=5):
    result = [None, None]
    for rank in (1, 0):
        rows, consumed = sampler.sample(
            list(partition.global_indexes),
            3,
            task_name="actor_train",
            partition_id="p",
            dp_rank=rank,
            batch_index=0,
            partition=partition,
            packing_geometry=geometry(capacity),
        )
        assert rows == consumed
        result[rank] = rows
    return result


@pytest.mark.parametrize("sft", [False, True])
def test_plan_replay_checkpoint_and_step_boundaries(sft):
    pytest.importorskip("transfer_queue")
    from relax.utils.data.identity_window_sampler import FlopsBalancedSampler, IdentityWindowSampler

    def make():
        return FlopsBalancedSampler(dp_size=2) if sft else IdentityWindowSampler(2, "balanced", balance_by_flops=True)

    partition = Partition()
    lengths = [9, 6, 6, 3, 3, 8]
    partition.metadata = {
        i: dict(sample_index=i, total_lengths=length, flops_cost=length + length * length)
        for i, length in enumerate(lengths)
    }
    sampler = make()
    expected = [[1, 3, 4, 5], [2, 0]]
    assert collect(sampler, partition, capacity=12) == expected
    restored = make()
    restored.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    for metadata in partition.metadata.values():
        metadata.pop(PLAN_METADATA_FIELD)
        metadata["producer_field"] = 42
    assert collect(restored, partition, capacity=12) == expected
    assert all(row["producer_field"] == 42 for row in partition.metadata.values())
    for rows in expected:
        batch = dict(total_lengths=[lengths[i] for i in rows])
        meta = SimpleNamespace(global_indexes=rows, get_all_custom_meta=lambda: [partition.metadata[i] for i in rows])
        restore_schedule(meta, batch)
        assert local_schedule(batch) == ([[0, 1], [2, 3]] if len(rows) == 4 else [[0], [1]], [2])
        assert batch["rollout_mini_local_sample_counts"] == [len(rows)]
        assert batch["rollout_mini_global_sample_counts"] == [6]
        doubled = {key: value + copy.deepcopy(value) for key, value in batch.items()}
        assert local_schedule(doubled)[1] == [2, 2]
        assert len(doubled[STEP_PLANS_FIELD]) == 2
    with pytest.raises(ValueError, match="geometry changed"):
        collect(restored, partition, capacity=24)
    restored.clear_cache("p")
    assert collect(restored, partition, capacity=24) == [[1, 3, 4, 5], [0, 2]]


def test_checkpoint_preserves_flops_mode():
    pytest.importorskip("transfer_queue")
    from relax.utils.data.identity_window_sampler import IdentityWindowSampler

    baseline = IdentityWindowSampler(2, "balanced")
    enabled = IdentityWindowSampler(2, "balanced", balance_by_flops=True)
    old_state = baseline.save_checkpoint()
    old_state.pop("balance_geometry")
    baseline.load_checkpoint(old_state)
    with pytest.raises(ValueError, match="FLOPs mode"):
        enabled.load_checkpoint(old_state)
    with pytest.raises(ValueError, match="FLOPs mode"):
        baseline.load_checkpoint(enabled.save_checkpoint())


def test_global_logical_count_is_independent_of_physical_rows():
    partition = Partition()
    ranks, updates = encode_schedule(list(range(6)), partition.metadata, geometry(), 2, 3, ("p", "train", 0))
    for rows in ranks:
        batch = dict(total_lengths=[1] * len(rows))
        restore_schedule(
            SimpleNamespace(global_indexes=rows, get_all_custom_meta=lambda: [updates[i] for i in rows]), batch
        )
        assert batch["rollout_mini_global_sample_counts"] == [3]
        assert local_schedule(batch)[1] == [1]


def test_allgather_cp_rounds_local_budget_before_multiplying():
    args = Namespace(
        balance_by_flops=True,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=12,
        allgather_cp=True,
        context_parallel_size=2,
        tensor_model_parallel_size=2,
        data_pad_size_multiplier=4,
    )
    assert get_dp_packing_geometry(args, "train")["capacity"] == 16
    assert get_dp_packing_geometry(args, "ref_log_probs") is None
    args.max_tokens_per_gpu = 7
    with pytest.raises(ValueError, match="padding must fit"):
        get_dp_packing_geometry(args, "train")
