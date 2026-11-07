# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Uses the real installed TransferQueue sampler and partition metadata."""

from copy import deepcopy

import pytest


pytest.importorskip("transfer_queue")

from transfer_queue.controller import DataPartitionStatus

from relax.utils.data.flops_sampler import FlopsBalancedSampler
from relax.utils.training.flops_counter import SequenceFlopsEstimator


def _partition(lengths, name="sft_0"):
    partition = DataPartitionStatus(partition_id=name)
    partition.set_custom_meta({i: {"total_lengths": length, "tag": f"sample-{i}"} for i, length in enumerate(lengths)})
    return partition


def test_flops_sampler_uses_workloads_and_does_not_modify_real_lengths():
    partition = _partition([1, 2, 3, 4, 5, 6])
    before = deepcopy(partition.custom_meta)
    sampler = FlopsBalancedSampler(SequenceFlopsEstimator(1, 1), dp_size=2)
    first, consumed = sampler.sample(list(range(6)), 3, "train", "sft_0", dp_rank=0, partition=partition)
    second, _ = sampler.sample([], 3, "train", "sft_0", dp_rank=1)
    assert consumed == first
    assert {frozenset(first), frozenset(second)} == {frozenset({0, 2, 5}), frozenset({1, 3, 4})}
    assert partition.custom_meta == before
    for rank, samples in enumerate((first, second)):
        assert sampler._states["sft_0"]["train"][rank][0] == (samples, samples)


def test_flops_sampler_keeps_prompt_groups_and_equal_counts():
    sampler = FlopsBalancedSampler(SequenceFlopsEstimator(1, 1), n_samples_per_prompt=2, dp_size=2)
    partition = _partition([1, 1, 2, 2, 3, 3, 4, 4])
    groups = [
        sampler.sample(list(range(8)), 4, "sft", "sft_0", dp_rank=rank, partition=partition)[0] for rank in (0, 1)
    ]
    assert sorted(i for group in groups for i in group) == list(range(8))
    for group in groups:
        assert len(group) == 4
        assert all((index ^ 1) in group for index in group)


def test_flops_sampler_checkpoint_and_clear_preserve_replay_protocol():
    estimator = SequenceFlopsEstimator(1, 1)
    sampler = FlopsBalancedSampler(estimator, dp_size=2)
    first = sampler.sample(list(range(6)), 3, "sft", "sft_0", dp_rank=0, partition=_partition([1, 2, 3, 4, 5, 6]))
    state = deepcopy(sampler.save_checkpoint())
    restored = FlopsBalancedSampler(estimator, dp_size=2)
    restored.load_checkpoint(state)
    assert restored.sample([], 3, "sft", "sft_0", dp_rank=0) == first
    restored.clear_cache("sft_0")
    assert restored._balanced_cache == {}
    assert "sft_0" not in restored._states
    with pytest.raises(ValueError, match="coefficients"):
        FlopsBalancedSampler(SequenceFlopsEstimator(2, 1), dp_size=2).load_checkpoint(state)
    for dp_size, group_size in ((4, 1), (2, 2)):
        with pytest.raises(ValueError, match="geometry"):
            FlopsBalancedSampler(estimator, dp_size=dp_size, n_samples_per_prompt=group_size).load_checkpoint(state)


def test_flops_sampler_never_crosses_step_partition_or_batch_index():
    sampler = FlopsBalancedSampler(SequenceFlopsEstimator(1, 1), dp_size=2)
    partition = _partition([1, 2, 3, 4, 5, 6, 7, 8])
    a = sampler.sample([0, 1, 2, 3], 2, "sft", "sft_0", dp_rank=0, batch_index=0, partition=partition)[0]
    b = sampler.sample([4, 5, 6, 7], 2, "sft", "sft_0", dp_rank=0, batch_index=1, partition=partition)[0]
    assert set(a) <= {0, 1, 2, 3} and set(b) <= {4, 5, 6, 7}
    assert sampler.sample([], 2, "sft", "sft_0", dp_rank=0, batch_index=0)[0] == a


@pytest.mark.parametrize("partition,rank", [(None, 0), (_partition([1]), 2), (_partition([1]), -1)])
def test_flops_sampler_rejects_missing_metadata_or_invalid_rank(partition, rank):
    with pytest.raises(ValueError):
        FlopsBalancedSampler(SequenceFlopsEstimator(1, 1), dp_size=2).sample(
            [0], 1, "sft", "sft_0", dp_rank=rank, partition=partition
        )


def test_flops_sampler_rejects_missing_lengths_instead_of_free_work():
    partition = _partition([1, 2])
    del partition.custom_meta[1]["total_lengths"]
    with pytest.raises(ValueError, match="total_lengths"):
        FlopsBalancedSampler(SequenceFlopsEstimator(1, 1), dp_size=2).sample(
            [0, 1], 1, "sft", "sft_0", dp_rank=0, partition=partition
        )
