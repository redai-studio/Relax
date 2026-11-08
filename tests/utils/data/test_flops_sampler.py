# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import inspect
from types import SimpleNamespace

import pytest

from relax.utils.training.flops_counter import SampleFlopsCost


@pytest.fixture
def cost():
    return SampleFlopsCost.from_config(
        SimpleNamespace(
            model_type="qwen3",
            hidden_size=128,
            vocab_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
        )
    )


@pytest.fixture
def sampler_type():
    from transfer_queue import SeqlenBalancedSampler

    if "assignment_transform" not in inspect.signature(SeqlenBalancedSampler.sample).parameters:
        pytest.fail("The real Docker-pinned TransferQueue assignment_transform API is required")
    return SeqlenBalancedSampler


class Partition:
    def __init__(self, lengths):
        self.metadata = {i: {"total_lengths": n} for i, n in enumerate(lengths)}

    def get_custom_meta(self, ids):
        return {i: self.metadata[i] for i in ids}


def test_real_sampler_assignment_cache_checkpoint_and_step_membership(sampler_type, cost):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    lengths = [n * 512 for n in [20, 21, 23, 10, 8, 10, 12, 9, 14, 3, 12, 16]] * 2
    partition = Partition(lengths)
    base = sampler_type(n_samples_per_prompt=1, dp_size=3)
    sampler = FlopsBalancedSampler(base, cost, 16384)
    original = copy.deepcopy(partition.metadata)
    outputs = []
    for batch_index in range(2):
        groups = {}
        for rank in [2, 0, 1]:
            ids, consumed = sampler.sample(
                list(range(batch_index * 12, 24)),
                4,
                "sft_train",
                "sft_0",
                dp_rank=rank,
                batch_index=batch_index,
                partition=partition,
            )
            assert ids == consumed
            groups[rank] = ids
            assert sampler.has_cached_result("sft_0", "sft_train", {"dp_rank": rank, "batch_index": batch_index})
        assert sorted(i for group in groups.values() for i in group) == list(
            range(batch_index * 12, (batch_index + 1) * 12)
        )
        assert all(len(group) == 4 for group in groups.values())
        outputs.append(groups)
    assert partition.metadata == original
    recovered = FlopsBalancedSampler(sampler_type(n_samples_per_prompt=1, dp_size=3), cost, 16384)
    recovered.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    for batch_index, groups in enumerate(outputs):
        for rank in [1, 2, 0]:
            ids, _ = recovered.sample(
                [], 4, "sft_train", "sft_0", dp_rank=rank, batch_index=batch_index, partition=partition
            )
            assert ids == groups[rank]
    sampler.clear_cache("sft_0")
    assert not sampler.has_cached_result("sft_0", "sft_train", {"dp_rank": 0, "batch_index": 0})


def test_real_sampler_improves_workload_against_token_assignment(sampler_type, cost):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    lengths = [n * 512 for n in [20, 21, 23, 10, 8, 10, 12, 9, 14, 3, 12, 16]]
    sampler = FlopsBalancedSampler(sampler_type(dp_size=3), cost, 16384)
    sampler.sample(list(range(12)), 4, "sft_train", "sft_0", dp_rank=0, partition=Partition(lengths))
    before = sampler.last_assignment_metrics["before"]
    after = sampler.last_assignment_metrics["after"]
    assert max(before["capacity_tokens"]) - min(before["capacity_tokens"]) <= 2 * 512
    assert after["max_flops"] - after["min_flops"] < before["max_flops"] - before["min_flops"]


def test_evaluation_delegates_without_new_assignment(sampler_type, cost):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    baseline = sampler_type(dp_size=2)
    adapter = FlopsBalancedSampler(sampler_type(dp_size=2), cost, 16384)
    kw = dict(dp_rank=0, batch_index=0, partition=Partition([8, 4, 2, 6]))
    assert adapter.sample(list(range(4)), 2, "sft_eval", "eval_0", **kw) == baseline.sample(
        list(range(4)), 2, "sft_eval", "eval_0", **kw
    )


def test_missing_length_fails_before_publication(sampler_type, cost):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    partition = Partition([4, 5, 6, 7])
    partition.metadata[1] = {}
    sampler = FlopsBalancedSampler(sampler_type(dp_size=2), cost, 16384)
    with pytest.raises(ValueError, match="total_lengths"):
        sampler.sample(list(range(4)), 2, "sft_train", "sft_0", dp_rank=0, partition=partition)
    assert not sampler.has_cached_result("sft_0", "sft_train", {"dp_rank": 0, "batch_index": 0})


def test_no_private_cache_access_and_explicit_api_check(cost):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    class Unpatched:
        def sample(self, *args, **kwargs):
            raise AssertionError("must fail before sampling")

    with pytest.raises(RuntimeError, match="assignment_transform"):
        FlopsBalancedSampler(Unpatched(), cost, 16384)


def test_real_sampler_K_rejection_is_published_and_replayed(sampler_type):
    from relax.utils.data.flops_sampler import FlopsBalancedSampler

    class QuadraticCost:
        def estimate(self, lengths):
            return [length**2 for length in lengths]

    lengths = [1, 1, 4, 7, 5, 9, 7, 2]
    partition = Partition(lengths)
    baseline = sampler_type(dp_size=2)
    adapter = FlopsBalancedSampler(sampler_type(dp_size=2), QuadraticCost(), 10)
    expected = []
    for rank in (0, 1):
        kwargs = dict(dp_rank=rank, partition=partition)
        expected.append(baseline.sample(list(range(8)), 4, "sft_train", "sft", **kwargs))
        assert adapter.sample(list(range(8)), 4, "sft_train", "sft", **kwargs) == expected[-1]
    assert adapter.last_assignment_metrics["decision"] == "rejected_by_k"
    assert adapter.last_assignment_metrics["counts"]["raw"] == {"baseline": [2, 2], "candidate": [3, 2]}
    restored = FlopsBalancedSampler(sampler_type(dp_size=2), QuadraticCost(), 10)
    restored.load_checkpoint(copy.deepcopy(adapter.save_checkpoint()))
    for rank in (1, 0):
        assert restored.sample([], 4, "sft_train", "sft", dp_rank=rank) == expected[rank]
