# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy

import pytest


pytest.importorskip("transfer_queue")

from relax.utils.data.flops_balancing import FlopsBalancedSampler, validated_cost_weights
from relax.utils.data.identity_window_sampler import IdentityWindowSampler


class Partition:
    def __init__(self, lengths: list[int]) -> None:
        self.global_indexes = set(range(len(lengths)))
        self.metadata = {
            i: {"sample_index": i, "total_lengths": length, "flops_cost": float(4 * length + length**2)}
            for i, length in enumerate(lengths)
        }

    def get_custom_meta(self, indexes: list[int]) -> dict[int, dict]:
        return {i: self.metadata[i] for i in indexes}


def collect(sampler, partition: Partition, order: list[int], batch_index: int = 0) -> list[list[int]]:
    groups = [None] * len(order)
    ready = list(partition.global_indexes)
    for rank in order:
        sampled, consumed = sampler.sample(
            ready,
            batch_size=4,
            task_name="actor_train",
            partition_id="train_0",
            dp_rank=rank,
            batch_index=batch_index,
            partition=partition,
        )
        assert sampled == consumed
        groups[rank] = sampled
        ready = [index for index in ready if index not in consumed]
    return groups


@pytest.mark.parametrize("sft", [False, True])
def test_cost_assignment_preserves_batch_identity_replay_and_checkpoint(sft):
    lengths = [8, 8, 20, 26, 22, 20, 1, 14, 14, 20, 6, 17]
    partition = Partition(lengths)

    def make():
        return (
            FlopsBalancedSampler(dp_size=3)
            if sft
            else IdentityWindowSampler(dp_size=3, placement="balanced", cost_key="flops_cost")
        )

    sampler = make()
    groups = collect(sampler, partition, [2, 0, 1])
    assert sorted(index for group in groups for index in group) == list(range(12))
    assert all(len(group) == 4 for group in groups)
    costs = [sum(partition.metadata[i]["flops_cost"] for i in group) for group in groups]
    assert max(costs) == 1309
    assert collect(sampler, partition, [1, 2, 0]) == groups
    restored = make()
    restored.load_checkpoint(copy.deepcopy(sampler.save_checkpoint()))
    assert collect(restored, partition, [0, 1, 2]) == groups


def test_cost_assignment_keeps_existing_length_mode_and_step_boundaries():
    lengths = [8, 8, 20, 26, 22, 20, 1, 14, 14, 20, 6, 17] * 2
    partition = Partition(lengths)
    baseline = IdentityWindowSampler(dp_size=3, placement="balanced")
    first = collect(baseline, partition, [0, 1, 2])
    assert max(sum(partition.metadata[i]["flops_cost"] for i in group) for group in first) == 1517
    sampler = IdentityWindowSampler(dp_size=3, placement="balanced", cost_key="flops_cost")
    optimized = collect(sampler, partition, [0, 1, 2])
    assert sorted(i for group in optimized for i in group) == list(range(12))
    # Remove the first window from the producer's currently ready rows.
    partition.global_indexes = set(range(12, 24))
    second = collect(sampler, partition, [2, 1, 0], batch_index=1)
    assert sorted(i for group in second for i in group) == list(range(12, 24))


@pytest.mark.parametrize("bad", [None, 0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_cost_never_silently_falls_back_to_tokens(bad):
    partition = Partition([2, 3])
    partition.metadata[1]["flops_cost"] = bad
    with pytest.raises(ValueError, match="flops_cost"):
        validated_cost_weights([0, 1], partition.metadata)


def test_checkpoint_rejects_changed_cost_mode():
    baseline = IdentityWindowSampler(dp_size=2, placement="balanced")
    optimized = IdentityWindowSampler(dp_size=2, placement="balanced", cost_key="flops_cost")
    with pytest.raises(ValueError, match="geometry or cost mode"):
        optimized.load_checkpoint(baseline.save_checkpoint())


def test_producer_tensordict_metadata_and_multirow_identity_admission(monkeypatch):
    from argparse import Namespace

    from relax.utils import misc
    from relax.utils.training.microbatch_cost import add_flops_cost_metadata
    from relax.utils.utils import build_rollout_custom_meta, dict_to_tensordict
    from tests.utils.test_microbatch_cost import tiny_config

    monkeypatch.setattr(misc, "get_hf_config", lambda _: tiny_config())
    lengths = [8, 8, 20, 26, 22, 20, 4, 14, 14, 20, 6, 17]
    identities = [i // 2 for i in range(12)]
    batch = {"tokens": [[1] * n for n in lengths], "total_lengths": lengths, "sample_indices": identities}
    args = Namespace(
        dp_balance="flops",
        microbatch_balance="tokens",
        loss_type="grpo",
        balance_data=True,
        use_dynamic_batch_size=True,
        hf_checkpoint="unused",
    )
    metadata = add_flops_cost_metadata(args, batch, [{"total_lengths": n} for n in lengths])
    batch["flops_cost"] = [row["flops_cost"] for row in metadata]
    published = build_rollout_custom_meta(dict_to_tensordict(batch, batch_size=12))
    assert [row["sample_index"] for row in published] == identities
    assert [row["flops_cost"] for row in published] == pytest.approx(batch["flops_cost"], rel=1e-6)
    partition = Partition(lengths)
    partition.metadata = dict(enumerate(published))
    sampler = IdentityWindowSampler(dp_size=3, placement="balanced", cost_key="flops_cost")
    groups = []
    for rank in range(3):
        ids, _ = sampler.sample(
            list(range(12)),
            2,
            task_name="actor_train",
            partition_id="train_0",
            dp_rank=rank,
            batch_index=0,
            partition=partition,
        )
        groups.append(ids)
    assert sorted(i for group in groups for i in group) == list(range(12))
    assert sampler._partition_task_states[("train_0", "actor_train")].windows[0].admitted_count == 6
