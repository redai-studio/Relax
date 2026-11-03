# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preference-row atomicity and dynamic batching tests."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import data as data_module
    from relax.backends.megatron import loss as loss_module
    from relax.backends.megatron.data import expand_preference_rollout_data
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


def _pair_rows(count=2):
    return {
        "pair_ids": list(range(100, 100 + count)),
        "chosen_tokens": [[1, 2]] * count,
        "rejected_tokens": [[1, 3]] * count,
        "chosen_loss_masks": [[0, 1]] * count,
        "rejected_loss_masks": [[0, 1]] * count,
        "chosen_total_lengths": [2] * count,
        "rejected_total_lengths": [2] * count,
    }


def test_expand_keeps_pairs_atomic_and_preserves_dynamic_denominator():
    rows = _pair_rows()
    rows["dynamic_global_batch_size"] = 2
    flat = expand_preference_rollout_data(rows)
    assert flat["dynamic_global_batch_size"] == 2
    assert flat["preference_branch_pair_ids"] == [100, 100, 101, 101]
    assert flat["preference_is_chosen"] == [True, False, True, False]
    assert flat["preference_pair_costs"] == [4, 4]


def test_preference_iterator_validates_step_global_pair_denominator(monkeypatch):
    flat = expand_preference_rollout_data(_pair_rows())
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group_gloo", lambda **kwargs: object())

    def all_gather_object(output, value, **_kwargs):
        output[:] = [value]

    monkeypatch.setattr(data_module.dist, "all_gather_object", all_gather_object)
    flat["dynamic_global_batch_size"] = 2
    args = Namespace(global_batch_size=8, max_tokens_per_gpu=16)
    iterators, counts = data_module._get_preference_data_iterator(args, flat, None)
    assert counts == [1]
    assert flat["dynamic_global_batch_size"] == 2
    assert flat[data_module.ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY] == [2]
    assert len(iterators) == 1

    invalid = expand_preference_rollout_data(_pair_rows())
    invalid["dynamic_global_batch_size"] = 4
    with pytest.raises(ValueError, match="step-global preference pair count"):
        data_module._get_preference_data_iterator(args, invalid, None)


@pytest.mark.parametrize("pair_ids", [[100, 101], [100, 100]])
@pytest.mark.parametrize(("capacity", "peer_bins"), [(3, 2), (4, 2), (8, 1), (8, 2)])
def test_dp2_pair_rows_remain_atomic_with_global_pair_denominator(monkeypatch, pair_ids, capacity, peer_bins):
    rows = _pair_rows()
    rows["pair_ids"] = pair_ids
    rows["chosen_tokens"] = [[1, 2], [4, 5]]
    rows["rejected_tokens"] = [[1, 3], [4, 6]]
    flat = expand_preference_rollout_data(rows)
    policy = torch.tensor([-1.0, -2.0, -0.5, -0.75], requires_grad=True)
    flat["policy_log_probs"] = [value.expand(2) for value in policy.unbind()]
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group_gloo", lambda **kwargs: object())
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module, "get_log_probs_and_entropy", lambda logits, **kwargs: (None, {"log_probs": list(logits.unbind())})
    )

    def all_gather_object(output, value, **_kwargs):
        output[:] = [value, (value[0], value[1], peer_bins)]

    monkeypatch.setattr(data_module.dist, "all_gather_object", all_gather_object)
    args = Namespace(
        global_batch_size=4,
        max_tokens_per_gpu=capacity,
        loss_type="dpo",
        dpo_reference_free=True,
        dpo_beta=0.2,
        qkv_format="thd",
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        allgather_cp=False,
    )
    iterators, counts = data_module._get_preference_data_iterator(args, flat, None)
    assert flat["dynamic_global_batch_size"] == 4
    assert counts == [peer_bins]
    seen = []
    seen_tokens = []
    losses = []
    for _ in range(counts[0]):
        batch = iterators[0].get_next(
            [
                "preference_branch_pair_ids",
                "preference_is_chosen",
                "tokens",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "policy_log_probs",
            ]
        )
        ids = batch["preference_branch_pair_ids"]
        assert ids
        if sum(map(len, batch["tokens"])) > capacity:
            assert len(ids) == 2
        assert ids[::2] == ids[1::2]
        assert batch["preference_is_chosen"] == [True, False] * (len(ids) // 2)
        seen.extend(batch["preference_branch_pair_ids"])
        seen_tokens.extend(batch["tokens"])
        batch["unconcat_tokens"] = batch["tokens"]
        batch["loss_masks"] = [torch.tensor(mask) for mask in batch["loss_masks"]]
        batch["dynamic_global_batch_size"] = flat["dynamic_global_batch_size"]
        loss, _, _ = loss_module.loss_function(
            args, batch, num_microbatches=counts[0], logits=torch.stack(batch["policy_log_probs"])
        )
        # Megatron averages microbatch gradients and DDP averages both ranks.
        losses.append(loss / (counts[0] * 2))
    assert sorted(seen) == sorted(pair_ids * 2)
    assert seen_tokens == [[1, 2], [1, 3], [4, 5], [4, 6]]
    actual = torch.stack(losses).sum()
    expected = -F.logsigmoid(0.2 * (policy[::2] - policy[1::2])).sum() / 4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, policy)[0], torch.autograd.grad(expected, policy)[0])


def test_preference_iterator_rejects_unequal_dp_pair_rows_via_gloo(monkeypatch):
    flat = expand_preference_rollout_data(_pair_rows())
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 2)
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group_gloo", lambda **kwargs: object())

    def all_gather_object(output, value, **_kwargs):
        output[:] = [value, (value[0] + 1, value[1], value[2])]

    monkeypatch.setattr(data_module.dist, "all_gather_object", all_gather_object)
    args = Namespace(global_batch_size=4, max_tokens_per_gpu=16)
    with pytest.raises(ValueError, match="equal local pair rows"):
        data_module._get_preference_data_iterator(args, flat, None)
