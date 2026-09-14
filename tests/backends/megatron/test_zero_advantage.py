# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import runpy
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
filter_rows = runpy.run_path(str(ROOT / "relax/backends/megatron/zero_advantage.py"))["filter_zero_advantage_rows"]


def make_args(**overrides):
    values = dict(
        advantage_estimator="grpo",
        loss_type="policy_loss",
        compute_advantages_and_returns=True,
        use_agentic_rollout=True,
        use_dynamic_batch_size=True,
        num_iters_per_train_update=1,
        entropy_coef=0.0,
        kl_coef=0.0,
        use_kl_loss=True,
        kl_loss_coef=0.02,
        kl_loss_type="k3",
        global_batch_size=592,
    )
    values.update(overrides)
    return Namespace(**values)


def make_batch(rewards, identities=None):
    count = len(rewards)
    identities = list(range(count)) if identities is None else identities
    lengths = [4 + i % 3 for i in range(count)]
    masks = [torch.ones(1 + i % 2, dtype=torch.float64) for i in range(count)]
    denominators = {identity: 0.0 for identity in identities}
    for identity, mask in zip(identities, masks, strict=True):
        denominators[identity] += float(mask.sum())
    return dict(
        total_lengths=lengths,
        tokens=[torch.arange(length) for length in lengths],
        rewards=list(rewards),
        sample_indices=list(identities),
        sample_index_mask_sums=[denominators[identity] for identity in identities],
        loss_masks=masks,
        response_lengths=[len(mask) for mask in masks],
        rollout_log_probs=tuple(torch.zeros(len(mask)) for mask in masks),
        multimodal_train_inputs=[{"pixel_values": torch.full((1, 4), float(i))} for i in range(count)],
        features=(torch.arange(count * 2, dtype=torch.float64).reshape(count, 2) + 1) / 10,
        rollout_mini_local_sample_counts=[count],
        rollout_mini_global_sample_counts=[592],
        rollout_mini_prompt_group_counts=[74],
        rollout_mini_batch_metas=[{"requested_trajectories": 592}],
        dynamic_global_batch_size=592,
        extra_scalar=None,
    )


def test_prunes_whole_identities_and_preserves_denominators_and_inputs():
    batch = make_batch([0.0, -0.0, 1.0, -1.0, -1.0], [10, 10, 20, 30, 30])
    original_tokens = batch["tokens"].copy()
    filtered, dropped = filter_rows(make_args(), batch, reference_matches_actor=True)
    assert dropped == 2
    assert filtered["sample_indices"] == [20, 30, 30]
    assert filtered["sample_index_mask_sums"] == batch["sample_index_mask_sums"][2:]
    assert filtered["rollout_mini_local_sample_counts"] == [3]
    assert filtered["rollout_mini_global_sample_counts"] is batch["rollout_mini_global_sample_counts"]
    assert filtered["dynamic_global_batch_size"] == 592
    assert filtered["rollout_mini_batch_metas"] is batch["rollout_mini_batch_metas"]
    assert filtered["multimodal_train_inputs"][0] is batch["multimodal_train_inputs"][2]
    assert isinstance(filtered["rollout_log_probs"], tuple)
    torch.testing.assert_close(filtered["features"], batch["features"][2:])
    assert len(batch["tokens"]) == 5
    assert all(old is new for old, new in zip(original_tokens, batch["tokens"], strict=True))
    assert batch["rollout_mini_local_sample_counts"] == [5]


@pytest.mark.parametrize("reward", [1e-30, float("nan"), float("inf"), -float("inf")])
def test_tiny_or_nonfinite_reward_keeps_the_entire_identity(reward):
    batch = make_batch([0.0, reward, 0.0], [1, 1, 2])
    filtered, dropped = filter_rows(make_args(), batch, reference_matches_actor=True)
    assert dropped == 1
    assert filtered["sample_indices"] == [1, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"advantage_estimator": "ppo"},
        {"loss_type": "custom_loss"},
        {"compute_advantages_and_returns": False},
        {"use_agentic_rollout": False},
        {"use_dynamic_batch_size": False},
        {"num_iters_per_train_update": 2},
        {"entropy_coef": 0.01},
        {"kl_coef": 0.01},
        {"attention_dropout": 0.1},
        {"hidden_dropout": 0.1},
        {"normalize_advantages": True},
        {"use_opd": True},
        {"partial_rollout": True},
        {"use_dynamic_global_batch_size": True},
        {"calculate_per_token_loss": True},
        {"use_rollout_routing_replay": True},
        {"use_routing_replay": True},
        {"num_experts": 8},
        {"mtp_num_layers": 1},
        {"custom_loss_function_path": "example.custom_loss"},
        {"custom_pg_loss_reducer_function_path": "example.custom_reducer"},
        {"use_tis": True, "custom_tis_function_path": "example.custom_tis"},
        {"custom_megatron_before_log_prob_hook_path": "example.log_prob_hook"},
        {"custom_megatron_before_train_step_hook_path": "example.train_hook"},
        {"rollout_data_postprocess_path": "example.train_hook"},
        {"kl_loss_type": "k1"},
    ],
)
def test_unsupported_semantics_keep_original_batch(overrides):
    batch = make_batch([0.0, 1.0])
    filtered, dropped = filter_rows(make_args(**overrides), batch, reference_matches_actor=True)
    assert filtered is batch
    assert dropped == 0


def test_unknown_or_stale_reference_keeps_original_batch():
    batch = make_batch([0.0, 1.0])
    assert filter_rows(make_args(), batch, reference_matches_actor=False) == (batch, 0)


@pytest.mark.parametrize("overrides", [{"use_kl_loss": False}, {"kl_loss_coef": 0.0}])
def test_no_kl_can_filter_without_reference_identity(overrides):
    batch = make_batch([0.0, 1.0])
    filtered, dropped = filter_rows(make_args(**overrides), batch, reference_matches_actor=False)
    assert dropped == 1
    assert filtered["sample_indices"] == [1]


@pytest.mark.parametrize("counts", [None, [1, 1], [3]])
def test_requires_valid_single_update_boundary(counts):
    batch = make_batch([0.0, 1.0])
    batch["rollout_mini_local_sample_counts"] = counts
    assert filter_rows(make_args(), batch, reference_matches_actor=True) == (batch, 0)


@pytest.mark.parametrize("bad_value", [[1], torch.ones(1, 3)])
def test_unrecognized_row_layout_falls_back_without_mutating(bad_value):
    batch = make_batch([0.0, 1.0])
    batch["other_field"] = bad_value
    filtered, dropped = filter_rows(make_args(), batch, reference_matches_actor=True)
    assert filtered is batch
    assert dropped == 0
    assert len(batch["tokens"]) == 2


def test_tensor_rewards_do_not_trigger_device_synchronization():
    batch = make_batch([0.0, 1.0])
    batch["rewards"] = torch.tensor([0.0, 1.0])
    filtered, dropped = filter_rows(make_args(), batch, reference_matches_actor=True)
    assert filtered is batch
    assert dropped == 0


def test_all_zero_dp_keeps_shortest_original_row_and_metadata():
    batch = make_batch([0.0, 0.0, 0.0])
    batch["total_lengths"] = [30, 4, 20]
    filtered, dropped = filter_rows(make_args(), batch, reference_matches_actor=True)
    assert dropped == 2
    assert filtered["sample_indices"] == [1]
    assert filtered["rewards"] == [0.0]
    assert filtered["loss_masks"][0] is batch["loss_masks"][1]
    assert filtered["rollout_mini_local_sample_counts"] == [1]
    assert filtered["rollout_mini_batch_metas"] is batch["rollout_mini_batch_metas"]
    assert filtered["rollout_mini_global_sample_counts"] == [592]


def policy_loss(weight, reference, batch):
    logits = batch["features"] @ weight
    log_probs = logits.log_softmax(dim=-1)[:, 0]
    old_log_probs = (batch["features"] @ reference).log_softmax(dim=-1)[:, 0]
    advantages = torch.tensor(batch["rewards"], dtype=torch.float64)
    ratio = (log_probs - old_log_probs).exp()
    pg = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(0.8, 1.2))
    log_ratio = old_log_probs - log_probs
    kl = log_ratio.exp() - 1.0 - log_ratio
    row_weights = torch.tensor(
        [
            float(mask.sum()) / denom
            for mask, denom in zip(batch["loss_masks"], batch["sample_index_mask_sums"], strict=True)
        ],
        dtype=torch.float64,
    )
    return ((pg + 0.02 * kl) * row_weights).sum() / batch["dynamic_global_batch_size"]


@pytest.mark.parametrize("all_zero", [False, True])
def test_loss_gradients_and_adam_update_match_unfiltered_data(all_zero):
    rewards = [0.0] * 5 if all_zero else [0.0, 0.0, 1.0, -1.0, -1.0]
    batch = make_batch(rewards, [10, 10, 20, 30, 30])
    filtered, _ = filter_rows(make_args(), batch, reference_matches_actor=True)
    initial = torch.tensor([[0.3, -0.4], [-0.2, 0.5]], dtype=torch.float64)
    outputs = []
    for current_batch in (batch, filtered):
        weight = initial.clone().requires_grad_()
        optimizer = torch.optim.AdamW([weight], lr=0.001, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        # Populate momentum: all-zero updates must still advance Adam and LR.
        weight.grad = torch.tensor([[0.1, 0.2], [0.3, -0.1]], dtype=torch.float64)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        reference = weight.detach().clone()
        loss = policy_loss(weight, reference, current_batch)
        loss.backward()
        gradient = weight.grad.detach().clone()
        optimizer.step()
        scheduler.step()
        outputs.append(
            (
                loss.detach(),
                gradient,
                weight.detach().clone(),
                copy.deepcopy(optimizer.state_dict()),
                scheduler.last_epoch,
            )
        )
    for index in range(3):
        torch.testing.assert_close(outputs[0][index], outputs[1][index], rtol=1e-12, atol=1e-14)
    assert outputs[0][4] == outputs[1][4] == 2
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(outputs[0][3]["state"][0][key], outputs[1][3]["state"][0][key])
    if all_zero:
        assert not torch.equal(outputs[0][2], initial)


def test_two_dp_partitions_keep_the_original_global_gradient_scale():
    batches = [make_batch([0.0, 0.0]), make_batch([1.0, 0.0, -1.0])]
    initial = torch.tensor([[0.3, -0.4], [-0.2, 0.5]], dtype=torch.float64)
    gradients = []
    for prune in (False, True):
        weight = initial.clone().requires_grad_()
        total = torch.zeros((), dtype=torch.float64)
        for batch in batches:
            current = filter_rows(make_args(), batch, reference_matches_actor=True)[0] if prune else batch
            assert current["rollout_mini_local_sample_counts"][0] > 0
            # Megatron's DP factor cancels DDP averaging; retain the global denominator.
            total = total + policy_loss(weight, initial, current)
        total.backward()
        gradients.append(weight.grad.clone())
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-12, atol=1e-14)
