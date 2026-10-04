# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production DPO loss regression tests."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from relax.engine.sft.eval.preference import finalize_pair_metrics, pair_metric_sums


try:
    from relax.backends.megatron import loss as loss_module
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


def _args(*, reference_free: bool = False, beta: float = 0.2) -> Namespace:
    return Namespace(
        loss_type="dpo",
        dpo_reference_free=reference_free,
        dpo_beta=beta,
        qkv_format="thd",
        calculate_per_token_loss=False,
        global_batch_size=2,
        recompute_loss_function=False,
        allgather_cp=False,
    )


def _run(
    monkeypatch,
    policy_values,
    *,
    order=None,
    reference_free=False,
    ref_values=None,
    num_samples=2,
    pair_ids=None,
    dispatch=False,
):
    if order is None:
        order = [0, 1, 2, 3]
    if pair_ids is None:
        pair_ids = [10, 10, 20, 20]
    is_chosen = [True, False, True, False]
    policy = [torch.as_tensor(policy_values[index]).reshape(1) for index in order]
    reference = None if ref_values is None else [torch.as_tensor(ref_values[index]).reshape(1) for index in order]
    monkeypatch.setattr(
        loss_module, "get_log_probs_and_entropy", lambda *args, **kwargs: (None, {"log_probs": policy})
    )
    logits = torch.ones(1, requires_grad=True)
    batch = {
        "response_lengths": [1] * 4,
        "unconcat_tokens": [torch.ones(1, dtype=torch.long)] * 4,
        "total_lengths": [1] * 4,
        "loss_masks": [torch.ones(1)] * 4,
        "preference_branch_pair_ids": [pair_ids[index] for index in order],
        "preference_is_chosen": [is_chosen[index] for index in order],
        "ref_log_probs": reference,
        "num_samples": num_samples,
    }
    args = _args(reference_free=reference_free)
    if not dispatch:
        return loss_module.dpo_loss_function(args, batch, logits, lambda value: value)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **kwargs: 1)
    loss, _, log = loss_module.loss_function(args, batch, num_microbatches=1, logits=logits)
    return loss, dict(zip(log["keys"], log["values"][1:], strict=True))


@pytest.mark.parametrize("pair_ids", [[10, 10, 20, 20], [10, 10, 10, 10]])
def test_production_dpo_loss_matches_independent_reference_and_gradients(monkeypatch, pair_ids):
    policy = torch.tensor([-1.0, -2.0, -0.5, -0.75], requires_grad=True)
    reference = torch.tensor([-1.2, -1.7, -0.4, -0.8])
    actual, metrics = _run(
        monkeypatch, list(policy.unbind()), ref_values=list(reference.unbind()), pair_ids=pair_ids, dispatch=True
    )
    expected = -F.logsigmoid(0.2 * ((policy[0::2] - policy[1::2]) - (reference[0::2] - reference[1::2]))).mean()
    torch.testing.assert_close(actual, expected)
    expected_rewards = 0.2 * (policy.detach() - reference)
    torch.testing.assert_close(metrics["dpo/reward_chosen"], expected_rewards[0::2].sum())
    torch.testing.assert_close(metrics["dpo/reward_rejected"], expected_rewards[1::2].sum())
    torch.testing.assert_close(metrics["dpo/reward_margin"], (expected_rewards[0::2] - expected_rewards[1::2]).sum())
    actual.backward()
    actual_grad = policy.grad.clone()
    policy.grad = None
    expected.backward()
    torch.testing.assert_close(actual_grad, policy.grad)
    assert {
        "dpo/logps_chosen",
        "dpo/logps_rejected",
        "dpo/ref_logps_chosen",
        "dpo/ref_logps_rejected",
        "dpo/tie_rate",
        "dpo/tie_aware_accuracy",
    }.issubset(metrics)


def test_whole_pair_reordering_preserves_loss(monkeypatch):
    policy = [-1.0, -2.0, -0.5, -0.75]
    reference = [-1.2, -1.7, -0.4, -0.8]
    baseline, baseline_metrics = _run(monkeypatch, policy, ref_values=reference)
    reordered, reordered_metrics = _run(monkeypatch, policy, ref_values=reference, order=[2, 3, 0, 1])
    torch.testing.assert_close(reordered, baseline)
    for key in baseline_metrics:
        torch.testing.assert_close(reordered_metrics[key], baseline_metrics[key])


def test_tie_metrics_are_epsilon_aware(monkeypatch):
    _, metrics = _run(
        monkeypatch,
        [-1.0, -2.0, -0.5, -0.75],
        ref_values=[-1.0, -2.0, -0.5, -0.75],
    )
    assert metrics["dpo/strict_accuracy"].item() == 0
    assert metrics["dpo/tie_rate"].item() == 2
    assert metrics["dpo/tie_aware_accuracy"].item() == 1


def test_dpo_train_and_eval_accuracy_match_near_ties(monkeypatch):
    policy = torch.tensor([-1.0 + 2.5e-6, -1.0, -1.0 - 2.5e-6, -1.0], dtype=torch.float64)
    reference = torch.full_like(policy, -1.0)
    _, train_metrics = _run(monkeypatch, list(policy.unbind()), ref_values=list(reference.unbind()))
    rewards = 0.2 * (policy - reference)
    eval_metrics = finalize_pair_metrics(pair_metric_sums(rewards[0::2], rewards[1::2], torch.zeros(2)), prefix="dpo")

    for metric, expected in [("strict_accuracy", 0.5), ("tie_rate", 1.0), ("tie_aware_accuracy", 0.5)]:
        assert train_metrics[f"dpo/{metric}"].item() / 2 == expected
        assert eval_metrics[f"eval/dpo_{metric}"] == expected
    torch.testing.assert_close(train_metrics["dpo/pair_accuracy"], train_metrics["dpo/strict_accuracy"])


def test_reference_free_partition_and_num_samples_do_not_change_pair_sum(monkeypatch):
    policy = [-1.0, -2.0, -0.5, -0.75]
    first, metrics = _run(monkeypatch, policy, reference_free=True, num_samples=1)
    second, _ = _run(monkeypatch, policy, reference_free=True, num_samples=999)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(metrics["dpo/reward_chosen"], torch.tensor(-0.3))
    torch.testing.assert_close(metrics["dpo/reward_rejected"], torch.tensor(-0.55))
    pair_losses = -F.logsigmoid(0.2 * (torch.tensor(policy)[0::2] - torch.tensor(policy)[1::2]))
    torch.testing.assert_close(first, pair_losses[:1].sum() + pair_losses[1:].sum())


@pytest.mark.parametrize(
    ("pair_ids", "chosen", "match"),
    [
        ([1, 1, 2, 2], [True, True, True, False], "ordered chosen/rejected"),
        ([1, 2, 2, 1], [True, False, True, False], "same pair ID"),
        ([1, 1], [False, True], "ordered chosen/rejected"),
        ([1, 1, 2], [True, False, True], "even number"),
    ],
)
def test_production_dpo_rejects_invalid_pair_identity(monkeypatch, pair_ids, chosen, match):
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (None, {"log_probs": [torch.zeros(1) for _ in pair_ids]}),
    )
    batch = {
        "response_lengths": [1] * len(pair_ids),
        "unconcat_tokens": [torch.ones(1, dtype=torch.long)] * len(pair_ids),
        "total_lengths": [1] * len(pair_ids),
        "loss_masks": [torch.ones(1)] * len(pair_ids),
        "preference_branch_pair_ids": pair_ids,
        "preference_is_chosen": chosen,
        "ref_log_probs": [torch.zeros(1) for _ in pair_ids],
    }
    with pytest.raises(ValueError, match=match):
        loss_module.dpo_loss_function(_args(), batch, torch.ones(1), lambda value: value)


def test_production_dpo_rejects_empty_completion_mask(monkeypatch):
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (None, {"log_probs": [torch.zeros(1), torch.zeros(1)]}),
    )
    batch = {
        "response_lengths": [1, 1],
        "unconcat_tokens": [torch.ones(1, dtype=torch.long)] * 2,
        "total_lengths": [1, 1],
        "loss_masks": [torch.zeros(1), torch.ones(1)],
        "preference_branch_pair_ids": [1, 1],
        "preference_is_chosen": [True, False],
        "ref_log_probs": [torch.zeros(1), torch.zeros(1)],
    }
    with pytest.raises(ValueError, match="at least one supervised token"):
        loss_module.dpo_loss_function(_args(), batch, torch.ones(1), lambda value: value)
