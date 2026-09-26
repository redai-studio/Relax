# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Production DPO loss regression tests."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import loss as loss_module
except Exception as exc:
    pytest.skip(f"relax.backends.megatron unavailable: {exc}", allow_module_level=True)


def _args(*, reference_free: bool = False, beta: float = 0.2) -> Namespace:
    return Namespace(dpo_reference_free=reference_free, dpo_beta=beta)


def _run(monkeypatch, policy_values, *, order=None, reference_free=False, ref_values=None, num_samples=2):
    if order is None:
        order = [0, 1, 2, 3]
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
    return loss_module.dpo_loss_function(_args(reference_free=reference_free), batch, logits, lambda value: value)


def test_production_dpo_loss_matches_independent_reference_and_gradients(monkeypatch):
    policy = torch.tensor([-1.0, -2.0, -0.5, -0.75], requires_grad=True)
    reference = torch.tensor([-1.2, -1.7, -0.4, -0.8])
    actual, metrics = _run(monkeypatch, list(policy.unbind()), ref_values=list(reference.unbind()))
    expected = -F.logsigmoid(0.2 * ((policy[0::2] - policy[1::2]) - (reference[0::2] - reference[1::2]))).sum()
    torch.testing.assert_close(actual, expected)
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


def test_pair_identity_restores_reordered_micro_batch(monkeypatch):
    policy = [-1.0, -2.0, -0.5, -0.75]
    reference = [-1.2, -1.7, -0.4, -0.8]
    baseline, baseline_metrics = _run(monkeypatch, policy, ref_values=reference)
    reordered, reordered_metrics = _run(monkeypatch, policy, ref_values=reference, order=[3, 0, 2, 1])
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


def test_reference_free_partition_and_num_samples_do_not_change_pair_sum(monkeypatch):
    policy = [-1.0, -2.0, -0.5, -0.75]
    first, _ = _run(monkeypatch, policy, reference_free=True, num_samples=1)
    second, _ = _run(monkeypatch, policy, reference_free=True, num_samples=999)
    torch.testing.assert_close(first, second)
    pair_losses = -F.logsigmoid(0.2 * (torch.tensor(policy)[0::2] - torch.tensor(policy)[1::2]))
    torch.testing.assert_close(first, pair_losses[:1].sum() + pair_losses[1:].sum())


@pytest.mark.parametrize(
    ("pair_ids", "chosen", "match"),
    [
        ([1, 1, 2, 2], [True, True, True, False], "duplicate chosen"),
        ([1, 2, 2, 3], [True, True, False, False], "exactly one"),
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
