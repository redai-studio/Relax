# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure numerical and batching tests for offline preference training."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.utils.training.preference_utils import (
    build_causal_lm_labels,
    dpo_pair_loss,
    pack_preference_pair_indices,
    require_tensor_condition,
)


def test_tensor_condition_uses_async_assert_without_python_bool_on_cuda(monkeypatch):
    class _CudaCondition:
        device = SimpleNamespace(type="cuda")

        def __bool__(self):
            raise AssertionError("CUDA conditions must not be converted to Python bool")

    calls = []
    monkeypatch.setattr(torch, "_assert_async", lambda condition, message: calls.append((condition, message)))
    condition = _CudaCondition()

    require_tensor_condition(condition, "finite")

    assert calls == [(condition, "finite")]


def test_build_causal_lm_labels_uses_next_token_mask():
    tokens = torch.tensor([10, 11, 12, 13, 14])
    raw_mask = torch.tensor([0, 0, 1, 1, 1])

    labels = build_causal_lm_labels(tokens, raw_mask)

    assert labels.tolist() == [-100, 12, 13, 14, -100]


@pytest.mark.parametrize("beta", [0.01, 0.1, 1.0])
def test_dpo_pair_loss_matches_independent_reference_and_gradient(beta: float):
    policy_chosen = torch.tensor([-2.0, -0.25, 4.0], dtype=torch.float32, requires_grad=True)
    policy_rejected = torch.tensor([-3.0, 0.75, -1.0], dtype=torch.float32, requires_grad=True)
    ref_chosen = torch.tensor([-2.5, 0.5, 1.0], dtype=torch.float32)
    ref_rejected = torch.tensor([-2.0, -0.5, -2.0], dtype=torch.float32)

    actual = dpo_pair_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen=ref_chosen,
        reference_rejected=ref_rejected,
        beta=beta,
    )
    expected = -F.logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)))
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward()
    actual_grad = policy_chosen.grad.detach().clone()
    policy_chosen.grad = None
    expected.sum().backward()
    assert torch.allclose(actual_grad, policy_chosen.grad, rtol=1e-6, atol=1e-6)


def test_dpo_pair_loss_reference_free_matches_independent_reference():
    chosen = torch.tensor([-1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.5, -2.0], requires_grad=True)

    actual = dpo_pair_loss(chosen, rejected, beta=0.1, reference_free=True)
    expected = -F.logsigmoid(0.1 * (chosen - rejected))

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_dpo_pair_loss_rejects_missing_reference_and_non_finite_values():
    finite = torch.tensor([0.0])
    with pytest.raises(ValueError, match="reference log-probabilities"):
        dpo_pair_loss(finite, finite)
    with pytest.raises(ValueError, match="finite"):
        dpo_pair_loss(torch.tensor([float("nan")]), finite, reference_free=True)


def test_pair_packer_is_deterministic_complete_and_capacity_safe():
    costs = [2, 4, 4, 5, 5]
    pair_ids = ["a", "b", "c", "d", "e"]

    bins = pack_preference_pair_indices(costs, pair_ids, capacity=10)

    assert sorted(index for group in bins for index in group) == list(range(len(costs)))
    assert all(sum(costs[index] for index in group) <= 10 for group in bins)
    assert bins == pack_preference_pair_indices(costs, pair_ids, capacity=10)


def test_pair_packer_reports_oversize_pair():
    with pytest.raises(ValueError, match="oversize.*pair-a.*11"):
        pack_preference_pair_indices([11], ["pair-a"], capacity=10)
