# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure numerical and batching tests for offline preference training."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.utils.training.preference_utils import (
    dpo_pair_loss,
    masked_sequence_sums,
    pack_preference_pair_indices,
    preference_accuracy,
    require_tensor_condition,
    reward_model_pair_loss,
    select_packed_sequence_scores,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_sequence_sums_preserves_values_dtype_and_gradients(dtype: torch.dtype):
    values = [
        torch.tensor([-100.0, -1.0, -2.0], dtype=dtype, requires_grad=True),
        torch.tensor([-7.0, -99.0], dtype=dtype, requires_grad=True),
    ]
    masks = [torch.tensor([False, True, True]), torch.tensor([True, False])]

    sums = masked_sequence_sums(values, masks, torch.device("cpu"))

    torch.testing.assert_close(sums, torch.tensor([-3.0, -7.0], dtype=dtype))
    sums.sum().backward()
    torch.testing.assert_close(values[0].grad, torch.tensor([0.0, 1.0, 1.0], dtype=dtype))
    torch.testing.assert_close(values[1].grad, torch.tensor([1.0, 0.0], dtype=dtype))


@pytest.mark.parametrize(
    ("masks", "match"),
    [
        ([], "branch aligned"),
        ([torch.ones(1)], "shape mismatch"),
        ([torch.zeros(2)], "at least one supervised token"),
    ],
)
def test_masked_sequence_sums_rejects_invalid_masks(masks: list[torch.Tensor], match: str):
    with pytest.raises(ValueError, match=match):
        masked_sequence_sums([torch.zeros(2)], masks, torch.device("cpu"))


@pytest.mark.parametrize(
    ("epsilon", "expected_ties", "expected_tie_aware"),
    [
        (1e-6, [False, True, True, True, True, True, False], [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]),
        (0.0, [False, False, False, True, False, False, False], [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]),
    ],
)
def test_preference_accuracy_distinguishes_strict_wins_and_ties(epsilon, expected_ties, expected_tie_aware):
    margins = torch.tensor([-2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6], dtype=torch.float64)

    strict, ties, tie_aware = preference_accuracy(margins, epsilon=epsilon)

    assert strict.tolist() == [False, False, False, False, True, True, True]
    assert ties.tolist() == expected_ties
    torch.testing.assert_close(tie_aware, torch.tensor(expected_tie_aware))


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


def test_reward_model_pair_loss_matches_independent_reference_and_gradient():
    chosen = torch.tensor([1.0, -1.0, 0.0], requires_grad=True)
    rejected = torch.tensor([0.0, 2.0, 0.0], requires_grad=True)

    actual = reward_model_pair_loss(chosen, rejected)
    expected = -F.logsigmoid(chosen - rejected)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward()
    actual_grad = chosen.grad.detach().clone()
    chosen.grad = None
    expected.sum().backward()
    assert torch.allclose(actual_grad, chosen.grad, rtol=1e-6, atol=1e-6)


def test_reward_model_pair_loss_rejects_non_finite_cpu_margin():
    with pytest.raises(ValueError, match="finite"):
        reward_model_pair_loss(torch.tensor([float("inf")]), torch.tensor([0.0]))


@pytest.mark.parametrize("shape", [(10,), (10, 1), (1, 10, 1)])
def test_select_packed_sequence_scores_accepts_scalar_head_shapes(shape):
    logits = torch.arange(10, dtype=torch.float32).reshape(shape)

    scores = select_packed_sequence_scores(logits, [3, 2, 4])

    torch.testing.assert_close(scores, torch.tensor([2.0, 4.0, 8.0]))


@pytest.mark.parametrize("shape", [(2, 4, 1), (4, 2)])
def test_select_packed_sequence_scores_rejects_non_scalar_head_shapes(shape):
    with pytest.raises(ValueError, match="logits must have shape"):
        select_packed_sequence_scores(torch.zeros(shape), [4])


@pytest.mark.parametrize("lengths", [[0], [-1], [3, 0]])
def test_select_packed_sequence_scores_rejects_non_positive_lengths(lengths):
    with pytest.raises(ValueError, match="non-positive"):
        select_packed_sequence_scores(torch.zeros(4), lengths)


def test_select_packed_sequence_scores_rejects_truncated_logits():
    with pytest.raises(ValueError, match="expected at least"):
        select_packed_sequence_scores(torch.zeros(4), [3, 2])


def test_pair_packer_is_deterministic_complete_and_capacity_safe():
    costs = [2, 4, 4, 5, 5]
    pair_ids = [0, 1, 2, 3, 4]

    bins = pack_preference_pair_indices(costs, pair_ids, capacity=10)

    assert sorted(index for group in bins for index in group) == list(range(len(costs)))
    assert all(sum(costs[index] for index in group) <= 10 for group in bins)
    assert bins == pack_preference_pair_indices(costs, pair_ids, capacity=10)


def test_pair_packer_keeps_oversize_pairs_in_separate_bins():
    bins = pack_preference_pair_indices([11, 2, 4, 4, 12], [0, 1, 2, 3, 4], capacity=10)

    assert bins == [[4], [0], [2, 3, 1]]
