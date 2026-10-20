# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GAGAR reward conversion through Relax's production group normalizer."""

from types import SimpleNamespace

import pytest

from examples.gagar.gagar import compute_group_rewards
from relax.algorithms.rewards import normalize_group_mean_std


def _advantages(rewards):
    args = SimpleNamespace(n_samples_per_prompt=len(rewards), grpo_std_normalization=False)
    samples = [SimpleNamespace(group_index=0) for _ in rewards]
    return normalize_group_mean_std(args, samples, rewards)


def test_uncapped_redistribution_preserves_positive_credit_and_failed_advantage():
    rewards = [1.0, 1.0, 1.0, 0.0]
    weights = [1.0, 0.8, 0.6, 0.0]
    result = compute_group_rewards(rewards, weights)

    assert result == pytest.approx([1.0625, 1.0, 0.9375, 0.0])
    assert _advantages(result) == pytest.approx([0.3125, 0.25, 0.1875, -0.75])
    assert rewards == [1.0, 1.0, 1.0, 0.0]
    assert weights == [1.0, 0.8, 0.6, 0.0]


def test_cap_is_applied_before_final_group_centering():
    result = compute_group_rewards([1, 1, 1, 0], [1, 0.5, 0.5, 0], cap=1.2)

    assert result == pytest.approx([1.05, 0.9, 0.9, 0])
    assert _advantages(result) == pytest.approx([0.3375, 0.1875, 0.1875, -0.7125])


def test_low_quality_group_does_not_regain_all_positive_credit():
    result = compute_group_rewards([1, 1, 1, 0], [0.2, 0.2, 0.2, 0])

    assert _advantages(result) == pytest.approx([0.20625, 0.20625, 0.20625, -0.61875])


@pytest.mark.parametrize("rewards", [[1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_unit_weights_preserve_original_rewards(rewards):
    assert compute_group_rewards(rewards, [1] * len(rewards)) == rewards


def test_candidate_order_and_failing_weights_do_not_change_assignment():
    result = compute_group_rewards([0, 1, 1, 1], [-1, 0.6, 1, 0.8])

    assert result == pytest.approx([0, 0.9375, 1.0625, 1])


@pytest.mark.parametrize(
    "rewards, weights, cap",
    [
        ([], [], 1.5),
        ([1, 0], [1], 1.5),
        ([0.5, 0], [1, 1], 1.5),
        ([1, 0], [-0.1, 0], 1.5),
        ([1, 0], [1.1, 0], 1.5),
        ([1, 0], [float("nan"), 0], 1.5),
        ([1, 0], [float("inf"), 0], 1.5),
        ([1, 0], [float("-inf"), 0], 1.5),
    ],
)
def test_invalid_inputs_are_rejected(rewards, weights, cap):
    with pytest.raises(ValueError):
        compute_group_rewards(rewards, weights, cap)


def test_cap_below_one_is_supported():
    result = compute_group_rewards([1, 0], [1, 0], cap=0.5)

    assert _advantages(result) == pytest.approx([0.375, -0.375])


def test_zero_weight_removes_credit_before_final_centering():
    result = compute_group_rewards([1, 1, 1, 0], [0, 1, 1, 0])

    assert _advantages(result) == pytest.approx([0, 0.375, 0.375, -0.75])


def test_all_zero_weights_skip_rescaling_but_keep_failed_advantage_before_centering():
    result = compute_group_rewards([1, 1, 1, 0], [0, 0, 0, 0])

    assert result == pytest.approx([0.75, 0.75, 0.75, 0])
    assert _advantages(result) == pytest.approx([0.1875, 0.1875, 0.1875, -0.5625])


def test_confirmed_hack_is_a_failed_reward_before_group_statistics():
    result = compute_group_rewards([1, 1, 0, 0], [1, 1, 0, 0])

    assert _advantages(result) == pytest.approx([0.5, 0.5, -0.5, -0.5])
