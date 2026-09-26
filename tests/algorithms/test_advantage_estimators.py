# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Behavior preserved when extracting the shared advantage estimators."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms import advantages as advantages_module  # noqa: E402
from relax.algorithms.advantages import compute_advantages_and_returns  # noqa: E402
from relax.utils.training import ppo_utils  # noqa: E402


def _args(estimator, **overrides):
    base = dict(advantage_estimator=estimator, kl_coef=0.0, gamma=1.0, lambd=1.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def _inputs(lengths=(3, 2)):
    return dict(
        kl=[torch.zeros(n, dtype=torch.float32) for n in lengths],
        loss_masks=[torch.ones(n, dtype=torch.float32) for n in lengths],
        response_lengths=list(lengths),
        total_lengths=[n + 2 for n in lengths],
        values=None,
    )


@pytest.fixture
def cp_disabled(monkeypatch):
    """Run GAE and REINFORCE++ on CPU without importing Megatron."""
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "sapo", "cispo", "m2po", "rloo"])
def test_grpo_family_broadcasts_rewards_and_preserves_return_list(estimator):
    advantages, returns = compute_advantages_and_returns(_args(estimator), rewards=[1.5, -2.0], **_inputs())
    expected = [torch.full((3,), 1.5), torch.full((2,), -2.0)]
    for actual, target in zip(advantages, expected, strict=True):
        assert torch.equal(actual, target)
    for actual, target in zip(returns, expected, strict=True):
        assert torch.equal(actual, target)
    assert advantages is not returns
    advantages[0] = torch.zeros(3)
    assert torch.equal(returns[0], expected[0])


@pytest.mark.parametrize("estimator", ["grpo", "m2po", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_tensor_rewards_preserve_mains_detached_advantages(cp_disabled, estimator):
    rewards = torch.tensor([1.5, -2.0], requires_grad=True)
    advantages, returns = compute_advantages_and_returns(_args(estimator), rewards=rewards, **_inputs())
    expected, _ = compute_advantages_and_returns(_args(estimator), rewards=[1.5, -2.0], **_inputs())
    for actual, target in zip(advantages, expected, strict=True):
        assert torch.equal(actual, target)
        assert not actual.requires_grad
    assert all(not value.requires_grad for value in returns)


def test_reward_tensor_conversion_preserves_mains_independent_storage():
    rewards = torch.tensor([1.5, -2.0])
    converted = advantages_module._as_reward_tensor(rewards, _inputs()["kl"])
    converted[0] = 99.0
    assert rewards[0] == 1.5


def test_reinforce_plus_plus_baseline_preserves_mask_kl_and_alias_semantics():
    inputs = _inputs(lengths=(3,))
    inputs["kl"] = [torch.tensor([2.0, 4.0, 6.0])]
    inputs["loss_masks"] = [torch.tensor([1.0, 0.0, 1.0])]
    advantages, returns = compute_advantages_and_returns(
        _args("reinforce_plus_plus_baseline", kl_coef=0.5), rewards=[3.0], **inputs
    )
    # The adapter must not introduce reward-side KL; startup validation owns
    # rejecting nonzero kl_coef for this estimator.
    assert torch.equal(advantages[0], torch.tensor([3.0, 0.0, 3.0]))
    assert returns is advantages


def test_reinforce_plus_plus_adapter_passes_mains_arguments(cp_disabled):
    rewards = [1.5, -2.0]
    inputs = _inputs()
    inputs["kl"] = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
    inputs["loss_masks"][0] = torch.tensor([1.0, 0.0, 1.0])
    args = _args("reinforce_plus_plus", kl_coef=0.3, gamma=0.95)
    advantages, returns = compute_advantages_and_returns(args, rewards=rewards, **inputs)
    expected = ppo_utils.get_reinforce_plus_plus_returns(
        rewards=torch.tensor(rewards),
        kl=inputs["kl"],
        loss_masks=inputs["loss_masks"],
        response_lengths=inputs["response_lengths"],
        total_lengths=inputs["total_lengths"],
        kl_coef=0.3,
        gamma=0.95,
    )
    for actual, target in zip(advantages, expected, strict=True):
        assert torch.equal(actual, target)
    for actual, target in zip(returns, expected, strict=True):
        assert torch.equal(actual, target)
    assert returns is not advantages


def _gae_inputs():
    """Fresh tensors because the original PPO branch mutates KL in place."""
    return dict(
        rewards=[1.5, -2.0],
        kl=[torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])],
        values=[torch.tensor([0.5, 0.25, 0.125]), torch.tensor([1.0, 2.0])],
        response_lengths=[3, 2],
        total_lengths=[3, 2],
    )


def _main_gae(kl_coef, gamma, lambd):
    """Frozen PPO reward shaping from main@5cec8ca1, followed by its kernel."""
    inputs = _gae_inputs()
    shaped_rewards = []
    for reward, kl in zip(inputs["rewards"], inputs["kl"], strict=True):
        kl *= -kl_coef
        kl[-1] += reward  # The fixture fixes context-parallel rank to zero.
        shaped_rewards.append(kl)
    return ppo_utils.get_advantages_and_returns_batch(
        inputs["total_lengths"],
        inputs["response_lengths"],
        inputs["values"],
        shaped_rewards,
        gamma,
        lambd,
        padded_total_lengths=None,
    )


@pytest.mark.parametrize("kl_coef,gamma,lambd", [(0.0, 1.0, 1.0), (0.05, 0.99, 0.95)])
def test_gae_adapter_matches_mains_numbers(cp_disabled, kl_coef, gamma, lambd):
    actual = compute_advantages_and_returns(_args("ppo", kl_coef=kl_coef, gamma=gamma, lambd=lambd), **_gae_inputs())
    expected = _main_gae(kl_coef, gamma, lambd)
    for actual_tensors, expected_tensors in zip(actual, expected, strict=True):
        for value, target in zip(actual_tensors, expected_tensors, strict=True):
            torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("padded_total_lengths", [None, [8, 8]])
def test_gae_adapter_forwards_shaped_rewards_and_padding(cp_disabled, monkeypatch, padded_total_lengths):
    """Check the two callers' different padding arguments, not CP slicing."""
    seen = {}

    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return [torch.zeros(3), torch.zeros(2)], [torch.zeros(3), torch.zeros(2)]

    monkeypatch.setattr(advantages_module, "get_advantages_and_returns_batch", spy)
    inputs = _gae_inputs()
    compute_advantages_and_returns(
        _args("ppo", kl_coef=0.5, gamma=0.9, lambd=0.8),
        **inputs,
        padded_total_lengths=padded_total_lengths,
    )
    assert seen["kwargs"]["padded_total_lengths"] == padded_total_lengths
    assert seen["args"][:2] == ([3, 2], [3, 2])
    assert seen["args"][2] is inputs["values"]
    assert seen["args"][4:] == (0.9, 0.8)
    expected_rewards = [torch.tensor([-0.05, -0.1, 1.35]), torch.tensor([-0.2, -2.25])]
    for actual, expected in zip(seen["args"][3], expected_rewards, strict=True):
        torch.testing.assert_close(actual, expected)
