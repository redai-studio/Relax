# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def post_process_rewards(monkeypatch):
    tensordict = ModuleType("tensordict")
    tensordict.TensorDict = dict
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)
    sys.modules.pop("relax.utils.utils", None)
    module = importlib.import_module("relax.utils.utils")
    yield module.post_process_rewards
    sys.modules.pop("relax.utils.utils", None)


class _Sample:
    def __init__(self, reward: float, group_index: int | None):
        self.reward = reward
        self.group_index = group_index
        self.custom_advantage = None

    def get_reward_value(self, _args):
        return self.reward


def _args(**overrides):
    defaults = dict(
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        advantage_estimator="reinforce_plus_plus_baseline",
        rewards_normalization=True,
        n_samples_per_prompt=2,
        grpo_std_normalization=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_baseline_subtracts_inclusive_group_mean_without_group_std(post_process_rewards):
    samples = [_Sample(1.0, 1), _Sample(2.0, 0), _Sample(3.0, 1), _Sample(6.0, 0)]

    centered = post_process_rewards(_args(), samples)

    assert centered == [-1.0, -2.0, 1.0, 2.0]


def test_baseline_all_zero_group_is_finite_zero(post_process_rewards):
    centered = post_process_rewards(_args(), [_Sample(0.0, 0), _Sample(0.0, 0)])

    assert centered == [0.0, 0.0]


def test_baseline_requires_complete_groups(post_process_rewards):
    with pytest.raises(ValueError, match="has 1 samples, expected 2"):
        post_process_rewards(_args(), [_Sample(1.0, 0)])


def test_baseline_requires_group_index(post_process_rewards):
    with pytest.raises(ValueError, match="group_index is required"):
        post_process_rewards(_args(), [_Sample(1.0, None), _Sample(2.0, None)])


def test_agentic_custom_advantage_does_not_require_raw_reward(post_process_rewards):
    sample = _Sample(None, 0)
    sample.custom_advantage = 0.5

    rewards = post_process_rewards(_args(agentic_custom_advantage_path="custom.py:advantage"), [sample])

    assert rewards == [0.5]


def test_agentic_converter_does_not_transfer_raw_reward(post_process_rewards):
    from relax.utils.types import Sample

    convert_samples_to_train_data = sys.modules["relax.utils.utils"].convert_samples_to_train_data
    sample = Sample(tokens=[1, 2], response_length=1, index=0, custom_advantage=0.5)
    args = _args(
        agentic_custom_advantage_path="custom.py:advantage",
        debug_train_only=True,
        multimodal_keys=None,
        use_opd=False,
    )

    train_data = convert_samples_to_train_data(args, [sample])

    assert train_data["rewards"] == [0.5]
    assert "raw_reward" not in train_data
