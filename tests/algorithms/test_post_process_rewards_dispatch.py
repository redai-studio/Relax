# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward entrypoint compatibility through the production registry
dispatcher."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


pytest.importorskip("torch")


@pytest.fixture()
def utils_mod(monkeypatch):
    """Load the real dispatcher without unrelated Ray, TensorDict, or HTTP
    imports."""
    tensordict = ModuleType("tensordict")
    tensordict.TensorDict = dict
    misc = ModuleType("relax.utils.misc")
    misc.load_function = Mock(side_effect=AssertionError("Custom callback loader must be configured by the test"))
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)
    monkeypatch.setitem(sys.modules, "relax.utils.misc", misc)

    # A separate module object keeps the production module cache untouched.
    path = Path(__file__).resolve().parents[2] / "relax/utils/utils.py"
    spec = importlib.util.spec_from_file_location("_reward_dispatch_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(estimator="grpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        n_samples_per_prompt=4,
        rewards_normalization=True,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        reward_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _Sample:
    def __init__(self, group_index, reward):
        self.group_index = group_index
        self.reward = reward

    def get_reward_value(self, args):
        return self.reward if not args.reward_key else self.reward[args.reward_key]


@pytest.mark.parametrize("estimator", ["grpo", "rloo"])
def test_returns_raw_and_normalized(utils_mod, estimator):
    args = _args(estimator)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]
    assert normalized != raw
    assert abs(sum(normalized)) < 1e-5


@pytest.mark.parametrize("estimator", ["ppo", "reinforce_plus_plus", "m2po"])
def test_identity_path_returns_raw_twice(utils_mod, estimator):
    args = _args(estimator)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_rewards_normalization_off_returns_raw_twice(utils_mod):
    args = _args("grpo", rewards_normalization=False)
    samples = [_Sample(0, r) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, normalized = utils_mod.post_process_rewards(args, samples)
    assert normalized is raw


def test_custom_path_still_short_circuits(utils_mod, monkeypatch):
    sentinel = (["raw"], ["norm"])
    monkeypatch.setattr(utils_mod, "load_function", lambda path: lambda a, s: sentinel)
    args = _args("grpo", custom_reward_post_process_path="pkg.mod.fn")
    assert utils_mod.post_process_rewards(args, []) is sentinel


def test_custom_path_may_return_only_processed_rewards(utils_mod, monkeypatch):
    processed = [10.0, 20.0]
    monkeypatch.setattr(utils_mod, "load_function", lambda path: lambda a, s: processed)
    args = _args("grpo", custom_reward_post_process_path="pkg.mod.fn")
    samples = [_Sample(0, 1.0), _Sample(0, 2.0)]
    raw, actual = utils_mod.post_process_rewards(args, samples)
    assert raw == [1.0, 2.0]
    assert actual is processed


def test_reward_key_selects_from_dict(utils_mod):
    args = _args("grpo", reward_key="score")
    samples = [_Sample(0, {"score": r, "other": 99.0}) for r in (0.0, 1.0, 2.0, 3.0)]
    raw, _ = utils_mod.post_process_rewards(args, samples)
    assert raw == [0.0, 1.0, 2.0, 3.0]


def test_unknown_estimator_raises_from_the_registry(utils_mod):
    args = _args("not_an_algorithm")
    samples = [_Sample(0, 1.0) for _ in range(4)]
    with pytest.raises(KeyError, match="Unknown advantage estimator"):
        utils_mod.post_process_rewards(args, samples)
