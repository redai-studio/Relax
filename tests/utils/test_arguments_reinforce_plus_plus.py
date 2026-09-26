# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def arguments_module(monkeypatch):
    router_pkg = ModuleType("sglang_router")
    launch_router = ModuleType("sglang_router.launch_router")
    launch_router.RouterArgs = object
    monkeypatch.setitem(sys.modules, "sglang_router", router_pkg)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router)

    sglang_arguments = ModuleType("relax.backends.sglang.arguments")
    sglang_arguments.sglang_parse_args = lambda: None
    sglang_arguments.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, "relax.backends.sglang.arguments", sglang_arguments)

    device = ModuleType("relax.utils.device")
    device.get_dist_backend = lambda: "gloo"
    monkeypatch.setitem(sys.modules, "relax.utils.device", device)

    eval_config = ModuleType("relax.utils.training.eval_config")
    eval_config.EvalDatasetConfig = dict
    eval_config.build_eval_dataset_configs = lambda args, datasets_config, defaults: []
    eval_config.build_named_prompt_data_configs = lambda values: []
    eval_config.ensure_dataset_list = lambda values: values or []
    monkeypatch.setitem(sys.modules, "relax.utils.training.eval_config", eval_config)

    sys.modules.pop("relax.utils.arguments", None)
    module = importlib.import_module("relax.utils.arguments")
    yield module
    sys.modules.pop("relax.utils.arguments", None)


def _args(estimator: str, **overrides) -> SimpleNamespace:
    defaults = dict(
        advantage_estimator=estimator,
        normalize_advantages=True,
        fully_async=False,
        hybrid=False,
        colocate=True,
        context_parallel_size=1,
        calculate_per_token_loss=False,
        kl_coef=0.01 if estimator == "reinforce_plus_plus" else 0.0,
        kl_loss_coef=0.0 if estimator == "reinforce_plus_plus" else 0.01,
        kl_loss_type="k1" if estimator == "reinforce_plus_plus" else "k2",
        use_kl_loss=estimator == "reinforce_plus_plus_baseline",
        n_samples_per_prompt=8,
        rewards_normalization=True,
        use_unbiased_kl=False,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_valid_reinforce_plus_plus_contracts(arguments_module, estimator):
    arguments_module._validate_reinforce_plus_plus_args(_args(estimator), is_sft=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"normalize_advantages": False}, "requires --normalize-advantages"),
        ({"fully_async": True}, "synchronous colocate"),
        ({"hybrid": True}, "synchronous colocate"),
        ({"colocate": False}, "requires --colocate"),
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"calculate_per_token_loss": True}, "response-mean"),
    ],
)
@pytest.mark.parametrize("estimator", ["reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_common_contract_rejections(arguments_module, estimator, overrides, message):
    with pytest.raises(ValueError, match=message):
        arguments_module._validate_reinforce_plus_plus_args(_args(estimator, **overrides), is_sft=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kl_coef": 0.0}, "kl-coef > 0"),
        ({"kl_loss_type": "k2"}, "kl-loss-type k1"),
        ({"use_kl_loss": True, "kl_loss_coef": 0.01}, "does not support a separate"),
    ],
)
def test_reinforce_plus_plus_kl_contract(arguments_module, overrides, message):
    with pytest.raises(ValueError, match=message):
        arguments_module._validate_reinforce_plus_plus_args(_args("reinforce_plus_plus", **overrides), is_sft=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_samples_per_prompt": 1}, "n-samples-per-prompt > 1"),
        ({"custom_reward_post_process_path": "custom.py:process"}, "custom-reward-post-process-path"),
        ({"agentic_custom_advantage_path": "custom.py:advantage"}, "agentic-custom-advantage-path"),
        ({"rewards_normalization": False}, "group-mean centering"),
        ({"kl_coef": 0.01, "kl_loss_coef": 0.0}, "does not put token KL"),
        ({"use_kl_loss": False, "kl_loss_coef": 0.0}, "independent k2"),
        ({"kl_loss_type": "k1"}, "kl-loss-type k2"),
        ({"use_unbiased_kl": True}, "plain k2"),
    ],
)
def test_reinforce_plus_plus_baseline_contract(arguments_module, overrides, message):
    with pytest.raises(ValueError, match=message):
        arguments_module._validate_reinforce_plus_plus_args(
            _args("reinforce_plus_plus_baseline", **overrides), is_sft=False
        )


def test_other_estimators_are_unchanged(arguments_module):
    arguments_module._validate_reinforce_plus_plus_args(
        _args("grpo", normalize_advantages=False, colocate=False), is_sft=False
    )
