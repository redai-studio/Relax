# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Contract tests for the declaration-driven test args factory."""

import copy

import pytest

from tests.helpers.relax_args import apply_safe_derivations, make_args, make_raw_args


def test_defaults_come_from_parser_registration():
    # Spot-check defaults owned by Relax's own argparse registration. If a
    # default changes at the declaration site, the factory changes with it.
    args = make_args()
    assert args.task_type == "causal_lm"
    assert args.advantage_estimator == "grpo"
    assert args.fully_async is False
    assert args.hybrid is False
    assert args.use_opd is False
    assert args.train_backend == "megatron"
    assert args.debug_train_only is False


def test_factory_returns_independent_namespaces():
    first = make_args()
    second = make_args()
    assert first is not second
    first.task_type = "seq_cls"
    first.lora_target_modules.append("custom")
    assert second.task_type == "causal_lm"
    assert second.lora_target_modules == ["linear_qkv", "linear_proj"]
    assert first.lora_target_modules is not second.lora_target_modules


def test_factory_does_not_read_pytest_argv():
    # A second collection must not be influenced by whatever argv pytest used.
    from tests.helpers import relax_args

    relax_args.reset_defaults_cache()
    args = make_args()
    assert args.advantage_estimator == "grpo"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("eval_size", None),
        ("use_health_check", False),
        ("n_samples_per_prompt", 0),
        ("notify_urls", []),
    ],
)
def test_explicit_empty_overrides_are_preserved(key, value):
    args = make_args(**{key: value})
    assert getattr(args, key) == value


def test_overrides_apply_after_derivations():
    args = make_args(use_critic=True, model_source=object())
    assert args.use_critic is True
    assert args.model_source is not None


def test_use_critic_follows_algorithm_spec():
    assert make_args().use_critic is False
    assert make_args(advantage_estimator="ppo").use_critic is True
    assert make_args(advantage_estimator="grpo").use_critic is False


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_offline_objectives_force_no_critic_even_for_ppo(loss_type):
    # Mirrors slime_validate_args: offline training never spawns a critic,
    # regardless of the algorithm spec (see relax/utils/arguments.py,
    # ``args.use_critic = False`` under ``is_offline``).
    args = make_args(loss_type=loss_type, advantage_estimator="ppo")
    assert args.use_critic is False


def test_explicit_use_critic_override_still_wins():
    assert make_args(loss_type="sft", use_critic=True).use_critic is True
    assert make_args(advantage_estimator="ppo", use_critic=False).use_critic is False


def test_safe_derivations_match_production_assignments():
    raw = make_raw_args()
    apply_safe_derivations(raw)
    assert raw.model_source is None
    assert raw._start_rollout_id_explicit is False
    assert raw.use_critic is False
    assert raw.lora_target_modules == ["linear_qkv", "linear_proj"]

    fsdp = make_raw_args(train_backend="fsdp")
    apply_safe_derivations(fsdp)
    assert fsdp.lora_target_modules is None

    explicit = make_raw_args(start_rollout_id=3)
    apply_safe_derivations(explicit)
    assert explicit._start_rollout_id_explicit is True


def test_raw_args_have_no_derived_fields():
    raw = make_raw_args()
    assert not hasattr(raw, "model_source")
    assert not hasattr(raw, "use_critic")
    assert not hasattr(raw, "_start_rollout_id_explicit")


def test_deepcopy_of_defaults_is_safe_after_cache_reset():
    # Resetting the cache must not corrupt previously returned namespaces.
    args = make_args()
    snapshot = copy.deepcopy(vars(args))
    from tests.helpers import relax_args

    relax_args.reset_defaults_cache()
    assert snapshot["task_type"] == "causal_lm"
