# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Argument parsing and validation must read the registry, not string lists."""

import argparse
import importlib.util
import pathlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest


ARGS_PATH = pathlib.Path(__file__).resolve().parents[2] / "relax" / "utils" / "arguments.py"


@pytest.fixture()
def arguments_module(monkeypatch):
    """Load real argument parsing while isolating optional training imports."""
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

    spec = importlib.util.spec_from_file_location("_algorithm_test_arguments", ARGS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(estimator="grpo", **overrides):
    base = dict(
        advantage_estimator=estimator,
        normalize_advantages=False,
        rewards_normalization=True,
        custom_reward_post_process_path=None,
        n_samples_per_prompt=4,
        reward_key="score",
        use_critic=False,
        fully_async=False,
        hybrid=False,
        dynamic_sampling_filter_path=None,
        m2po_kl2_budget=0.01,
        m2po_miniclip_low=0.3,
        m2po_miniclip_high=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------- behaviour ----------------


def test_parser_rejects_an_unregistered_estimator(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--advantage-estimator", "not_an_algorithm"])


def test_parser_accepts_newly_registered_algorithms(arguments_module, monkeypatch):
    from relax.algorithms import ALGORITHM_SPECS, list_algorithm_names

    name = "new_algorithm"
    monkeypatch.setitem(ALGORITHM_SPECS, name, replace(ALGORITHM_SPECS["grpo"], name=name))
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)

    for name in list_algorithm_names():
        parsed = parser.parse_args(["--advantage-estimator", name])
        assert parsed.advantage_estimator == name


# ---------------- --custom-config-path override timing ----------------


def _write_yaml(tmp_path, body):
    path = tmp_path / "override.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _overridable_args(tmp_path, body, **overrides):
    """Args as they look when the YAML merge runs: already validated once."""
    base = _args("grpo", reward_key=None)
    base.loss_type = "policy_loss"
    base.custom_config_path = _write_yaml(tmp_path, body)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_yaml_cannot_switch_to_an_estimator_that_needs_a_critic(arguments_module, tmp_path):
    """Role composition and the offload flags were derived from the pre-
    override `use_critic`, so accepting a YAML switch to PPO would leave the
    run neither fully critic nor fully critic-free."""
    args = _overridable_args(tmp_path, "advantage_estimator: ppo\n")

    with pytest.raises(ValueError, match="critic setup"):
        arguments_module.apply_custom_config_overrides(args)


# The three validators below were split out of `validate_algorithm_args`
# because the main path has a derivation order, and only two of the four were
# wired back into the override path -- so a YAML file could select rloo and
# then move any value the missing three read. Each case here fails on the
# pre-fix code.


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param("advantage_estimator: rloo\nkl_coef: 0.01\n", "nonzero --kl-coef", id="reward-side-kl"),
        pytest.param(
            "advantage_estimator: rloo\nnum_steps_per_rollout: 4\n",
            "num-steps-per-rollout 1",
            id="update-schedule",
        ),
        pytest.param(
            "advantage_estimator: rloo\nglobal_batch_size: 999\n",
            "one optimizer update per rollout",
            id="batch-shape",
        ),
    ],
)
def test_yaml_cannot_bypass_the_late_running_validators(arguments_module, tmp_path, body, expected):
    args = _overridable_args(
        tmp_path,
        body,
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=128,
        num_steps_per_rollout=None,
        kl_coef=0.0,
        max_staleness=0,
        calculate_per_token_loss=True,
        rewards_normalization=True,
        normalize_advantages=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        hybrid=False,
        fully_async=False,
    )

    with pytest.raises(ValueError, match=expected):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_that_changes_the_update_schedule_gets_a_rederived_batch(arguments_module, tmp_path):
    """Re-running the validator without its derivation rejected a legal config.

    `validate_batch_shape` reads `global_batch_size`, which the main path
    derives from `num_steps_per_rollout` *before* the merge. A YAML switching
    grpo@4-steps to rloo@1-step should get `rollout * n = 128`; the first
    version of this fix compared against the stale 32 and refused it.
    """
    args = _overridable_args(
        tmp_path,
        "advantage_estimator: rloo\nnum_steps_per_rollout: 1\n",
        n_samples_per_prompt=8,
        rollout_batch_size=16,
        global_batch_size=32,
        num_steps_per_rollout=4,
        kl_coef=0.0,
        max_staleness=0,
        calculate_per_token_loss=True,
        rewards_normalization=True,
        normalize_advantages=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        hybrid=False,
        fully_async=False,
    )

    arguments_module.apply_custom_config_overrides(args)

    assert args.global_batch_size == 128, "the merge should re-derive it, not keep the pre-merge value"


def test_yaml_without_algorithm_changes_is_accepted(arguments_module, tmp_path):
    args = _overridable_args(tmp_path, "lr: 0.5\n")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5
    assert args.advantage_estimator == "grpo"


@pytest.mark.parametrize(
    ("initial_loss_type", "yaml_loss_type"),
    [
        pytest.param("policy_loss", "sft", id="rl-to-sft"),
        pytest.param("sft", "policy_loss", id="sft-to-rl"),
    ],
)
def test_yaml_cannot_change_the_training_mode(arguments_module, tmp_path, initial_loss_type, yaml_loss_type):
    args = _overridable_args(
        tmp_path,
        f"loss_type: {yaml_loss_type}\n",
        loss_type=initial_loss_type,
        advantage_estimator="ppo",
    )

    with pytest.raises(ValueError, match="cannot change loss_type"):
        arguments_module.apply_custom_config_overrides(args)


def test_no_yaml_is_a_no_op(arguments_module):
    args = _args("grpo", reward_key=None)
    args.loss_type = "policy_loss"
    args.custom_config_path = None
    arguments_module.apply_custom_config_overrides(args)


def test_sft_runs_skip_the_algorithm_recheck(arguments_module, tmp_path):
    """SFT never selects an estimator, so a stale one must not block it."""
    args = _overridable_args(tmp_path, "lr: 0.5\n", loss_type="sft", advantage_estimator="ppo")
    arguments_module.apply_custom_config_overrides(args)
    assert args.lr == 0.5


@pytest.mark.parametrize("field", ["reward_normalizer", "advantage_fn", "policy_loss_fn"])
def test_spec_with_an_unregistered_implementation_is_rejected_at_startup(arguments_module, monkeypatch, field):
    """Reject an invalid implementation before starting workers."""
    from relax.algorithms.spec import ALGORITHM_SPECS

    broken = replace(ALGORITHM_SPECS["grpo"], **{field: "typo_does_not_exist"})
    monkeypatch.setitem(ALGORITHM_SPECS, "grpo", broken)

    with pytest.raises(ValueError, match="typo_does_not_exist"):
        arguments_module.validate_algorithm_args(_args("grpo", reward_key=None))


# ---------------- context parallel capabilities ----------------


@pytest.mark.parametrize("estimator", ["m2po", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_cp_limited_algorithms_accept_unsharded_responses(arguments_module, estimator):
    arguments_module.validate_algorithm_args(
        _args(
            estimator,
            context_parallel_size=1,
            dynamic_context_parallel=False,
            normalize_advantages=True,
            reward_key=None,
        )
    )


@pytest.mark.parametrize("estimator", ["m2po", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize(("cp_size", "dynamic_cp"), [(2, False), (1, True)])
def test_cp_limited_algorithms_reject_static_and_dynamic_sharding(arguments_module, estimator, cp_size, dynamic_cp):
    with pytest.raises(ValueError, match="context-parallel-size 1.*dynamic-context-parallel disabled"):
        arguments_module.validate_algorithm_args(
            _args(
                estimator,
                context_parallel_size=cp_size,
                dynamic_context_parallel=dynamic_cp,
                normalize_advantages=True,
                reward_key=None,
            )
        )


@pytest.mark.parametrize(("cp_size", "dynamic_cp"), [(2, False), (1, True)])
def test_cp_capable_algorithm_keeps_parallel_configurations(arguments_module, cp_size, dynamic_cp):
    arguments_module.validate_algorithm_args(
        _args("grpo", context_parallel_size=cp_size, dynamic_context_parallel=dynamic_cp)
    )


@pytest.mark.parametrize(
    ("body", "estimator", "cp_size"),
    [
        ("context_parallel_size: 2\n", "m2po", 1),
        ("dynamic_context_parallel: true\n", "m2po", 1),
        ("advantage_estimator: m2po\n", "grpo", 2),
    ],
)
def test_yaml_cannot_bypass_m2po_context_parallel_limit(arguments_module, tmp_path, body, estimator, cp_size):
    args = _overridable_args(tmp_path, body, advantage_estimator=estimator, context_parallel_size=cp_size)
    with pytest.raises(ValueError, match="context-parallel-size 1.*dynamic-context-parallel disabled"):
        arguments_module.apply_custom_config_overrides(args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"m2po_kl2_budget": 0.0},
        {"m2po_miniclip_low": 1.2},
        {"m2po_miniclip_high": 0.0},
    ],
)
def test_m2po_registration_preserves_mains_numeric_config_validation(arguments_module, overrides):
    """Retain previously accepted parameters without adding algorithm
    constraints."""
    arguments_module.validate_algorithm_args(_args("m2po", reward_key=None, **overrides))


# ---------------- a YAML global_batch_size must not be derived over ----------------


def _batch_args(tmp_path, body, **overrides):
    """Args shaped for the batch-size derivation, already validated once."""
    base = _overridable_args(tmp_path, body, **overrides)
    base.rollout_batch_size = 32
    base.n_samples_per_prompt = 4
    base.num_steps_per_rollout = 4
    base.global_batch_size = 32  # 32 * 4 // 4, i.e. what the pre-merge derivation wrote
    base.micro_batch_size = 1
    base.use_dynamic_batch_size = False
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_yaml_global_batch_size_conflicting_with_the_derivation_is_refused(arguments_module, tmp_path):
    """The override used to be written and then silently replaced.

    `apply_custom_config_overrides` merges the YAML, then re-derives
    `global_batch_size` from `num_steps_per_rollout`. With the derivation
    unconditional, a YAML that names `global_batch_size` had its value assigned
    by the merge loop and overwritten one statement later -- so the run used
    neither the configured number nor an error, which is the single outcome the
    "YAML key overrides the argument" contract does not allow.
    """
    args = _batch_args(tmp_path, "global_batch_size: 999\n")

    with pytest.raises(ValueError, match="sets global_batch_size to 999"):
        arguments_module.apply_custom_config_overrides(args)


def test_yaml_global_batch_size_agreeing_with_the_derivation_survives(arguments_module, tmp_path):
    """Naming the value the derivation would reach anyway is not a conflict."""
    args = _batch_args(tmp_path, "global_batch_size: 32\n")

    arguments_module.apply_custom_config_overrides(args)

    assert args.global_batch_size == 32


def test_new_algorithm_validation_uses_declared_capabilities(arguments_module, monkeypatch):
    from relax.algorithms import ALGORITHM_SPECS

    name = "new_algorithm"
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        name,
        replace(ALGORITHM_SPECS["grpo"], name=name, needs_critic=True, requires_normalize_advantages=True),
    )
    args = _args(name)
    with pytest.raises(ValueError, match="requires advantage normalization"):
        arguments_module.validate_algorithm_args(args)
    args.normalize_advantages = True
    arguments_module.validate_algorithm_args(args)
    assert args.use_critic


def test_training_argument_entrypoint_revalidates_yaml_algorithm(arguments_module, tmp_path):
    from tests.utils.test_arguments_opd_teacher_colocate import _opd_args

    args = _opd_args()
    args.custom_config_path = _write_yaml(tmp_path, "advantage_estimator: m2po\ncontext_parallel_size: 2\n")
    with pytest.raises(ValueError, match="context-parallel-size 1"):
        arguments_module.slime_validate_args(args)
