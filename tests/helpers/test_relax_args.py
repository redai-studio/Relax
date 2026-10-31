# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from tests.helpers.relax_args import make_args, slime_arg_defaults


def test_make_args_uses_declared_task_type_default():
    defaults = slime_arg_defaults()
    assert defaults["task_type"] == "causal_lm"
    assert make_args().task_type == "causal_lm"


def test_make_args_applies_argparse_type_converters():
    # type=json.loads + default="{}" must yield a dict, matching real parse_args.
    args = make_args()
    assert args.train_env_vars == {}
    assert args.apply_chat_template_kwargs == {}


def test_make_args_override_task_type():
    assert make_args(task_type="seq_cls").task_type == "seq_cls"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("fully_async", False),
        ("n_samples_per_prompt", 1),
        ("n_samples_per_prompt", 0),
        ("num_labels", None),
        ("prompt_data", None),
        ("prompt_data", []),
        ("eval_prompt_data", None),
        ("eval_prompt_data", []),
        ("kl_coef", 0.0),
        ("use_opd", False),
        ("lora_rank", 0),
    ],
)
def test_make_args_preserves_explicit_boundary_overrides(name, value):
    args = make_args(**{name: value})
    assert getattr(args, name) is value or getattr(args, name) == value


def test_make_args_isolates_mutable_defaults():
    # Mutate declared list defaults (not fresh override lists) so a shallow
    # cache copy would fail this test the way production fixtures would.
    a = make_args()
    b = make_args()
    assert a.agent_env == []
    assert a.profile_target == ["train_overall"]
    a.agent_env.append("FOO=bar")
    a.profile_target.append("train_actor")
    assert b.agent_env == []
    assert b.profile_target == ["train_overall"]
    cached = slime_arg_defaults()
    assert cached["agent_env"] == []
    assert cached["profile_target"] == ["train_overall"]


def test_make_args_runtime_fields_match_parse_init():
    args = make_args()
    # Same writer as ``_parse_args_impl``: runtime-only markers start unset/False.
    assert args.model_source is None
    assert args.is_vl_model is False
    assert args.uses_unsplit_forward is False
    # Default estimator is grpo → no critic (same as validate_algorithm_args).
    assert args.use_critic is False


def test_make_args_runtime_field_overrides_win():
    args = make_args(is_vl_model=True, uses_unsplit_forward=True, model_source="s3://bucket/model")
    assert args.is_vl_model is True
    assert args.uses_unsplit_forward is True
    assert args.model_source == "s3://bucket/model"


def test_make_args_derives_use_critic_from_advantage_estimator():
    assert make_args(advantage_estimator="ppo").use_critic is True
    assert make_args(advantage_estimator="grpo").use_critic is False
    # Explicit override wins over derivation.
    assert make_args(advantage_estimator="ppo", use_critic=False).use_critic is False


@pytest.mark.parametrize("loss_type", ["sft", "dpo", "rm"])
def test_make_args_offline_objectives_never_use_critic(loss_type):
    # slime_validate_args forces use_critic=False for offline objectives even
    # when the (ignored) estimator would need one.
    assert make_args(loss_type=loss_type, advantage_estimator="ppo").use_critic is False


def test_make_args_clears_parent_package_arguments_attr():
    # After collecting defaults under stubs, ``from relax.utils import arguments``
    # must not keep returning the stub-bound module via the package attribute.
    import sys

    import relax.utils as utils_pkg
    from tests.helpers import relax_args as helper

    sys.modules.pop("relax.utils.arguments", None)
    if hasattr(utils_pkg, "arguments"):
        delattr(utils_pkg, "arguments")
    helper._DEFAULTS = None
    make_args()
    assert "relax.utils.arguments" not in sys.modules
    assert not hasattr(utils_pkg, "arguments")


def test_make_args_adapts_unusable_router_args_on_cached_module():
    # Mirrors arguments_module fixtures that set RouterArgs=object before make_args.
    import importlib
    import sys
    from types import ModuleType

    import relax.utils as utils_pkg
    from tests.helpers import relax_args as helper

    watched = (
        "sglang_router",
        "sglang_router.launch_router",
        "relax.backends.sglang.arguments",
        "relax.utils.arguments",
    )
    previous_modules = {name: sys.modules.get(name) for name in watched}
    previous_defaults = helper._DEFAULTS
    previous_utils_attr = getattr(utils_pkg, "arguments", None)
    had_utils_attr = hasattr(utils_pkg, "arguments")

    try:
        helper._DEFAULTS = None
        sys.modules.pop("relax.utils.arguments", None)
        if had_utils_attr:
            delattr(utils_pkg, "arguments")

        launch = ModuleType("sglang_router.launch_router")
        launch.RouterArgs = object
        router_pkg = ModuleType("sglang_router")
        sys.modules["sglang_router"] = router_pkg
        sys.modules["sglang_router.launch_router"] = launch

        sglang_arguments = ModuleType("relax.backends.sglang.arguments")
        sglang_arguments.sglang_parse_args = lambda: None
        sglang_arguments.validate_args = lambda args: args
        sglang_arguments.add_sglang_arguments = lambda parser: parser
        sglang_arguments.ServerArgs = object
        sys.modules["relax.backends.sglang.arguments"] = sglang_arguments

        module = importlib.import_module("relax.utils.arguments")
        module.RouterArgs = object
        assert module.RouterArgs is object

        args = make_args()
        assert args.task_type == "causal_lm"
        assert module.RouterArgs is object  # restored after collection
    finally:
        helper._DEFAULTS = previous_defaults
        for name, mod in previous_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        if had_utils_attr:
            setattr(utils_pkg, "arguments", previous_utils_attr)
        elif hasattr(utils_pkg, "arguments"):
            delattr(utils_pkg, "arguments")


def test_missing_task_type_raises_without_getattr_default():
    args = make_args()
    delattr(args, "task_type")
    with pytest.raises(AttributeError):
        _ = args.task_type
