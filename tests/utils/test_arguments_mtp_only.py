# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def test_mtp_detach_paths_defaults_to_all_paths(arguments_module):
    parser = argparse.ArgumentParser()
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    arguments_module.get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).mtp_detach_paths == arguments_module._MTP_DETACH_PATHS

    args = parser.parse_args(["--mtp-detach-paths", "lm-head", "embedding"])
    arguments_module._normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ("embedding", "lm-head")

    args = parser.parse_args(["--mtp-detach-paths", "none"])
    arguments_module._normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["embedding"], ("embedding",)),
        (["lm-head", "embedding"], ("embedding", "lm-head")),
        (["backbone", "backbone"], ("backbone",)),
        (["none"], ()),
    ],
)
def test_normalize_mtp_detach_paths(arguments_module, values, expected):
    args = SimpleNamespace(mtp_detach_paths=values)

    arguments_module._normalize_mtp_detach_paths(args)

    assert args.mtp_detach_paths == expected


def test_normalize_mtp_detach_paths_rejects_none_with_another_path(arguments_module):
    with pytest.raises(ValueError, match="none cannot be combined"):
        arguments_module._normalize_mtp_detach_paths(SimpleNamespace(mtp_detach_paths=["none", "embedding"]))


@pytest.mark.parametrize("flag", ["--mtp-detach-main-model", "--no-mtp-detach-main-model"])
def test_removed_mtp_detach_flags_fail_fast_even_when_unknown_args_are_ignored(arguments_module, flag):
    with pytest.raises(ValueError, match="has been removed"):
        arguments_module._reject_removed_mtp_detach_flags([flag])


def _args(arguments_module, **overrides) -> SimpleNamespace:
    defaults = {
        "mtp_only_training": True,
        "loss_type": "sft",
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "lora_rank": 0,
        "sft_chunked_logits": False,
        "overlap_moe_expert_parallel_comm": False,
        "fully_async": False,
        "hybrid": False,
        "mtp_num_layers": None,
        "mtp_loss_scaling_factor": 0.2,
        "enable_mtp_training": False,
        "mtp_detach_paths": arguments_module._MTP_DETACH_PATHS,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_normalize_mtp_only_training_enables_mtp_and_freezes_non_mtp_params(arguments_module):
    args = _args(arguments_module)

    arguments_module._normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is True
    assert args.mtp_detach_paths == arguments_module._MTP_DETACH_PATHS
    assert args.mtp_num_layers == 1
    assert args.only_train_params_name_list == [arguments_module._MTP_ONLY_PARAM_PATTERN]


@pytest.mark.parametrize(
    ("override", "expected_flag"),
    [
        ({"loss_type": "policy_loss"}, "--loss-type sft"),
        ({"only_train_params_name_list": ["decoder"]}, "--only-train-params-name-list"),
        ({"freeze_params_name_list": ["output_layer"]}, "--freeze-params-name-list"),
        ({"lora_rank": 8}, "--lora-rank"),
        ({"sft_chunked_logits": True}, "--sft-chunked-logits"),
        ({"overlap_moe_expert_parallel_comm": True}, "--overlap-moe-expert-parallel-comm"),
        ({"fully_async": True}, "--fully-async"),
        ({"hybrid": True}, "--hybrid"),
        ({"mtp_detach_paths": ()}, "--mtp-detach-paths"),
        ({"mtp_num_layers": 2}, "exactly one"),
        ({"mtp_loss_scaling_factor": 0.0}, "greater than 0"),
    ],
)
def test_normalize_mtp_only_training_rejects_unsafe_combinations(arguments_module, override, expected_flag):
    with pytest.raises(ValueError, match=expected_flag):
        arguments_module._normalize_mtp_only_training_args(_args(arguments_module, **override))


def test_normalize_mtp_only_training_is_noop_when_disabled(arguments_module):
    args = _args(arguments_module, mtp_only_training=False)

    arguments_module._normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is False
    assert args.mtp_detach_paths == arguments_module._MTP_DETACH_PATHS
    assert args.mtp_num_layers is None
    assert args.only_train_params_name_list is None
