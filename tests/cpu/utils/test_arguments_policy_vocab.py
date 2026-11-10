# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from relax.utils.policy_vocab import resolve_policy_vocab_size
from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


@pytest.fixture()
def parser(arguments_module):
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    return parser


@pytest.mark.parametrize("argv,expected", [([], None), (["--policy-vocab-size", "32000"], 32000)])
def test_policy_vocab_cli_resolves_output_range(parser, argv, expected):
    args = parser.parse_args(argv)
    args.train_backend = "megatron"
    assert args.policy_vocab_size == expected
    assert resolve_policy_vocab_size(args, model_vocab_size=32768) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true"])
def test_policy_vocab_cli_rejects_invalid_size(parser, value):
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--policy-vocab-size", value])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"loss_type": "sft"}, "online RL"),
        ({"loss_type": "dpo"}, "online RL"),
        ({"loss_type": "rm"}, "online RL"),
        ({"train_backend": "fsdp"}, "Megatron"),
    ],
)
def test_policy_vocab_cli_rejects_unsupported_training_paths(parser, overrides, message):
    args = parser.parse_args(["--policy-vocab-size", "32000"])
    args.train_backend = "megatron"
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        resolve_policy_vocab_size(args, model_vocab_size=32768)


def test_policy_vocab_cli_rejects_range_larger_than_model(parser):
    args = parser.parse_args(["--policy-vocab-size", "32000"])
    args.train_backend = "megatron"
    with pytest.raises(ValueError, match="integer in"):
        resolve_policy_vocab_size(args, model_vocab_size=31000)


@pytest.mark.parametrize("selection", ["teacher_topk", "union"])
def test_policy_vocab_cli_rejects_unrestricted_opd_student_prefill(parser, selection):
    args = parser.parse_args(
        [
            "--policy-vocab-size",
            "32000",
            "--use-opd",
            "--opd-type",
            "sglang",
            "--opd-token-selection",
            selection,
            "--opd-kl-coef",
            "1.0",
            "--opd-loss-coef",
            "0.0",
        ]
    )
    args.train_backend = "megatron"
    with pytest.raises(ValueError, match="student prefill.*full vocabulary"):
        resolve_policy_vocab_size(args, model_vocab_size=32768)


@pytest.mark.parametrize(
    "limit,overrides",
    [
        (32000, {"opd_token_selection": "student_sampled"}),
        (32000, {"opd_token_selection": "student_topk"}),
        (32000, {"opd_kl_coef": 0.0, "opd_loss_coef": 1.0}),
        (32000, {"opd_fkl_entropy_gate": True, "opd_loss_coef": 1.0}),
        (32000, {"use_opd": False}),
        (32768, {}),
        (None, {}),
    ],
)
def test_policy_vocab_keeps_opd_paths_without_mixed_student_probabilities(limit, overrides):
    args = SimpleNamespace(
        policy_vocab_size=limit,
        train_backend="megatron",
        use_opd=True,
        opd_type="sglang",
        opd_token_selection="union",
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        opd_fkl_entropy_gate=False,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    assert resolve_policy_vocab_size(args, model_vocab_size=32768) == limit
