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
