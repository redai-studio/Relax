# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Runtime defaults must exist before validation and worker serialization."""

import argparse
import sys
from types import ModuleType

import pytest

from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture

_EXPECTED = {
    "score_centering_vocab_size": None,
    "rollout_request_timeout": 3600.0,
    "scale_out_max_concurrent_weight_syncs": 4,
    "mtp_loss_normalization": None,
    "final_eval_wait_timeout": None,
    "final_eval_poll_interval": None,
}


@pytest.fixture
def parse_module(arguments_module, monkeypatch):
    # Isolate optional backend/I/O imports while using the real Relax argument
    # declarations, pre-parser, merge order and runtime initializer.
    backend = ModuleType("relax.backends.megatron.arguments")

    def parse_backend(*, extra_args_provider, **_kwargs):
        parser = argparse.ArgumentParser(add_help=False)
        extra_args_provider(parser)
        return parser.parse_known_args()[0]

    backend.megatron_parse_args = parse_backend
    backend.validate_args = lambda args: args
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    source = ModuleType("relax.utils.s3_model_loader")
    source.is_s3_uri = lambda uri: False
    monkeypatch.setitem(sys.modules, source.__name__, source)
    monkeypatch.setattr(arguments_module, "sglang_parse_args", argparse.Namespace)
    monkeypatch.setattr(arguments_module, "teacher_sglang_parse_args", argparse.Namespace)
    return arguments_module


@pytest.mark.parametrize("flags", [[], ["--debug-train-only"]], ids=["rollout", "skip_sglang"])
def test_parse_initializes_runtime_fields_before_validation(parse_module, monkeypatch, flags):
    def validate(args):
        assert {name: getattr(args, name) for name in _EXPECTED} == _EXPECTED
        raise RuntimeError("reached first validator")

    monkeypatch.setattr(sys, "argv", ["train.py", *flags])
    monkeypatch.setattr(parse_module, "slime_validate_args", validate)
    with pytest.raises(RuntimeError, match="reached first validator"):
        parse_module._parse_args_impl()


def test_runtime_initialization_preserves_supplied_values_and_cache(arguments_module):
    supplied = {
        "score_centering_vocab_size": 32000,
        "rollout_request_timeout": 0.0,
        "scale_out_max_concurrent_weight_syncs": 2,
        "mtp_loss_normalization": "main_update_tokens",
        "final_eval_wait_timeout": 90.0,
        "final_eval_poll_interval": 0.0,
    }
    args = argparse.Namespace(**supplied)
    arguments_module.init_runtime_args(args)
    arguments_module.init_runtime_args(args)
    assert {name: getattr(args, name) for name in supplied} == supplied


def test_late_yaml_overrides_runtime_defaults_and_custom_cli(parse_module, monkeypatch, tmp_path):
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "score_centering_vocab_size: 64000\n"
        "rollout_request_timeout: 42.0\n"
        "scale_out_max_concurrent_weight_syncs: 3\n"
        "mtp_loss_normalization: main_update_tokens\n"
        "final_eval_wait_timeout: 180.0\n"
        "final_eval_poll_interval: 0.5\n",
        encoding="utf-8",
    )

    def custom_arguments(parser):
        # Exercise the supported custom CLI extension without adding a public
        # framework option for a historical YAML/runtime field.
        parser.add_argument("--rollout-request-timeout", type=float)
        return parser

    def validate(args):
        assert args.rollout_request_timeout == 12.5
        args.use_critic = False  # Normally derived before the late YAML merge.
        parse_module.apply_custom_config_overrides(args)
        assert args.score_centering_vocab_size == 64000
        assert args.rollout_request_timeout == 42.0
        assert args.scale_out_max_concurrent_weight_syncs == 3
        assert args.mtp_loss_normalization == "main_update_tokens"
        assert args.final_eval_wait_timeout == 180.0
        assert args.final_eval_poll_interval == 0.5
        # Reinitialization must preserve YAML values and a resolved cache.
        parse_module.init_runtime_args(args)
        assert args.rollout_request_timeout == 42.0
        assert args.score_centering_vocab_size == 64000
        raise RuntimeError("checked late YAML")

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--loss-type", "sft", "--custom-config-path", str(config), "--rollout-request-timeout", "12.5"],
    )
    monkeypatch.setattr(parse_module, "slime_validate_args", validate)
    with pytest.raises(RuntimeError, match="checked late YAML"):
        parse_module._parse_args_impl(add_custom_arguments=custom_arguments)
