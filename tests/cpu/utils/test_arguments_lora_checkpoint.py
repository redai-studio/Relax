# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


def _parse_checkpoint_args(arguments_module, tmp_path, *extra_flags: str) -> argparse.Namespace:
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    return parser.parse_args(
        [
            "--lora-rank",
            "8",
            "--save",
            str(tmp_path / "native"),
            "--save-lora-only",
            "--save-hf",
            str(tmp_path / "hf"),
            *extra_flags,
        ],
        namespace=argparse.Namespace(
            train_backend="megatron", ckpt_format="torch_dist", bf16=True, fp16=False, fp8=None, no_save_rng=False
        ),
    )


def test_save_lora_only_accepts_full_hf_export(arguments_module, tmp_path):
    args = _parse_checkpoint_args(arguments_module, tmp_path)

    arguments_module.validate_save_lora_only_args(args)

    assert args.save_lora_only is True
    assert args.save == str(tmp_path / "native")
    assert args.save_hf == str(tmp_path / "hf")
    assert args.save_hf_dtype == "bf16"


@pytest.mark.parametrize("flag", ["--async-save", "--no-save-optim", "--rotate-ckpt"])
def test_save_lora_only_with_hf_keeps_resume_guards(arguments_module, tmp_path, flag):
    args = _parse_checkpoint_args(arguments_module, tmp_path, flag)

    with pytest.raises(ValueError, match=flag):
        arguments_module.validate_save_lora_only_args(args)
