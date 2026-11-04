# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.cpu.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


@pytest.mark.parametrize("legacy", ["0", "1"])
@pytest.mark.parametrize("cli", [False, True])
@pytest.mark.parametrize("loss_type", ["sft", "policy_loss"])
def test_fused_linear_ce_cli_is_explicit_and_independent_of_legacy_env(
    arguments_module, monkeypatch, legacy, cli, loss_type
):
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", legacy)
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    argv = ["--loss-type", loss_type] + (["--fused-linear-ce"] if cli else [])
    args = parser.parse_args(argv)
    assert args.fused_linear_ce is cli
    assert not args.sft_chunked_logits  # The kernel selector does not change SFT routing.
