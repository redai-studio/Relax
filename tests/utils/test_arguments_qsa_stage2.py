# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture


def test_qsa_stage2_loss_coefficient_defaults_off_and_parses_positive(arguments_module):
    parser = argparse.ArgumentParser()
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **kwargs: parser)
    arguments_module.get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).qsa_indexer_loss_coeff == 0.0
    assert parser.parse_args(["--qsa-indexer-loss-coeff", "0.01"]).qsa_indexer_loss_coeff == 0.01


def test_qsa_stage2_loss_coefficient_reuses_megatron_registered_option(arguments_module):
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsa-indexer-loss-coeff", type=float, default=0.25)
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **kwargs: parser)
    arguments_module.get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).qsa_indexer_loss_coeff == 0.0
    assert parser.parse_args(["--qsa-indexer-loss-coeff", "0.7"]).qsa_indexer_loss_coeff == 0.7
    with pytest.raises(SystemExit):
        parser.parse_args(["--qsa-indexer-loss-coeff", "nan"])


@pytest.mark.parametrize("value", ["-0.01", "nan", "inf", "-inf"])
def test_qsa_stage2_loss_coefficient_rejects_invalid_values(value: str, arguments_module):
    parser = argparse.ArgumentParser()
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **kwargs: parser)
    arguments_module.get_slime_extra_args_provider()(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([f"--qsa-indexer-loss-coeff={value}"])
