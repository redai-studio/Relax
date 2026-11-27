# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from tests.cpu.utils.test_arguments_opd_teacher_colocate import (
    arguments_module as _arguments_module_fixture,
)


arguments_module = _arguments_module_fixture


@pytest.mark.parametrize(
    "argv",
    [
        ["--use-slime-router"],
        ["--slime-router-sticky"],
        ["--slime-router-sticky-idle-secs=600"],
    ],
)
def test_arguments_reject_removed_router_options_before_backend_parsing(arguments_module, monkeypatch, argv):
    monkeypatch.setattr(arguments_module.sys, "argv", ["train.py", *argv])

    with pytest.raises(ValueError, match=argv[0].split("=", 1)[0]) as error:
        arguments_module._parse_args_impl()

    assert "--sglang-router-policy manual" in str(error.value)
    assert "--router-max-idle-secs" in str(error.value)


def test_arguments_removed_router_guard_allows_native_and_backend_options(arguments_module):
    arguments_module._reject_removed_slime_router_flags(
        ["--sglang-router-policy", "manual", "--router-max-idle-secs=600", "--tensor-model-parallel-size", "2"]
    )
