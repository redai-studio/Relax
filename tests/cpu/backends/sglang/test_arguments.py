# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture()
def sglang_arguments_module(monkeypatch):
    sglang = ModuleType("sglang")
    sglang_srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    http_utils = ModuleType("relax.utils.http_utils")
    http_utils._wrap_ipv6 = lambda host: host
    monkeypatch.setitem(sys.modules, "relax.utils.http_utils", http_utils)

    sys.modules.pop("relax.backends.sglang.arguments", None)
    module = importlib.import_module("relax.backends.sglang.arguments")
    yield module
    sys.modules.pop("relax.backends.sglang.arguments", None)


def _args(**overrides):
    values = {
        "sglang_data_parallel_size": 1,
        "sglang_pipeline_parallel_size": 1,
        "sglang_expert_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
        "sglang_enable_dp_attention": False,
        "router_dp_aware": False,
        "sglang_router_ip": None,
        "prefill_num_servers": None,
        "sglang_config": None,
        "rollout_external": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dp_attention_enables_router_dp_aware_with_warning(sglang_arguments_module, monkeypatch):
    logger = Mock()
    monkeypatch.setattr(sglang_arguments_module, "logger", logger)
    args = _args(
        sglang_data_parallel_size=4,
        rollout_num_gpus_per_engine=4,
        sglang_enable_dp_attention=True,
    )

    sglang_arguments_module.validate_args(args)

    assert args.router_dp_aware is True
    logger.warning.assert_called_once_with(
        "sglang_enable_dp_attention=True requires router_dp_aware=True; overriding router_dp_aware=False with True."
    )


def test_dp_attention_does_not_warn_when_router_dp_aware_is_enabled(sglang_arguments_module, monkeypatch):
    logger = Mock()
    monkeypatch.setattr(sglang_arguments_module, "logger", logger)
    args = _args(sglang_enable_dp_attention=True, router_dp_aware=True)

    sglang_arguments_module.validate_args(args)

    assert args.router_dp_aware is True
    logger.warning.assert_not_called()


@pytest.mark.parametrize("router_dp_aware", [False, True])
def test_router_dp_aware_is_unchanged_without_dp_attention(sglang_arguments_module, monkeypatch, router_dp_aware):
    logger = Mock()
    monkeypatch.setattr(sglang_arguments_module, "logger", logger)
    args = _args(router_dp_aware=router_dp_aware)

    sglang_arguments_module.validate_args(args)

    assert args.router_dp_aware is router_dp_aware
    logger.warning.assert_not_called()
