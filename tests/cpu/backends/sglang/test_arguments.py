# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import json
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


@pytest.mark.parametrize("prefix", ["router", "sglang-router"])
def test_router_arguments_preserve_explicit_manual_options(sglang_arguments_module, monkeypatch, prefix):
    from sglang_router.router_args import RouterArgs

    # Isolate only the unrelated inference-server CLI. Router registration,
    # parsing and construction all use the real implementations.
    monkeypatch.setattr(sglang_arguments_module, "ServerArgs", SimpleNamespace(add_cli_args=lambda parser: None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", f"--{prefix}-assignment-mode", "min_load", f"--{prefix}-max-idle-secs", "600"],
    )

    router_args = RouterArgs.from_cli_args(sglang_arguments_module.sglang_parse_args(), use_router_prefix=True)

    assert router_args.assignment_mode == "min_load"
    assert router_args.max_idle_secs == 600


def test_router_arguments_mixed_prefixes_use_last_value(sglang_arguments_module, monkeypatch):
    from sglang_router.router_args import RouterArgs

    monkeypatch.setattr(sglang_arguments_module, "ServerArgs", SimpleNamespace(add_cli_args=lambda parser: None))
    monkeypatch.setattr(
        sys, "argv", ["train", "--sglang-router-max-idle-secs", "300", "--router-max-idle-secs", "600"]
    )

    router_args = RouterArgs.from_cli_args(sglang_arguments_module.sglang_parse_args(), use_router_prefix=True)

    assert router_args.max_idle_secs == 600


@pytest.mark.parametrize("build", ["stock", "unrecorded-fork", "recorded-fork", "other-source"])
def test_routing_replay_requires_raw_generate_router(sglang_arguments_module, monkeypatch, tmp_path, build):
    router = ModuleType("sglang_router")
    router.__file__ = str(tmp_path / "__init__.py")
    router.__version__ = "0.3.2" if build == "stock" else "0.3.2+slime.dpaware"
    monkeypatch.setitem(sys.modules, "sglang_router", router)
    if build in ("recorded-fork", "other-source"):
        (tmp_path / "_relax_build.json").write_text(
            json.dumps(
                {
                    "version": router.__version__,
                    "source_commit": "9daabcd801c861088d93b8ce8a2a831490ce4bee"
                    if build == "recorded-fork"
                    else "another-source",
                }
            )
        )
    args = _args(use_rollout_routing_replay=True)
    if build == "recorded-fork":
        sglang_arguments_module.validate_args(args)
    else:
        with pytest.raises(ValueError, match="return_routed_experts"):
            sglang_arguments_module.validate_args(args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_rollout_routing_replay": False},
        {"debug_train_only": True},
        {"load_debug_rollout_data": "saved-rollout.pt"},
        {"sglang_router_ip": "external-router"},
    ],
)
def test_routing_replay_does_not_gate_unused_local_router(sglang_arguments_module, monkeypatch, overrides):
    # No local router is launched in these modes. In particular, a local stock
    # installation says nothing about an externally managed router's capability.
    monkeypatch.setitem(sys.modules, "sglang_router", None)
    args = _args(**{"use_rollout_routing_replay": True, **overrides})
    sglang_arguments_module.validate_args(args)
