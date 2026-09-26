# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Controller decisions layered on top of the TQ config contracts."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.core.test_controller_s3_model_cleanup import controller
from tests.utils.test_arguments_opd_teacher_colocate import arguments_module as _arguments_module_fixture


arguments_module = _arguments_module_fixture
_MASTER = "master.invalid:50051"


def _config(**overrides: Any) -> SimpleNamespace:
    values = {
        "tq_rdma_mode": "auto",
        "tq_rdma_device": "",
        "num_data_storage_units": 1,
        "n_samples_per_prompt": 1,
        "rollout_batch_size": 8,
        "seq_length": 8192,
        "multimodal_keys": None,
        "max_staleness": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve(config: SimpleNamespace) -> dict[str, Any]:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = config
    return instance._resolve_tq_backend()


def _is_simple(backend: dict[str, Any]) -> bool:
    return backend["storage_backend"] == "SimpleStorage"


class _DecisionHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, failure: str | None, *, stub_backend: bool = True) -> None:
        self.calls: list[str] = []

        def contract() -> None:
            self.calls.append("contract")
            if failure == "contract":
                raise RuntimeError("contract unavailable")

        def master() -> str:
            self.calls.append("master")
            if failure == "master":
                raise RuntimeError("master unavailable")
            return _MASTER

        def backend(_args: Any, *, device: str, master_address: str):
            self.calls.append(f"build:{device}:{master_address}")
            if failure == "capacity-error":
                return {}, "capacity insufficient"
            if failure == "capacity-config":
                raise RuntimeError("capacity configuration unusable")
            return {
                "storage_backend": "MooncakeStore",
                "MooncakeStore": {"protocol": "rdma", "device_name": device, "master_server_address": master_address},
            }, None

        monkeypatch.setattr(controller, "validate_mooncake_runtime_contract", contract)
        monkeypatch.setattr(controller, "resolve_mooncake_master_address", master)
        if stub_backend:
            monkeypatch.setattr(controller, "build_backend_config", backend)


@pytest.mark.parametrize(
    ("mode", "failure", "expect_simple"),
    [("off", None, True), ("auto", None, False), ("required", None, False)]
    + [("auto", failure, True) for failure in ("contract", "master", "capacity-error", "capacity-config")]
    + [("required", failure, False) for failure in ("contract", "master", "capacity-error", "capacity-config")],
)
def test_backend_decision_matrix(
    monkeypatch: pytest.MonkeyPatch, mode: str, failure: str | None, expect_simple: bool
) -> None:
    harness = _DecisionHarness(monkeypatch, failure)
    context = pytest.raises(RuntimeError) if mode == "required" and failure else nullcontext()
    with context:
        backend = _resolve(_config(tq_rdma_mode=mode, tq_rdma_device="rdma0"))
        assert _is_simple(backend) is expect_simple
        if not expect_simple:
            assert backend["MooncakeStore"] == {
                "protocol": "rdma",
                "device_name": "rdma0",
                "master_server_address": _MASTER,
            }
    if mode == "off":
        assert harness.calls == []
    elif failure == "contract":
        assert harness.calls == ["contract"]
    elif failure == "master":
        assert harness.calls == ["contract", "master"]
    else:
        assert harness.calls == ["contract", "master", f"build:rdma0:{_MASTER}"]


@pytest.mark.parametrize("overrides", [{"tq_rdma_mode": "mooncake"}, {"tq_rdma_device": "bad\ndevice"}])
def test_invalid_config_fails_before_preconditions(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any]) -> None:
    harness = _DecisionHarness(monkeypatch, None)
    with pytest.raises(ValueError, match="TransferQueue RDMA configuration"):
        _resolve(_config(**overrides))
    assert harness.calls == []


def test_missing_mode_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _DecisionHarness(monkeypatch, None)
    config = _config()
    del config.tq_rdma_mode
    assert _is_simple(_resolve(config))
    assert harness.calls == []


def test_off_mode_simple_storage_is_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real builder runs here: production must never hand SimpleStorage a
    # logical-batch-derived row cap, because agent/tool fan-out writes more
    # physical rows than the rollout batch holds identities.
    _DecisionHarness(monkeypatch, None, stub_backend=False)

    backend = _resolve(_config(tq_rdma_mode="off"))

    assert _is_simple(backend)
    assert backend["SimpleStorage"] == {"total_storage_size": None, "num_data_storage_units": 1}


@pytest.mark.parametrize("failure", ["contract", "master"])
def test_auto_fallback_simple_storage_is_unbounded(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    _DecisionHarness(monkeypatch, failure, stub_backend=False)

    backend = _resolve(_config(tq_rdma_mode="auto", tq_rdma_device="rdma0"))

    assert _is_simple(backend)
    assert backend["SimpleStorage"]["total_storage_size"] is None


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_capacity_failure_builds_fallback_only_in_auto(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from relax.utils.tq import config as tq_config

    _DecisionHarness(monkeypatch, None, stub_backend=False)
    monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "0.001")
    simple_builder = MagicMock(wraps=tq_config.build_simple_storage_config)
    monkeypatch.setattr(tq_config, "build_simple_storage_config", simple_builder)
    warning = MagicMock()
    monkeypatch.setattr(controller.logger, "warning", warning)

    if mode == "auto":
        backend = _resolve(_config(tq_rdma_mode=mode))
        assert _is_simple(backend)
        simple_builder.assert_called_once_with(total_storage_size=None, num_data_storage_units=1)
        warning.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="segment capacity insufficient"):
            _resolve(_config(tq_rdma_mode=mode))
        simple_builder.assert_not_called()
        warning.assert_not_called()


def test_initialized_data_system_keeps_simple_storage_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = _config(
        tq_rdma_mode="off",
        fully_async=False,
        balance_data=False,
        polling_mode=False,
    )
    instance._tq_owner = None
    instance._tq_legacy_init = False
    captured: dict[str, Any] = {}

    monkeypatch.setattr(controller, "resolve_sft_algo_key", lambda _config: "grpo")
    monkeypatch.setattr(controller, "compute_dp_size", lambda _config: 1)
    monkeypatch.setattr(controller, "IdentityWindowSampler", lambda **_kwargs: object())
    monkeypatch.setattr(controller, "reap_unusable_tq_controller", lambda: None)

    def init(conf: Any) -> Any:
        captured["conf"] = conf
        return conf

    monkeypatch.setattr(controller.tq, "init", init)

    instance._initialize_data_system()

    backend = captured["conf"]["backend"]
    assert _is_simple(backend)
    assert backend["SimpleStorage"]["total_storage_size"] is None


def test_cli_exposes_only_mode_and_device(arguments_module: Any) -> None:
    arguments_module.RouterArgs = SimpleNamespace(add_cli_args=lambda parser, **_kwargs: parser)
    parser = argparse.ArgumentParser()
    arguments_module.get_slime_extra_args_provider()(parser)
    tq_flags = sorted(
        option for action in parser._actions for option in action.option_strings if option.startswith("--tq-")
    )
    assert tq_flags == ["--tq-rdma-device", "--tq-rdma-mode"]
    args = parser.parse_args([])
    assert (args.tq_rdma_mode, args.tq_rdma_device) == ("off", "")


def test_off_mode_rejects_a_healthy_existing_controller_before_legacy_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = _config(
        tq_rdma_mode="off",
        fully_async=False,
        balance_data=False,
        polling_mode=False,
    )
    instance._tq_owner = None
    instance._tq_legacy_init = False
    monkeypatch.setattr(controller, "resolve_sft_algo_key", lambda _config: "grpo")
    monkeypatch.setattr(controller, "compute_dp_size", lambda _config: 1)
    monkeypatch.setattr(controller, "IdentityWindowSampler", lambda **_kwargs: object())
    resolved_calls: list[bool] = []

    def resolve_backend() -> dict[str, Any]:
        resolved_calls.append(True)
        return {"storage_backend": "SimpleStorage"}

    monkeypatch.setattr(instance, "_resolve_tq_backend", resolve_backend)
    monkeypatch.setattr(
        controller,
        "reap_unusable_tq_controller",
        lambda: (_ for _ in ()).throw(RuntimeError("exclusive cluster is not clean")),
    )
    monkeypatch.setattr(controller.tq, "init", lambda **_kwargs: pytest.fail("tq.init must not run"))
    with pytest.raises(RuntimeError, match="exclusive cluster"):
        instance._initialize_data_system()
    assert resolved_calls == [True]
    assert instance._tq_owner is None and instance._tq_legacy_init is False


class _AttachHarness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        failures: list[str] | None = None,
        verify_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.events: list[str] = []

        def verify(_conf: Any, **_kwargs: Any) -> list[str]:
            self.events.append("handshake")
            if verify_error:
                raise verify_error
            return failures or []

        def close(_owner: Any, **_kwargs: Any) -> None:
            self.events.append("close")
            if close_error:
                raise close_error

        def fallback(conf: Any, **_kwargs: Any) -> Any:
            self.events.append("init_simple")
            return controller.TqInitResult(config=conf, owner="fallback-owner")

        monkeypatch.setattr(controller, "verify_cluster_attach", verify)
        monkeypatch.setattr(controller, "close_tq_owner", close)
        monkeypatch.setattr(controller, "initialize_tq_with_fallback", fallback)


def _confirm(mode: str = "auto") -> tuple[Any, Any]:
    instance = controller.Controller.__new__(controller.Controller)
    instance.config = _config(tq_rdma_mode=mode)
    instance._tq_owner = "mooncake-owner"
    result = controller.TqInitResult(config="mooncake-conf", owner="mooncake-owner")
    return instance, result


@pytest.mark.parametrize(
    ("mode", "failures", "verify_error", "expected_events", "error_match"),
    [
        ("auto", [], None, ["handshake"], None),
        ("auto", ["timeout"], None, ["handshake", "close", "init_simple"], None),
        ("required", ["protocol=tcp"], None, ["handshake", "close"], "reported"),
        ("auto", None, RuntimeError("private detail"), ["handshake", "close"], "orchestration failed"),
        (
            "auto",
            None,
            controller.TqHandshakeIsolationError("private detail"),
            ["handshake", "close"],
            "could not be confirmed stopped",
        ),
    ],
)
def test_attach_handshake_cleanup_matrix(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    failures: list[str] | None,
    verify_error: BaseException | None,
    expected_events: list[str],
    error_match: str | None,
) -> None:
    harness = _AttachHarness(monkeypatch, failures=failures, verify_error=verify_error)
    instance, result = _confirm(mode)
    context = pytest.raises(RuntimeError, match=error_match) if error_match else nullcontext()
    with context:
        resolved = instance._confirm_mooncake_attach(result, "simple-conf")
        if failures:
            assert resolved.config == "simple-conf" and resolved.owner == "fallback-owner"
        else:
            assert resolved is result
    assert harness.events == expected_events


def test_attach_cleanup_failure_aborts_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _AttachHarness(
        monkeypatch,
        failures=["timeout"],
        close_error=RuntimeError("owner cleanup failed"),
    )
    instance, result = _confirm()
    with pytest.raises(RuntimeError, match="owner cleanup failed"):
        instance._confirm_mooncake_attach(result, "simple-conf")
    assert harness.events == ["handshake", "close"] and instance._tq_owner == "mooncake-owner"


def test_constructor_and_repeated_cleanup_preserve_owner_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Any] = []
    close_data_system = controller.Controller._close_data_system
    monkeypatch.setattr(controller, "resolve_sft_num_rollout", lambda _config: None)
    monkeypatch.setattr(controller, "HealthManager", lambda **_kwargs: object())

    def fail_init(instance: Any) -> None:
        instance._tq_owner = "owner"
        raise RuntimeError("initialization failed")

    def close(instance: Any) -> None:
        events.append(instance._tq_owner)
        instance._tq_owner = None

    monkeypatch.setattr(controller.Controller, "_initialize_data_system", fail_init)
    monkeypatch.setattr(controller.Controller, "_close_data_system", close)
    with pytest.raises(RuntimeError, match="initialization failed"):
        controller.Controller(SimpleNamespace(use_health_check=False, max_global_restart=3))
    assert events == ["owner"]

    monkeypatch.setattr(controller.Controller, "_close_data_system", close_data_system)
    instance = controller.Controller.__new__(controller.Controller)
    instance._tq_legacy_init = False
    instance._tq_owner = "owner"
    monkeypatch.setattr(controller, "close_tq_owner", lambda owner: events.append(owner) if owner else None)
    instance._close_data_system()
    instance._close_data_system()
    assert events == ["owner", "owner"] and instance._tq_owner is None
