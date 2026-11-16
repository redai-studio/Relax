# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only contracts for TransferQueue configuration construction."""

from __future__ import annotations

import argparse
import os
import sys
from types import ModuleType
from typing import Any

import pytest

from relax.utils.tq.config import (
    _split_host_port,
    build_backend_config,
    build_mooncake_config,
    build_simple_storage_config,
    estimate_payload_bytes,
    resolve_global_segment_size,
    resolve_mooncake_master_address,
    resolve_tq_capacity_batch_size,
    validate_config,
    validate_mooncake_runtime_contract,
    validate_segment_capacity,
)


_MASTER = "master.example:50051"


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "tq_rdma_mode": "auto",
        "tq_rdma_device": "",
        "num_data_storage_units": 1,
        "max_staleness": 0,
        "n_samples_per_prompt": 1,
        "rollout_batch_size": 32,
        "multimodal_keys": None,
        "seq_length": 8192,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [({"tq_rdma_mode": mode}, None) for mode in ("off", "auto", "required")]
    + [({"tq_rdma_device": device}, None) for device in ("", "rdma0", "mlx5_0")]
    + [({"tq_rdma_mode": mode}, "--tq-rdma-mode") for mode in ("mooncake", "auto\nprivate", None, ["auto"])]
    + [({"tq_rdma_device": device}, "--tq-rdma-device") for device in (None, ["rdma0"], "rdma0\nforged", "   ")],
)
def test_validate_config_matrix(overrides: dict[str, Any], expected_error: str | None) -> None:
    errors = validate_config(_args(**overrides))
    if expected_error is None:
        assert errors == []
    else:
        assert len(errors) == 1 and expected_error in errors[0]
        assert repr(next(iter(overrides.values()))) not in errors[0]


def test_missing_config_attributes_default_to_off() -> None:
    assert validate_config(argparse.Namespace()) == []


@pytest.mark.parametrize(
    "address",
    [
        "master.example",
        "master.example:",
        "master.example:abc",
        ":50051",
        "master.example:0",
        "master.example:65536",
        "fe80::1",
        "fe80::1:50051",
        "[fe80::1]",
        "[fe80::1]50051",
        "master\nprivate:50051",
        "master name:50051",
    ],
)
def test_master_endpoint_rejects_malformed_values_without_echo(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setenv("MC_MASTER_ADDRESS", address)
    with pytest.raises(RuntimeError, match="not a usable endpoint") as excinfo:
        resolve_mooncake_master_address()
    assert address not in str(excinfo.value)


def test_master_endpoint_is_required_and_accepts_hostname_or_bracketed_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MC_MASTER_ADDRESS", raising=False)
    with pytest.raises(RuntimeError, match="MC_MASTER_ADDRESS"):
        resolve_mooncake_master_address()
    monkeypatch.setenv("MC_MASTER_ADDRESS", _MASTER)
    assert resolve_mooncake_master_address() == _MASTER
    assert _split_host_port(_MASTER) == ("master.example", 50051)
    assert _split_host_port("[2001:db8::1]:50051") == ("2001:db8::1", 50051)


@pytest.mark.parametrize("total_storage_size", [1000, None], ids=["bounded", "unlimited"])
def test_simple_storage_config(total_storage_size: int | None) -> None:
    assert build_simple_storage_config(total_storage_size, 2) == {
        "storage_backend": "SimpleStorage",
        "SimpleStorage": {"total_storage_size": total_storage_size, "num_data_storage_units": 2},
    }


@pytest.mark.parametrize(
    ("kwargs", "protocol", "device"),
    [({}, "rdma", ""), ({"device": "rdma0"}, "rdma", "rdma0"), ({"protocol": "tcp"}, "tcp", "")],
    ids=["production-default", "explicit-device", "benchmark-tcp"],
)
def test_mooncake_config_contract(kwargs: dict[str, str], protocol: str, device: str) -> None:
    config = build_mooncake_config(master_address=_MASTER, **kwargs)["MooncakeStore"]
    assert (config["protocol"], config["device_name"]) == (protocol, device)
    assert config["master_server_address"] == _MASTER
    assert config["hard_pin"] is True and config["auto_init"] is False and config["use_gdr"] is False
    assert "gdr_staging_buffer_mb" not in config


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"master_address": value}, "master_address") for value in ("not-an-endpoint", "host:0", "fe80::1", 50051)]
    + [
        ({"master_address": _MASTER, "global_segment_size": value}, "positive integer") for value in (0, -1, True, 1.5)
    ],
)
def test_mooncake_builder_rejects_invalid_direct_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_mooncake_config(**kwargs)


@pytest.mark.parametrize("override", [None, "0"], ids=["default", "explicit-disable"])
@pytest.mark.parametrize("contract_version", [1, 2], ids=["required", "forward-compatible"])
def test_runtime_contract_accepts_safe_memcpy(
    monkeypatch: pytest.MonkeyPatch, override: str | None, contract_version: int
) -> None:
    tq_stub = ModuleType("transfer_queue")
    tq_stub.MOONCAKE_CORRECTNESS_CONTRACT_VERSION = contract_version
    monkeypatch.setitem(sys.modules, "transfer_queue", tq_stub)
    if override is None:
        monkeypatch.delenv("MC_STORE_MEMCPY", raising=False)
    else:
        monkeypatch.setenv("MC_STORE_MEMCPY", override)
    validate_mooncake_runtime_contract()
    assert os.environ["MC_STORE_MEMCPY"] == "0"


@pytest.mark.parametrize(
    ("override", "private_marker"),
    [("1", None), ("1\nprivate deployment detail", "private deployment detail")],
)
def test_runtime_contract_rejects_unsafe_memcpy_without_echo(
    monkeypatch: pytest.MonkeyPatch, override: str, private_marker: str | None
) -> None:
    monkeypatch.setenv("MC_STORE_MEMCPY", override)
    with pytest.raises(RuntimeError, match="MC_STORE_MEMCPY") as excinfo:
        validate_mooncake_runtime_contract()
    if private_marker is not None:
        assert private_marker not in str(excinfo.value)


@pytest.mark.parametrize("contract_version", [None, 0, "1", True], ids=["missing", "old", "string", "bool"])
def test_runtime_contract_rejects_missing_or_invalid_marker(
    monkeypatch: pytest.MonkeyPatch, contract_version: int | str | bool | None
) -> None:
    monkeypatch.setenv("MC_STORE_MEMCPY", "0")
    tq_stub = ModuleType("transfer_queue")
    if contract_version is not None:
        tq_stub.MOONCAKE_CORRECTNESS_CONTRACT_VERSION = contract_version
    monkeypatch.setitem(sys.modules, "transfer_queue", tq_stub)
    with pytest.raises(RuntimeError, match="required Mooncake correctness contract"):
        validate_mooncake_runtime_contract()


@pytest.mark.parametrize(
    ("overrides", "expected_fragment"),
    [
        ({"multimodal_keys": None}, None),
        ({"multimodal_keys": ["pixel_values"], "rollout_batch_size": 256, "n_samples_per_prompt": 8}, "insufficient"),
        ({"multimodal_keys": ["pixel_values"], "rollout_batch_size": 32, "max_staleness": 1}, "staleness+1=2"),
        (
            {
                "multimodal_keys": ["pixel_values"],
                "rollout_batch_size": 16,
                "partial_rollout": True,
                "use_dynamic_global_batch_size": True,
                "over_sampling_batch_size": 64,
            },
            "effective_batch=64",
        ),
    ],
)
def test_segment_capacity_matrix(overrides: dict[str, Any], expected_fragment: str | None) -> None:
    error = validate_segment_capacity(_args(**overrides))
    if expected_fragment is None:
        assert error is None
    else:
        assert error is not None and expected_fragment.lower() in error.lower()


@pytest.mark.parametrize("value", ["four", "-1", "nan", "inf", "-inf", "1e-20"])
def test_segment_size_env_rejects_unusable_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", value)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_global_segment_size()
    assert value not in str(excinfo.value)


@pytest.mark.parametrize("segment_gib", ["0.001", "4"], ids=["insufficient", "sufficient"])
def test_backend_config_capacity_contract(monkeypatch: pytest.MonkeyPatch, segment_gib: str) -> None:
    monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", segment_gib)
    args = _args()

    backend, error = build_backend_config(args, device="rdma0", master_address=_MASTER)

    if segment_gib == "0.001":
        assert backend == {}
        assert error is not None and "capacity insufficient" in error
    else:
        assert error is None
        assert backend == build_mooncake_config(master_address=_MASTER, device="rdma0")


def test_capacity_override_and_payload_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    multimodal = _args(multimodal_keys=["pixel_values"], rollout_batch_size=1)
    assert estimate_payload_bytes(_args()) == 32 * 32 * 8192
    assert estimate_payload_bytes(multimodal) == 8192 * (32 + 24_576)
    dynamic = _args(
        rollout_batch_size=16,
        partial_rollout=True,
        use_dynamic_global_batch_size=True,
        over_sampling_batch_size=64,
    )
    assert resolve_tq_capacity_batch_size(dynamic) == 64
    monkeypatch.setenv("RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB", "16")
    assert validate_segment_capacity(_args(multimodal_keys=["pixel_values"], max_staleness=1)) is None
    with pytest.raises(RuntimeError, match="seq_length"):
        estimate_payload_bytes(_args(seq_length=None))
