# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fallback owner validation and strict warm delivery without file fallback."""

import copy
import json
from pathlib import Path

import pytest

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.fallback_receipts import (
    owner_keys,
    read_receipt,
    route_receipts,
    serialize_receipt,
)


def fixture(version=3, audit=False):
    keys = [f"{engine}-{tp}" for engine in ("3", "7") for tp in range(2)]
    schemas = {
        key: dict(generation="generation:" + key, params={"bias", "weight"}, loader_buffers={"norm.gemma_weight"})
        for key in keys
    }
    receipts = [
        dict(
            generation=schemas[key]["generation"],
            version=version,
            params=["bias", "weight"],
            digests=dict.fromkeys(("bias", "weight"), "a" * 64) if audit else {},
            loader_buffers=["norm.gemma_weight"],
        )
        for key in keys
    ]
    return keys, schemas, [serialize_receipt(rank, key, receipts[rank]) for rank, key in enumerate(keys)]


@pytest.mark.parametrize("audit", [False, True])
def test_original_receipts_are_routed_without_reconstruction(audit):
    keys, schemas, payloads = fixture(audit=audit)
    assert owner_keys(keys, "7", 2) == keys
    assert route_receipts(payloads, keys, schemas, 3, audit) == {
        key: json.loads(payload) for key, payload in zip(keys, payloads, strict=True)
    }


@pytest.mark.parametrize(
    "fault",
    [
        "missing_owner",
        "owner",
        "bool_owner",
        "key",
        "delivery",
        "version",
        "bool_version",
        "generation",
        "missing_param",
        "extra_param",
        "duplicate_param",
        "buffer",
        "duplicate_buffer",
        "digest",
        "extra_field",
    ],
)
def test_router_rejects_stale_and_incomplete_originals(fault):
    keys, schemas, payloads = fixture()
    item = json.loads(payloads[1])
    receipt = item["receipt"]
    if fault == "missing_owner":
        payloads.pop()
    elif fault in ("owner", "bool_owner"):
        item["source_rank"] = True if fault == "bool_owner" else 0
    elif fault in ("key", "delivery"):
        item[fault] = "wrong"
    elif fault in ("version", "bool_version"):
        receipt["version"] = True if fault == "bool_version" else 2
    elif fault == "generation":
        receipt["generation"] = "stale"
    elif fault == "missing_param":
        receipt["params"].pop()
    elif fault == "extra_param":
        receipt["params"].append("z")
    elif fault == "duplicate_param":
        receipt["params"].append("weight")
    elif fault == "buffer":
        receipt["loader_buffers"] = []
    elif fault == "duplicate_buffer":
        receipt["loader_buffers"] *= 2
    elif fault == "digest":
        receipt["digests"] = {"bias": "a" * 64}
    else:
        receipt["unexpected"] = True
    payloads[1] = json.dumps(item)
    with pytest.raises(RuntimeError):
        route_receipts(payloads, keys, schemas, 3, False)


def test_serialization_rejects_nonfinite_data_before_collective():
    with pytest.raises(ValueError):
        serialize_receipt(0, "0-0", {"bad": float("nan")})


@pytest.mark.parametrize("fault", ["missing_tp", "duplicate", "reorder", "missing_engine", "bad_rank"])
def test_owner_order_is_complete_and_stable(fault):
    keys, _, _ = fixture()
    if fault == "missing_tp":
        keys.pop()
    elif fault == "duplicate":
        keys[1] = keys[0]
    elif fault == "reorder":
        keys[0], keys[1] = keys[1], keys[0]
    elif fault == "bad_rank":
        keys[1] = "3-01"
    with pytest.raises(RuntimeError):
        owner_keys(keys, "unknown" if fault == "missing_engine" else "3", 2)


@pytest.fixture
def warm(tmp_path):
    keys, schemas, payloads = fixture()
    routed = route_receipts(payloads, keys, schemas, 3, False)
    state = dict(
        active=True,
        audit=False,
        version=3,
        initial_version=2,
        completed_version=2,
        key="3-0",
        generation=schemas["3-0"]["generation"],
        directory=str(tmp_path),
        receipt_tp_size=2,
        fallback_receipt_keys=keys,
        fallback_loader_buffers=["norm.gemma_weight"],
        params={"weight": None, "bias": None, "expert": None},
        direct_params={"expert"},
    )
    command = dict(engine="3", version=3, fallback_receipts={key: routed[key] for key in keys[:2]})
    return state, command, tmp_path


def test_warm_never_reads_files_and_copies_original(warm, monkeypatch):
    state, command, path = warm
    (path / "3-0.done.json").write_text("stale")
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("Warm RPC read a file"))
    original = copy.deepcopy(command["fallback_receipts"]["3-0"]["receipt"])
    actual = read_receipt(state, command, path, "3-0")
    command["fallback_receipts"]["3-0"]["receipt"]["params"].clear()
    assert actual == original


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "missing_tp",
        "extra_tp",
        "owner",
        "key",
        "version",
        "generation",
        "coverage",
        "buffers",
        "inactive",
        "completed",
        "directory",
        "command_version",
        "ipc",
    ],
)
def test_warm_fails_closed_even_with_valid_file(warm, fault):
    state, command, path = warm
    original = command["fallback_receipts"]["3-0"]["receipt"]
    (path / "3-0.done.json").write_text(json.dumps(original))
    envelope = command["fallback_receipts"]["3-0"]
    if fault == "missing":
        command.pop("fallback_receipts")
    elif fault == "missing_tp":
        command["fallback_receipts"].pop("3-1")
    elif fault == "extra_tp":
        command["fallback_receipts"]["7-0"] = envelope
    elif fault == "owner":
        envelope["source_rank"] = 2
    elif fault == "key":
        envelope["key"] = "3-1"
    elif fault in ("version", "generation"):
        original[fault] = 4 if fault == "version" else "stale"
    elif fault == "coverage":
        original["params"] = []
    elif fault == "buffers":
        original["loader_buffers"] = []
    elif fault == "inactive":
        state["active"] = False
    elif fault == "completed":
        state["completed_version"] = 1
    elif fault == "directory":
        state["directory"] += "/elsewhere"
    elif fault == "command_version":
        command["version"] = 4
    elif fault == "ipc":
        state["ipc"] = True
    with pytest.raises(RuntimeError):
        read_receipt(state, command, path, "3-0")


def test_cold_keeps_files_and_rejects_rpc(tmp_path):
    value = {"original": "file receipt"}
    (tmp_path / "0-0.done.json").write_text(json.dumps(value))
    state = dict(version=2, initial_version=2)
    assert read_receipt(state, {}, tmp_path, "0-0") == value
    with pytest.raises(RuntimeError, match="Unexpected"):
        read_receipt(state, {"fallback_receipts": {}}, tmp_path, "0-0")
