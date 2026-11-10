# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deliver original fallback completions through the existing commit RPC."""

import copy
import json
from pathlib import Path


RPC_FALLBACK_DELIVERY = "fallback-commit-rpc-v1"
_RECEIPT_FIELDS = {"generation", "version", "params", "digests", "loader_buffers"}
_ENVELOPE_FIELDS = {"delivery", "source_rank", "key", "receipt"}


def checked_names(value: list) -> set[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(name, str) or not name for name in value)
        or value != sorted(set(value))
    ):
        raise RuntimeError("Fallback receipt names must be unique sorted strings")
    return set(value)


def owner_keys(value: list, engine: str, tp_size: int) -> list[str]:
    """Freeze the Actor group-slot ordering; engine identifiers need not be
    contiguous."""
    if (
        type(tp_size) is not int
        or tp_size < 1
        or not isinstance(value, list)
        or not value
        or any(not isinstance(key, str) or "-" not in key for key in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeError("Malformed fallback receiver owner keys")
    engines = [key.rsplit("-", 1)[0] for key in value[::tp_size]]
    expected = [f"{peer}-{tp}" for peer in engines for tp in range(tp_size)]
    if value != expected or len(set(engines)) != len(engines) or engine not in engines:
        raise RuntimeError("Incomplete or misordered fallback receiver ownership")
    return list(value)


def validate_receipt(receipt: dict, schema: dict, version: int, audit: bool) -> None:
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _RECEIPT_FIELDS
        or type(receipt["version"]) is not int
        or receipt["version"] != version
        or receipt["generation"] != schema["generation"]
    ):
        raise RuntimeError("Stale or malformed fallback completion receipt")
    if checked_names(receipt["params"]) != schema["params"]:
        raise RuntimeError("Fallback completion parameter coverage differs")
    if checked_names(receipt["loader_buffers"]) != schema["loader_buffers"]:
        raise RuntimeError("Fallback completion loader-buffer coverage differs")
    digests = receipt["digests"]
    if (
        not isinstance(digests, dict)
        or set(digests) != (schema["params"] if audit else set())
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests.values()
        )
    ):
        raise RuntimeError("Malformed fallback completion digests")


def serialize_receipt(rank: int, key: str, receipt: dict) -> str:
    """Call inside Actor error consensus before entering the object gather."""
    if type(rank) is not int or rank < 0 or not isinstance(key, str):
        raise RuntimeError("Malformed fallback receipt owner")
    return json.dumps(
        dict(delivery=RPC_FALLBACK_DELIVERY, source_rank=rank, key=key, receipt=receipt),
        allow_nan=False,
        separators=(",", ":"),
    )


def route_receipts(payloads: list[str], keys: list[str], schemas: dict, version: int, audit: bool) -> dict:
    """Validate against receiver schemas, never against the supplied
    completions."""
    if len(payloads) != len(keys) or len(set(keys)) != len(keys) or set(schemas) != set(keys):
        raise RuntimeError("Incomplete fallback receipt gather")
    routed = {}
    for rank, (key, payload) in enumerate(zip(keys, payloads, strict=True)):
        envelope = json.loads(payload)
        if (
            not isinstance(envelope, dict)
            or set(envelope) != _ENVELOPE_FIELDS
            or envelope["delivery"] != RPC_FALLBACK_DELIVERY
            or type(envelope["source_rank"]) is not int
            or envelope["source_rank"] != rank
            or envelope["key"] != key
        ):
            raise RuntimeError("Fallback receipt does not match its Actor gather slot")
        validate_receipt(envelope["receipt"], schemas[key], version, audit)
        routed[key] = envelope
    return routed


def read_receipt(state: dict, command: dict, directory: Path, key: str) -> dict:
    """Warm RPC sessions never read a completion file, including on failure."""
    warm = state["version"] != state["initial_version"]
    if not warm:
        if "fallback_receipts" in command:
            raise RuntimeError("Unexpected fallback RPC receipt in a cold or file session")
        return json.loads((directory / f"{key}.done.json").read_text())
    keys = state["fallback_receipt_keys"]
    values = command.get("fallback_receipts")
    expected = {f"{command['engine']}-{tp}" for tp in range(state["receipt_tp_size"])}
    if (
        not state["active"]
        or state.get("ipc")
        or state.get("completed_version") != state["version"] - 1
        or state["version"] != command["version"]
        or state["directory"] != str(directory)
        or state["key"] != key
        or not isinstance(values, dict)
        or set(values) != expected
        or key not in expected
    ):
        raise RuntimeError("Missing or misrouted fallback RPC completions")
    for target, envelope in values.items():
        if (
            not isinstance(envelope, dict)
            or set(envelope) != _ENVELOPE_FIELDS
            or envelope["delivery"] != RPC_FALLBACK_DELIVERY
            or envelope["key"] != target
            or type(envelope["source_rank"]) is not int
            or target not in keys
            or envelope["source_rank"] != keys.index(target)
        ):
            raise RuntimeError("Fallback RPC completion owner changed")
    receipt = values[key]["receipt"]
    schema = dict(
        generation=state["generation"],
        params=set(state["params"]) - state["direct_params"],
        loader_buffers=set(state["fallback_loader_buffers"]),
    )
    validate_receipt(receipt, schema, state["version"], state["audit"])
    return copy.deepcopy(receipt)
