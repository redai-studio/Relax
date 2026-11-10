# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Route original source completions without weakening the remote shard commit
gate."""

import json
from pathlib import Path

from .plan import RPC_RECEIPT_DELIVERY, ReceiptValidator


def index_receipts(values: list[dict]) -> dict[int, dict]:
    """Preserve originals and reject duplicate owners before indexing."""
    if not isinstance(values, list) or any(
        not isinstance(value, dict) or type(value.get("source_rank")) is not int for value in values
    ):
        raise RuntimeError("Malformed RPC source receipts")
    receipts = {value["source_rank"]: value for value in values}
    if len(receipts) != len(values):
        raise RuntimeError("Duplicate RPC source receipt")
    return receipts


def route_receipts(
    payloads: list[str], validators: dict[str, ReceiptValidator], generations: dict[str, str], version: int
) -> dict[str, list[dict]]:
    """Gather slots are group ranks; never infer a missing completion from a
    plan."""
    routed = {key: [] for key in validators}
    for rank, payload in enumerate(payloads):
        local = json.loads(payload)
        expected = {key for key, validator in validators.items() if rank in validator.expected}
        if not isinstance(local, dict) or set(local) != expected:
            raise RuntimeError("RPC owner supplied incomplete or misrouted target receipts")
        for key, receipt in local.items():
            if not isinstance(receipt, dict) or type(receipt.get("source_rank")) is not int:
                raise RuntimeError("Malformed RPC owner receipt")
            if receipt["source_rank"] != rank:
                raise RuntimeError("RPC source receipt does not match its gather slot")
            routed[key].append(receipt)
    for key, validator in validators.items():
        validator.validate(index_receipts(routed[key]), generations[key], version)
    return routed


def read_owner_receipts(
    command: dict,
    directory: Path,
    key: str,
    validator,
    generation: str,
    version: int,
    tp_size: int,
    *,
    warm_rpc: bool,
) -> dict[int, dict]:
    """Read original completions with identical routing and generation
    checks."""
    field = "source_receipts"
    if warm_rpc:
        envelope = command.get(field)
        targets = {f"{command['engine']}-{tp}" for tp in range(tp_size)}
        if not isinstance(envelope, dict) or set(envelope) != targets or key not in envelope:
            raise RuntimeError("Missing or misrouted commit RPC owner receipts")
        receipts = index_receipts(envelope[key])
    else:
        if field in command:
            raise RuntimeError("Unexpected commit RPC receipts in a cold/file session")
        stem = "source"
        receipts = {
            rank: json.loads((directory / f"{key}.{stem}-{rank}.done.json").read_text()) for rank in validator.expected
        }
    validator.validate(receipts, generation, version)
    return receipts


def read_commit_receipts(state: dict, command: dict, directory: Path, key: str) -> dict[int, dict]:
    """Warm RPC sessions fail closed; cold and default sessions retain
    files."""
    if state["plan"].get("receipt_delivery") != RPC_RECEIPT_DELIVERY:
        raise RuntimeError("Expert shard plan lacks its required RPC receipt protocol")
    return read_owner_receipts(
        command,
        directory,
        key,
        state["receipt_validator"],
        state["generation"],
        state["version"],
        state["receipt_tp_size"],
        warm_rpc=state["version"] != state["initial_version"],
    )
