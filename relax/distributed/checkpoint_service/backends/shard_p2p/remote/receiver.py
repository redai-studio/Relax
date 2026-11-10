# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate incoming shard updates and restore native parameter layouts before
commit."""

import copy
import json
from pathlib import Path


RPC_PLAN_DELIVERY = "plan-rpc-v1"


def engine_plan_envelope(
    command: dict,
    key: str,
    generation: str,
    version: int,
    tp_size: int,
) -> dict:
    """Require this engine's original warm expert plan envelopes."""
    values = command.get("plan_envelopes")
    keys = {f"{command['engine']}-{tp}" for tp in range(tp_size)}
    fields = {"generation", "version", "cached_plan_hash"}
    if (
        not isinstance(values, dict)
        or set(values) != keys
        or key not in keys
        or any(not isinstance(value, dict) or set(value) != fields for value in values.values())
    ):
        raise RuntimeError("Missing or misrouted RPC plan envelopes")
    envelope = values[key]
    if type(envelope["version"]) is not int or envelope["version"] != version or envelope["generation"] != generation:
        raise RuntimeError("Stale plan envelope")
    return envelope


def read_plan_envelope(state: dict, command: dict, directory: Path, key: str) -> dict:
    """A warm RPC session never falls back to a plan file or synthesized
    data."""
    warm = state["version"] != state["initial_version"]
    if not warm:
        if "plan_envelopes" in command:
            raise RuntimeError("Unexpected RPC plan in a cold or file session")
        return json.loads((directory / f"{key}.plan.json").read_text())
    envelope = engine_plan_envelope(
        command,
        key,
        state["generation"],
        state["version"],
        state["receipt_tp_size"],
    )
    plan = state.get("validated_plan")
    plan_hash = state.get("validated_plan_hash")
    validator = state.get("receipt_validator")
    if (
        not state.get("active")
        or state.get("ipc")
        or state.get("key") != key
        or state.get("directory") != str(directory)
        or command.get("version") != state["version"]
        or type(envelope["version"]) is not int
        or envelope["version"] != state["version"]
        or state.get("completed_version") != state["version"] - 1
        or envelope["generation"] != state["generation"]
        or not isinstance(plan_hash, str)
        or not plan_hash
        or envelope["cached_plan_hash"] != plan_hash
        or not isinstance(plan, dict)
        or not state.get("validated_direct_params")
        or getattr(validator, "plan_hash", None) != plan_hash
        or not getattr(validator, "expected", None)
        or plan.get("receipt_delivery") != "commit-rpc-v1"
        or plan.get("delivery") not in (None, "nccl_allgather")
        or plan.get("relay_consumers")
    ):
        raise RuntimeError("Incomplete, stale or unsupported cached RPC plan")
    if state.get("collective_allgather"):
        peers = set(state.get("collective_keys", []))
        if (
            state.get("collective_failed")
            or state.get("collective_last_version") != state["version"] - 1
            or state.get("collective_cached_plan_hash") != plan_hash
            or not peers
            or set(state.get("collective_validators", {})) != peers
            or set(state.get("collective_plan_hashes", {})) != peers
            or any(
                getattr(state["collective_validators"][peer], "plan_hash", None)
                != state["collective_plan_hashes"][peer]
                for peer in peers
            )
            or not state.get("collective_schema_hash")
        ):
            raise RuntimeError("Incomplete cached collective RPC plan")
    # Preserve what arrived, independently of later mutations to command/cache.
    return copy.deepcopy(envelope)
