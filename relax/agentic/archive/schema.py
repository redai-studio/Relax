# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Task-independent event schema, validation and deterministic path
identity."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
ASSET_PREFIX = "asset://"


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def asset_ids(value: Any) -> set[str]:
    if isinstance(value, str) and value.startswith(ASSET_PREFIX):
        return {value[len(ASSET_PREFIX) :]}
    if isinstance(value, dict):
        return set().union(*(asset_ids(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(asset_ids(v) for v in value)) if value else set()
    return set()


def asset_digest(key: str) -> str:
    """Validate an archive-relative image path and return its content hash."""
    if re.fullmatch(r"[a-f0-9]{64}", key):
        return key
    parts = key.split("/")
    if (
        len(parts) not in (4, 5)
        or any(p in ("", ".", "..") for p in parts)
        or not re.fullmatch(r"[A-Za-z0-9_.%\-]+", parts[0])
        or not re.fullmatch(r"[a-f0-9]{32}", parts[1])
        or (len(parts) == 4 and parts[2] != "screenshots")
        or (
            len(parts) == 5
            and (parts[2] not in ("model_inputs", "events") or not re.fullmatch(r"[0-9]{6,}", parts[3]))
        )
    ):
        raise ValueError("invalid asset reference")
    match = re.fullmatch(r"[0-9]{6,}-([a-f0-9]{64})\.[a-z0-9]{1,10}", parts[-1])
    if match is None:
        raise ValueError("invalid asset reference")
    return match[1]


def _content_identity(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(ASSET_PREFIX):
        return ASSET_PREFIX + asset_digest(value[len(ASSET_PREFIX) :])
    if isinstance(value, dict):
        return {k: _content_identity(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_content_identity(v) for v in value]
    return value


def materialize(events: list[dict]) -> dict:
    if not events or events[0].get("kind") != "trajectory":
        raise ValueError("missing trajectory header")
    trajectory = dict(events[0]["data"])
    trajectory.update(observations=[], model_calls=[], actions=[], outcome=None)
    indexes: dict[str, dict] = {"model_call": {}, "action": {}}
    for event in events[1:]:
        kind, data = event["kind"], dict(event["data"])
        if kind == "observation":
            trajectory["observations"].append(data)
        elif kind in indexes:
            key = "call_id" if kind == "model_call" else "action_id"
            if data[key] in indexes[kind]:
                raise ValueError("duplicate event identity")
            indexes[kind][data[key]] = data
            trajectory["model_calls" if kind == "model_call" else "actions"].append(data)
        elif kind in ("call_result", "call_parse", "action_result"):
            target, key = ("action", "action_id") if kind == "action_result" else ("model_call", "call_id")
            indexes[target][data[key]].update(data)
        elif kind == "outcome":
            trajectory.update(
                outcome=data, ended_at=data.get("ended_at"), termination_reason=data.get("termination_reason")
            )
        elif kind == "commit":
            trajectory["integrity"] = data["integrity"]
        elif kind == "incomplete":
            trajectory.setdefault("incomplete_reasons", []).append(data["reason"])
        else:
            raise ValueError(f"unknown event kind: {kind}")
    for call in trajectory["model_calls"]:
        call["model_input_asset_ids"] = sorted(asset_ids(call["request"]))
        call["action_ids"] = [a["action_id"] for a in trajectory["actions"] if a.get("call_id") == call["call_id"]]
    return trajectory


def validate(trajectory: dict) -> None:
    for key in ("run_id", "trajectory_id", "session_id", "task_source", "task_id", "task_version", "instruction"):
        if not trajectory.get(key):
            raise ValueError(f"missing {key}")
    if trajectory.get("schema_version") != SCHEMA_VERSION or trajectory.get("incomplete_reasons"):
        raise ValueError("incomplete trajectory")
    if not trajectory.get("started_at") or not (trajectory.get("task_config") or trajectory.get("task_ref")):
        raise ValueError("missing task provenance or start time")
    for key in ("model", "policy_version", "agent_config", "environment_version"):
        if key not in trajectory:
            raise ValueError(f"missing provenance: {key}")
    outcome = trajectory.get("outcome") or {}
    if "raw_score" not in outcome or "evaluation_error" not in outcome:
        raise ValueError("missing evaluation evidence")
    if outcome.get("status") != "success" or outcome.get("evaluation_error"):
        raise ValueError("outcome is not confirmed success")
    if not outcome.get("basis") or not trajectory.get("ended_at") or not trajectory.get("termination_reason"):
        raise ValueError("missing outcome evidence")
    observations = {o["observation_id"]: o for o in trajectory["observations"]}
    calls = {c["call_id"]: c for c in trajectory["model_calls"]}
    if not observations or not calls or not trajectory["actions"]:
        raise ValueError("missing observations, calls or actions")
    if len(observations) != len(trajectory["observations"]):
        raise ValueError("duplicate observation identity")
    for obs in observations.values():
        if (
            not obs.get("valid")
            or not isinstance(obs.get("screenshot"), str)
            or not obs["screenshot"].startswith(ASSET_PREFIX)
        ):
            raise ValueError("missing or placeholder screenshot")
    for call in calls.values():
        if not call.get("request", {}).get("messages") or "response" not in call:
            raise ValueError("incomplete model call")
        if call.get("parse_status") not in ("parsed", "failed", "not_parsed"):
            raise ValueError("missing parse status")
        if not call.get("observation_ids") or any(o not in observations for o in call["observation_ids"]):
            raise ValueError("broken model observation link")
        if call.get("selected"):
            reply = call.get("assistant_message") or {}
            response = call.get("response") or {}
            if not isinstance(reply, dict) or reply.get("role") != "assistant":
                raise ValueError("selected call has no assistant reply")
            if isinstance(response, dict) and (response.get("error") or response.get("http_status", 200) != 200):
                raise ValueError("selected call has failed response")
        if call.get("selected") and (call["parse_status"] != "parsed" or not call["action_ids"]):
            raise ValueError("selected call has no parsed action")
    if [a.get("order") for a in trajectory["actions"]] != list(range(len(trajectory["actions"]))):
        raise ValueError("invalid action ordering")
    previous = None
    for action in trajectory["actions"]:
        cid = action.get("call_id")
        if cid not in calls or (not calls[cid].get("selected") and action.get("origin") != "harness"):
            raise ValueError("broken action call link")
        if action.get("step") != calls[cid].get("step"):
            raise ValueError("action and call step disagree")
        if not action.get("type") or "payload" not in action:
            raise ValueError("missing action semantics")
        if previous is not None and previous != action.get("before_observation_id"):
            raise ValueError("broken observation chain")
        previous = action.get("after_observation_id")
        if action.get("status") not in ("success", "failure", "unknown"):
            raise ValueError("unfinished action")
        if any(action.get(k) not in observations for k in ("before_observation_id", "after_observation_id")):
            raise ValueError("broken action observation link")
    if outcome.get("final_observation_id") not in observations:
        raise ValueError("missing final observation")


def task_key(trajectory: dict) -> tuple[str, str, str]:
    return tuple(trajectory[k] for k in ("task_source", "task_version", "task_id"))


def path_fingerprint(trajectory: dict) -> str:
    # IDs, timestamps, scores and response prose are not behavior. Preserve actual
    # action semantics, order and observed states; retries do not create new paths.
    observations = {o["observation_id"]: o for o in trajectory["observations"]}
    path = [
        {
            "type": a["type"],
            "payload": a["payload"],
            "before": observations[a["before_observation_id"]]["screenshot"],
            "after": observations[a["after_observation_id"]]["screenshot"],
            "before_fields": observations[a["before_observation_id"]].get("fields", {}),
            "after_fields": observations[a["after_observation_id"]].get("fields", {}),
        }
        for a in trajectory["actions"]
    ]
    return hashlib.sha256(encode([task_key(trajectory), _content_identity(path)])).hexdigest()


def quality(trajectory: dict) -> tuple:
    # Lower is better. Unknown execution evidence ranks below confirmed success.
    return (
        sum(a["status"] == "failure" or bool(a.get("error")) for a in trajectory["actions"]),
        len({a["step"] for a in trajectory["actions"]}),
        len(trajectory["actions"]),
        len(trajectory["model_calls"]),
        sum(a["status"] == "unknown" for a in trajectory["actions"]),
        trajectory["trajectory_id"],
    )
