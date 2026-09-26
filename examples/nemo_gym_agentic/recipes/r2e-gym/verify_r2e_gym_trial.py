# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Verify one prepared R2E-Gym recipe SIF with the upstream golden-patch
evaluator."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


TERMINAL_STATUSES = {"completed", "truncated", "aborted", "failed"}


def _read_task(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        task = json.loads(handle.readline())
    metadata = task.get("responses_create_params", {}).get("metadata", {})
    if not isinstance(metadata, dict) or "instance_dict" not in metadata:
        raise ValueError("Expected a prepared R2E-Gym JSONL row")
    return task


def _trial_payload(
    task: dict[str, Any],
    *,
    request_id: str,
    callback_base_url: str,
    deadline_s: int,
) -> dict[str, Any]:
    return {
        "protocol_version": "relax-nemo-gym/v1",
        "request_id": request_id,
        "session": {
            "session_id": uuid.uuid4().hex,
            "group_id": "r2e-gym-golden",
            "rollout_mode": "eval",
            "attempt": 1,
        },
        "environment": {
            "name": "r2e_gym",
            "config": "r2e-gym-v1",
            "task": task,
        },
        "model_endpoint": {
            "base_url": callback_base_url.rstrip("/"),
            "api_key": uuid.uuid4().hex,
            "model": "unused-in-golden-verification",
        },
        "generation": {},
        "interrupt_policy": "protected",
        "deadline_s": deadline_s,
        "lease_s": 120,
        "metadata": {"integration_test": "r2e-gym-golden"},
    }


def _wait_for_trial(
    client: httpx.Client,
    gateway_url: str,
    request_id: str,
    *,
    deadline_s: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + deadline_s
    next_renewal = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = client.get(f"{gateway_url}/v1/trials/{request_id}")
        response.raise_for_status()
        result = response.json()
        if result["status"] in TERMINAL_STATUSES:
            return result
        if time.monotonic() >= next_renewal:
            renewal = client.post(f"{gateway_url}/v1/trials/{request_id}/renew")
            renewal.raise_for_status()
            next_renewal = time.monotonic() + 60
        time.sleep(2)
    raise TimeoutError(f"R2E-Gym golden trial did not finish within {deadline_s} seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:28100")
    parser.add_argument("--callback-base-url", required=True)
    parser.add_argument("--task-jsonl", type=Path, required=True)
    parser.add_argument("--deadline-s", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = _read_task(args.task_jsonl)
    request_id = f"r2e-golden-{uuid.uuid4().hex}"
    gateway_url = args.gateway_url.rstrip("/")
    payload = _trial_payload(
        task,
        request_id=request_id,
        callback_base_url=args.callback_base_url,
        deadline_s=args.deadline_s,
    )
    with httpx.Client(timeout=15, trust_env=False) as client:
        created = client.post(f"{gateway_url}/v1/trials", json=payload)
        created.raise_for_status()
        result = _wait_for_trial(
            client,
            gateway_url,
            request_id,
            deadline_s=args.deadline_s,
        )
        health = client.get(f"{gateway_url}/readyz")
        health.raise_for_status()

    if result["status"] != "completed" or float(result["reward"]) != 1.0:
        raise RuntimeError(f"R2E-Gym golden verification failed: {json.dumps(result, sort_keys=True)}")
    if health.json().get("active_trials") != 0:
        raise RuntimeError(f"Gateway retained an active trial: {health.json()}")

    instance_id = task["responses_create_params"]["metadata"]["instance_id"]
    print(f"R2E-Gym golden verification passed: instance_id={instance_id} reward={result['reward']}")


if __name__ == "__main__":
    main()
