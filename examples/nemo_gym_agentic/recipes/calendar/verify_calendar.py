# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Assert that the Calendar verifier distinguishes a valid schedule from an
invalid one."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

import httpx


def _response(start_time: str) -> dict[str, Any]:
    identifier = uuid.uuid4().hex
    answer = [
        {
            "event_id": 0,
            "event_name": "Planning",
            "start_time": start_time,
            "duration": 60,
        }
    ]
    return {
        "id": f"resp_{identifier}",
        "created_at": time.time(),
        "model": "verifier-contract-test",
        "object": "response",
        "output": [
            {
                "id": f"msg_{identifier}",
                "content": [
                    {
                        "annotations": [],
                        "text": json.dumps(answer),
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _verify(client: httpx.Client, resource_url: str, start_time: str) -> dict[str, Any]:
    payload = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Return the calendar as JSON."},
                {"role": "user", "content": "Schedule Planning at 10am for one hour."},
            ],
            "tools": [],
        },
        "exp_cal_state": {
            "0": {
                "event_id": 0,
                "event_name": "Planning",
                "duration": 60,
                "constraint": "at 10am",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        },
        "response": _response(start_time),
    }
    response = client.post(f"{resource_url.rstrip('/')}/verify", json=payload)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError("Verifier response must be a JSON object")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource-url",
        default="http://127.0.0.1:29102",
        help="Base URL of the Calendar resource server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        correct = _verify(client, args.resource_url, "10:00")
        incorrect = _verify(client, args.resource_url, "11:00")

    correct_reward = float(correct["reward"])
    incorrect_reward = float(incorrect["reward"])
    if correct_reward != 1.0 or incorrect_reward != 0.0:
        raise RuntimeError(
            f"Calendar verifier contract failed: correct_reward={correct_reward} incorrect_reward={incorrect_reward}"
        )

    sys.stdout.write(
        f"Calendar verifier contract passed: correct_reward={correct_reward} incorrect_reward={incorrect_reward}\n"
    )


if __name__ == "__main__":
    main()
