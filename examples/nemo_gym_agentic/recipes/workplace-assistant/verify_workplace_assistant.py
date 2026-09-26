# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Assert that the Workplace Assistant recipe scores equivalent final state,
not a fixed action trace."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any

import httpx


GROUND_TRUTH = [
    {
        "name": "email_reply_email",
        "arguments": json.dumps(
            {
                "email_id": "00000057",
                "body": "Thanks for the update - I will get back to you tomorrow.",
            }
        ),
    }
]


def _response(actions: list[dict[str, str]]) -> dict[str, Any]:
    identifier = uuid.uuid4().hex
    output = []
    for index, action in enumerate(actions):
        call_id = f"call_{identifier}_{index}"
        output.append(
            {
                "arguments": action["arguments"],
                "call_id": call_id,
                "id": call_id,
                "name": action["name"],
                "status": "completed",
                "type": "function_call",
            }
        )
    return {
        "id": f"resp_{identifier}",
        "created_at": time.time(),
        "model": "verifier-contract-test",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def _verify(client: httpx.Client, resource_url: str, actions: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Reply to carlos's last email about 'Task Update on Develop prototype for report generation'."
                    ),
                }
            ]
        },
        "ground_truth": GROUND_TRUTH,
        "id": 0,
        "category": "workplace_assistant_email",
        "environment_name": "workplace_assistant",
        "response": _response(actions),
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
        default="http://127.0.0.1:29002",
        help="Base URL of the Workplace Assistant resource server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_only_search = {
        "name": "email_search_emails",
        "arguments": json.dumps({"query": "Carlos Task Update"}),
    }
    wrong_reply = {
        "name": "email_reply_email",
        "arguments": json.dumps(
            {
                "email_id": "00000057",
                "body": "This is the wrong reply.",
            }
        ),
    }
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        equivalent = _verify(client, args.resource_url, [read_only_search, *GROUND_TRUTH])
        incorrect = _verify(client, args.resource_url, [wrong_reply])

    equivalent_reward = float(equivalent["reward"])
    incorrect_reward = float(incorrect["reward"])
    if equivalent_reward != 1.0 or incorrect_reward != 0.0:
        raise RuntimeError(
            "Workplace Assistant verifier contract failed: "
            f"equivalent_reward={equivalent_reward} incorrect_reward={incorrect_reward}"
        )

    sys.stdout.write(
        "Workplace Assistant verifier contract passed: "
        f"equivalent_reward={equivalent_reward} incorrect_reward={incorrect_reward}\n"
    )


if __name__ == "__main__":
    main()
