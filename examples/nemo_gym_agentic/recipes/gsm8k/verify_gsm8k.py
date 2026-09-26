# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Assert that the GSM8K recipe's symbolic verifier distinguishes correct and
wrong answers."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

import httpx


def _response(answer: str) -> dict[str, Any]:
    identifier = uuid.uuid4().hex
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
                        "text": answer,
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


def _verify(client: httpx.Client, resource_url: str, answer: str) -> dict[str, Any]:
    payload = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem. Make sure to put the answer "
                        "(and only answer) inside \\\\boxed{}.\\n\\nWhat is 3 + 5?"
                    ),
                }
            ]
        },
        "question": "What is 3 + 5?",
        "expected_answer": "8",
        "response": _response(answer),
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
        default="http://127.0.0.1:28102",
        help="Base URL of the gsm8k math_with_judge resource server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with httpx.Client(timeout=30.0, trust_env=False) as client:
        correct = _verify(client, args.resource_url, r"\boxed{8}")
        incorrect = _verify(client, args.resource_url, r"\boxed{9}")

    correct_reward = float(correct["reward"])
    incorrect_reward = float(incorrect["reward"])
    if correct_reward != 1.0 or incorrect_reward != 0.0:
        raise RuntimeError(
            f"GSM8K verifier contract failed: correct_reward={correct_reward}, incorrect_reward={incorrect_reward}"
        )

    sys.stdout.write(
        f"GSM8K verifier contract passed: correct_reward={correct_reward} incorrect_reward={incorrect_reward}\n"
    )


if __name__ == "__main__":
    main()
