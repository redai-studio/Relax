# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Linear, two-turn fixture for a dedicated immutable LoRA acceptance
deployment.

The shared barrier is test input, not a model-visible instruction. The agent
replays the complete assistant message (including reasoning/tool fields).
"""

import json
import os
import time
from pathlib import Path

import httpx


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


def main() -> None:
    data = json.loads(Path(os.environ["RELAX_INPUT_JSON"]).read_text())
    case = data["metadata"]["lora_verification"]
    directory = Path(case["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    messages = list(data["messages"])
    deadline = time.monotonic() + case["timeout_seconds"]
    base = os.environ["RELAX_BASE_URL"].rstrip("/")
    with httpx.Client(timeout=httpx.Timeout(10, read=None), trust_env=False) as client:
        for turn in range(2):
            response = client.post(
                base + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["RELAX_SESSION_ID"]},
                json={
                    "model": "policy",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": case.get("max_tokens", 32),
                    "logprobs": True,
                },
            )
            response.raise_for_status()
            result = response.json()
            messages.append(result["choices"][0]["message"])
            write_json(
                directory / f"turn-{turn}.json",
                {"session_id": os.environ["RELAX_SESSION_ID"], "response": result, "completed_at": time.time()},
            )
            if turn == 0:
                while not (directory / "continue").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("fixture barrier was not released")
                    time.sleep(0.05)
                messages.append({"role": "user", "content": "Continue with one more concise explanation."})
    write_json(Path(os.environ["RELAX_OUTPUT_JSON"]), {"reward": 0.0})


if __name__ == "__main__":
    main()
