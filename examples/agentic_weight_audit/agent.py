# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-turn text agent for the real-model smoke recipe."""

import json
import os
import time
from pathlib import Path

import httpx


def main() -> None:
    payload = json.loads(Path(os.environ["RELAX_INPUT_JSON"]).read_text())
    messages = payload["messages"]
    metadata = payload.get("metadata") or {}
    url = os.environ["RELAX_BASE_URL"].rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['RELAX_SESSION_ID']}"}
    timeout = httpx.Timeout(connect=30, read=None, write=30, pool=30)
    completions = []
    with httpx.Client(timeout=timeout, headers=headers) as client:
        for turn in range(2):
            response = client.post(
                url,
                json={"model": "smoke", "messages": messages, "max_completion_tokens": 48, "stream": False},
            )
            response.raise_for_status()
            completion = response.json()
            messages.append(completion["choices"][0]["message"])
            completions.append(completion)
            if turn == 0:
                # Optional deterministic tool hold for the partial-rollout run.
                # Train dumps open the tool holds; the runtime's generation
                # gate then keeps the next request behind weight publication.
                control_dir = os.environ.get("TASK15_CONTROL_DIR")
                wait_for = metadata.get("wait_for")
                if control_dir and metadata.get("signal_first_turn"):
                    (Path(control_dir) / f"long-ready.{os.environ['RELAX_SESSION_ID']}").touch()
                if wait_for and control_dir:
                    release = Path(control_dir) / wait_for
                    deadline = time.monotonic() + 1800
                    while (
                        len(list(Path(control_dir).glob("long-ready.*"))) < metadata.get("first_turn_barrier_size", 2)
                        if wait_for == "long-ready"
                        else not release.exists()
                    ):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Tool wait was not released by the validation driver")
                        time.sleep(0.2)
                value = int(metadata.get("value", 1)) * 2
                messages.append({"role": "user", "content": f"The doubling tool returned {value}. State this result."})
    result = {
        "reward": float(sum(c["usage"]["completion_tokens"] for c in completions)),
        "metadata": {"audit_smoke": True, "tool_result": value, "model_turns": 2},
    }
    output = Path(os.environ["RELAX_OUTPUT_JSON"])
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result))
    temporary.replace(output)


if __name__ == "__main__":
    main()
