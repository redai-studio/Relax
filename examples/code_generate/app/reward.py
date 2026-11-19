"""Final-answer-only binary reward for thinking models, judged by
SandboxFusion."""

import re
from argparse import Namespace
from typing import TYPE_CHECKING

import httpx


if TYPE_CHECKING:
    from relax.utils.types import Sample


def extract_code(response: str) -> str | None:
    """Keep the baseline's final-answer-only extraction contract."""
    _, closed, answer = response.rpartition("</think>")
    if not closed:
        return None
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", answer, re.DOTALL)
    if not blocks or not blocks[-1].strip():
        return None
    return blocks[-1].strip()


def judge_request(task_id: str, label: str, code: str) -> dict:
    return {
        "dataset": "live_code_bench_v1",
        "id": task_id,
        "completion": f"```python\n{code}\n```",
        "config": {
            "language": "python",
            "run_timeout": 6,
            "provided_data": {
                "id": task_id,
                "labels": {},
                "content": "",
                "test": {"input_output": label},
            },
        },
    }


def outcome_from_result(task_id: str, result: dict) -> float:
    if any(test["exec_info"]["status"] == "SandboxError" for test in result["tests"]):
        raise RuntimeError(f"SandboxFusion infrastructure error for {task_id}")
    return float(result["accepted"])


async def reward(args: Namespace, sample: "Sample") -> float:
    """Score the final code without changing training tokens."""
    code = extract_code(sample.response)
    if code is None:
        return 0.0
    task_id = sample.metadata["task_id"]
    payload = judge_request(task_id, sample.label, code)
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        response = await client.post(f"{args.rm_url.rstrip('/')}/submit", json=payload)
        response.raise_for_status()
        result = response.json()
    return outcome_from_result(task_id, result)
