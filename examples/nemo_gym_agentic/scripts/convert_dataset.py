# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Convert NeMo Gym task rows into Relax agentic rollout rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GSM8K_USER_PROMPT = """Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.

{question}"""


def convert_row(row: dict[str, Any]) -> dict[str, Any]:
    responses_params = row.get("responses_create_params")
    is_gsm8k_row = responses_params is None
    if responses_params is None:
        question = row.get("question")
        if not isinstance(question, str) or "expected_answer" not in row:
            raise ValueError(
                "NeMo Gym row must contain responses_create_params, or GSM8K question and expected_answer fields"
            )
        responses_params = {
            "input": [
                {
                    "role": "user",
                    "content": GSM8K_USER_PROMPT.format(question=question),
                }
            ]
        }
    elif not isinstance(responses_params, dict):
        raise ValueError("responses_create_params must be an object")

    messages = responses_params.get("input")
    if not isinstance(messages, list):
        raise ValueError("responses_create_params.input must be a messages array")

    metadata = {key: value for key, value in row.items() if key != "responses_create_params"}
    preserved_responses_params = {
        key: value
        for key, value in responses_params.items()
        if key not in {"input", "tools", "parallel_tool_calls", "instructions"}
    }
    relax_messages = messages
    if not messages and isinstance(row.get("problem_statement"), str) and row["problem_statement"].strip():
        relax_messages = [{"role": "user", "content": row["problem_statement"]}]
        preserved_responses_params["input"] = []
    if preserved_responses_params:
        metadata["responses_create_params"] = preserved_responses_params
    if is_gsm8k_row:
        # The pinned Gym prepare script writes numeric GSM8K answers, while
        # LibraryJudgeMathRunRequest intentionally validates this field as a
        # string before passing it to math-verify.
        metadata["expected_answer"] = str(metadata["expected_answer"])
    for key in ("tools", "parallel_tool_calls"):
        if key in responses_params:
            metadata[key] = responses_params[key]
    if instructions := responses_params.get("instructions"):
        metadata["developer_message"] = instructions

    return {
        "input": relax_messages,
        "metadata": metadata,
    }


def convert_file(input_path: Path, output_path: Path, *, limit: int | None = None) -> int:
    if input_path.resolve() == output_path.resolve() or (output_path.exists() and input_path.samefile(output_path)):
        raise ValueError(
            f"Input and output must refer to different files: {input_path} -> {output_path}. "
            "Set NEMO_GYM_PROMPT_DATA to a different path from NEMO_GYM_SOURCE_DATA."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, start=1):
            if limit is not None and converted >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {input_path}:{line_number}")
            destination.write(json.dumps(convert_row(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            converted += 1
    if converted == 0:
        raise ValueError(f"No rows converted from {input_path}")
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be greater than zero")
    converted = convert_file(args.input, args.output, limit=args.limit)
    print(f"Converted {converted} NeMo Gym row(s) to {args.output}")


if __name__ == "__main__":
    main()
