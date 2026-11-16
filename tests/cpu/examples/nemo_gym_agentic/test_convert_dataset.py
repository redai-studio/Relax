# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for converting NeMo Gym datasets into Relax rollout rows."""

import json
from pathlib import Path

import pytest
from relax_nemo_gym_example.scripts.convert_dataset import convert_file, convert_row


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink"])
def test_convert_file_rejects_same_file_without_truncating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    source = tmp_path / "source.jsonl"
    original = b'{"responses_create_params":{"input":[{"role":"user","content":"question"}]}}\n'
    source.write_bytes(original)
    destination = source
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        destination = Path("source.jsonl")
    elif alias == "symlink":
        destination = tmp_path / "alias.jsonl"
        destination.symlink_to(source)
    elif alias == "hardlink":
        destination = tmp_path / "alias.jsonl"
        destination.hardlink_to(source)

    with pytest.raises(ValueError, match="Input and output must refer to different files"):
        convert_file(source, destination)

    assert source.read_bytes() == original
    assert destination.read_bytes() == original


def test_convert_file_writes_separate_output_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    row = {"responses_create_params": {"input": [{"role": "user", "content": "question"}]}, "id": 7}
    original = json.dumps(row) + "\n"
    source.write_text(original, encoding="utf-8")
    destination = tmp_path / "converted" / "output.jsonl"

    assert convert_file(source, destination) == 1
    assert source.read_text(encoding="utf-8") == original
    assert json.loads(destination.read_text(encoding="utf-8")) == convert_row(row)


def test_convert_row_preserves_task_fields_in_metadata() -> None:
    row = {
        "id": 7,
        "responses_create_params": {
            "input": [{"role": "user", "content": "question"}],
            "tools": [{"type": "function", "name": "lookup"}],
            "parallel_tool_calls": False,
            "metadata": {"task_id": "task-7"},
            "temperature": 0.7,
        },
        "expected_answer": "answer",
    }

    converted = convert_row(row)

    assert converted == {
        "input": [{"role": "user", "content": "question"}],
        "metadata": {
            "id": 7,
            "expected_answer": "answer",
            "tools": [{"type": "function", "name": "lookup"}],
            "parallel_tool_calls": False,
            "responses_create_params": {
                "metadata": {"task_id": "task-7"},
                "temperature": 0.7,
            },
        },
    }


def test_convert_row_formats_gsm8k_benchmark_row() -> None:
    row = {
        "question": "What is 3 + 5?",
        "expected_answer": 8,
        "reference_solution": "Three plus five is eight.",
    }

    converted = convert_row(row)

    assert converted == {
        "input": [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem. Make sure to put the answer "
                    "(and only answer) inside \\boxed{}.\n\nWhat is 3 + 5?"
                ),
            }
        ],
        "metadata": {
            "question": "What is 3 + 5?",
            "expected_answer": "8",
            "reference_solution": "Three plus five is eight.",
        },
    }


def test_convert_row_uses_r2e_problem_statement_as_relax_prompt() -> None:
    row = {
        "responses_create_params": {
            "input": [],
            "metadata": {"instance_dict": '{"instance_id":"task-one"}'},
            "temperature": 0.7,
        },
        "problem_statement": "Fix the route parser.",
        "instance_id": "task-one",
    }

    converted = convert_row(row)

    assert converted["input"] == [{"role": "user", "content": "Fix the route parser."}]
    assert converted["metadata"]["responses_create_params"] == {
        "input": [],
        "metadata": {"instance_dict": '{"instance_id":"task-one"}'},
        "temperature": 0.7,
    }


def test_convert_row_preserves_workplace_assistant_verifier_and_tools() -> None:
    row = {
        "id": 11,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Use workplace tools."},
                {"role": "user", "content": "Reply to the latest email."},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "email_reply_email",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
            "parallel_tool_calls": False,
            "temperature": 1.0,
        },
        "ground_truth": [
            {
                "name": "email_reply_email",
                "arguments": '{"email_id":"00000057","body":"done"}',
            }
        ],
        "category": "workplace_assistant_email",
        "environment_name": "workplace_assistant",
    }

    converted = convert_row(row)

    assert converted["input"] == row["responses_create_params"]["input"]
    assert converted["metadata"] == {
        "id": 11,
        "ground_truth": row["ground_truth"],
        "category": "workplace_assistant_email",
        "environment_name": "workplace_assistant",
        "responses_create_params": {"temperature": 1.0},
        "tools": row["responses_create_params"]["tools"],
        "parallel_tool_calls": False,
    }


def test_convert_row_preserves_calendar_verifier_state() -> None:
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Return the calendar as JSON."},
                {"role": "user", "content": "Schedule planning at 10am."},
            ],
            "tools": [],
        },
        "exp_cal_state": {
            "0": {
                "event_id": 0,
                "duration": 60,
                "constraint": "at 10am",
                "min_time": "10:00",
                "max_time": "16:00",
            }
        },
    }

    converted = convert_row(row)

    assert converted == {
        "input": row["responses_create_params"]["input"],
        "metadata": {
            "exp_cal_state": row["exp_cal_state"],
            "tools": [],
        },
    }
