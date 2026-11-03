# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json

import pytest

from examples.code_generate.app.reward import extract_code, judge_request, outcome_from_result


def test_final_code_extraction_preserves_baseline_contract():
    assert extract_code("```python\nprint(1)\n```") is None
    assert extract_code("<think>```python\nwrong\n```</think>```python\nprint(2)\n```") == "print(2)"
    assert extract_code("</think>```python\nfirst\n```\n```python\nlast\n```") == "last"
    assert extract_code("</think>```python\n \n```") is None
    payload = judge_request("t", '{"inputs":[]}', "print(1)")
    assert payload["config"]["run_timeout"] == 6
    assert payload["config"]["provided_data"]["test"]["input_output"] == '{"inputs":[]}'
    assert payload["completion"] == "```python\nprint(1)\n```"


def test_infrastructure_failure_is_not_an_incorrect_solution():
    assert outcome_from_result("t", {"accepted": False, "tests": []}) == 0
    with pytest.raises(RuntimeError, match="infrastructure"):
        outcome_from_result("t", {"accepted": False, "tests": [{"exec_info": {"status": "SandboxError"}}]})


@pytest.mark.parametrize("subset", ["primeintellect", "taco", "lcbv5"])
def test_prepared_task_preserves_problem_and_starter(subset):
    from examples.code_generate.prepare_data import convert_row

    row = {
        "problem": "Print the literal text <think>.",
        "tests": json.dumps({"inputs": ["1"], "outputs": ["2"]})
        if subset == "taco"
        else json.dumps([{"input": "1", "output": "2"}]),
        "starter_code": "class Solution:\n    def solve(self, n): ...",
        "metadata": {"func_name": "solve"},
    }
    converted = convert_row(row, subset, 0)
    task = converted["input"][0]["content"]
    assert row["problem"] in task
    if subset == "lcbv5":
        assert row["starter_code"] in task
