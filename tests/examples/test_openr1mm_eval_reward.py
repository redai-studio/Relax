# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import json
from types import SimpleNamespace

import pytest

from examples.openr1mm.eval_reward import reward, score


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("reasoning 50</think><answer>110</answer>", 0),
        ("<answer>50° + ∠2</answer>", 0),
        ("<answer>50</answer>", 1),
        ("<think>50 is intermediate", 0),
        ("<answer>50</answer><answer>110</answer>", 0),
    ],
)
def test_eval_uses_whole_final_value(response, expected):
    assert score(response, {"answer": "50"}) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("<answer>B</answer>", 1),
        ("<answer>(B)</answer>", 1),
        ("<answer>second choice text</answer>", 1),
        ("<answer>A or B</answer>", 0),
        ("<answer>A</answer>", 0),
        ("<answer>second choice text plus other options</answer>", 0),
        ("<answer>B. second choice text</answer>", 1),
        ("<answer>(B): second choice text</answer>", 1),
        ("<answer>A. second choice text</answer>", 0),
        ("<answer>B. first</answer>", 0),
        ("<answer>Z. second choice text</answer>", 0),
        ("<answer>B. second choice text or A. first</answer>", 0),
    ],
)
def test_eval_choice_is_unambiguous(response, expected):
    assert score(response, {"answer": "B", "choices": ["first", "second choice text"], "correct_index": 1}) == expected


def test_eval_numeric_precision_and_answer_aliases():
    assert score("<answer>1.24</answer>", {"answer": "1.2", "precision": 1}) == 1
    assert score("<answer>1.26</answer>", {"answer": "1.2", "precision": 1}) == 0
    assert score("<answer>1e100</answer>", {"answer": "1.2", "precision": 1}) == 0
    assert score("<answer>1,000</answer>", {"answer": "1000"}) == 1
    assert score("<answer>1,2</answer>", {"answer": "12"}) == 0
    assert score("<answer>1,23</answer>", {"answer": "123"}) == 0
    assert score("<answer>NaN</answer>", {"answer": "1000"}) == 0
    assert score("<answer>alternative</answer>", {"answer": ["primary", "alternative"]}) == 1


def test_eval_route_preserves_train_reward_but_excludes_format_point():
    response = "<think>Compute it.</think><answer>2</answer>"
    train = SimpleNamespace(response=response, label="2", metadata={})
    assert reward(None, train) == 2
    evaluation = SimpleNamespace(
        response=response, label=json.dumps({"answer": "2"}), metadata={"eval_benchmark": "mathvista_testmini"}
    )
    assert reward(None, evaluation) == 1
