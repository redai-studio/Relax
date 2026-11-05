# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest

from relax.engine.rewards import openr1mm as _module
from relax.engine.rewards.openr1mm import get_openr1mm_benchmark_accuracy as score
from relax.engine.rewards.openr1mm import get_openr1mm_benchmark_reward as reward
from relax.engine.rewards.openr1mm import get_openr1mm_rule_based_reward


@pytest.mark.parametrize(
    "response,label,expected",
    [
        ("<think>Compute ML.</think><answer>36</answer><|im_end|>", "<answer>36</answer>", 1.0),
        ("Compute ML.</think><answer>36</answer><|im_end|>", "36", 1.0),
        ("<answer>35</answer><|im_end|>", "36", 0.0),
        ("<answer>Paris</answer><|im_end|>", "Paris", 1.0),
        ("<answer>London</answer><|im_end|>", "Paris", 0.0),
        (
            "<think> $1,141$ </think><answer> $1,141 </answer><|im_end|>",
            "<think> $1141$ </think><answer>$1141</answer>",
            1.0,
        ),
        (
            "<think> $5.35$ </think><answer> $5.35 </answer><|im_end|>",
            "<think> $5.35$ </think><answer>5.35</answer>",
            1.0,
        ),
        (r"<answer>\frac{1}{2}</answer><|im_end|>", "0.5", 1.0),
        ("42", "42", 1.0),
        ("Paris", "Paris", 1.0),
        (r"\invalid{command}", r"\invalid{command}", 1.0),
        ("", "", 0.0),
        ("<answer> </answer><|im_end|>", "", 0.0),
        ("<answer>36</answer><answer>36</answer><|im_end|>", "36", 0.0),
        ("reasoning</think></think><answer>36</answer>", "36", 0.0),
        ("<answer>36</answer></answer><|im_end|>", "36", 0.0),
        ("<answer>$(1,2)$</answer><|im_end|>", "<answer>$1 < x < 2$</answer>", 1.0),
        ("<answer>$1 < x < 2$</answer><|im_end|>", "<answer>$(1,2)$</answer>", 0.0),
    ],
)
def test_openr1mm_preserves_original_scoring(response: str, label: str, expected: float) -> None:
    assert get_openr1mm_rule_based_reward(response, label) == expected


def test_openr1mm_verifies_original_text_with_reference_as_gold(monkeypatch: pytest.MonkeyPatch) -> None:
    import math_verify

    response = "<think>Calculate.</think><answer>36</answer><|im_end|>"
    label = "<think>Reference reasoning.</think><answer>36</answer>"
    parsed_texts: list[str] = []
    verified_pairs: list[tuple[list[str], list[str]]] = []

    def parse(text: str) -> list[str]:
        parsed_texts.append(text)
        return [text]

    def verify(gold: list[str], target: list[str]) -> bool:
        verified_pairs.append((gold, target))
        return True

    monkeypatch.setattr(math_verify, "parse", parse)
    monkeypatch.setattr(math_verify, "verify", verify)
    assert get_openr1mm_rule_based_reward(response, label) == 1.0
    assert parsed_texts == [response, label]
    assert verified_pairs == [([label], [response])]


def test_openr1mm_keeps_string_fallback_on_symbolic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import math_verify

    def parse(text: str) -> list[str]:
        raise ValueError("symbolic parsing failed")

    monkeypatch.setattr(math_verify, "parse", parse)
    assert get_openr1mm_rule_based_reward("<answer>Paris</answer><|im_end|>", "Paris") == 1.0
    assert get_openr1mm_rule_based_reward("<answer>London</answer><|im_end|>", "Paris") == 0.0


@pytest.mark.parametrize("symbolic_failure", ["no_match", "exception"])
@pytest.mark.parametrize(
    "response,label,expected",
    [
        ("<answer>\nParis\n</answer><|im_end|>", "Paris", 1.0),
        ("<answer>Paris</answer><|im_end|>", "<answer>\nParis\n</answer>", 1.0),
        (
            "<think>Identify the city.</think>\n<answer>\nParis\n</answer><|im_end|>",
            "<think>Reference reasoning.</think>\n<answer>\nParis\n</answer>",
            1.0,
        ),
        ("<answer>\nLondon\n</answer><|im_end|>", "<answer>\nParis\n</answer>", 0.0),
        ("<answer>\n \t\n</answer><|im_end|>", "<answer>\n \n</answer>", 0.0),
        ("<answer>\nParis\n</answer><answer>Paris</answer>", "Paris", 0.0),
        ("<answer>\nNew\nYork\n</answer><|im_end|>", "<answer>New\nYork</answer>", 1.0),
        ("<answer>\nNew\nYork\n</answer><|im_end|>", "New York", 0.0),
    ],
)
def test_openr1mm_string_fallback_supports_multiline_answers(
    monkeypatch: pytest.MonkeyPatch, symbolic_failure: str, response: str, label: str, expected: float
) -> None:
    import math_verify

    def parse(text: str) -> list[str]:
        if symbolic_failure == "exception":
            raise ValueError("symbolic parsing failed")
        return []

    monkeypatch.setattr(math_verify, "parse", parse)
    monkeypatch.setattr(math_verify, "verify", lambda gold, target: False)
    assert get_openr1mm_rule_based_reward(response, label) == expected


@pytest.mark.parametrize(
    "response,label",
    [
        (r"<think>Try \boxed{42}.</think><answer>9</answer>", "42"),
        (r"Try \boxed{42}.</think><answer>9</answer>", "42"),
        ("9", r"<think>Try \boxed{9}.</think><answer>42</answer>"),
        (r"<think>Try \boxed{42}.</think>\boxed{9}", "42"),
        ("<think>42</think><answer>9</answer>", "<think>42</think><answer>42</answer>"),
    ],
)
def test_openr1mm_wrong_final_answer_cannot_match_reasoning(response, label):
    assert _module._accuracy_reward(response, label) == 0.0


@pytest.mark.parametrize(
    "response,label",
    [
        ("42", "42"),
        ("Paris", "Paris"),
        (r"\frac{1}{2}", "0.5"),
        ("<think>Ignore 9.</think><answer>42</answer>", "42"),
        ("Ignore 9.</think><answer>42</answer>", "42"),
        ("<answer>\n42\n</answer>", "<think>Ignore 9.</think><answer>42</answer>"),
        (r"<think>Try 9.</think>The final answer is \boxed{42}.", "42"),
    ],
)
def test_openr1mm_valid_final_answer_scores(response, label):
    assert _module._accuracy_reward(response, label) == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "<think>42",
        "<think><answer>42</answer>",
        "<answer>42",
        "42</answer>",
        "</answer>42<answer>",
        "<answer>42</answer><answer>42</answer>",
        "<answer>42</answer></answer>",
        "<think>42</think></think><answer>42</answer>",
        "<think><think>42</think><answer>42</answer>",
        "</think><think>42<answer>42</answer>",
    ],
)
def test_openr1mm_malformed_or_repeated_blocks_do_not_score(response):
    assert _module._accuracy_reward(response, "42") == 0.0


@pytest.mark.parametrize(
    "response,label,expected",
    [
        ("B", "The answer is option B: 26.", 1.0),
        ("<answer>B</answer>", "<think>Try A.</think><answer>The answer is option B: 26.</answer>", 1.0),
        ("26", "The answer is option B: 26.", 1.0),
        ("Option B: 26", "26", 1.0),
        ("(B)", "choice B", 1.0),
        ("A: 26", "The answer is option B: 26.", 0.0),
        ("B: 99", "The answer is option B: 26.", 0.0),
        ("B or C", "The answer is option B: 26.", 0.0),
        ("A-1", "1", 0.0),
        ("(A", "A", 0.0),
        ("A. foo\nB. bar", "A", 0.0),
        ("A. foo\n(B) bar", "A", 0.0),
        ("(A) 26\n(B) 42", "The answer is option B: 42.", 0.0),
        ("A. 26\nB. 42", "The answer is option A: 26.", 0.0),
        ("<think>B</think><answer>A</answer>", "The answer is option B: 26.", 0.0),
    ],
)
def test_openr1mm_choice_final_answer(response, label, expected):
    assert _module._accuracy_reward(response, label) == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        ("<think>line one\nline two</think>\n\n<answer>\nB\n</answer><|im_end|>", 1.0),
        ("line one\nline two</think>\n<answer>B</answer>", 1.0),
        ("reasoning<answer>B</answer>", 0.0),
        ("reasoning</think><answer>B", 0.0),
        ("reasoning</think><answer>B</answer><answer>C</answer>", 0.0),
        ("reasoning</think></think><answer>B</answer>", 0.0),
        ("<think></think><answer>B</answer>", 0.0),
        ("reasoning</think><answer> </answer>", 0.0),
        ("reasoning</think><answer>B</answer>extra", 0.0),
        ("reasoning</think>extra reasoning<answer>B</answer>", 0.0),
    ],
)
def test_openr1mm_format_handles_prefill_and_multiline(response, expected):
    assert _module._accuracy_format(response, "unused") == expected


def test_openr1mm_combined_reward_keeps_accuracy_and_format_independent():
    combined = _module.get_openr1mm_accuracy_format_reward
    label = "The answer is option B: 26."
    assert combined("reasoning</think><answer>B</answer>", label) == 2.0
    assert combined("reasoning</think><answer>A</answer>", label) == 1.0
    assert combined("B", label) == 1.0
    assert combined("reasoning<answer>A", label) == 0.0


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
