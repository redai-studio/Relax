# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("math_verify")
_MODULE_PATH = Path(__file__).resolve().parents[3] / "relax/engine/rewards/openr1mm_accuracy_format.py"
_spec = importlib.util.spec_from_file_location("openr1mm_final_answer", _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
reward = _module.get_openr1mm_rule_based_reward


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
    assert reward(response, label) == 0.0


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
    assert reward(response, label) == 1.0


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
    assert reward(response, "42") == 0.0


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
    assert reward(response, label) == expected


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
    assert _module.MiniR1Format(response, "unused") == expected


def test_openr1mm_combined_reward_keeps_accuracy_and_format_independent():
    combined = _module.get_openr1mm_accuracy_format_reward
    label = "The answer is option B: 26."
    assert combined("reasoning</think><answer>B</answer>", label) == 2.0
    assert combined("reasoning</think><answer>A</answer>", label) == 1.0
    assert combined("B", label) == 1.0
    assert combined("reasoning<answer>A", label) == 0.0
