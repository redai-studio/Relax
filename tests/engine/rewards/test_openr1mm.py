# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest

from relax.engine.rewards import openr1mm as _module
from relax.engine.rewards.openr1mm import get_openr1mm_benchmark_accuracy as score
from relax.engine.rewards.openr1mm import get_openr1mm_benchmark_reward as reward
from relax.engine.rewards.openr1mm import get_openr1mm_rule_based_reward


def native_final(answer: str, reasoning: str = "Reasoning mentions <answer> and 42.") -> str:
    return (
        reasoning
        + "<|close|>think<|sep|><|open|>response<|sep|>"
        + answer
        + "<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
    )


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [
        # Real final-answer pairs: rollout/sample 2/355, 10/452, 0/161, 0/297.
        ("A. closely resembles a rectangle", "C", 0.0),
        ("B. Scalene", "C", 0.0),
        ("A. 1/2 * base * height", "A", 1.0),
        ("A. sine", "The correct answer is A, sine.", 1.0),
        ("**Answer: C. Obtuse**", "B", 0.0),
        ("B", "Therefore, the angle is 75.0°, which corresponds to option B.", 1.0),
        ("D", "The answer is option D: 18°", 1.0),
        ("B", "180° (Option B)", 1.0),
        ("B", "The degree measure of angle A is 50°, so the answer is option B.", 1.0),
        ("A or B", "A", 0.0),
        ("The answer is A or b", "A", 0.0),
        ("The answer is a square", "A", 0.0),
        ("The answer is a square", "The answer is a triangle", 0.0),
        ("The answer is a", "A", 1.0),
        ("\nB\n", "B", 1.0),
        ("1/2", "0.5", 1.0),
        (r"\frac{1}{2}", "0.5", 1.0),
        (r"\boxed{\frac{1}{2}}", "0.5", 1.0),
        ("$1,000$", "$1000$", 1.0),
        ("$1,234.56$", "$1234.56$", 1.0),
        (r"$\frac{x}{y}$", "$x/y$", 1.0),
        (r"$\sin(\pi/6)$", "$0.5$", 1.0),
        (r"$\pi$", r"$2*\pi/2$", 1.0),
        ("$(1,2)$", "$(1, 2)$", 1.0),
        ("$4,7$", "$4$", 0.0),
        (r"$\frac{x}{y}$", "$x*y$", 0.0),
        ("**23**", "23", 1.0),
        ("2**3", "23", 0.0),
        ("**2**3**", "23", 0.0),
        ("2**3 + 4**5", "23 + 45", 0.0),
        ("(2)**3 + (4)**5", "(2)3 + (4)5", 0.0),
        ("7.0", "7", 1.0),
        ("60°", "60", 1.0),
        (r"60^\circ", "60", 1.0),
        (r"\boxed{x=60^\circ}", "60", 1.0),
        (r"4\,\mathrm{cm}", "4", 1.0),
        (r"18\pi", "18", 0.0),
        ("not $4$", "4", 0.0),
        ("25%", "2013, 2014, and 2015 with a percentage above 25%", 0.0),
        ("2", "Total surface area is 2(A_triangle) + 3(A_rectangle)", 0.0),
        ("Option A is incorrect; option B is correct", "A", 0.0),
        (r"A. option B also works, \boxed{4}", "4", 0.0),
        ("A. is wrong; the answer is B", "A", 0.0),
        (r"Not \boxed{4}, the answer is 5", "4", 0.0),
        (r"\boxed{4} is incorrect; answer 5", "4", 0.0),
        (r"\not\boxed{4}", "4", 0.0),
        (r"\not\boxed 4", "4", 0.0),
        (r"\boxed{4}\neq 4", "4", 0.0),
        (r"\boxed{4} \lor 5", "4", 0.0),
        (r"It is not $$4$$", "4", 0.0),
        (r"It is not \[\boxed{4}\]", "4", 0.0),
        ("Small intestine", "small intestine", 1.0),
        ("There are **3 signs** visible in total. The speed sign reads 20 mph.", "3", 1.0),
        ("There are **two cows** in the water.", "2", 1.0),
        ("There are 3 apples and 4 pears.", "3", 0.0),
        ("There are 3.5 apples.", "3", 0.0),
        ("There are 4 red hearts. There are 3 blue hearts. The total is 7.", "4", 0.0),
        ("There are 4 hearts. Actually, there are 5 hearts.", "4", 0.0),
        ("There are 4 red hearts. The answer is 7.", "4", 0.0),
        ("There are 4 red hearts. The answer is 7.", "7", 1.0),
        ("There are 4 red hearts. The answer is 4.", "4", 1.0),
        ("There are 4 red hearts. The total number of hearts is 7.", "4", 0.0),
        ("There are 4 red hearts. The total number of hearts is 7.", "7", 1.0),
        ("There are 4 red hearts. $$7$$", "4", 0.0),
        ("There are 4 red hearts. $$7$$", "7", 1.0),
        ("The length of **BC is 4**.", "4", 1.0),
        (r"Solving gives: $$BC = 4$$", "4", 1.0),
    ],
)
def test_openr1mm_compares_final_answers(prediction: str, reference: str, expected: float) -> None:
    label = f"<think>Intermediate calculation: $42$.</think><answer>{reference}</answer>"
    for response in (
        f"<think>Intermediate calculation: $42$.</think><answer>{prediction}</answer>",
        native_final(prediction),
        native_final(f"<answer>{prediction}</answer>"),
        native_final(f"<think>Intermediate calculation: $42$.</think><answer>{prediction}</answer>"),
    ):
        assert get_openr1mm_rule_based_reward(response, label) == expected


@pytest.mark.parametrize(
    "response",
    [
        "Reasoning ends in $4$",  # truncated reasoning, no final channel
        r"Reasoning ends in \boxed{4}",
        "<think>I found 4, but must still check",
        "<think><answer>4</answer>",
        "<think>4</think><answer>4",
        "<answer>4</answer><answer>5</answer>",
        "<answer>4</answer></answer>",
        "<answer> </answer>",
        "<answer>4</answer> actually 5",
        native_final("<answer>4</answer><answer>5</answer>"),
        native_final("4") + native_final("5"),
        "<|close|>think<|sep|><|open|>response<|sep|>4",  # incomplete native final
        native_final("4") + "actually 5",
    ],
)
def test_openr1mm_rejects_unfinished_or_ambiguous_answers(response: str) -> None:
    assert get_openr1mm_rule_based_reward(response, "<answer>4</answer>") == 0.0


def test_openr1mm_supports_native_answer_closure() -> None:
    response = native_final("<answer>4").replace("<|close|>response", "<|close|>answer")
    assert get_openr1mm_rule_based_reward(response, "4") == 1.0


def test_openr1mm_ignores_reference_and_prediction_reasoning() -> None:
    assert (
        get_openr1mm_rule_based_reward("<think>$4$</think><answer>5</answer>", "<think>$5$</think><answer>4</answer>")
        == 0.0
    )


def test_openr1mm_passes_gold_before_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    import math_verify

    def parse(text: str, **kwargs) -> list[str]:
        assert kwargs["fallback_mode"] == "no_fallback"
        return [text]

    def verify(gold: list[str], target: list[str]) -> bool:
        assert gold == ["$0.5$"]
        assert target == ["$1/2$"]
        return True

    monkeypatch.setattr(math_verify, "parse", parse)
    monkeypatch.setattr(math_verify, "verify", verify)
    assert get_openr1mm_rule_based_reward("<answer>1/2</answer>", "0.5") == 1.0


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
