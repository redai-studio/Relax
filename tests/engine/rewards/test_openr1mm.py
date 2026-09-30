# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

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
