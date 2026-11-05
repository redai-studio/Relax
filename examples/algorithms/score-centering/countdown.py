# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Countdown data generation, prompts and arithmetic grading.

Generation, prompt and 0/1 grading follow martin-marek/score-centering,
envs/countdown.py at 1de793964e9cba9487176775fb3e9c7bc8ad7702.
"""

import json
import random
import re
from collections import Counter
from fractions import Fraction
from typing import Any


SYSTEM_PROMPT = (
    "You are a helpful assistant. You first thinks about the reasoning process in the mind "
    "and then provides the user with the answer."
)


def generate(rng: random.Random) -> dict[str, Any]:
    for _ in range(200):
        numbers = [rng.randint(1, 100) for _ in range(4)]
        items = [(Fraction(n), str(n)) for n in numbers]
        while len(items) > 1:
            i, j = rng.sample(range(len(items)), 2)
            av, ae = items[i]
            bv, be = items[j]
            op = rng.choice(("+", "-", "*", "/"))
            if op == "/" and bv == 0:
                op = "+"
            value = {"+": av + bv, "-": av - bv, "*": av * bv, "/": av / bv if bv else av}[op]
            items = [item for k, item in enumerate(items) if k not in (i, j)]
            items.append((value, f"({ae} {op} {be})"))
        value, expression = items[0]
        if value.denominator == 1 and 1 <= value <= 1000:
            return {"numbers": numbers, "target": int(value), "solution": expression[1:-1]}
    raise RuntimeError("No solvable in-range Countdown problem in 200 attempts")


def messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    instruction = (
        f" Using the numbers {problem['numbers']}, create an equation that equals {problem['target']}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
        "for example <answer> (1 + 2) / 3 </answer>."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}]


def grade(completion: str, numbers: list[int], target: int) -> float:
    matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    expression = matches[-1].strip() if matches else ""
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression) or "**" in expression or "//" in expression:
        return 0.0
    if Counter(int(x) for x in re.findall(r"\d+", expression)) != Counter(numbers):
        return 0.0
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # Arithmetic-only grammar, as in the reference.
        return float(abs(result - target) < 1e-9)
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError, TypeError):
        return 0.0


def reward(args: Any, sample: Any, **kwargs: Any) -> float | list[float]:
    if isinstance(sample, list):
        return [reward(args, item, **kwargs) for item in sample]
    label = json.loads(sample.label) if isinstance(sample.label, str) else sample.label
    return grade(sample.response, label["numbers"], label["target"])
