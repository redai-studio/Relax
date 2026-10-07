# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Strict local benchmark accuracy; not the Open-R1 GPT-4o judge protocol."""

import ast
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any


def final_answer(response: str) -> str | None:
    text = response.replace("<|im_end|>", "").strip()
    if text.count("<answer>") == text.count("</answer>") == 1:
        match = re.search(r"<answer>(.*?)</answer>\s*$", text, re.S)
        return match[1].strip() if match else None
    if "<answer>" in text or "</answer>" in text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return None
    return text or None


def normalized(text: Any) -> str:
    return " ".join(str(text).strip().casefold().split())


def number(text: str) -> Decimal | None:
    text = text.strip().strip("$").replace("−", "-")
    if "," in text:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
            return None
        text = text.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return None
    try:
        result = Decimal(text)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def score(response: str, spec: dict[str, Any]) -> float:
    answer = final_answer(response)
    if answer is None:
        return 0.0
    gold = spec["answer"]
    choices = spec.get("choices") or []
    if choices:
        match = re.fullmatch(r"(?:[Tt]he answer is\s+|[Oo]ption\s+)?\(?([A-Z])\)?[.]?", answer)
        if match:
            return float(ord(match[1]) - ord("A") == spec["correct_index"])
        match = re.fullmatch(r"\(?([A-Z])\)?[.:：、)]\s*(.+)", answer, re.S)
        if match:
            index = ord(match[1]) - ord("A")
            return float(
                index == spec["correct_index"] and normalized(match[2]) == normalized(choices[spec["correct_index"]])
            )
        return float(normalized(answer) == normalized(choices[spec["correct_index"]]))
    candidates = gold if isinstance(gold, list) else [gold]
    for expected in candidates:
        if normalized(answer) == normalized(expected):
            return 1.0
        a, b = number(answer), number(str(expected))
        if a is not None and b is not None:
            precision = spec.get("precision")
            if precision is not None:
                try:
                    if round(a, int(precision)) == round(b, int(precision)):
                        return 1.0
                except InvalidOperation:
                    # Finite model outputs can exceed Decimal's rounding context.
                    continue
            elif a == b:
                return 1.0
        if spec.get("answer_type") == "list":
            try:
                if ast.literal_eval(answer) == ast.literal_eval(str(expected)):
                    return 1.0
            except (ValueError, SyntaxError):
                pass
    return 0.0


def reward(args: Any, sample: Any, **kwargs: Any) -> float:
    metadata = sample.metadata or {}
    if metadata.get("eval_benchmark") in ("mathvista_testmini", "mmmu_validation"):
        return score(sample.response, json.loads(sample.label))
    # Preserve the active training reward exactly; no format point in benchmark eval.
    from relax.engine.rewards.openr1mm_accuracy_format import get_openr1mm_accuracy_format_reward

    return get_openr1mm_accuracy_format_reward(sample.response, sample.label)
