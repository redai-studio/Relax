# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def get_openr1mm_rule_based_reward(response: str, label: str) -> float:
    """Preserve the original symbolic verification and XML string fallback."""
    if response.count("</think>") > 1 or response.count("<answer>") > 1 or response.count("</answer>") > 1:
        return 0.0

    from math_verify import parse, verify

    # -------------------------
    # 1. symbolic verification
    # -------------------------
    try:
        answer = parse(response)
        solution = parse(label)
        # math_verify is asymmetric: the reference must be its gold argument.
        if verify(solution, answer):
            return 1.0
    except Exception:
        logger.exception("OpenR1-MM symbolic verification failed")

    # -------------------------
    # 2. string-based matching
    # -------------------------
    try:
        # extract ground truth
        sol_match = _ANSWER.search(label)
        ground_truth = sol_match.group(1).strip() if sol_match else label.strip()
        # extract model answer
        content_match = _ANSWER.search(response)
        student_answer = content_match.group(1).strip() if content_match else response.strip()

        return float(bool(student_answer) and student_answer == ground_truth)
    except Exception:
        logger.exception("OpenR1-MM answer matching failed")
        return 0.0


def MiniR1Format(response, label):
    try:
        completion = ensure_think_prefix(response)
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer><|im_end|>$"

        m = re.search(regex, completion, re.DOTALL)
        # if the format is not correct, reward is 0
        if m is None or len(m.groups()) != 2:
            return 0.0
        else:
            return 1.0
    except Exception:
        return 0.0


def ensure_think_prefix(s):
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s


_ACCURACY_CHOICE_MARKER = r"(?:^|\s)(?:\([A-Z]\)(?=\s|$)|[A-Z][.:)](?=\s|$)|(?i:option|choice)\s+[A-Z]\b)"


def _accuracy_final_answer(text: str) -> str | None:
    # Allow a prompt-prefilled <think>, but reject repeated or unfinished blocks.
    if any(text.count(tag) > 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    if "<think>" in text:
        return None
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer:
        return answer.group(1).strip()
    if "<answer>" in text or "</answer>" in text:
        return None
    return text.strip()


def _accuracy_choice_answer(text: str) -> tuple[str, str] | None:
    # Only explicit final-answer forms, never letters occurring in prose/reasoning.
    text = text.strip()
    bare = re.fullmatch(r"([A-Z])|\(([A-Z])\)", text)
    if bare:
        return bare[1] or bare[2], ""
    match = re.fullmatch(
        r"(?:[Tt]he answer is\s+)?(?:(?i:option|choice)\s+)?"
        r"([A-Z])(?:[.:)]\s+(.+)|[.]?)",
        text,
        re.DOTALL,
    )
    if match is None:
        return None
    payload = (match[2] or "").strip()
    # Reject a list of alternatives instead of accepting its first option.
    if re.search(_ACCURACY_CHOICE_MARKER, payload):
        return None
    return match[1], payload


def _accuracy_reward(response: str, label: str) -> float:
    # Both sides can include reasoning. Never let an intermediate boxed number
    # override the final answer, including in a reasoning-bearing dataset label.
    student_answer, ground_truth = _accuracy_final_answer(response), _accuracy_final_answer(label)
    if student_answer is None or ground_truth is None:
        return 0.0
    if len(re.findall(_ACCURACY_CHOICE_MARKER, student_answer)) > 1:
        return 0.0
    if student_answer == ground_truth:
        return 1.0

    student_choice, gold_choice = _accuracy_choice_answer(student_answer), _accuracy_choice_answer(ground_truth)
    if gold_choice is not None and student_choice is None and re.search(_ACCURACY_CHOICE_MARKER, student_answer):
        return 0.0
    if student_choice is not None and gold_choice is not None:
        if student_choice[0] != gold_choice[0]:
            return 0.0
        if not student_choice[1] or not gold_choice[1]:
            return 1.0
        student_answer, ground_truth = student_choice[1], gold_choice[1]
    elif gold_choice is not None and gold_choice[1]:
        ground_truth = gold_choice[1]
    elif student_choice is not None and student_choice[1]:
        student_answer = student_choice[1]
    if student_answer == ground_truth:
        return 1.0

    from math_verify import parse, verify

    try:
        answer = parse(student_answer)
        solution = parse(ground_truth)
        if verify(solution, answer):
            return 1.0
    except Exception:
        pass
    return 0.0


def _accuracy_format(response: str, label: str) -> float:
    """Score one complete reasoning/answer pair, including a prefilled think
    opener."""
    completion = _accuracy_think_prefix(response)
    completion = completion.removesuffix("<|im_end|>").rstrip()
    if any(completion.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        return 0.0
    match = re.fullmatch(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", completion, re.DOTALL)
    return float(match is not None and bool(match[1].strip()) and bool(match[2].strip()))


def get_openr1mm_accuracy_format_reward(response: str, label: str) -> float:
    """Add independent accuracy and format rewards, each in [0, 1]."""
    return _accuracy_reward(response, label) + _accuracy_format(response, label)


def _accuracy_think_prefix(s: str) -> str:
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s


def _benchmark_final_answer(response: str) -> str | None:
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


def _benchmark_normalized(text: Any) -> str:
    return " ".join(str(text).strip().casefold().split())


def _benchmark_number(text: str) -> Decimal | None:
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


def get_openr1mm_benchmark_accuracy(response: str, spec: dict[str, Any]) -> float:
    answer = _benchmark_final_answer(response)
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
                index == spec["correct_index"]
                and _benchmark_normalized(match[2]) == _benchmark_normalized(choices[spec["correct_index"]])
            )
        return float(_benchmark_normalized(answer) == _benchmark_normalized(choices[spec["correct_index"]]))
    candidates = gold if isinstance(gold, list) else [gold]
    for expected in candidates:
        if _benchmark_normalized(answer) == _benchmark_normalized(expected):
            return 1.0
        a, b = _benchmark_number(answer), _benchmark_number(str(expected))
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


def get_openr1mm_benchmark_reward(args: Any, sample: Any, **kwargs: Any) -> float:
    metadata = sample.metadata or {}
    if metadata.get("eval_benchmark") in ("mathvista_testmini", "mmmu_validation"):
        return get_openr1mm_benchmark_accuracy(sample.response, json.loads(sample.label))
    # Preserve the training reward; benchmark evaluation has no format point.
    return get_openr1mm_accuracy_format_reward(sample.response, sample.label)
