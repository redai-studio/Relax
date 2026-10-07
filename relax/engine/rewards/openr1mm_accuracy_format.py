# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re


_CHOICE_MARKER = r"(?:^|\s)(?:\([A-Z]\)(?=\s|$)|[A-Z][.:)](?=\s|$)|(?i:option|choice)\s+[A-Z]\b)"


def _final_answer(text: str) -> str | None:
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


def _choice_answer(text: str) -> tuple[str, str] | None:
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
    if re.search(_CHOICE_MARKER, payload):
        return None
    return match[1], payload


def get_openr1mm_rule_based_reward(response: str, label: str) -> float:
    # Both sides can include reasoning. Never let an intermediate boxed number
    # override the final answer, including in a reasoning-bearing dataset label.
    student_answer, ground_truth = _final_answer(response), _final_answer(label)
    if student_answer is None or ground_truth is None:
        return 0.0
    if len(re.findall(_CHOICE_MARKER, student_answer)) > 1:
        return 0.0
    if student_answer == ground_truth:
        return 1.0

    student_choice, gold_choice = _choice_answer(student_answer), _choice_answer(ground_truth)
    if gold_choice is not None and student_choice is None and re.search(_CHOICE_MARKER, student_answer):
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


def MiniR1Format(response: str, label: str) -> float:
    """Score one complete reasoning/answer pair, including a prefilled think
    opener."""
    completion = ensure_think_prefix(response)
    completion = completion.removesuffix("<|im_end|>").rstrip()
    if any(completion.count(tag) != 1 for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        return 0.0
    match = re.fullmatch(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", completion, re.DOTALL)
    return float(match is not None and bool(match[1].strip()) and bool(match[2].strip()))


def get_openr1mm_accuracy_format_reward(response: str, label: str) -> float:
    """Add independent accuracy and format rewards, each in [0, 1]."""
    return get_openr1mm_rule_based_reward(response, label) + MiniR1Format(response, label)


def ensure_think_prefix(s: str) -> str:
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s
