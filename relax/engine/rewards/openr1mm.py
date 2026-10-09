# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_NATIVE_FINAL = re.compile(r"<\|open\|>(response|answer)<\|sep\|>")
_NATIVE_END = re.compile(r"<\|close\|>(?:response|answer)<\|sep\|>")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_UNITS = r"(?:°|degrees?|cm|mm|km|m|meters?|metres?|units?|ft|feet|inches?|sq\s+km)"
_COUNTS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve".split()))


def _final_answer(text: str, *, allow_plain: bool = False) -> str | None:
    """Isolate XML or Kimi-native final content before inspecting its answers.

    The prompt may already contain the opening think token. Never fall back to
    extracting numbers from reasoning when the final block is unfinished.
    """
    native = list(_NATIVE_FINAL.finditer(text))
    native_complete = False
    if native:
        if len(native) != 1:
            return None
        text = text[native[0].end() :]
        ends = list(_NATIVE_END.finditer(text))
        if len(ends) != 1:
            return None
        suffix = text[ends[0].end() :].strip()
        if suffix not in ("", "<|close|>message<|sep|><|end_of_msg|>"):
            return None
        text = text[: ends[0].start()].strip()
        native_complete = True
    # Native final channels can themselves contain the requested XML pair.
    if text.count("</think>") > 1:
        return None
    xml_final = "</think>" in text
    if xml_final:
        text = text.split("</think>", 1)[1]
    if "<think>" in text or "<|" in text:
        return None
    matches = list(_ANSWER.finditer(text))
    if matches:
        if len(matches) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return None
        if text[matches[0].end() :].strip():
            return None
        return matches[0][1].strip() or None
    # Some Kimi completions close an XML opening with the native final token.
    # Only accept this when the independently delimited native final is complete.
    if native_complete and text.startswith("<answer>") and text.count("<answer>") == 1:
        text = text[len("<answer>") :]
    if "<answer>" in text or "</answer>" in text:
        return None
    text = text.strip()
    if native_complete or xml_final or allow_plain:
        return text or None
    # Preserve bare-answer callers without treating unframed reasoning as a
    # final channel. Only a compact expression/text or explicit answer assertion
    # is eligible; arbitrary prose with an embedded box is not.
    if "\n" in text:
        return None
    if re.match(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s+|:\s*)", text, re.I):
        return text or None
    residue = re.sub(r"\\[a-zA-Z]+", "", text)
    if re.fullmatch(r"[\d\s.{}()\[\]+*/^=_%$\\-]+", residue) or re.fullmatch(r"[\w -]+", text):
        return text or None
    return None


def _boxed_answer(text: str) -> str | None:
    """Read one balanced boxed expression without dropping nested fractions."""
    if text.count(r"\boxed{") != 1:
        return None
    start = text.index(r"\boxed{") + len(r"\boxed{")
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            # A box in a rejected calculation is not the final answer.
            if re.search(r"\b(?:not|incorrect|wrong)\b", text, re.I):
                return None
            suffix = text[index + 1 :].strip().strip(".$ ")
            suffix = re.sub(r"^\\[)\]]\s*", "", suffix)
            if suffix and not re.fullmatch(_UNITS, suffix, re.I):
                return None
            return text[start:index]
    return None


def _canonical_answer(text: str) -> tuple[str, str]:
    """Compare complete answers; never search arbitrary prose for a number."""
    text = " ".join(text.split())
    # Strip paired Markdown emphasis, preserving exponentiation such as 2**3.
    text = re.sub(r"(?<![\w)\]}])\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\w)", r"\1", text)
    # Choices may carry an explanation, or be explicitly named at the end of
    # a reference answer. Do not interpret incidental letters in prose as choices.
    choices = [
        choice.upper()
        for choice in re.findall(
            r"\b(?:option\s+|choice\s+|answer\s*(?:is\s+|:\s*)(?:option\s+)?)"
            # Lowercase letters need punctuation/end, otherwise the article
            # in "the answer is a square" would become choice A.
            r"\(?((?-i:[A-Z])|(?-i:[a-z])(?=[.,:;)]|$))\)?(?=[\s.,:;)]|$)",
            text,
            re.I,
        )
    ]
    leading = re.match(
        r"^(?:(?:Therefore,?\s*)?(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*"
        r"(?:is\s*(?:option\s+)?|:\s*))?"
        r"\(?([A-Z])\)?(?:[.:,]\s|[.)]?$)",
        text,
        re.IGNORECASE,
    )
    if leading:
        choices.append(leading[1].upper())
    if choices:
        choices.extend(re.findall(r"\b(?:or|and)\s+\(?([A-Z])\)?\b", text))
        choices.extend(choice.upper() for choice in re.findall(r"\b(?:or|and)\s+\(?([a-z])\)?(?=[.,;)]|$)", text))
    if choices and len(set(choices)) == 1 and not re.search(r"\b(?:not|incorrect|wrong)\b", text, re.I):
        # A trailing "option B" is affirmative only in explicit answer wording.
        if leading or re.search(
            r"\b(?:corresponds to (?:option|choice)|answer\s*(?:is|:)(?:\s+option)?)\s+[A-Z]\b"
            r"|\((?:option|choice)\s+[A-Z]\)\.?$|^(?:option|choice)\s+[A-Z]\.?$",
            text,
            re.I,
        ):
            return "choice", choices[0]
    if choices:
        # Conflicting choices must not fall through to a boxed numeric match.
        return "ambiguous", text
    boxed = _boxed_answer(text)
    if boxed is not None:
        return _canonical_answer(boxed)
    if re.search(r"\\boxed\b", text):
        # Do not let math_verify extract a box rejected by the final-answer checks.
        return "ambiguous", text
    # An explicit conclusion takes precedence over counts in its explanation.
    conclusion = re.search(
        r"(?:^|[.!?]\s+)(?:Therefore,?\s*)?(?:the\s+)?"
        r"(?:(?:(?:correct|final)\s+)?answer|total(?:\s+number(?:\s+of\s+[A-Za-z\s]+)?)?)"
        r"\s*(?:is\s+|:\s*)(.+)$",
        text,
        re.I,
    )
    if conclusion:
        return _canonical_answer(conclusion[1])
    # A final displayed expression or an explicit final quantity statement
    # is also a conclusion; intermediate equations remain outside scoring.
    display = re.search(r"(?:\$\$([^$]*)\$\$|\\\[((?:(?!\\\]).)*)\\\])\.?$", text)
    if display and not re.search(r"\b(?:not|incorrect|wrong)\b", text, re.I):
        # Restrict to the last display, rather than spanning earlier blocks.
        expression = next(part for part in display.groups() if part is not None)
        if "$" not in expression and r"\]" not in expression:
            return _canonical_answer(expression)
    # Native final channels often answer counts in a sentence. Accept a
    # leading count assertion, not a number found in its later explanation.
    count = re.match(
        r"^There (?:are|is|will be) (\d+|" + "|".join(_COUNTS.values()) + r")\b(?!\.\d)([^.!?:]*)(?:[.!?:]|$)",
        text,
        re.I,
    )
    if (
        count
        and not re.search(r"\d|\b(?:and|or|not)\b", count[2], re.I)
        and not re.search(r"\b(?:there (?:are|is|will be)|total is|actually|instead)\b", text[count.end() :], re.I)
    ):
        value = count[1].lower()
        return "math", str(next((n for n, word in _COUNTS.items() if word == value), value))
    quantity = re.search(
        r"(?:^|[.!?]\s+)(?:The\s+)?(?:length|area|volume|measure|value)"
        r"(?:\s+of)?\s+[A-Za-z\s]+\s+is\s+([^.!?]+)\.?$",
        text,
        re.I,
    )
    if quantity:
        kind, value = _canonical_answer(quantity[1])
        if kind == "math":
            return kind, value
    text = re.sub(
        r"^(?:Therefore,?\s*)?(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s+|:\s*)", "", text, flags=re.I
    ).rstrip(".")
    if (text.startswith(r"\(") and text.endswith(r"\)")) or (text.startswith(r"\[") and text.endswith(r"\]")):
        text = text[2:-2].strip()
    elif text.startswith("$") and text.endswith("$"):
        text = text.strip("$").strip()
    text = text.replace("−", "-").replace("π", r"\pi")
    text = re.sub(r"\\[,;! ]", "", text)
    text = re.sub(r"\^\s*\{?\\circ\}?", "°", text)
    text = re.sub(r"\\(?:text|mathrm)\{\s*(" + _UNITS + r")\s*\}", r"\1", text, flags=re.I)
    text = re.sub(r"(?<=[\d}])\s*" + _UNITS + r"(?:\^?\{?[23]\}?)?$", "", text, flags=re.I).strip()
    text = re.sub(r"^(?:\\angle\s*)?(?:[A-Z]{1,3}|[a-z])\s*=\s*", "", text)
    # Let math_verify interpret LaTeX commands, while rejecting surrounding prose.
    residue = re.sub(r"\\[a-zA-Z]+|\bpi\b", " ", text)
    if not re.search(r"[a-zA-Z]{2,}", residue) and re.fullmatch(r"[\da-zA-Z\s.,{}()\[\]+*/^=_%\\-]+", residue):
        return "math", text
    return "text", text.casefold()


def get_openr1mm_rule_based_reward(response: str, label: str) -> float:
    """Score only final answers, with the reference as math_verify's gold."""
    prediction, reference = _final_answer(response), _final_answer(label, allow_plain=True)
    if prediction is None or reference is None:
        return 0.0
    gold_kind, gold = _canonical_answer(reference)
    pred_kind, pred = _canonical_answer(prediction)
    if gold_kind != pred_kind or gold_kind == "ambiguous":
        return 0.0
    if gold == pred:
        return 1.0
    if gold_kind != "math":
        return 0.0
    # math_verify uses signal timeouts: rewards run in worker main threads.
    from math_verify import LatexExtractionConfig, parse, verify

    try:
        config = [LatexExtractionConfig()]
        gold_expr = parse("$" + gold + "$", extraction_config=config, fallback_mode="no_fallback")
        pred_expr = parse("$" + pred + "$", extraction_config=config, fallback_mode="no_fallback")
        return float(bool(gold_expr and pred_expr and verify(gold_expr, pred_expr)))
    except Exception:
        logger.exception("OpenR1-MM final-answer verification failed")
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
