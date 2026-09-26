# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

from .math_dapo_utils import compute_score as compute_score_dapo
from .math_dapo_utils import normalize_final_answer
from .math_utils import grade_answer_verl
from .openr1mm import get_openr1mm_rule_based_reward


# EOS markers that Qwen3-VL may emit before the real stop token.
_EOS_MARKERS = ("</s>", "<|endoftext|>", "<|im_end|>")

# GSM8K answer separator pattern: "#### 42", "#### $1,200", etc.
_GSM8K_PATTERN = re.compile(r"####[^\S\n]*\n*[^\S\n]*(\$?[\d,]+(?:\.\d+)?)")

# Minerva-style "Answer: <value>" extractor (same as math_dapo_utils default).
_MINERVA_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")


def _strip_eos(text: str) -> str:
    """Remove everything after the first EOS marker (if any)."""
    for marker in _EOS_MARKERS:
        idx = text.find(marker)
        if idx > 0:
            return text[:idx]
    return text


def _is_correct_gsm8k(solution_str: str, gt: str) -> tuple[bool, str]:
    """Check correctness for GSM8K-style responses.

    Extraction order:
      1. Minerva "Answer: <val>" pattern
      2. GSM8K "#### <val>" separator (fallback)

    GT normalisation handles integers, decimals, and non-numeric LaTeX.
    Comparison uses exact string match, then numeric fallback, then
    unit-suffix fallback ("852 BC" → 852, "100 miles" → 100).
    """
    match = _MINERVA_PATTERN.findall(solution_str)
    if not match:
        match = _GSM8K_PATTERN.findall(solution_str)
    extracted = match[-1].strip() if match else "[INVALID]"
    pred = normalize_final_answer(extracted)

    gt_norm = normalize_final_answer(gt)
    try:
        gt_float = float(gt_norm)
        gt_norm = str(int(gt_float)) if gt_float == int(gt_float) else str(gt_float)
    except (ValueError, OverflowError):
        pass  # non-numeric gt (e.g. LaTeX); keep as-is

    if pred == gt_norm:
        return True, pred
    try:
        if float(pred) == float(gt_norm):
            return True, pred
    except (ValueError, TypeError):
        pass
    num_m = re.match(r"^(\d[\d,]*(?:\.\d+)?)", pred.strip())
    if num_m:
        try:
            if float(num_m.group(1).replace(",", "")) == float(gt_norm):
                return True, pred
        except (ValueError, TypeError):
            pass
    return False, pred


def _compute_gsm8k_score(response: str, label) -> float:
    """MOPD scorer for GSM8K-style (text math) data.

    Improvements over base dapo scorer:
      - Strips EOS noise (</s>, <|im_end|>, etc.) before parsing
      - Penalises reward-hacking via repetitive #### output (>5 markers → -1)
      - Accepts both "Answer:" and "####" extraction formats
      - Handles decimal ground-truth labels and unit-suffix predictions
    """
    gt = str(label.get("ground_truth") or label.get("answer", "") if isinstance(label, dict) else label)

    response = _strip_eos(response)

    if response.count("####") > 5:
        return -1.0

    # Only scan the tail for efficiency (longest MATH-500 answer ≈ 159 chars).
    response = response[-300:]

    correct, _ = _is_correct_gsm8k(response, gt)
    return 1.0 if correct else -1.0


def get_mopd_reward(response: str, label, metadata: dict | None = None) -> float:
    """Per-data-source reward routing for MOPD.

    Routes each sample to its domain-appropriate scorer based on ``data_source``
    in metadata, always returning a scalar (rm_type="mopd" carries no reward_key).
    Each source delegates to the same integrated reward the standalone rm_type
    would use, so MOPD scoring matches single-source training:
      - geometry3k / geo3k         : rm_type="math" scorer (grade_answer_verl, {0,1})
      - multimodal-open-r1 / openr1mm : openr1mm rule-based scorer
      - dapo-math                  : DAPO scorer's scalar "score" (dict → scalar)
      - all others                 : GSM8K scorer with EOS/repeat hardening (-1/1)
    """
    data_source = (metadata or {}).get("data_source", "").lower()
    if "geometry3k" in data_source or "geo3k" in data_source:
        return 1.0 if grade_answer_verl(response, label) else 0.0
    if "multimodal-open-r1" in data_source or "openr1mm" in data_source:
        return get_openr1mm_rule_based_reward(response, label)
    if "dapo" in data_source:
        return float(compute_score_dapo(response, label)["score"])
    return _compute_gsm8k_score(response, label)
