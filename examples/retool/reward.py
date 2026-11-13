# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Final answer reward minus 0.1 per tool failure, without a floor."""

from typing import Any

from examples.retool.math_dapo import compute_score


FAILURE_STATUSES = frozenset(
    {"code_error", "timeout", "invalid_call", "invalid_arguments", "unknown_tool", "batch_rejected"}
)


def score(final_answer: str, label: str, counts: dict[str, int]) -> dict:
    if set(counts) - FAILURE_STATUSES - {"ok"} or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Require actual executor feedback; infrastructure faults are not model penalties")
    result = compute_score(final_answer, label, strict_box_verify=False)
    failures = sum(n for status, n in counts.items() if status != "ok")
    penalty = round(-0.1 * failures, 10)
    return {
        **result,
        "math_score": result["score"],
        "score": round(result["score"] + penalty, 10),
        "tool_failure_count": failures,
        "tool_penalty": penalty,
        "tool_submission_count": sum(counts.values()),
        "tool_execution_ok": counts.get("ok", 0),
        "reward_floor": None,
    }


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> dict:
    return score(sample.metadata["retool_final_answer"], str(sample.label), sample.metadata["tool_feedback_counts"])
