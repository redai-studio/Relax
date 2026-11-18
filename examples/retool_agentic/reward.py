# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve ReTool's episode status and reward after Agentic export."""

from typing import Any

from examples.retool_agentic.math_dapo import compute_score
from relax.agentic.profile import TRACE_KEY
from relax.utils.types import Sample


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


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> dict:
    # A native stop at </code> ends an API response, not necessarily the
    # episode. Agent-side turn/observation budgets can also end an episode.
    if sample.metadata["retool_stop_reason"] != "final_answer":
        sample.status = Sample.Status.TRUNCATED
        sample.metadata[TRACE_KEY]["terminal_status"] = sample.status.value
    # Session export records the actual rollout that generated this context.
    # Draw indices can advance further when a failed group is replaced.
    if getattr(args, "hybrid", False):
        if not sample.weight_versions or any(
            not str(version).isdigit() or int(version) < 1 for version in sample.weight_versions
        ):
            raise RuntimeError(f"Invalid hybrid policy versions: {sample.weight_versions}")
        versions = list(map(int, sample.weight_versions))
        if versions != sorted(versions):
            raise RuntimeError(f"Hybrid policy versions went backwards: {versions}")
        sample.metadata.update(
            retool_policy_version_min=versions[0],
            retool_policy_version_max=versions[-1],
            retool_policy_version_span=versions[-1] - versions[0],
        )
    else:
        expected = sample.metadata["start_rollout_id"] + 1
        if not sample.weight_versions or any(
            not str(version).isdigit() or int(version) + (args.start_rollout_id or 0) != expected
            for version in sample.weight_versions
        ):
            raise RuntimeError(f"Rollout policy version mismatch: expected {expected}, got {sample.weight_versions}")
    return score(sample.metadata["retool_final_answer"], str(sample.label), sample.metadata["tool_feedback_counts"])
