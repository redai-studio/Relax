# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import math
from typing import Any


REQUIRED_EVIDENCE = (
    "publication_idempotence",
    "content_conflict",
    "old_session_logprobs",
    "new_session_logprobs",
    "missing_adapter_rejected",
    "capacity",
    "c_generation_and_b_continuation",
    "baseline_repeatability",
    "fixture_separation",
    "cuda_graph",
    "real_export",
    "partial_readiness",
    "failure_late_ack",
    "cancel_finish_race",
    "abort_resume",
    "native_pin",
    "last_gpu_use",
    "slot_clear_completion",
    "mixed_batch",
    "slot_reuse",
    "slot_reuse_negative_control",
    "kv_sensitivity",
    "cache_a_to_b",
    "cache_b_to_a",
    "wrong_kv_negative_control",
    "continuous_traffic",
    "steady_state_overhead",
)


def scores(output: dict[str, Any]) -> dict[int, float]:
    entries = output["meta_info"]["output_token_ids_logprobs"][0]
    result = {int(entry[1]): float(entry[0]) for entry in entries}
    if not result or len(result) != len(entries) or not all(math.isfinite(value) for value in result.values()):
        raise AssertionError("missing, duplicate or nonfinite fixed-token logprobs")
    return result


def compare(actual: dict[int, float], expected: dict[int, float], *, atol: float, rtol: float) -> float:
    if not all(math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    if not actual or actual.keys() != expected.keys():
        raise AssertionError("scored token positions/IDs changed or are empty")
    if not all(math.isfinite(value) for value in (*actual.values(), *expected.values())):
        raise AssertionError("nonfinite score")
    worst = max(actual, key=lambda key: abs(actual[key] - expected[key]))
    error = abs(actual[worst] - expected[worst])
    if any(abs(actual[key] - expected[key]) > atol + rtol * abs(expected[key]) for key in actual):
        raise AssertionError(
            f"logprob mismatch, maximum absolute error {error}, position/ID {worst}: "
            f"actual={actual[worst]}, baseline={expected[worst]}"
        )
    return error


def sample_scores(sample: Any) -> dict[int, float]:
    """Mask applies to continuation, including inter-turn observation
    tokens."""
    count = sample.response_length
    if count <= 0 or len(sample.loss_mask) != count or len(sample.rollout_log_probs) != count:
        raise AssertionError("invalid exported token/logprob alignment")
    offset = len(sample.tokens) - count
    if offset < 1:
        raise AssertionError("missing model-visible prompt")
    result = {offset + i: float(value) for i, value in enumerate(sample.rollout_log_probs) if sample.loss_mask[i]}
    compare(result, result, atol=0, rtol=0)
    return result


def compare_decode(actual: dict, expected: dict, *, atol: float, rtol: float) -> list[float]:
    """Compare matching autoregressive paths, never score past a token fork."""
    tokens = actual["output_ids"]
    if not tokens or tokens != expected["output_ids"]:
        raise AssertionError("independent decode generated different tokens")
    left = actual["meta_info"]["output_token_ids_logprobs"]
    right = expected["meta_info"]["output_token_ids_logprobs"]
    if len(left) != len(tokens) or len(right) != len(tokens):
        raise AssertionError("decode token scores missing")
    return [
        compare(
            scores({"meta_info": {"output_token_ids_logprobs": [a]}}),
            scores({"meta_info": {"output_token_ids_logprobs": [b]}}),
            atol=atol,
            rtol=rtol,
        )
        for a, b in zip(left, right, strict=True)
    ]


def audit_evidence(report: dict) -> list[str]:
    """Missing experiments remain explicit; a runner's existence is no PASS."""
    checks = report.get("checks", {})
    required = (*REQUIRED_EVIDENCE, *(("memory_handoff",) if report.get("profile", {}).get("memory_handoff") else ()))
    return [name for name in required if checks.get(name, {}).get("status") != "PASS"]


def worker_observations(receipt: dict) -> dict[str, dict]:
    observation = receipt.get("observation") or {}
    if "workers" not in observation:
        if not observation:
            raise AssertionError("missing native resource observation")
        return {"0": observation}
    boots, workers = observation.get("worker_boots"), observation["workers"]
    if (
        not boots
        or len(set(boots)) != len(boots)
        or set(workers) != {str(rank) for rank in range(len(boots))}
        or not all(workers.values())
    ):
        raise AssertionError("incomplete native worker observations")
    return workers
