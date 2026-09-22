# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Reward function for the Search-QA agentic OPD recipe.

The environment computes binary exact match during rollout. For training, score
also applies the SDAR invalid-action penalty. Raw EM remains available as
em/acc/won for evaluation.
"""

from __future__ import annotations

import os
from math import sqrt
from typing import Any

from relax.utils.types import Sample


INVALID_ACTION_PENALTY_COEF = float(os.environ.get("SEARCH_INVALID_ACTION_PENALTY_COEF", "0.01"))


def compute_score(metadata: dict) -> dict:
    em = float(metadata.get("score", metadata.get("em", 1.0 if metadata.get("won") else 0.0)))
    num_turns = float(metadata.get("num_turns", 0.0))
    num_valid_actions = float(metadata.get("num_valid_actions", num_turns))
    num_invalid_actions = max(0.0, num_turns - num_valid_actions)
    invalid_action_penalty = INVALID_ACTION_PENALTY_COEF * num_invalid_actions

    result = {
        "score": em - invalid_action_penalty,
        "acc": em,
        "won": em,
        "em": em,
        "num_invalid_actions": num_invalid_actions,
        "invalid_action_penalty": invalid_action_penalty,
    }
    # Rollout-health counters (searches issued, invalid/truncated turns) so the
    # calibration run can watch them alongside EM without extra plumbing.
    for key in (
        "num_turns",
        "num_searches",
        "num_valid_actions",
        "num_truncated_turns",
        "num_retrieval_errors",
        "num_llm_errors",
    ):
        if key in metadata:
            result[key] = float(metadata[key])
    return result


def grpo_turn_advantage(metadata_by_slot: list[dict[Any, dict[str, Any]]]):
    """Match SDAR GRPO normalization across every active turn in a prompt
    group."""

    scored_turns: list[tuple[int, Any, float]] = []
    for slot_index, slot in enumerate(metadata_by_slot):
        for name, metadata in slot.items():
            scored_turns.append((slot_index, name, float(compute_score(metadata)["score"])))

    if not scored_turns:
        return None

    scores = [score for _, _, score in scored_turns]
    if len(scores) == 1:
        mean = 0.0
        std = 1.0
    else:
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        std = sqrt(variance)

    result: list[dict[Any, float]] = [dict() for _ in metadata_by_slot]
    for slot_index, name, score in scored_turns:
        result[slot_index][name] = (score - mean) / (std + 1e-6)
    return result


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return compute_score(metadata)
