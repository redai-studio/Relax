# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GAGAR Appendix A.1: evidence-backed quality tiers and tied-rank weights."""

from typing import Any


CRITERIA = {"approach": 30, "precision": 25, "minimality": 20, "side_effects": 15, "consistency": 10}
ISSUES = {"unrequested_rewrite", "test_specific_workaround", "severe_process_issue", "unresolved_regression"}


def quality_tier(scores: dict[str, int], issues: list[str]) -> int:
    if (
        "unrequested_rewrite" in issues
        or "test_specific_workaround" in issues
        or scores["approach"] == 1
        or (scores["minimality"] <= 2 and scores["side_effects"] <= 2)
    ):
        return 3
    if min(scores.values()) >= 4 and not {"severe_process_issue", "unresolved_regression"}.intersection(issues):
        return 1
    return 2


def grade_to_weights(rewards: list[float], grading: dict[str, Any]) -> dict[str, Any]:
    """Validate a model's assessment, then apply the fixed paper mapping.

    Only original passes are assessed. Confirmed leaked/external solutions
    change the outcome before tiering; test-specific workarounds alone are T3.
    Ranking is a list of tied groups, highest quality first.
    """
    passing = {i for i, reward in enumerate(rewards) if reward == 1}
    evaluations = grading.get("evaluations", [])
    if not isinstance(evaluations, list):
        raise ValueError("evaluations must be a list")
    assessed: dict[int, dict] = {}
    corrected = rewards.copy()
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ValueError("each evaluation must be an object")
        index = evaluation.get("rollout_n")
        if type(index) is not int or index not in passing or index in assessed:
            raise ValueError("evaluate each originally passing rollout_n exactly once")
        scores = evaluation.get("scores", {})
        issues = evaluation.get("issues", [])
        if (
            not isinstance(scores, dict)
            or set(scores) != set(CRITERIA)
            or any(type(score) is not int or not 1 <= score <= 5 for score in scores.values())
        ):
            raise ValueError("each criterion needs an integer score from 1 to 5")
        if not isinstance(issues, list) or any(issue not in ISSUES for issue in issues):
            raise ValueError("unknown quality issue")
        if not isinstance(evaluation.get("evidence"), str) or not evaluation["evidence"].strip():
            raise ValueError("each assessment needs task-specific evidence")
        hack = evaluation.get("confirmed_external_solution", False)
        if type(hack) is not bool:
            raise ValueError("confirmed_external_solution must be boolean")
        if hack:
            if not isinstance(evaluation.get("hack_evidence"), str) or not evaluation["hack_evidence"].strip():
                raise ValueError("outcome correction requires evidence of reliance on leaked/external solutions")
            corrected[index] = 0
        assessed[index] = {
            **evaluation,
            "tier": quality_tier(scores, issues) if not hack else None,
            "initial_score": sum(CRITERIA[name] * score for name, score in scores.items()) / 100,
        }
    if set(assessed) != passing:
        raise ValueError("missing passing candidates")

    remaining = {i for i in passing if corrected[i] == 1}
    ranking = grading.get("ranking", [])
    if not isinstance(ranking, list):
        raise ValueError("ranking must be a list of tied groups")
    seen: set[int] = set()
    tiers: dict[int, list[list[int]]] = {1: [], 2: [], 3: []}
    last_tier = 1
    for tied in ranking:
        if not isinstance(tied, list) or not tied:
            raise ValueError("ranking must contain nonempty tied groups")
        for index in tied:
            if type(index) is not int or index not in remaining or index in seen:
                raise ValueError(
                    "rank every corrected pass exactly once using rollout_n IDs: "
                    f"expected {sorted(remaining)}, got {index!r}"
                )
            seen.add(index)
        tier = assessed[tied[0]]["tier"]
        if tier < last_tier or any(assessed[i]["tier"] != tier for i in tied):
            candidate_tiers = {
                f"T{number}": [i for i in sorted(remaining) if assessed[i]["tier"] == number] for number in (1, 2, 3)
            }
            raise ValueError(
                f"ranking and ties must respect quality tiers T1, T2, T3; candidate IDs by tier: {candidate_tiers}"
            )
        last_tier = tier
        tiers[tier].append(tied)
    if seen != remaining:
        raise ValueError(f"ranking is missing corrected passing candidates: {sorted(remaining - seen)}")

    weights = [0.0] * len(rewards)
    for tier, tied_groups in tiers.items():
        initial_scores = sorted({assessed[i]["initial_score"] for tied in tied_groups for i in tied}, reverse=True)
        for rank, tied in enumerate(tied_groups):
            if tier == 1:
                weight = 1.0 if rank == 0 else 0.9
            elif tier == 2:
                weight = 0.85 if len(tied_groups) == 1 else 0.85 - 0.45 * rank / (len(tied_groups) - 1)
            else:
                weight = 0.2
            for index in tied:
                initial_rank = initial_scores.index(assessed[index]["initial_score"])
                reason = assessed[index].get("rank_reason", "")
                if rank != initial_rank and (not isinstance(reason, str) or not reason.strip()):
                    raise ValueError("each change from the initial ordering needs a rank_reason")
                weights[index] = weight
    return {"weights": weights, "rewards": corrected, "evaluations": list(assessed.values()), "ranking": ranking}
