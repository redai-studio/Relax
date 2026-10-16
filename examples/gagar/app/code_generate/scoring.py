# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Code-generation quality weights, independent of the SWE rubric."""


def grade_to_weights(rewards: list[float], grading: dict) -> dict:
    passing = {i for i, reward in enumerate(rewards) if reward == 1}
    weights = [0.0] * len(rewards)
    seen = set()
    evaluations = grading.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("evaluations must be a list")
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            raise ValueError("each evaluation must be an object")
        index, weight = evaluation.get("rollout_n"), evaluation.get("weight")
        if type(index) is not int or index not in passing or index in seen:
            raise ValueError("evaluate each originally passing rollout_n exactly once")
        if type(weight) not in (int, float) or not 0 <= weight <= 1:
            raise ValueError("weight must be a number in [0, 1]")
        if not isinstance(evaluation.get("reason"), str) or not evaluation["reason"].strip():
            raise ValueError("each weight needs a brief evidence-based reason")
        seen.add(index)
        weights[index] = float(weight)
    if seen != passing:
        raise ValueError("missing passing candidates")
    return {"weights": weights, "rewards": rewards.copy(), "evaluations": evaluations}
