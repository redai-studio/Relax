# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GAGAR as equivalent group rewards (paper Appendix A.2 and A.3).

The caller computes binary test rewards and obtains aligned quality weights
from an independent grading service, then returns compute_group_rewards(...)
from its custom group RM. This module only implements the arithmetic.

Use --group-rm and --disable-grpo-std-normalization, keeping reward mean
centering enabled. Do not clip the returned rewards; they can exceed 1.
Keep original test outcomes separately for pass-rate metrics and filtering.
"""


def compute_group_rewards(rewards: list[float], weights: list[float], cap: float = 1.5) -> list[float]:
    """Convert one mix-outcome rollout group's rewards and weights into RM
    scores.

    Rewards are 0/1 after hack correction. Passing weights are in [0, 1];
    failing weights are ignored. The caller handles invalid trajectories and
    unavailable grading results. The final group centering is performed by
    Relax, not by this function.
    """
    if not rewards or len(rewards) != len(weights):
        raise ValueError("rewards and weights must be nonempty and have equal lengths")
    if any(reward not in (0, 1) for reward in rewards):
        raise ValueError("rewards must be binary test outcomes (0 or 1)")

    passing = [i for i, reward in enumerate(rewards) if reward == 1]
    if not passing or len(passing) == len(rewards):
        return rewards.copy()
    if any(not (0 <= weights[i] <= 1) for i in passing):
        raise ValueError("passing weights must be within [0, 1]")

    # 1. Dr.GRPO:  mean-centered outcome advantage.
    reward_mean = sum(rewards) / len(rewards)
    advantages = [reward - reward_mean for reward in rewards]

    # 2. Discount passing advantages, then restore their sum up to the cap.
    positive_sum = sum(advantages[i] for i in passing)
    weighted_sum = sum(weights[i] * advantages[i] for i in passing)
    # All-zero weights leave passing B values at zero; no rescaling is needed.
    scale = min(positive_sum / weighted_sum, cap) if weighted_sum else 1.0

    # 3. Return B + reward_mean. Relax subtracts its new mean to obtain B - mean(B).
    # Failed rewards stay zero; their final advantages can change when the cap binds.
    result = rewards.copy()
    for i in passing:
        result[i] = reward_mean + scale * weights[i] * advantages[i]
    return result
