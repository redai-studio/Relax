# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reward normalisation strategies, one per algorithm family.

These run on the rollout side (CPU) from
``relax.utils.utils.post_process_rewards``.  Every normaliser takes the raw
per-sample scalar rewards and returns the per-sample scalars written into the
TransferQueue ``rewards`` column, so adding a strategy never changes the data
schema.
"""

from typing import Any, Callable

import torch

from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards


GROUP_EPS = 1e-6
"""Epsilon added to a group standard deviation before dividing by it.

Matches the value the pre-registry GRPO path used, so GRPO, GSPO, SAPO and
CISPO keep producing exactly the numbers they produced before the registry
existed. That parity is what the equivalence tests assert, so this constant is
not free to move.
"""


def group_positions(samples: list[Any], expected_size: int) -> dict[int, list[int]]:
    """Map ``Sample.group_index`` to the positions it occupies in ``samples``.

    Grouping follows ``group_index`` rather than position, so how the caller
    orders the batch does not affect the result.
    """
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("Sample.group_index is required for group reward normalization.")
        if sample.group_index not in positions_by_group:
            positions_by_group[sample.group_index] = []
        positions_by_group[sample.group_index].append(position)

    for group_index, positions in positions_by_group.items():
        if len(positions) != expected_size:
            raise ValueError(f"Reward group {group_index} has {len(positions)} samples, expected {expected_size}.")
    return positions_by_group


def _group_normalize(args: Any, samples: list[Any], raw_rewards: list[float], *, use_std: bool) -> list[float]:
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized_rewards = torch.empty_like(rewards)
    for positions in positions_by_group.values():
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if use_std:
            group_rewards = group_rewards / (group_rewards.std() + GROUP_EPS)
        normalized_rewards[positions] = group_rewards

    return normalized_rewards.tolist()


def normalize_none(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """No normalisation — the estimator consumes raw rewards (REINFORCE++,
    PPO)."""
    return raw_rewards


def normalize_group_mean(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean only (REINFORCE++ baseline)."""
    return _group_normalize(args, samples, raw_rewards, use_std=False)


def normalize_group_mean_std(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """Subtract the group mean, then optionally divide by the group std.

    ``--disable-grpo-std-normalization`` (Dr.GRPO) turns the division off.
    """
    return _group_normalize(args, samples, raw_rewards, use_std=args.grpo_std_normalization)


def _reject_non_finite_group(group_index: int, positions: list[int], group_rewards: torch.Tensor) -> None:
    """Raise if any reward in one group is NaN or infinite, naming the sample.

    The other normalisers do not check: an infinite reward there poisons only
    the sample that carries it (``group_mean_std`` divides it away, and the
    caller sees one bad advantage). A leave-one-out baseline averages the
    *other* samples, so one bad value silently propagates into every advantage
    in the group -- which is why this reports the offender's position instead
    of letting the arithmetic swallow it.
    """
    finite_mask = torch.isfinite(group_rewards)
    if finite_mask.all():
        return
    invalid_group_positions = (~finite_mask).nonzero(as_tuple=False).flatten().tolist()
    invalid_sample_positions = [positions[position] for position in invalid_group_positions]
    invalid_values = group_rewards[~finite_mask].tolist()
    raise ValueError(
        f"RLOO group_index={group_index} contains non-finite reward(s) at "
        f"group position(s) {invalid_group_positions}, sample position(s) "
        f"{invalid_sample_positions}: {invalid_values}."
    )


def normalize_group_leave_one_out(args: Any, samples: list[Any], raw_rewards: list[float]) -> list[float]:
    """RLOO's leave-one-out baseline (Ahmadian et al. 2024, arXiv:2402.14740).

    Each sample is centred on the mean of the *others* in its prompt group::

        A_i = r_i - mean_{j != i}(r_j) = G / (G - 1) * (r_i - mean(r))

    Unlike :func:`normalize_group_mean_std` there is no division by the group
    standard deviation, so the advantage keeps the reward's scale. The
    right-hand identity is what
    :func:`~relax.utils.training.ppo_utils.compute_rloo_leave_one_out_rewards`
    computes; that helper also owns the ``G >= 2`` and finiteness contracts,
    and it is shared with the rollout diagnostics so the numbers reported and
    the numbers trained on cannot drift apart.
    """
    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)

    normalized_rewards = torch.empty_like(rewards)
    for group_index, positions in positions_by_group.items():
        group_rewards = rewards[positions]
        _reject_non_finite_group(group_index, positions, group_rewards)
        normalized_rewards[positions] = compute_rloo_leave_one_out_rewards(group_rewards)

    return normalized_rewards.tolist()


REWARD_NORMALIZERS: dict[str, Callable[[Any, list[Any], list[float]], list[float]]] = {
    "none": normalize_none,
    "group_mean": normalize_group_mean,
    "group_mean_std": normalize_group_mean_std,
    "group_leave_one_out": normalize_group_leave_one_out,
}
