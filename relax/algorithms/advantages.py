# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Advantage estimators as pure functions shared by both execution paths.

The colocate path calls these from ``relax.backends.megatron.loss`` inside the
Megatron worker; the fully-async path calls them from the
``relax.components.advantages`` Ray Serve deployment.  Keeping the maths here
means the two call sites differ only in what surrounds them: the pipeline-stage
early return, in-place write-back versus nested-tensor packing, and the
optional advantage whitening.

Any group-wise reward standardisation happens earlier, on the rollout side
(see :mod:`relax.algorithms.rewards`). By the time an estimator runs,
``rewards`` holds one processed scalar per sample.
"""

from typing import Any, Callable

import torch

from relax.algorithms.spec import get_algorithm
from relax.utils.training.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)


def _as_reward_tensor(rewards: Any, kl: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(rewards, torch.Tensor):
        return rewards.detach().to(dtype=torch.float32, device=kl[0].device, copy=True)
    return torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)


def advantage_grpo_broadcast(args: Any, *, rewards, kl, **_unused):
    """Broadcast the processed scalar reward over tokens."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_grpo_returns(reward_tensor, kl)
    advantages = list(returns)  # separate list so rebinding one does not move the other
    return advantages, returns


def advantage_reinforce_plus_plus(args: Any, *, rewards, kl, loss_masks, response_lengths, total_lengths, **_unused):
    """Discounted returns for REINFORCE++
    (https://arxiv.org/pdf/2501.03262)."""
    reward_tensor = _as_reward_tensor(rewards, kl)
    returns = get_reinforce_plus_plus_returns(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
    )
    advantages = list(returns)
    return advantages, returns


def advantage_reinforce_plus_plus_baseline(args: Any, *, rewards, kl, loss_masks, **_unused):
    """REINFORCE++ with a group baseline already subtracted upstream.

    No ``kl_coef``: this estimator keeps the KL penalty out of the advantage
    (``_validate_reinforce_plus_plus_args`` requires ``--kl-coef 0`` and an
    independent k2 loss instead), and the helper dropped the parameter to
    match.
    """
    reward_tensor = _as_reward_tensor(rewards, kl)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=reward_tensor,
        kl=kl,
        loss_masks=loss_masks,
    )
    # NOTE(dev): returns aliases the same list object, matching the pre-refactor
    # behaviour. On-policy distillation rebinds slots of the advantages list and
    # both views are expected to observe that.
    return advantages, advantages


def advantage_gae(
    args: Any,
    *,
    rewards,
    kl,
    values,
    response_lengths,
    total_lengths,
    padded_total_lengths=None,
    **_unused,
):
    """Generalised advantage estimation (PPO).

    ``padded_total_lengths`` is what the two call sites disagree on, and it is
    load-bearing rather than cosmetic: under bshd ``qkv_format``, VL models or
    unsplit forward, the sequences were padded before the forward pass, so CP
    un-sharding inside ``get_advantages_and_returns_batch`` has to slice at the
    padded offsets.  Passing ``None`` there does not raise — it silently reads
    the wrong token positions.  The Megatron path computes the value via
    ``maybe_padded_total_lengths`` and passes it; the ``Advantages`` deployment
    has no equivalent and passes ``None``, which is the behaviour it had before
    this handler existed.
    """
    from megatron.core import mpu

    shaped_rewards = []
    cp_rank = mpu.get_context_parallel_rank()
    for reward, k in zip(rewards, kl, strict=False):
        k *= -args.kl_coef
        if cp_rank == 0:
            k[-1] += reward
        shaped_rewards.append(k)
    return get_advantages_and_returns_batch(
        total_lengths,
        response_lengths,
        values,
        shaped_rewards,
        args.gamma,
        args.lambd,
        padded_total_lengths=padded_total_lengths,
    )


ADVANTAGE_FNS: dict[str, Callable[..., tuple[list[torch.Tensor], list[torch.Tensor]]]] = {
    "grpo_broadcast": advantage_grpo_broadcast,
    "reinforce_plus_plus": advantage_reinforce_plus_plus,
    "reinforce_plus_plus_baseline": advantage_reinforce_plus_plus_baseline,
    "gae": advantage_gae,
}


def compute_advantages_and_returns(
    args: Any,
    *,
    rewards,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor] | None = None,
    response_lengths: list[int] | None = None,
    total_lengths: list[int] | None = None,
    values: list[torch.Tensor] | None = None,
    padded_total_lengths: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the estimator registered for ``args.advantage_estimator``.

    ``padded_total_lengths`` is likewise call-site specific: only the Megatron
    path can compute it, and only GAE consumes it. Every parameter here is
    keyword-only and every estimator absorbs the rest, so a call site that
    cannot supply one omits it rather than inventing a value.
    """
    spec = get_algorithm(args.advantage_estimator)
    fn = ADVANTAGE_FNS[spec.advantage_fn]
    return fn(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        values=values,
        padded_total_lengths=padded_total_lengths,
    )
