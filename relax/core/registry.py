# Copyright (c) 2026 Relax Authors. All Rights Reserved.

try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


from relax.algorithms import ALGORITHM_SPECS, algorithm_needs_critic
from relax.components.actor import Actor
from relax.components.actor_fwd import ActorFwd
from relax.components.advantages import Advantages
from relax.components.critic import Critic
from relax.components.rollout import Rollout
from relax.components.sft import SFT


# NOTE(dev): Use StrEnum and keep visiting order with definition order
class ROLES(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"
    actor_fwd: str = "actor_fwd"
    sft: str = "sft"


class ROLES_TRAIN_ONLY(StrEnum):
    actor: str = "actor"


class ROLES_ROLLOUT_ONLY(StrEnum):
    rollout: str = "rollout"


class ROLES_COLOCATE(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"


class ROLES_SFT_ONLY(StrEnum):
    sft: str = "sft"
    actor: str = "actor"


class ROLES_FULLY_ASYNC_ON_POLICY(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"


class ROLES_PPO_COLOCATE(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"


class ROLES_PPO_FULLY_ASYNC(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"
    actor_fwd: str = "actor_fwd"


class ROLES_PPO_FULLY_ASYNC_ON_POLICY(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"


def _rl_roles(*, needs_critic: bool) -> dict:
    """Component classes one RL algorithm binds to each role.

    Every RL algorithm starts the same services except for the critic, which
    only value-based estimators need.  Deriving the table from the registry is
    what keeps ``ALGOS`` and ``--advantage-estimator`` from drifting apart: an
    estimator argument parsing accepts can no longer fail here at
    service-registration time, because both sides read the same dict.

    This decides which class a role maps to, *not* which roles the controller
    walks -- that is ``process_role``'s job, which reads the same
    ``needs_critic`` through ``algorithm_needs_critic``.  ``controller.py``
    iterates ``list(process_role(config))`` and
    skips any role missing from this dict, so an algorithm without a critic
    simply never matches the ``critic`` member the role sets already carry.
    """
    roles = {
        ROLES.rollout: Rollout,
        ROLES.actor: Actor,
    }
    if needs_critic:
        roles[ROLES.critic] = Critic
    roles[ROLES.advantages] = Advantages
    roles[ROLES.reference] = ActorFwd
    roles[ROLES.actor_fwd] = ActorFwd
    return roles


# NOTE(dev): `ALGOS` keys live in a different namespace from AlgorithmSpec names.
# "sft" is selected by `loss_type`, not by `--advantage-estimator`, so it stays a
# separate literal entry rather than being folded into the algorithm registry.
ALGOS = {name: _rl_roles(needs_critic=spec.needs_critic) for name, spec in ALGORITHM_SPECS.items()}
ALGOS["sft"] = {
    ROLES.sft: SFT,
    ROLES.actor: Actor,
}


def process_role(config):
    if config.debug_rollout_only:
        return ROLES_ROLLOUT_ONLY
    if config.debug_train_only:
        return ROLES_TRAIN_ONLY
    if getattr(config, "loss_type", None) == "sft":
        return ROLES_SFT_ONLY
    if algorithm_needs_critic(config):
        if config.fully_async:
            if getattr(config, "true_on_policy_mode", False):
                return ROLES_PPO_FULLY_ASYNC_ON_POLICY
            return ROLES_PPO_FULLY_ASYNC
        return ROLES_PPO_COLOCATE
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        if getattr(config, "true_on_policy_mode", False):
            # actor_fwd's log_probs equal the train forward's log_probs in this regime
            # (same weights, deterministic Megatron forward), so we recompute inline.
            return ROLES_FULLY_ASYNC_ON_POLICY
        return ROLES
    else:
        return ROLES_COLOCATE
