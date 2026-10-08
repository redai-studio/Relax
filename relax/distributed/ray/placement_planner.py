# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Placement planning for the inference roles (rollout, GenRM, OPD teacher).

From the run configuration alone, work out which bundle range of which
placement group every inference model occupies, and reject layouts in which
two models would hold GPU memory on the same bundles at the same time.

Two layers:

- Rule layer: immutable pools/claims plus the capacity and conflict checks.
  It knows nothing about ``args``.
- Adapter layer: ``plan_placement(args)`` turns the run configuration into
  pools and claims and runs the checks.

Coordinates are indices into a placement group's bundle list sorted by
``(node, gpu id)``, i.e. the ``reordered_*`` lists ``create_placement_group``
returns. Validation is logical only: node boundaries and GPU-id contiguity
depend on the real placement group and are not checked here.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from relax.core.optional_roles import GENRM_ROLE


ROLLOUT_ROLE = "rollout"
TEACHER_ROLE = "teacher"

# Model key of a role's sole model when it has no route keys.
DEFAULT_MODEL_KEY = "__default__"

# The placement group actor and rollout share in colocate mode.
ACTOR_POOL = "actor"


class PlacementError(ValueError):
    """The run configuration asks for an inference layout that cannot be
    placed."""


class Phase(str, Enum):
    """A window of the rollout step during which a model holds GPU memory."""

    GENERATE = "generate"
    SCORE = "score"


class PoolOwner(str, Enum):
    """Who creates, and therefore removes, a pool's placement group."""

    CONTROLLER = "controller"
    SERVICE = "service"
    MANAGER = "manager"


@dataclass(frozen=True)
class PlacementPool:
    """One placement group, as a number of GPU bundles."""

    name: str
    size: int
    owner: PoolOwner


@dataclass(frozen=True)
class PlacementClaim:
    """The bundle range ``[start, stop)`` of ``pool`` one model occupies, and
    the phases during which it holds GPU memory there."""

    role: str
    model: str
    pool: str
    start: int
    size: int
    phases: frozenset[Phase]

    @property
    def stop(self) -> int:
        return self.start + self.size


@dataclass(frozen=True)
class PlacementPlan:
    pools: tuple[PlacementPool, ...]
    claims: tuple[PlacementClaim, ...]

    def pool(self, name: str) -> PlacementPool:
        for pool in self.pools:
            if pool.name == name:
                return pool
        raise PlacementError(f"No placement pool named '{name}'; known pools: {[p.name for p in self.pools]}.")

    def claim(self, role: str, model: str = DEFAULT_MODEL_KEY) -> PlacementClaim:
        """Return the single claim of ``model`` under ``role``."""
        matches = [claim for claim in self.claims if claim.role == role and claim.model == model]
        if len(matches) != 1:
            raise PlacementError(f"Expected exactly one placement claim for {role}/{model}, found {len(matches)}.")
        return matches[0]

    def describe(self) -> str:
        """One line per claim, for logging."""
        return "\n".join(f"  {_describe(claim)} phases={_phase_names(claim.phases)}" for claim in self.claims)


# ----------------------------------------------------------------------
# Rule layer.
# ----------------------------------------------------------------------

# Role pairs that may hold GPU memory on the same bundles in the same phase.
# Rollout + GenRM keeps the documented GenRM "Shared / Co-resident" layout;
# RFC #71 rejects every other co-resident combination. This is the only place
# the exception lives -- empty the set to reject co-residence outright.
_CO_RESIDENT_ROLE_PAIRS: frozenset[frozenset[str]] = frozenset({frozenset({ROLLOUT_ROLE, GENRM_ROLE})})

# Roles whose bundle range the planner owns. The rollout claim is advisory: its
# real engine-group layout comes from --sglang-config and is bounds-checked by
# rollout_validation.py, so it only takes part in overlap checks.
_CAPACITY_CHECKED_ROLES: frozenset[str] = frozenset({GENRM_ROLE, TEACHER_ROLE})


def _may_co_reside(a: PlacementClaim, b: PlacementClaim) -> bool:
    return frozenset({a.role, b.role}) in _CO_RESIDENT_ROLE_PAIRS


def _phase_names(phases: frozenset[Phase]) -> list[str]:
    return sorted(phase.value for phase in phases)


def _describe(claim: PlacementClaim) -> str:
    return f"{claim.role}/{claim.model} pool={claim.pool} [{claim.start}, {claim.stop})"


def claims_overlap(a: PlacementClaim, b: PlacementClaim) -> bool:
    """Whether two claims cover at least one common bundle of the same pool."""
    return a.pool == b.pool and a.start < b.stop and b.start < a.stop


def validate_placement(plan: PlacementPlan) -> None:
    """Raise ``PlacementError`` if a claim does not fit its pool, or two claims
    hold GPU memory on the same bundles in the same phase."""
    for claim in plan.claims:
        if claim.role not in _CAPACITY_CHECKED_ROLES:
            continue
        pool = plan.pool(claim.pool)
        if claim.start < 0 or claim.stop > pool.size:
            raise PlacementError(
                f"Placement out of range: {_describe(claim)} does not fit pool '{pool.name}' "
                f"of {pool.size} GPU bundle(s)."
            )

    for a, b in itertools.combinations(plan.claims, 2):
        if not claims_overlap(a, b):
            continue
        shared_phases = a.phases & b.phases
        if not shared_phases or _may_co_reside(a, b):
            continue
        raise PlacementError(
            f"Placement conflict in pool '{a.pool}': {_describe(a)} and {_describe(b)} both hold GPU memory "
            f"in phase(s) {_phase_names(shared_phases)}. Only rollout and GenRM may share bundles in the "
            f"same phase; give the two models disjoint bundle ranges or run them on separate placement groups."
        )


# ----------------------------------------------------------------------
# Adapter layer.
# ----------------------------------------------------------------------


def _resource_gpus(resource: dict, role: str) -> int:
    # --resource maps role -> [num_serves, num_gpus].
    return int(resource[role][1])


def _shares_actor_pool(args: Any, resource: dict) -> bool:
    """True when rollout/GenRM/teacher live inside the actor placement group.

    Same condition ``is_managed_opd_teacher_colocate`` applies to the teacher,
    so all three roles follow one rule.
    """
    return (
        bool(getattr(args, "colocate", False))
        and not getattr(args, "hybrid", False)
        and "actor" in resource
        and ROLLOUT_ROLE in resource
    )


def _genrm_claims(
    args: Any, resource: dict, pools: dict[str, PlacementPool], shared: bool, rollout_stop: int
) -> list[PlacementClaim]:
    instances = getattr(args, "_genrm_instances_resolved", None) or {}
    if not instances:
        return []

    if shared:
        pool = ACTOR_POOL
        # Shared bundles (co-resident or defer-swap) start at 0; split starts
        # right after the rollout region.
        start = 0 if getattr(args, "_genrm_colocate_with_rollout", False) else rollout_stop
    else:
        pool = GENRM_ROLE
        total = sum(int(spec["num_gpus"]) for spec in instances.values())
        size = _resource_gpus(resource, GENRM_ROLE) if GENRM_ROLE in resource else total
        pools[pool] = PlacementPool(pool, size, PoolOwner.SERVICE)
        start = 0

    # With --defer-reward-to-post-process GenRM stays offloaded while rollout
    # generates and is only woken to score the finished batch.
    deferred = getattr(args, "defer_reward_to_post_process", False)
    phases = frozenset({Phase.SCORE if deferred else Phase.GENERATE})

    claims = []
    for key, spec in instances.items():
        num_gpus = int(spec["num_gpus"])
        claims.append(PlacementClaim(GENRM_ROLE, key, pool, start, num_gpus, phases))
        # Prefix sum: instances may have unequal GPU budgets.
        start += num_gpus
    return claims


def _teacher_claims(
    args: Any, resource: dict, pools: dict[str, PlacementPool], rollout_stop: int
) -> list[PlacementClaim]:
    if not getattr(args, "use_opd", False):
        return []

    # Deferred: opd_utils pulls in torch, and it imports this module back.
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_colocate, is_managed_opd_teacher_enabled

    if not is_managed_opd_teacher_enabled(args):
        return []

    routes_json = getattr(args, "opd_teacher_routes", None)
    keys = list(json.loads(routes_json)) if routes_json is not None else [DEFAULT_MODEL_KEY]
    if not keys:
        return []
    gpus_per_teacher = _resource_gpus(resource, TEACHER_ROLE) // len(keys)

    # A managed teacher is woken with rollout and only offloaded before
    # training, so it is still resident while a deferred GenRM scores.
    phases = frozenset({Phase.GENERATE, Phase.SCORE})

    claims = []
    if is_managed_opd_teacher_colocate(args):
        start = rollout_stop
        for key in keys:
            claims.append(PlacementClaim(TEACHER_ROLE, key, ACTOR_POOL, start, gpus_per_teacher, phases))
            start += gpus_per_teacher
        return claims

    # Dedicated: every replica creates and owns a placement group of its own.
    gpus_per_replica = getattr(args, "teacher_num_gpus_per_engine", None) or gpus_per_teacher
    num_replicas = gpus_per_teacher // gpus_per_replica if gpus_per_replica > 0 else 0
    for key in keys:
        for replica in range(num_replicas):
            pool = f"{TEACHER_ROLE}/{key}/{replica}"
            pools[pool] = PlacementPool(pool, gpus_per_replica, PoolOwner.MANAGER)
            claims.append(PlacementClaim(TEACHER_ROLE, key, pool, 0, gpus_per_replica, phases))
    return claims


def _check_deferred_scoring_layout(args: Any, claims: list[PlacementClaim]) -> None:
    """Scoring deferred to the score phase swaps GPU memory with rollout, which
    only means something for models that live in the pool rollout lives in."""
    # Deferred: the module imports this one back.
    from relax.engine.inference.deferred import is_framework_deferred_reward

    if not is_framework_deferred_reward(args):
        return
    for claim in claims:
        if claim.role == GENRM_ROLE and claim.pool != ACTOR_POOL:
            raise PlacementError(
                f"--defer-reward-to-post-process needs GenRM to share the actor placement group with rollout, "
                f"but {_describe(claim)} has its own. With GPUs of its own GenRM has nothing to swap with: drop "
                f"the flag, or run colocate with 'actor' and 'rollout' in --resource."
            )


def plan_placement(args: Any, *, validate: bool = True) -> PlacementPlan:
    """Plan where every inference model of this run is placed, and validate it.

    A pure function of the run configuration: it needs no Ray connection and
    never writes to ``args``, so the controller preflight and the managers that
    consume the plan can each call it and get the same answer. Missing
    attributes fall back to "feature off".

    Args:
        validate: Pass ``False`` to only read a layout that was already
            validated when the run started.

    Raises:
        PlacementError: the layout does not fit or two models conflict. Not
            raised under ``--debug-train-only``, which starts no inference
            engine.
    """
    resource = getattr(args, "resource", None) or {}
    pools: dict[str, PlacementPool] = {}
    claims: list[PlacementClaim] = []
    generate = frozenset({Phase.GENERATE})

    shared = _shares_actor_pool(args, resource)
    rollout_stop = 0
    if shared:
        pools[ACTOR_POOL] = PlacementPool(ACTOR_POOL, _resource_gpus(resource, "actor"), PoolOwner.CONTROLLER)
        rollout_num_gpus = getattr(args, "rollout_num_gpus", None)
        if rollout_num_gpus is None:
            rollout_num_gpus = _resource_gpus(resource, ROLLOUT_ROLE)
        rollout_stop = int(rollout_num_gpus)
        claims.append(PlacementClaim(ROLLOUT_ROLE, DEFAULT_MODEL_KEY, ACTOR_POOL, 0, rollout_stop, generate))
    elif ROLLOUT_ROLE in resource:
        size = _resource_gpus(resource, ROLLOUT_ROLE)
        pools[ROLLOUT_ROLE] = PlacementPool(ROLLOUT_ROLE, size, PoolOwner.SERVICE)
        claims.append(PlacementClaim(ROLLOUT_ROLE, DEFAULT_MODEL_KEY, ROLLOUT_ROLE, 0, size, generate))

    claims.extend(_genrm_claims(args, resource, pools, shared, rollout_stop))
    claims.extend(_teacher_claims(args, resource, pools, rollout_stop))

    plan = PlacementPlan(pools=tuple(pools.values()), claims=tuple(claims))
    if validate and not getattr(args, "debug_train_only", False):
        validate_placement(plan)
        _check_deferred_scoring_layout(args, claims)
    return plan
