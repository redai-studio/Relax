# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Placement planning for rollout / GenRM / OPD teacher, without a Ray cluster.

The rule-layer tests build claims by hand. The adapter-layer tests feed
``plan_placement`` the sparse ``Namespace`` shapes the launch path produces
after argument parsing.
"""

import copy
import json
import sys
from argparse import Namespace

import pytest

from relax.distributed.ray.placement_planner import (
    Phase,
    PlacementClaim,
    PlacementError,
    PlacementPlan,
    PlacementPool,
    PoolOwner,
    plan_placement,
    validate_placement,
)


GENERATE = frozenset({Phase.GENERATE})
SCORE = frozenset({Phase.SCORE})
GENERATE_AND_SCORE = frozenset({Phase.GENERATE, Phase.SCORE})


def _claim(role, model="__default__", *, pool="actor", start=0, size=4, phases=GENERATE):
    return PlacementClaim(role, model, pool, start, size, phases)


def _plan(*claims, pool_size=8):
    pools = (
        PlacementPool("actor", pool_size, PoolOwner.CONTROLLER),
        PlacementPool("genrm", pool_size, PoolOwner.SERVICE),
    )
    return PlacementPlan(pools=pools, claims=tuple(claims))


def _genrm_spec(num_gpus, num_gpus_per_engine=1):
    return {
        "model_path": "/judge",
        "num_gpus": num_gpus,
        "num_gpus_per_engine": num_gpus_per_engine,
        "engine_config": {},
        "sampling_config": {},
    }


def _colocate_args(**overrides):
    values = dict(
        colocate=True,
        hybrid=False,
        fully_async=False,
        rollout_num_gpus=4,
        resource={"actor": [1, 8], "rollout": [1, 4]},
    )
    values.update(overrides)
    return Namespace(**values)


def _teacher_args(**overrides):
    values = dict(
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="/teacher",
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
    )
    values.update(overrides)
    return _colocate_args(**values)


def _claims_of(plan, role):
    return [claim for claim in plan.claims if claim.role == role]


# ----------------------------------------------------------------------
# Rule layer: data model.
# ----------------------------------------------------------------------


def test_placement_planner_plan_returns_claim_by_role_and_model():
    quality = _claim("genrm", "quality", start=4, size=2)
    safety = _claim("genrm", "safety", start=6, size=2)
    plan = _plan(_claim("rollout"), quality, safety)

    assert plan.claim("genrm", "safety") is safety
    assert plan.claim("rollout").stop == 4
    assert plan.pool("actor").owner is PoolOwner.CONTROLLER


def test_placement_planner_plan_unknown_claim_raises_placement_error():
    plan = _plan(_claim("rollout"))

    with pytest.raises(PlacementError, match="genrm/quality"):
        plan.claim("genrm", "quality")
    assert issubclass(PlacementError, ValueError)


# ----------------------------------------------------------------------
# Rule layer: conflicts.
# ----------------------------------------------------------------------


def test_placement_planner_same_phase_overlap_is_rejected():
    plan = _plan(_claim("genrm", "judge", start=4), _claim("teacher", "math", start=4))

    with pytest.raises(PlacementError) as excinfo:
        validate_placement(plan)

    message = str(excinfo.value)
    assert "pool 'actor'" in message
    assert "genrm/judge pool=actor [4, 8)" in message
    assert "teacher/math pool=actor [4, 8)" in message
    assert "['generate']" in message


def test_placement_planner_disjoint_ranges_are_accepted():
    validate_placement(_plan(_claim("genrm", start=0), _claim("teacher", start=4)))


def test_placement_planner_different_phases_are_accepted():
    validate_placement(_plan(_claim("genrm", start=4, phases=SCORE), _claim("teacher", start=4, phases=GENERATE)))


def test_placement_planner_different_pools_are_accepted():
    validate_placement(_plan(_claim("genrm", pool="genrm"), _claim("teacher", pool="actor")))


def test_placement_planner_rollout_and_genrm_may_co_reside():
    validate_placement(_plan(_claim("rollout", size=8), _claim("genrm", size=8)))


def test_placement_planner_teacher_overlapping_rollout_is_rejected():
    plan = _plan(_claim("rollout", size=8), _claim("teacher", start=4, phases=GENERATE_AND_SCORE))

    with pytest.raises(PlacementError, match="rollout/__default__.*teacher/__default__"):
        validate_placement(plan)


def test_placement_planner_two_genrm_instances_overlapping_is_rejected():
    plan = _plan(_claim("genrm", "quality", start=4), _claim("genrm", "safety", start=6, size=2))

    with pytest.raises(PlacementError, match="genrm/quality.*genrm/safety"):
        validate_placement(plan)


# ----------------------------------------------------------------------
# Rule layer: capacity.
# ----------------------------------------------------------------------


def test_placement_planner_genrm_exceeding_pool_is_rejected():
    plan = _plan(_claim("genrm", "judge", start=8, size=8))

    with pytest.raises(PlacementError) as excinfo:
        validate_placement(plan)

    message = str(excinfo.value)
    assert "genrm/judge pool=actor [8, 16)" in message
    assert "8 GPU bundle(s)" in message


def test_placement_planner_teacher_exceeding_pool_is_rejected():
    plan = _plan(_claim("teacher", "math", start=6, size=4))

    with pytest.raises(PlacementError, match=r"teacher/math pool=actor \[6, 10\)"):
        validate_placement(plan)


def test_placement_planner_rollout_exceeding_pool_is_not_rejected():
    validate_placement(_plan(_claim("rollout", size=16)))


# ----------------------------------------------------------------------
# Adapter layer: pools and the rollout region.
# ----------------------------------------------------------------------


def test_placement_planner_colocate_uses_shared_pool_owned_by_controller():
    plan = plan_placement(_colocate_args(rollout_num_gpus=8, resource={"actor": [1, 8], "rollout": [1, 8]}))

    assert plan.pools == (PlacementPool("actor", 8, PoolOwner.CONTROLLER),)
    assert plan.claims == (PlacementClaim("rollout", "__default__", "actor", 0, 8, GENERATE),)


def test_placement_planner_hybrid_uses_role_owned_pools():
    args = _colocate_args(
        hybrid=True,
        fully_async=True,
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 2]},
        _genrm_instances_resolved={"__default__": _genrm_spec(2)},
    )

    plan = plan_placement(args)

    assert plan.pools == (
        PlacementPool("rollout", 4, PoolOwner.SERVICE),
        PlacementPool("genrm", 2, PoolOwner.SERVICE),
    )
    assert plan.claim("rollout").pool == "rollout"
    assert plan.claim("genrm").pool == "genrm"


def test_placement_planner_fully_async_uses_role_owned_pools():
    args = Namespace(colocate=False, hybrid=False, fully_async=True, resource={"actor": [1, 4], "rollout": [1, 4]})

    plan = plan_placement(args)

    assert plan.pools == (PlacementPool("rollout", 4, PoolOwner.SERVICE),)
    assert plan.claims == (PlacementClaim("rollout", "__default__", "rollout", 0, 4, GENERATE),)


def test_placement_planner_resource_without_rollout_has_no_shared_pool():
    # SFT-only style: colocate is set but no rollout role is launched.
    plan = plan_placement(Namespace(colocate=True, hybrid=False, resource={"actor": [1, 8]}))

    assert plan.pools == ()
    assert plan.claims == ()


def test_placement_planner_rollout_region_falls_back_to_resource():
    args = Namespace(colocate=True, hybrid=False, resource={"actor": [1, 8], "rollout": [1, 6]})

    assert plan_placement(args).claim("rollout").stop == 6


def test_placement_planner_sparse_namespace_is_accepted():
    assert plan_placement(Namespace()) == PlacementPlan(pools=(), claims=())

    plan = plan_placement(Namespace(_genrm_instances_resolved={"quality": _genrm_spec(1), "safety": _genrm_spec(1)}))

    # No --resource entry: the dedicated GenRM pool is sized by its instances.
    assert plan.pool("genrm") == PlacementPool("genrm", 2, PoolOwner.SERVICE)
    assert (plan.claim("genrm", "quality").start, plan.claim("genrm", "safety").start) == (0, 1)


# ----------------------------------------------------------------------
# Adapter layer: GenRM.
# ----------------------------------------------------------------------


def test_placement_planner_genrm_split_layout():
    plan = plan_placement(_colocate_args(_genrm_instances_resolved={"__default__": _genrm_spec(4)}))

    rollout, genrm = plan.claim("rollout"), plan.claim("genrm")
    assert (rollout.pool, rollout.start, rollout.stop) == ("actor", 0, 4)
    assert (genrm.pool, genrm.start, genrm.stop) == ("actor", 4, 8)


def test_placement_planner_genrm_shared_layout():
    args = _colocate_args(
        rollout_num_gpus=8,
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        _genrm_instances_resolved={"__default__": _genrm_spec(8)},
        _genrm_colocate_with_rollout=True,
    )

    plan = plan_placement(args)

    for claim in (plan.claim("rollout"), plan.claim("genrm")):
        assert (claim.pool, claim.start, claim.stop) == ("actor", 0, 8)


def test_placement_planner_genrm_multi_instance_prefix_sum():
    equal = plan_placement(
        _colocate_args(_genrm_instances_resolved={"quality": _genrm_spec(2), "safety": _genrm_spec(2)})
    )
    assert (equal.claim("genrm", "quality").start, equal.claim("genrm", "quality").stop) == (4, 6)
    assert (equal.claim("genrm", "safety").start, equal.claim("genrm", "safety").stop) == (6, 8)

    # Unequal budgets: the second region starts where the first one ended.
    unequal = plan_placement(
        _colocate_args(_genrm_instances_resolved={"quality": _genrm_spec(3), "safety": _genrm_spec(1)})
    )
    assert (unequal.claim("genrm", "safety").start, unequal.claim("genrm", "safety").stop) == (7, 8)


def test_placement_planner_genrm_own_pool_starts_at_zero():
    args = Namespace(
        colocate=False,
        hybrid=False,
        fully_async=True,
        rollout_num_gpus=4,
        resource={"actor": [1, 4], "rollout": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"quality": _genrm_spec(2), "safety": _genrm_spec(2)},
    )

    plan = plan_placement(args)

    assert plan.pool("genrm") == PlacementPool("genrm", 4, PoolOwner.SERVICE)
    assert (plan.claim("genrm", "quality").start, plan.claim("genrm", "quality").stop) == (0, 2)
    assert (plan.claim("genrm", "safety").start, plan.claim("genrm", "safety").stop) == (2, 4)


def test_placement_planner_genrm_defer_resides_in_score_phase():
    instances = {"__default__": _genrm_spec(4)}

    inline = plan_placement(_colocate_args(_genrm_instances_resolved=instances))
    deferred = plan_placement(_colocate_args(_genrm_instances_resolved=instances, defer_reward_to_post_process=True))

    assert inline.claim("genrm").phases == GENERATE
    assert deferred.claim("genrm").phases == SCORE
    assert deferred.claim("rollout").phases == GENERATE


# ----------------------------------------------------------------------
# Adapter layer: managed OPD teacher.
# ----------------------------------------------------------------------


def test_placement_planner_teacher_single_colocate_layout():
    teacher = plan_placement(_teacher_args()).claim("teacher")

    assert (teacher.pool, teacher.start, teacher.stop) == ("actor", 4, 8)


def test_placement_planner_teacher_multi_colocate_layout():
    args = _teacher_args(
        teacher_hf_checkpoint=None,
        opd_teacher_routes=json.dumps({"math": "/ckpt/math", "code": "/ckpt/code"}),
        rollout_num_gpus=8,
        resource={"actor": [1, 16], "rollout": [1, 8], "teacher": [1, 8]},
    )

    plan = plan_placement(args)

    assert (plan.claim("teacher", "math").start, plan.claim("teacher", "math").stop) == (8, 12)
    assert (plan.claim("teacher", "code").start, plan.claim("teacher", "code").stop) == (12, 16)
    assert plan.pool("actor").owner is PoolOwner.CONTROLLER


def test_placement_planner_teacher_dedicated_replicas_use_private_pools():
    args = _teacher_args(
        colocate=False,
        fully_async=True,
        teacher_num_gpus_per_engine=4,
        resource={"actor": [1, 4], "rollout": [1, 4], "teacher": [1, 8]},
    )

    plan = plan_placement(args)

    replicas = _claims_of(plan, "teacher")
    assert [(claim.start, claim.stop) for claim in replicas] == [(0, 4), (0, 4)]
    assert len({claim.pool for claim in replicas}) == 2
    for claim in replicas:
        assert plan.pool(claim.pool) == PlacementPool(claim.pool, 4, PoolOwner.MANAGER)


def test_placement_planner_teacher_resides_in_generate_and_score():
    assert plan_placement(_teacher_args()).claim("teacher").phases == GENERATE_AND_SCORE


def test_placement_planner_non_opd_run_does_not_touch_opd_utils(monkeypatch):
    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"non-OPD planning touched opd_utils.{name}")

    monkeypatch.setitem(sys.modules, "relax.utils.opd.opd_utils", _Exploding())

    plan = plan_placement(_colocate_args(_genrm_instances_resolved={"__default__": _genrm_spec(4)}))

    assert _claims_of(plan, "teacher") == []


# ----------------------------------------------------------------------
# Adapter layer: validation wired in.
# ----------------------------------------------------------------------


def _genrm_with_colocate_teacher_args(**overrides):
    # Argument parsing skips the GenRM split/shared decision when a managed
    # teacher is enabled, so _genrm_colocate_with_rollout stays False.
    values = dict(
        resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4], "genrm": [1, 4]},
        _genrm_instances_resolved={"__default__": _genrm_spec(4)},
        _genrm_colocate_with_rollout=False,
    )
    values.update(overrides)
    return _teacher_args(**values)


def test_placement_planner_genrm_with_colocate_teacher_is_rejected():
    with pytest.raises(PlacementError) as excinfo:
        plan_placement(_genrm_with_colocate_teacher_args())

    message = str(excinfo.value)
    assert "pool 'actor'" in message
    assert "genrm/__default__ pool=actor [4, 8)" in message
    assert "teacher/__default__ pool=actor [4, 8)" in message


def test_placement_planner_deferred_genrm_with_colocate_teacher_is_rejected():
    # The teacher stays resident while a deferred GenRM scores.
    with pytest.raises(PlacementError, match=r"\['score'\]"):
        plan_placement(_genrm_with_colocate_teacher_args(defer_reward_to_post_process=True))


def test_placement_planner_genrm_region_past_shared_pool_is_rejected():
    args = _colocate_args(rollout_num_gpus=8, _genrm_instances_resolved={"__default__": _genrm_spec(8)})

    with pytest.raises(PlacementError, match=r"genrm/__default__ pool=actor \[8, 16\).*8 GPU bundle"):
        plan_placement(args)


def _shared_bundle_args(num_gpus, **overrides):
    values = dict(
        rollout_num_gpus=num_gpus,
        resource={"actor": [1, num_gpus], "rollout": [1, num_gpus], "genrm": [1, num_gpus]},
        _genrm_instances_resolved={"__default__": _genrm_spec(num_gpus)},
        _genrm_colocate_with_rollout=True,
    )
    values.update(overrides)
    return _colocate_args(**values)


def test_placement_planner_co_resident_layout_is_accepted():
    plan = plan_placement(_shared_bundle_args(8))

    assert plan.claim("rollout").phases == plan.claim("genrm").phases == GENERATE


def test_placement_planner_defer_swap_layout_is_accepted():
    plan = plan_placement(_shared_bundle_args(16, defer_reward_to_post_process=True))

    assert plan.claim("rollout").phases.isdisjoint(plan.claim("genrm").phases)


def test_placement_planner_rollout_only_is_accepted():
    plan_placement(_colocate_args())
    # Advisory claim: a rollout region past the pool is left to the existing
    # rollout engine-group validation.
    plan_placement(_colocate_args(rollout_num_gpus=16))


def test_placement_planner_debug_train_only_skips_validation():
    plan = plan_placement(_genrm_with_colocate_teacher_args(debug_train_only=True))

    assert plan.claim("genrm").start == plan.claim("teacher").start == 4


def test_placement_planner_same_config_plans_identically():
    args = _teacher_args()
    before = copy.deepcopy(vars(args))

    assert plan_placement(args) == plan_placement(args)
    assert vars(args) == before


def test_placement_planner_genrm_without_shared_pool_starts_at_zero():
    """Neither colocate nor fully-async (e.g. --debug-rollout-only): GenRM runs
    on its own 2-bundle placement group.

    Before the planner, GenRMManager computed ``rollout_num_gpus + rank * k``
    here -- bundles 4 and 5 -- which lies past that placement group.
    """
    args = Namespace(
        colocate=False,
        hybrid=False,
        fully_async=False,
        rollout_num_gpus=4,
        resource={"rollout": [1, 4], "genrm": [1, 2]},
        _genrm_instances_resolved={"__default__": _genrm_spec(2)},
    )

    genrm = plan_placement(args).claim("genrm")

    assert (genrm.pool, genrm.start, genrm.stop) == ("genrm", 0, 2)
