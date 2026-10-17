# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A scorer takes its own placement phase only when it defers on bundles shared
with generation; sharing them without deferring is rejected."""

from types import SimpleNamespace

import pytest

from relax.engine.inference.phase_plans import (
    PHASE_GENERATE,
    PHASE_GENRM,
    PHASE_TEACHER,
    deferred_genrm_enabled,
    deferred_opd_enabled,
    placement_phase,
    reject_shared_co_resident,
)
from relax.engine.inference.placement import (
    PlacementGroupView,
    PlacementOwner,
    PlacementPlanner,
    PlacementRequest,
)


def build_args(**overrides):
    args = SimpleNamespace(
        colocate=True,
        hybrid=False,
        fully_async=False,
        rollout_num_gpus=4,
        sglang_config=None,
        prefill_num_servers=None,
        _genrm_instances_resolved={},
        opd_teacher_routes=None,
        teacher_hf_checkpoint=None,
        resource={},
        defer_reward_to_post_process=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_managed_teacher_defers_only_when_it_shares_the_rollout_bundles():
    split = build_args(
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    assert not deferred_opd_enabled(split)
    assert placement_phase(split, PHASE_TEACHER) == PHASE_GENERATE

    shared = build_args(
        rollout_num_gpus=8,
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 8], "actor": [1, 8], "rollout": [1, 8]},
        use_opd=True,
        opd_type="sglang",
    )
    assert deferred_opd_enabled(shared)
    assert placement_phase(shared, PHASE_TEACHER) == PHASE_TEACHER

    dedicated = build_args(
        colocate=False,
        hybrid=True,
        teacher_hf_checkpoint="/ckpt",
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        use_opd=True,
        opd_type="sglang",
    )
    assert not deferred_opd_enabled(dedicated)


def test_inline_genrm_occupies_the_generation_phase():
    inline = build_args(_genrm_instances_resolved={"a": {}})
    assert placement_phase(inline, PHASE_GENRM) == PHASE_GENERATE
    deferred = build_args(
        _genrm_instances_resolved={"a": {}}, _genrm_colocate_with_rollout=True, defer_reward_to_post_process=True
    )
    assert placement_phase(deferred, PHASE_GENRM) == PHASE_GENRM


def test_phase_plans_split_genrm_scores_inline_even_when_deferred():
    """Only a GenRM sharing the rollout bundles takes turns with generation.

    In split and fully-async layouts no role shares the GenRM slice, so a
    ``genrm`` phase would have nothing to switch and ``enter_phase`` would fail
    at the first publish.
    """
    split = build_args(
        _genrm_instances_resolved={"a": {}}, _genrm_colocate_with_rollout=False, defer_reward_to_post_process=True
    )
    assert not deferred_genrm_enabled(split)
    assert placement_phase(split, PHASE_GENRM) == PHASE_GENERATE


def test_shared_co_resident_genrm_is_rejected_before_startup():
    shared = build_args(_genrm_instances_resolved={"a": {}}, _genrm_colocate_with_rollout=True)
    with pytest.raises(ValueError, match="shared co-resident"):
        reject_shared_co_resident(shared)
    # Deferring makes the same bundles legal: the roles take turns.
    reject_shared_co_resident(
        build_args(
            _genrm_instances_resolved={"a": {}}, _genrm_colocate_with_rollout=True, defer_reward_to_post_process=True
        )
    )
    reject_shared_co_resident(build_args(_genrm_instances_resolved={"a": {}}))


def test_shared_co_resident_genrm_overlap_is_rejected_by_the_planner():
    args = build_args(_genrm_instances_resolved={"a": {}})
    planner = PlacementPlanner()
    view = PlacementGroupView(tuple(range(4)), tuple(range(4)), PlacementOwner.CONTROLLER, identity="pg")
    for group_id, phase in (("rollout/model", PHASE_GENERATE), ("genrm/a", placement_phase(args, PHASE_GENRM))):
        request = PlacementRequest(
            group_id=group_id,
            worker_type="regular",
            num_gpus=4,
            num_gpus_per_engine=4,
            num_gpus_per_node=4,
            phase=phase,
            bundle_offset=0,
        )
        if group_id == "genrm/a":
            with pytest.raises(ValueError, match="overlap"):
                planner.plan((request,), view)
        else:
            planner.plan((request,), view)
