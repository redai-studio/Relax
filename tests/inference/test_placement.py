# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import copy
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from relax.inference.placement import (
    InferencePlacement,
    model_placement,
    plan_inference_placement,
    validate_bound_placement,
)


def _args(**overrides):
    args = dict(
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 2], "teacher": [1, 2]},
        num_gpus_per_node=8,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        genrm_model_path="judge",
        genrm_num_gpus=2,
        genrm_num_gpus_per_engine=2,
        genrm_engine_config={},
        teacher_hf_checkpoint="teacher",
        teacher_num_gpus_per_engine=2,
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        fully_async=False,
        hybrid=False,
        offload_train=True,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_placement_split_has_combined_prefix_offsets_and_no_args_mutation():
    args = _args()
    before = copy.deepcopy(vars(args))

    plan = plan_inference_placement(args)

    assert vars(args) == before
    assert plan.mode == "split"
    assert [(p.role, p.bundle_start, p.bundle_stop) for p in plan.placements] == [
        ("rollout", 0, 4),
        ("genrm", 4, 6),
        ("teacher", 6, 8),
    ]
    assert plan.total_required_gpus == 8
    assert dict(plan.pool_sizes) == {"actor": 8}
    assert all(p.owner == "controller" for p in plan.placements)
    with pytest.raises(FrozenInstanceError):
        plan.mode = "defer"
    with pytest.raises(TypeError):
        plan.pool_sizes["actor"] = 16
    assert json.loads(json.dumps(plan.to_dict()))["mode"] == "split"


def test_placement_decoupled_counts_all_training_and_managed_teacher_pools():
    args = _args(colocate=False, fully_async=True, offload_train=False, offload_rollout=False)
    args.resource.update(reference=[1, 4], critic=[1, 2], actor_fwd=[1, 4])

    plan = plan_inference_placement(args)

    assert plan.mode == "decoupled"
    assert plan.total_required == 26
    assert {p.pool for p in plan.placements} == {"rollout", "genrm", "teacher"}
    assert all(p.bundle_start == 0 for p in plan.placements)


def test_placement_shared_critic_uses_actor_pool_but_independent_reference_counts():
    args = _args()
    args.resource.update(critic=[1, 8], reference=[1, 4])

    assert plan_inference_placement(args).total_required == 12
    args.resource["critic"] = [1, 2]
    assert plan_inference_placement(args).total_required == 14


def test_placement_unmanaged_teacher_resource_is_not_allocated():
    args = _args(use_opd=False, colocate=False)
    plan = plan_inference_placement(args)
    assert plan.for_role("teacher") == ()
    assert plan.total_required == 14


def test_placement_decoupled_external_rollout_does_not_forbid_managed_teacher():
    args = _args(colocate=False, rollout_external=True, genrm_model_path=None)
    plan = plan_inference_placement(args)
    assert plan.for_role("teacher")[0].pool == "teacher"


def test_placement_train_only_skips_inference_and_offload_requirements():
    args = _args(debug_train_only=True, offload_train=False, offload_rollout=False)
    plan = plan_inference_placement(args)
    assert plan.placements == ()
    assert plan.total_required == 8


def test_placement_named_static_models_get_separate_nonoverlapping_slices():
    args = _args(
        resource={"actor": [1, 12], "rollout": [1, 4], "genrm": [1, 4], "teacher": [1, 4]},
        _genrm_instances_resolved={
            "math": {"num_gpus": 2, "num_gpus_per_engine": 2, "engine_config": {}},
            "code": {"num_gpus": 2, "num_gpus_per_engine": 1, "engine_config": {}},
        },
        opd_teacher_routes=json.dumps({"text": "text-checkpoint", "vision": "vision-checkpoint"}),
    )
    plan = plan_inference_placement(args)
    assert [(p.role, p.model_id, p.bundle_start) for p in plan.placements] == [
        ("rollout", "default", 0),
        ("genrm", "math", 4),
        ("genrm", "code", 6),
        ("teacher", "text", 8),
        ("teacher", "vision", 10),
    ]


def test_placement_explicit_genrm_defer_reuses_rollout_bundles_in_later_phase():
    args = _args(
        resource={"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]},
        rollout_num_gpus=8,
        genrm_num_gpus=8,
        defer_reward_to_post_process=True,
        use_opd=False,
    )
    plan = plan_inference_placement(args)
    rollout, genrm = plan.placements
    assert plan.mode == "defer"
    assert rollout.bundle_start == genrm.bundle_start == 0
    assert rollout.phase != genrm.phase
    assert genrm.deferred is True
    assert plan.total_required == 8


def test_deferred_ppo_rejects_persistent_ray_actor_reservation_conflict():
    args = _args(
        resource={"actor": [1, 8], "critic": [1, 8], "rollout": [1, 8], "teacher": [1, 8]},
        rollout_num_gpus=8,
        teacher_num_gpus_per_engine=8,
        genrm_model_path=None,
        inference_defer_roles=["teacher"],
    )

    with pytest.raises(ValueError, match=r"actor\[0\].*1.2.*actor=0.4.*critic=0.4.*rollout/default=0.2.*teacher"):
        plan_inference_placement(args)


def test_deferred_grpo_three_roles_fit_persistent_fractional_reservations():
    args = _args(
        resource={"actor": [1, 8], "rollout": [1, 8], "teacher": [1, 8], "genrm": [1, 8]},
        rollout_num_gpus=8,
        genrm_num_gpus=8,
        teacher_num_gpus_per_engine=8,
        inference_defer_roles=["teacher", "genrm"],
    )

    assert plan_inference_placement(args).mode == "defer"


def test_explicit_genrm_ray_fraction_is_validated_at_its_actual_slice():
    args = _args(genrm_ray_num_gpus=0.7)

    with pytest.raises(ValueError, match=r"actor\[4\].*1.1.*genrm"):
        plan_inference_placement(args)
    args.genrm_ray_num_gpus = 0.6
    assert plan_inference_placement(args).mode == "split"


def test_ppo_split_keeps_persistent_inference_heads_disjoint():
    args = _args()
    args.resource["critic"] = [1, 8]

    assert plan_inference_placement(args).mode == "split"


@pytest.mark.parametrize("fraction", [-0.1, float("inf"), float("nan"), True])
def test_invalid_genrm_ray_fraction_rejected_before_resource_creation(fraction):
    with pytest.raises(ValueError, match="genrm_ray_num_gpus"):
        plan_inference_placement(_args(genrm_ray_num_gpus=fraction))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"offload_train": False}, "offload"),
        ({"offload_rollout": False}, "offload"),
        ({"rollout_num_gpus": 8}, "rollout_num_gpus"),
        ({"genrm_num_gpus": 3}, "GenRM resource"),
        ({"teacher_num_gpus_per_engine": 3}, "divide"),
        ({"_genrm_colocate_with_rollout": True}, "Same-phase"),
        ({"inference_defer_roles": ["actor"]}, "Only GenRM"),
        ({"fully_async": True}, "hybrid"),
        ({"rollout_external": True}, "External Rollout"),
        ({"num_gpus_per_node": 0}, "positive"),
        ({"rollout_num_gpus_per_engine": -1}, "positive"),
        ({"sglang_pp_size": 3}, "positive|TP x PP"),
        ({"sglang_config": "unread-config.yaml"}, "parsed"),
    ],
)
def test_placement_rejects_invalid_global_layouts_before_side_effects(override, match):
    with pytest.raises(ValueError, match=match):
        plan_inference_placement(_args(**override))


def test_placement_mixed_three_role_overflow_is_rejected_together():
    args = _args(resource={"actor": [1, 6], "rollout": [1, 4], "genrm": [1, 2], "teacher": [1, 2]})
    with pytest.raises(ValueError, match="Combined inference"):
        plan_inference_placement(args)


def test_placement_dp_attention_does_not_multiply_gpu_world_size():
    args = _args(sglang_dp_size=2, sglang_enable_dp_attention=True)
    assert plan_inference_placement(args).for_role("rollout")[0].tp_size == 2
    args.sglang_dp_size = 3
    with pytest.raises(ValueError, match="DP attention"):
        plan_inference_placement(args)


def test_placement_multinode_counts_worker_slots_and_validates_whole_nodes():
    args = _args(
        colocate=False,
        use_opd=False,
        genrm_model_path=None,
        num_gpus_per_node=4,
        resource={"actor": [1, 8], "rollout": [1, 16]},
        rollout_num_gpus=16,
        rollout_num_gpus_per_engine=8,
    )
    model = plan_inference_placement(args).for_role("rollout")[0]
    assert (model.replicas, model.nodes_per_engine, model.num_worker_slots) == (2, 2, 4)
    args.resource["rollout"] = [1, 12]
    args.rollout_num_gpus = 12
    args.rollout_num_gpus_per_engine = 6
    with pytest.raises(ValueError, match="whole configured nodes"):
        plan_inference_placement(args)


def test_placement_resolves_pd_engine_groups_without_reading_config_file():
    config = {
        "sglang": [
            {
                "name": "policy",
                "engine_groups": [
                    {"worker_type": "prefill", "num_gpus": 2, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 2, "num_gpus_per_engine": 2},
                ],
            }
        ]
    }
    plan = plan_inference_placement(_args(sglang_config="ignored.yaml", _inference_rollout_config=config))
    assert [(p.worker_type, p.bundle_start) for p in plan.for_role("rollout")] == [("prefill", 0), ("decode", 2)]


def test_placement_lookup_accepts_serialized_plan_and_legacy_single_default():
    args = _args()
    plan = plan_inference_placement(args)
    assert model_placement(args, "teacher") is None
    args._inference_placement_plan = plan.to_dict()
    assert model_placement(args, "teacher").bundle_start == 6
    assert model_placement(args, "rollout").model_id == "default"
    assert model_placement(args, "genrm", "missing") is None


def _placement():
    return InferencePlacement(
        "teacher", "model", 8, 8, 0, "teacher", "teacher", "inference", num_gpus_per_node=4, tp_size=8
    )


def _topology():
    return [{"bundle_index": index, "node_id": f"node-{index // 4}", "gpu_id": index % 4} for index in range(8)]


def test_bound_placement_uses_reordered_bundle_indices_and_logical_multinode_heads():
    topology = _topology()
    for index, item in enumerate(topology):
        item["bundle_index"] = 7 - index
    result = validate_bound_placement(_placement(), topology, bundle_indices=list(reversed(range(8))))
    assert [item["bundle_index"] for item in result] == list(reversed(range(8)))
    assert len(result) == 8


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_gpu", "same physical"),
        ("mixed_node", "one node"),
        ("noncontiguous", "contiguous"),
        ("same_node", "distinct nodes"),
        ("missing_node", "node identity"),
    ],
)
def test_bound_placement_rejects_illegal_physical_layout(mutation, match):
    topology = _topology()
    if mutation == "duplicate_gpu":
        topology[1]["gpu_id"] = 0
    elif mutation == "mixed_node":
        topology[1]["node_id"] = "node-other"
    elif mutation == "noncontiguous":
        topology[1]["gpu_id"] = 9
    elif mutation == "same_node":
        for item in topology[4:]:
            item["node_id"] = "node-0"
            item["gpu_id"] += 4
    else:
        topology[0].pop("node_id")
    with pytest.raises(ValueError, match=match):
        validate_bound_placement(_placement().to_dict(), topology)


def test_bound_placement_refuses_pool_overflow_and_unknown_bundle_order():
    with pytest.raises(ValueError, match="exceeds"):
        validate_bound_placement(replace(_placement(), bundle_start=1), _topology())
    with pytest.raises(ValueError, match="unknown indices"):
        validate_bound_placement(_placement(), _topology(), bundle_indices=[99])
