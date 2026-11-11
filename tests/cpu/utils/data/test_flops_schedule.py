# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.data.flops_schedule import (
    build_workload_microbatch_plan,
    expand_workload_microbatch_plan,
    flops_microbatch_geometry,
    get_flops_estimator,
    unequal_flops_dp_enabled,
    validate_flops_scheduling_args,
    validate_text_rollout,
)
from relax.utils.training.flops_counter import SequenceFlopsEstimator


def _args(**overrides):
    values = dict(loss_type="sft", use_dynamic_batch_size=True, flops_microbatch_packing=True)
    values.update(overrides)
    return Namespace(**values)


def test_flops_flags_disabled_preserve_other_training_modes():
    validate_flops_scheduling_args(Namespace())
    assert get_flops_estimator(Namespace()) is None


def test_flops_microbatch_geometry_is_shared_by_iterator_and_prepack():
    args = Namespace(data_pad_size_multiplier=256, is_vl_model=False, uses_unsplit_forward=True, allgather_cp=True)
    assert flops_microbatch_geometry(args, 2, 4) == {
        "tp_size": 2,
        "cp_size": 4,
        "pad_multiplier": 256,
        "is_vl_model": True,
        "allgather_cp": True,
    }


def test_variable_dp_counts_require_a_supported_fixed_sft_window():
    args = _args(flops_dp_balancing=True, calculate_per_token_loss=True, global_batch_size=16, rollout_batch_size=16)
    assert unequal_flops_dp_enabled(args)
    for name, value in (
        ("flops_dp_balancing", False),
        ("calculate_per_token_loss", False),
        ("pipeline_model_parallel_size", 2),
        ("sft_async_prepack", True),
        ("debug_train_only", True),
        ("rollout_batch_size", 32),
        ("use_dynamic_global_batch_size", True),
        ("polling_mode", False),
    ):
        changed = Namespace(**{**vars(args), name: value})
        assert not unequal_flops_dp_enabled(changed)
    assert not unequal_flops_dp_enabled(Namespace())


@pytest.mark.parametrize(
    "override",
    [
        {"loss_type": "policy_loss"},
        {"task_type": "seq_cls"},
        {"train_backend": "fsdp"},
        {"use_dynamic_batch_size": False},
        {"sft_wholepack": True},
        {"sft_packing_strategy": "binpack"},
        {"qkv_format": "bshd"},
        {"dynamic_context_parallel": True},
        {"fully_async": True},
        {"multimodal_keys": {"image": "images"}},
        {"mtp_num_layers": 1},
        {"moe_aux_loss_coeff": 0.01},
        {"moe_z_loss_coeff": [0, 0.1]},
        {"moe_expert_capacity_factor": 1.0},
        {"moe_router_enable_expert_bias": True},
        {"moe_router_load_balancing_type": "sinkhorn"},
        {"moe_router_load_balancing_type": ["none", "sinkhorn"]},
    ],
)
def test_flops_flags_reject_unmodeled_paths(override):
    with pytest.raises(ValueError):
        validate_flops_scheduling_args(_args(**override))


def test_flops_flags_allow_supported_prepack_and_reuse_cached_coefficients():
    args = _args(fully_async=True, sft_async_prepack=True)
    validate_flops_scheduling_args(args)
    args._flops_workload_estimator = SequenceFlopsEstimator(10, 1)
    assert get_flops_estimator(args) is args._flops_workload_estimator


def test_dp_only_preserves_upstream_binpack_but_rejects_wholepack():
    args = _args(flops_microbatch_packing=False, flops_dp_balancing=True, sft_packing_strategy="binpack")
    validate_flops_scheduling_args(args)
    args.sft_wholepack = True
    with pytest.raises(ValueError, match="wholepack"):
        validate_flops_scheduling_args(args)


def test_workload_plan_keeps_real_samples_once_with_dummy_execution_slots():
    plan = build_workload_microbatch_plan([3, 5], SequenceFlopsEstimator(10, 1), 128, num_microbatches=6)
    assert len(plan.partitions) == 6
    assert plan.dummy_offsets == frozenset({2, 3, 4, 5})
    real = [i for offset, group in enumerate(plan.partitions) if offset not in plan.dummy_offsets for i in group]
    assert sorted(real) == [0, 1]
    assert all(sum(plan.capacity_lengths[i] for i in group) <= plan.max_capacity for group in plan.partitions)


def test_workload_plan_accounts_for_final_tp_padding():
    plan = build_workload_microbatch_plan([129, 129], SequenceFlopsEstimator(10, 1), 511, tp_size=2)
    assert plan.max_capacity == 256
    assert len(plan.partitions) == 2
    for group in plan.partitions:
        raw = sum(plan.capacity_lengths[i] for i in group)
        assert (raw + 255) // 256 * 256 <= 511


@pytest.mark.parametrize("count", [1, 2, 3, 8])
def test_expansion_preserves_capacity_and_single_real_coverage(count):
    initial = build_workload_microbatch_plan([3, 5, 7], SequenceFlopsEstimator(10, 1), 128)
    plan = expand_workload_microbatch_plan(initial, count)
    assert len(plan.partitions) == count
    real = [i for offset, group in enumerate(plan.partitions) if offset not in plan.dummy_offsets for i in group]
    assert sorted(real) == [0, 1, 2]
    assert all(
        group and sum(plan.capacity_lengths[i] for i in group) <= plan.max_capacity for group in plan.partitions
    )
    assert initial.partitions == [[0, 1, 2]]


def test_expansion_rejects_shrinking_the_agreed_execution_count():
    plan = build_workload_microbatch_plan([128, 128], SequenceFlopsEstimator(10, 1), 128)
    with pytest.raises(ValueError, match="must not shrink"):
        expand_workload_microbatch_plan(plan, 1)


@pytest.mark.parametrize(
    "vl,allgather,expected",
    [(False, False, [12, 20]), (True, False, [16, 24]), (True, True, [16, 24]), (False, True, [9, 17])],
)
def test_workload_plan_accounts_for_cp_representation(vl, allgather, expected):
    plan = build_workload_microbatch_plan(
        [9, 17],
        SequenceFlopsEstimator(10, 1),
        256,
        tp_size=2,
        cp_size=2,
        is_vl_model=vl,
        allgather_cp=allgather,
    )
    assert plan.capacity_lengths == expected
    assert plan.max_capacity == 512


@pytest.mark.parametrize("lengths,budget", [([], 128), ([129], 128), ([1], 127)])
def test_workload_plan_rejects_unexecutable_steps(lengths, budget):
    with pytest.raises(ValueError):
        build_workload_microbatch_plan(lengths, SequenceFlopsEstimator(10, 1), budget)


def test_workload_plan_rejects_too_few_execution_slots():
    with pytest.raises(ValueError):
        build_workload_microbatch_plan([128, 128], SequenceFlopsEstimator(10, 1), 128, num_microbatches=1)


def test_text_scope_allows_empty_media_field_and_rejects_real_media():
    validate_text_rollout({"multimodal_train_inputs": [None, {}]})
    with pytest.raises(ValueError, match="multimodal"):
        validate_text_rollout({"multimodal_train_inputs": [{"pixel_values": object()}]})
