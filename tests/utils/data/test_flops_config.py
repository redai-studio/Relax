# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils.data.flops_config import validate_flops_balancing


def make_args(**changes):
    values = dict(
        balance_by_flops=True,
        loss_type="sft",
        train_backend="megatron",
        task_type="causal_lm",
        use_dynamic_batch_size=True,
        n_samples_per_prompt=1,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_disabled_does_not_inspect_or_change_existing_configuration():
    args = SimpleNamespace(balance_by_flops=False, fully_async=True)
    before = vars(args).copy()
    validate_flops_balancing(args)
    assert vars(args) == before


def test_supported_configuration_is_not_mutated():
    args = make_args(max_staleness=0)
    before = vars(args).copy()
    validate_flops_balancing(args)
    assert vars(args) == before


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"loss_type": "grpo"}, "SFT"),
        ({"loss_type": "dpo"}, "SFT"),
        ({"train_backend": "fsdp"}, "Megatron"),
        ({"use_dynamic_batch_size": False}, "dynamic"),
        ({"task_type": "seq_cls"}, "causal"),
        ({"n_samples_per_prompt": 2}, "n-samples"),
        ({"fully_async": True}, "fully_async"),
        ({"hybrid": True}, "hybrid"),
        ({"sft_wholepack": True}, "wholepack"),
        ({"dynamic_context_parallel": True}, "dynamic_context_parallel"),
        ({"multimodal_keys": {"image": "image"}}, "multimodal"),
        ({"generation_task": "i2i"}, "multimodal"),
        ({"sft_async_prepack": True}, "sft-max-in-flight-steps 1"),
        ({"lora_rank": 8}, "LoRA"),
        ({"enable_mtp_training": True}, "enable_mtp_training"),
        ({"moe_aux_loss_coeff": 0.01}, "auxiliary loss"),
        ({"moe_aux_loss_coeff": [0.0, 0.01]}, "auxiliary loss"),
        ({"moe_expert_capacity_factor": 1.0}, "expert capacity"),
        ({"moe_z_loss_coeff": 0.001}, "dummy zero-gradient"),
        ({"moe_router_load_balancing_type": "sinkhorn"}, "microbatch-dependent"),
        ({"moe_router_load_balancing_type": ["aux_loss", "quantile_balancing"]}, "microbatch-dependent"),
        ({"moe_router_enable_expert_bias": True}, "expert-bias updates"),
        ({"moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 0.001}, "expert-bias updates"),
    ],
)
def test_unsupported_modes_fail_explicitly(changes, match):
    with pytest.raises(ValueError, match=match):
        validate_flops_balancing(make_args(**changes))


def test_static_parallel_settings_are_preserved():
    args = make_args(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    before = vars(args).copy()
    validate_flops_balancing(args)
    assert vars(args) == before


@pytest.mark.parametrize("coeff", [None, 0.0, [0.0, 0.0], (0.0,)])
@pytest.mark.parametrize("z_coeff", [None, 0.0])
def test_moe_without_auxiliary_loss_or_token_dropping_is_accepted(coeff, z_coeff):
    validate_flops_balancing(
        make_args(moe_aux_loss_coeff=coeff, moe_z_loss_coeff=z_coeff, moe_expert_capacity_factor=None)
    )


def test_disabled_preserves_existing_moe_configuration():
    args = make_args(
        balance_by_flops=False,
        moe_aux_loss_coeff=[0.01],
        moe_z_loss_coeff=0.001,
        moe_expert_capacity_factor=1.0,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="sinkhorn",
    )
    before = vars(args).copy()
    validate_flops_balancing(args)
    assert vars(args) == before


@pytest.mark.parametrize(
    "changes",
    [
        {"moe_router_enable_expert_bias": False, "moe_router_bias_update_rate": 0.001},
        {"moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 0},
        {"moe_router_load_balancing_type": "aux_loss", "moe_aux_loss_coeff": 0.0},
        {"moe_router_load_balancing_type": ["none"]},
    ],
)
def test_microbatch_independent_moe_routing_is_accepted(changes):
    validate_flops_balancing(make_args(**changes))


@pytest.mark.parametrize("allgather,expected", [(False, 44), (True, 32)])
def test_controller_capacity_matches_worker_formula(allgather, expected):
    from relax.utils.data.flops_config import get_flops_capacity_settings

    args = make_args(
        max_tokens_per_gpu=11,
        context_parallel_size=4,
        tensor_model_parallel_size=2,
        data_pad_size_multiplier=4,
        allgather_cp=allgather,
        qkv_format="thd",
    )
    assert get_flops_capacity_settings(args) == (expected, 16)
