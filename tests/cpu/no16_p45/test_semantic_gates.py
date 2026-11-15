# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Review F1/F2/F3: batch-coupled objectives are rejected at the scheduling gate.

M2PO clipping, direct OPD loss normalization and batch-coupled MoE routing
(aux_loss statistics / Sinkhorn / token dropping) all average or normalize over
the current microbatch; FLOPs repacking changes their statistics domain. The
gate must reject these combinations while flops_scheduling is enabled and keep
them untouched in off mode.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from relax.utils.data.flops_scheduling import build_estimator


_QWEN3_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "vocab_size": 151936,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
}

_BASE_ARGS = dict(
    flops_scheduling="dp",
    train_backend="megatron",
    use_dynamic_batch_size=True,
    loss_type="sft",
    task_type="causal_lm",
    enable_mtp_training=False,
    generation_task=None,
    advantage_estimator="grpo",
    use_opd=False,
    opd_loss_coef=0.0,
    moe_aux_loss_coeff=0.0,
    moe_expert_capacity_factor=None,
    moe_router_load_balancing_type="none",
)


def _args(**override):
    return SimpleNamespace(**{**_BASE_ARGS, **override})


@pytest.mark.parametrize(
    "override, message",
    [
        (dict(advantage_estimator="m2po"), "M2PO clipping"),
        (dict(use_opd=True, opd_loss_coef=1.0), "direct OPD loss"),
        (dict(moe_aux_loss_coeff=0.01), "auxiliary-loss coefficients"),
        (dict(moe_aux_loss_coeff=[0.0, 0.01]), "auxiliary-loss coefficients"),
        (dict(moe_expert_capacity_factor=4), "token dropping"),
        (dict(moe_router_load_balancing_type="sinkhorn"), "token-independent MoE routing"),
    ],
)
def test_batch_coupled_objectives_are_rejected_with_flops(override, message):
    with pytest.raises(ValueError, match=message):
        build_estimator(_args(**override), _QWEN3_CONFIG)


def test_same_objectives_are_untouched_in_off_mode():
    args = _args(
        flops_scheduling="off",
        advantage_estimator="m2po",
        use_opd=True,
        opd_loss_coef=1.0,
        moe_aux_loss_coeff=0.01,
        moe_router_load_balancing_type="sinkhorn",
    )
    assert build_estimator(args, _QWEN3_CONFIG) is None


def test_token_independent_routing_modes_stay_supported():
    for mode in ("none", "aux_loss", "seq_aux_loss", "global_aux_loss"):
        assert build_estimator(_args(moe_router_load_balancing_type=mode), _QWEN3_CONFIG) is not None
