# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.training import microbatch_cost
from relax.utils.training.qwen35_cost import estimate_sample_costs


def tiny_config(*, moe=False):
    text = SimpleNamespace(
        model_type="qwen3_5_moe_text" if moe else "qwen3_5_text",
        hidden_size=8,
        vocab_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        linear_num_key_heads=1,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
    )
    if moe:
        text.num_experts, text.num_experts_per_tok = 4, 2
        text.moe_intermediate_size, text.shared_expert_intermediate_size = 4, 4
    return text


def test_dense_and_moe_costs():
    # Independent counts for two tiny layers, including gated projections,
    # active experts, causal attention, three GDN contractions and the head.
    assert estimate_sample_costs(tiny_config(), {"total_lengths": [2, 4]}, Namespace()) == [14304, 28992]
    assert estimate_sample_costs(tiny_config(moe=True), {"total_lengths": [2, 4]}, Namespace()) == [15264, 30912]


def test_cost_scope_is_explicit():
    config = tiny_config()
    with pytest.raises(ValueError, match="text-only"):
        estimate_sample_costs(
            config, dict(total_lengths=[2], multimodal_train_inputs=[{"image_grid_thw": [[1, 2, 2]]}]), Namespace()
        )
    with pytest.raises(ValueError, match="without MTP"):
        estimate_sample_costs(config, {"total_lengths": [2]}, Namespace(enable_mtp_training=True, mtp_num_layers=1))


@pytest.mark.parametrize("enabled", [False, True])
def test_costs_are_published_only_as_sampler_metadata(monkeypatch, enabled):
    from relax.utils import misc
    from relax.utils.training import qwen35_cost
    from relax.utils.utils import build_rollout_custom_meta, dict_to_tensordict

    costs = [1_000_000_000_000.125, 1_000_000_000_000.25]
    monkeypatch.setattr(misc, "get_hf_config", lambda _: tiny_config())
    monkeypatch.setattr(qwen35_cost, "estimate_sample_costs", lambda *args: costs)
    batch = dict_to_tensordict(
        dict(total_lengths=[2, 4], sample_indices=[8, 2], multimodal_train_inputs=[None, None]), 2
    )
    args = Namespace(balance_by_flops=enabled, use_dynamic_batch_size=True, hf_checkpoint="unused")
    metadata = build_rollout_custom_meta(batch, args)
    assert [row["sample_index"] for row in metadata] == [8, 2]
    assert "flops_cost" not in batch
    if enabled:
        assert [row["flops_cost"] for row in metadata] == costs
    else:
        assert all("flops_cost" not in row for row in metadata)


def test_disabled_feature_does_not_load_model_or_change_metadata(monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(misc, "get_hf_config", lambda *_: pytest.fail("OFF must not load a model"))
    args = Namespace()
    metadata = [{"total_lengths": 2}]
    assert not microbatch_cost.validate_flops_balance(args)
    assert microbatch_cost.add_flops_cost_metadata(args, {"total_lengths": [2]}, metadata) is metadata
    assert vars(args) == {}


@pytest.mark.parametrize(
    "override", [dict(use_dynamic_batch_size=False), dict(polling_mode=False), dict(moe_aux_loss_coeff=0.1)]
)
def test_unsupported_scheduler_configuration_is_rejected(override):
    args = Namespace(balance_by_flops=True, use_dynamic_batch_size=True)
    vars(args).update(override)
    with pytest.raises(ValueError):
        microbatch_cost.validate_flops_balance(args)


def test_unknown_model_and_invalid_cost_fail_at_producer(monkeypatch):
    from relax.utils import misc
    from relax.utils.training import qwen35_cost

    args = Namespace(balance_by_flops=True, use_dynamic_batch_size=True, hf_checkpoint="unused")
    monkeypatch.setattr(misc, "get_hf_config", lambda _: SimpleNamespace(model_type="unknown"))
    with pytest.raises(ValueError, match="model_type"):
        microbatch_cost.sample_costs(args, {"total_lengths": [2]})
    monkeypatch.setattr(misc, "get_hf_config", lambda _: tiny_config())
    monkeypatch.setattr(qwen35_cost, "estimate_sample_costs", lambda *_: [float("nan")])
    with pytest.raises(ValueError, match="positive and finite"):
        microbatch_cost.sample_costs(args, {"total_lengths": [2]})
