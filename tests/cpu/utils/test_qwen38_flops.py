# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numeric Qwen3.8 performance oracles retained from the baseline branch."""

import pytest

from relax.utils.training.flops_counter import FlopsCounter
from relax.utils.training.qwen38_flops import _qwen38_indexer_pairs, _qwen38_selected_pairs


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


@pytest.fixture
def qwen38_config():
    # Published 48-layer text dimensions; vision uses the existing Qwen3.5 fixture.
    return Config(
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 12,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4,
                "mtp": {"layer_types": ["full_attention"]},
            },
            "vision_config": {
                **{
                    "num_heads": 16,
                    "depth": 27,
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "out_hidden_size": 4096,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "patch_size": 16,
                },
                "out_hidden_size": 2560,
            },
        }
    )


_QWEN38_POLICY = {
    "freeze_params_name_list": [r"(^|\.)per_layer_embedding\.|self_attention\.indexer\."],
    "freeze_vision_model": True,
    "freeze_vision_projection": True,
}
_QWEN38_MTP = {"enable_mtp_training": True, "mtp_num_layers": 1, "mtp_detach_paths": ()}


@pytest.mark.parametrize("budget,ratio", [(4, 1), (4, 2), (8, 4)])
def test_qwen38_flops_sparse_pairs_match_causal_block_enumeration(budget, ratio):
    for length in range(4 * budget + 1):
        selected = indexer = 0
        for position in range(1, length + 1):
            full_blocks, tail = divmod(position, ratio)
            selected += min(full_blocks, budget // ratio) * ratio + tail
            if full_blocks > budget // ratio:
                indexer += full_blocks
        assert _qwen38_selected_pairs(length, budget, ratio) == selected
        assert _qwen38_indexer_pairs(length, budget, ratio) == indexer


@pytest.mark.parametrize("mtp_on,expected", [(False, 233.897266774016), (True, 261.24036079616)])
@pytest.mark.parametrize("text_only,sparse_label", [(False, False), (True, True)])
def test_flops_counter_qwen38_reference_and_mtp_golden(qwen38_config, mtp_on, expected, text_only, sparse_label):
    text = qwen38_config.text_config
    if sparse_label:
        text.layer_types = ["qwen_sparse_attention" if t == "full_attention" else t for t in text.layer_types]
        text.mtp = {"layer_types": ["qwen_sparse_attention"]}
    counter = FlopsCounter(text if text_only else qwen38_config)
    # Baseline from a15d94a3; independently audited MTP1 adds 27.343094022144 TFLOPs:
    # one QSA/MoE block, three HC mixers, two HC injections, e/h projections,
    # and a full-token LM head. No vocabulary-sized embedding lookup GEMM.
    policy = {**_QWEN38_POLICY, **(_QWEN38_MTP if mtp_on else {})}
    result, _ = counter.estimate([512, 1024, 4096], delta_time=2.0, **policy)
    assert result == pytest.approx(expected / 2)
    doubled, _ = counter.estimate([512, 1024, 4096] * 2, delta_time=2.0, **policy)
    assert doubled == pytest.approx(2 * result)  # No attention across packed document boundaries.


def test_flops_counter_qwen38_frozen_vision_forward_cost(qwen38_config):
    counter = FlopsCounter(qwen38_config)
    baseline, _ = counter.estimate([1024], 1.0, **_QWEN38_POLICY)
    with_image, _ = counter.estimate([1024], 1.0, images_seqlens=[256], **_QWEN38_POLICY)
    # Frozen ViT: patch projection + 27 full-attention/GELU blocks + one merger.
    expected = (
        2 * 256 * 1152 * 3 * 2 * 16**2
        + 27 * (2 * 256 * (4 * 1152**2 + 2 * 1152 * 4304) + 4 * 256**2 * 1152)
        + 2 * 64 * ((4 * 1152) ** 2 + 4 * 1152 * 2560)
    ) / 1e12
    assert with_image - baseline == pytest.approx(expected)


@pytest.mark.parametrize("policy", [{"lora_rank": 8}, {"only_train_params_name_list": ["output_layer"]}])
def test_qwen38_flops_rejects_unvalidated_trainable_subset(qwen38_config, policy):
    result, _ = FlopsCounter(qwen38_config).estimate([1024], 1.0, **{**_QWEN38_POLICY, **policy})
    assert result == 0.0
