# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small, independently counted V4.1 GEMM/causal-pair regressions; no GPU."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from relax.utils.training.flops_counter import FlopsCounter


@pytest.fixture(autouse=True)
def cpu_peak(monkeypatch):
    monkeypatch.setattr("relax.utils.training.flops_counter.get_device_peak_flops", lambda **_: 989.0)


def config():
    # Deliberately lacks intermediate_size: V4.1 has MoE layers throughout.
    return SimpleNamespace(
        model_type="deepseek_v41",
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        q_lora_rank=2,
        o_lora_rank=2,
        o_groups=1,
        moe_intermediate_size=3,
        n_routed_experts=3,
        num_experts_per_tok=1,
        n_shared_experts=1,
        vocab_size=7,
        sliding_window=2,
        compress_ratios=[0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        index_n_heads=2,
        index_head_dim=2,
        index_topk=2,
        hc_mult=2,
        engram_layer_ids=[],
        candidate_source_layer_id=-1,
    )


def flops(cfg, lengths=(3, 5), **kwargs):
    return FlopsCounter(cfg).estimate(list(lengths), 1.0, **kwargs)[0] * 1e12


@pytest.mark.parametrize("forward_only, expected", [(True, 5216), (False, 15456)])
def test_v41_counts_concrete_gemm_shapes_and_frozen_router(forward_only, expected):
    # Eight tokens: attention/expert/HC/head weights = 40+72+128+28=268.
    # SWA has 5+9=14 valid pairs -> 224 FLOPs; HC mixing -> 512 FLOPs.
    # Trainable forward=5024, frozen-router forward=192 (still needs dX).
    cfg = config()
    assert flops(cfg, forward_only=forward_only) == pytest.approx(expected)
    # Composite HF config and the standalone text model share the formula.
    cfg.model_type = "deepseek_v41_text"
    assert flops(cfg, forward_only=forward_only) == pytest.approx(expected)
    assert flops(
        SimpleNamespace(model_type="deepseek_v41", text_config=vars(cfg)), forward_only=forward_only
    ) == pytest.approx(expected)


@pytest.mark.parametrize("forward_only, expected_extra", [(True, 1748), (False, 3348)])
def test_v41_reuses_compression_and_indexing_across_layers(forward_only, expected_extra):
    baseline = config()
    baseline.num_hidden_layers = 3
    baseline.compress_ratios = [0, 0, 0]
    compressed = deepcopy(baseline)
    compressed.compress_ratios = [2, 2, 1]
    compressed.kv_source_layer_ids = [0, 2]
    compressed.index_source_layer_ids = [0, 2]
    # Lengths 3/5: ratio2 has 8 pairs, ratio1 has 21 (14 after topk2).
    # Three attention layers add 480 FLOPs. The two compressors add 320,
    # including discarding each odd tail. Two no_grad indexers add 948.
    # The middle layer reuses KV and indices; it only adds attention work.
    assert flops(compressed, forward_only=forward_only) - flops(baseline, forward_only=forward_only) == pytest.approx(
        expected_extra
    )
    # HF may retain extra MTP schedule entries after the text backbone.
    compressed.compress_ratios.extend([0, 0, 0])
    assert flops(compressed, forward_only=forward_only) - flops(baseline, forward_only=forward_only) == pytest.approx(
        expected_extra
    )


@pytest.mark.parametrize("forward_only, expected_extra", [(True, 384), (False, 768)])
def test_v41_engram_projection_has_weight_gradient_but_no_lookup_gradient(forward_only, expected_extra):
    baseline = config()
    engram = deepcopy(baseline)
    engram.engram_layer_ids = [0]
    engram.engram_max_ngram_size = 2
    engram.engram_n_heads = 2
    engram.engram_head_dim = 1
    engram.engram_num_embeddings = [1_000_000_000]
    # Projection 2 -> 12: forward=2*8*24, dW only. Table size is irrelevant.
    assert flops(engram, forward_only=forward_only) - flops(baseline, forward_only=forward_only) == pytest.approx(
        expected_extra
    )


def test_v41_document_boundaries_and_elapsed_time_are_preserved():
    cfg = config()
    separate = flops(cfg, (3, 5))
    assert separate == pytest.approx(flops(cfg, (3,)) + flops(cfg, (5,)))
    assert flops(cfg, (8,)) > separate  # packing must not create cross-document pairs
    assert FlopsCounter(cfg).estimate([3, 5], 2.0)[0] * 1e12 == pytest.approx(separate / 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lora_rank": 8},
        {"freeze_params_name_list": ["*.mlp.*"]},
        {"has_image_inputs": True},
        {"mtp_num_layers": 1},
    ],
)
def test_v41_unsupported_training_policies_do_not_report_fake_mfu(kwargs):
    assert math.isnan(flops(config(), **kwargs))


def test_v41_rejects_insufficient_candidate_budget():
    cfg = config()
    cfg.num_hidden_layers = 2
    cfg.compress_ratios = [1, 1]
    cfg.kv_source_layer_ids = [0]
    cfg.index_source_layer_ids = [0, 1]
    cfg.candidate_source_layer_id = 0
    cfg.candidate_topk_blocks = 1
    cfg.candidate_block_size = 8
    assert math.isnan(flops(cfg))
