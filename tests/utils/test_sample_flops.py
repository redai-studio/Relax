# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from relax.utils.training.flops_counter import (
    SampleFlopsCost,
    _estimate_qwen2_flops,
    _estimate_qwen2_moe_flops,
    _estimate_qwen3_5_flops,
)


def model_config(model_type="qwen3"):
    config = SimpleNamespace(
        model_type=model_type,
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    if "moe" in model_type:
        config.num_experts = 4
        config.num_experts_per_tok = 2
        config.moe_intermediate_size = 6
        config.shared_expert_intermediate_size = 3
    else:
        config.intermediate_size = 16
    if "3_5" in model_type:
        config.linear_num_key_heads = 1
        config.linear_num_value_heads = 2
        config.linear_key_head_dim = 2
        config.linear_value_head_dim = 3
        config.linear_conv_kernel_dim = 4
        config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return config


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "qwen2_moe", "qwen3_moe", "qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("lengths", [[], [0], [1], [3, 11, 20], [1000, 2000]])
def test_sample_flops_matches_existing_training_formula(model_type, lengths):
    config = model_config(model_type)
    model = SampleFlopsCost.from_config(config)
    estimator = (
        _estimate_qwen3_5_flops
        if "3_5" in model_type
        else _estimate_qwen2_moe_flops
        if "moe" in model_type
        else _estimate_qwen2_flops
    )
    expected = estimator(config, sum(lengths), lengths, 1) * 1e12 / 3
    assert sum(model.estimate(lengths)) == pytest.approx(expected, rel=1e-14)


def test_sample_flops_dense_independent_formula():
    # H=8, Q=8, K=V=4, MLP=8*16*3; embedding+output=32*8*2.
    linear = 2 * ((8 * 16 * 3 + 8 * (8 + 4 + 4 + 8)) * 4 + 32 * 8 * 2)
    quadratic = 2 * 4 * 2 * 4
    assert SampleFlopsCost.from_config(model_config()).estimate([1, 7]) == [
        linear + quadratic,
        linear * 7 + quadratic * 49,
    ]


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"])
def test_sample_flops_distinguishes_same_token_total(model_type):
    model = SampleFlopsCost.from_config(model_config(model_type))
    assert sum(model.estimate([4, 12])) > sum(model.estimate([8, 8]))
    costs = model.estimate([1, 3, 10])
    assert 0 < costs[0] < costs[1] < costs[2]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_sample_flops_hybrid_wrapper_and_interval(model_type):
    config = model_config(model_type)
    expected = SampleFlopsCost.from_config(config)
    wrapper = SimpleNamespace(model_type=model_type, text_config=config)
    assert SampleFlopsCost.from_config(wrapper) == expected
    config.model_type += "_text"
    assert SampleFlopsCost.from_config(config) == expected
    del config.layer_types
    config.full_attention_interval = 4
    assert SampleFlopsCost.from_config(config) == expected


def test_sample_flops_immutable_precomputed_and_device_independent(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Device query is forbidden")

    monkeypatch.setattr("relax.utils.training.flops_counter.get_device_peak_flops", forbidden)
    config = model_config()
    model = SampleFlopsCost.from_config(config)
    expected = model.estimate([1, 5, 9])
    config.__dict__.clear()
    assert model.estimate([1, 5, 9]) == expected
    assert model.estimate([1, 5, 9]) == expected
    with pytest.raises(FrozenInstanceError):
        model.linear_flops = 0


@pytest.mark.parametrize("config", [None, SimpleNamespace(), SimpleNamespace(model_type="unknown")])
def test_sample_flops_rejects_unsupported_config(config):
    with pytest.raises(ValueError, match="model_type"):
        SampleFlopsCost.from_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 0),
        ("vocab_size", -1),
        ("num_attention_heads", None),
        ("head_dim", 1.5),
        ("num_hidden_layers", True),
    ],
)
def test_sample_flops_rejects_invalid_dimensions(field, value):
    config = model_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        SampleFlopsCost.from_config(config)


def test_sample_flops_rejects_missing_field():
    config = model_config()
    del config.intermediate_size
    with pytest.raises(ValueError, match="intermediate_size"):
        SampleFlopsCost.from_config(config)


@pytest.mark.parametrize("change", ["unknown_layer", "missing_layer", "invalid_interval", "topk"])
def test_sample_flops_rejects_invalid_hybrid_or_moe_config(change):
    config = model_config("qwen3_5_moe")
    if change == "unknown_layer":
        config.layer_types[0] = "unknown"
    elif change == "missing_layer":
        config.layer_types.pop()
    elif change == "invalid_interval":
        del config.layer_types
        config.full_attention_interval = 0
    else:
        config.num_experts_per_tok = 5
    with pytest.raises(ValueError):
        SampleFlopsCost.from_config(config)


@pytest.mark.parametrize("length", [-1, True, 1.5, None])
def test_sample_flops_rejects_invalid_lengths(length):
    with pytest.raises(ValueError, match="length"):
        SampleFlopsCost.from_config(model_config()).estimate([length])


@pytest.mark.parametrize(
    "model_type,linear,quadratic",
    [("qwen3_moe", 5120, 64), ("qwen3_5", 6276, 16), ("qwen3_5_moe", 6404, 16)],
)
def test_sample_flops_moe_and_hybrid_golden_coefficients(model_type, linear, quadratic):
    # Independent small-model arithmetic includes router/shared experts and
    # three GDN layers; one full-attention layer supplies the quadratic term.
    model = SampleFlopsCost.from_config(model_config(model_type))
    assert (model.linear_flops, model.quadratic_flops) == (linear, quadratic)
    assert model.estimate([1, 10]) == [linear + quadratic, 10 * linear + 100 * quadratic]


def test_sample_flops_default_head_dim():
    config = model_config()
    expected = SampleFlopsCost.from_config(config)
    del config.head_dim
    assert SampleFlopsCost.from_config(config) == expected
    config.hidden_size = 9
    with pytest.raises(ValueError, match="divisible"):
        SampleFlopsCost.from_config(config)


def test_sample_flops_malformed_model_type():
    with pytest.raises(ValueError, match="model_type"):
        SampleFlopsCost.from_config(SimpleNamespace(model_type=[]))
