# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regress the Gemma4 dense fallback error using pinned official configs."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.utils.training.flops_counter import FlopsCounter
from relax.utils.training.gemma4_flops import _causal_pairs, estimate_gemma4_flops, gemma4_active_weights


def _config(name="26b"):
    return json.loads((Path(__file__).parent / "fixtures" / f"gemma4_{name}_config.json").read_text())


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (235, 5.43648497664),
        (1024, 24.384148340736),
        (1025, 24.4088414208),
        (4096, 102.559817662464),
        (8192, 214.009588482048),
    ],
)
def test_26b_official_useful_flops(length, expected):
    config = _config()
    # Independently audited GEMM weights: active experts + shared MLP +
    # router + mixed local/global QKO(V) + one vocabulary output projection.
    assert gemma4_active_weights(config) == 3_821_830_144
    assert estimate_gemma4_flops(config, length, [length], 1.0) == pytest.approx(expected)


def test_window_includes_current_token_and_resets_between_packed_sequences():
    assert _causal_pairs(1024, 1024) == 524800
    assert _causal_pairs(1025, 1024) == 525824
    config = _config()
    separate = estimate_gemma4_flops(config, 8192, [4096, 4096], 1)
    assert separate == pytest.approx(2 * 102.559817662464)
    assert separate < estimate_gemma4_flops(config, 8192, [8192], 1)


def test_dense_31b_does_not_activate_experts_from_residual_config_fields():
    config = _config("31b")
    config["text_config"].update(num_experts=128, top_k_experts=8, moe_intermediate_size=704)
    assert estimate_gemma4_flops(config, 4096, [4096], 1) == pytest.approx(788.923315912704)


def test_global_k_eq_v_does_not_remove_sliding_v_projection():
    config = _config()
    untied = copy.deepcopy(config)
    untied["text_config"]["attention_k_eq_v"] = False
    # Fix the global KV head override so this test isolates tying V.
    untied["text_config"]["per_layer_config"] = {str(i): {"num_key_value_heads": 2} for i in range(5, 30, 6)}
    tied_n = gemma4_active_weights(config)
    untied_n = gemma4_active_weights(untied)
    assert untied_n - tied_n == 5 * 2816 * 2 * 512


@pytest.mark.parametrize("root", [True, False])
def test_counter_registers_root_and_standalone_text_and_forward_only(monkeypatch, root):
    from relax.utils.training import flops_counter

    monkeypatch.setattr(flops_counter, "get_device_peak_flops", lambda unit: 989.0)
    config = _config()
    counter = FlopsCounter(_namespace(config if root else config["text_config"]))
    train, peak = counter.estimate([4096], 2)
    forward, _ = counter.estimate([4096], 2, forward_only=True)
    assert peak == 989
    assert train == pytest.approx(102.559817662464 / 2)
    assert forward == pytest.approx(train / 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lora_rank": 8},
        {"freeze_language_model": True},
        {"freeze_params_name_list": [".*"]},
        {"only_train_params_name_list": ["head"]},
        {"enable_mtp_training": True},
        {"images_seqlens": [256]},
        {"audio_seqlens": [100]},
    ],
)
def test_unsupported_training_paths_omit_mfu(kwargs):
    assert math.isnan(estimate_gemma4_flops(_config(), 4096, [4096], 1, **kwargs))


@pytest.mark.parametrize("field", ["hidden_size_per_layer_input", "num_kv_shared_layers"])
def test_unsupported_architecture_does_not_fall_back_to_dense(field):
    config = _config()
    config["text_config"][field] = 1
    assert math.isnan(estimate_gemma4_flops(config, 4096, [4096], 1))


def test_pure_text_with_frozen_vision_uses_no_vision_flops():
    config = _config()
    assert estimate_gemma4_flops(config, 4096, [4096], 1, freeze_vision_model=True) == pytest.approx(102.559817662464)


def test_resolved_hf_layers_never_read_ambiguous_global_shapes():
    class HeterogeneousConfig(SimpleNamespace):
        def __getattribute__(self, name):
            if name in ("head_dim", "num_key_value_heads"):
                raise RuntimeError("AmbiguousGlobalPerLayerAttributeError")
            return super().__getattribute__(name)

    text = _config()["text_config"]
    text["per_layer_config"] = [
        {"head_dim": 512, "num_key_value_heads": 2}
        if kind == "full_attention"
        else {"head_dim": 256, "num_key_value_heads": 8}
        for kind in text["layer_types"]
    ]
    config = HeterogeneousConfig(**vars(_namespace(text)))
    assert estimate_gemma4_flops(config, 4096, [4096], 1) == pytest.approx(102.559817662464)


@pytest.mark.parametrize("namespace", [True, False])
def test_hf_zero_padded_layer_override_indices(namespace):
    text = _config()["text_config"]
    text["attention_k_eq_v"] = False
    text["per_layer_config"] = {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 2} for i in range(5, 30, 6)}
    config = _namespace(text) if namespace else text
    assert gemma4_active_weights(config) == 3_836_248_064


@pytest.mark.parametrize(
    "tokens,lengths,seconds",
    [(4096, [4096], 0), (4096, [4096], -1), (4096, [4096], float("inf")), (3, [2], 1), (-1, [-1], 1)],
)
def test_invalid_measurements_do_not_report_mfu(tokens, lengths, seconds):
    assert math.isnan(estimate_gemma4_flops(_config(), tokens, lengths, seconds))


def test_modality_arrays_do_not_require_a_boolean_tensor_conversion():
    class Grid:
        def __len__(self):
            return 2

        def __bool__(self):
            raise RuntimeError("ambiguous tensor truth value")

    assert math.isnan(estimate_gemma4_flops(_config(), 4096, [4096], 1, image_grid_thw=Grid()))
