# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prevent silently loading K3.1 as dense K3 or omitting its indexer
weights."""

import importlib
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest

from tests.cpu.models import test_kimi_k3


@pytest.fixture
def k3_modules(monkeypatch):
    return test_kimi_k3.k3_modules.__wrapped__(monkeypatch)


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=6,
        mla_use_nope=True,
        linear_attn_config={"kda_layers": [1, 2, 3, 5], "full_attn_layers": [4, 6]},
        dsa_config={
            "dsa_layers": [4, 6],
            "index_topk": 8,
            "index_head_dim": 4,
            "index_num_heads": 2,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    )


def test_kimi_k31_bridge_selects_sparse_spec_and_preserves_multimodal_metadata(monkeypatch, k3_modules):
    dsa_module = ModuleType(f"{k3_modules.bridge.__package__}.dsa")
    dsa_module.build_kimi_k31_spec = object()
    monkeypatch.setitem(sys.modules, dsa_module.__name__, dsa_module)
    text_config = _text_config()
    original_dsa_config = deepcopy(text_config.dsa_config)
    vision_config = object()
    hf = SimpleNamespace(
        config=SimpleNamespace(text_config=text_config, vision_config=vision_config, media_placeholder_token_id=123),
        model_name_or_path="test-k31-checkpoint",
        trust_remote_code=True,
    )

    bridge = k3_modules.bridge.KimiK31VLBridge()
    provider = bridge.provider_bridge(hf)

    assert provider.kimi_dsa_config == {**original_dsa_config, "dsa_layers": (4, 6)}
    assert text_config.dsa_config == original_dsa_config
    assert provider.kimi_dsa_config is not text_config.dsa_config
    assert provider.transformer_layer_spec is dsa_module.build_kimi_k31_spec
    assert provider.vision_config is vision_config
    assert provider.hf_model_path == hf.model_name_or_path
    assert provider.image_token_id == 123
    assert provider.trust_remote_code
    assert not bridge.USE_MODEL_CONFIG_FOR_CONVERSION
    assert k3_modules.provider.KimiK3VLModelProvider().kimi_dsa_config is None
    assert (
        dict(
            source="KimiK31ForConditionalGeneration",
            target=k3_modules.model.KimiK3VLModel,
            provider=k3_modules.provider.KimiK3VLModelProvider,
            model_type="kimi_k31",
        )
        in k3_modules.registrations
    )


def test_kimi_k31_bridge_includes_all_indexer_parameters_without_replacing_k3_mappings(k3_modules):
    base = k3_modules.bridge.KimiK3VLBridge().mapping_registry().mappings
    mappings = k3_modules.bridge.KimiK31VLBridge().mapping_registry().mappings
    assert [(mapping.megatron_param, mapping.hf_param) for mapping in mappings[: len(base)]] == [
        (mapping.megatron_param, mapping.hf_param) for mapping in base
    ]
    indexer = {
        mapping.megatron_param.removeprefix(
            "language_model.decoder.layers.*.self_attention.core_attention.indexer."
        ): mapping.hf_param
        for mapping in mappings[len(base) :]
    }
    assert indexer == {
        name: f"language_model.model.layers.*.self_attn.indexer.{name}"
        for name in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight")
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"index_rope_dim": 64}, "index_rope_dim"),
        ({"index_k_norm_type": "rmsnorm"}, "index_k_norm_type"),
        ({"index_topk_freq": 2}, "index_topk_freq"),
        ({"index_topk": 0}, "positive integer index_topk"),
        ({"index_num_heads": True}, "positive integer index_num_heads"),
        ({"index_head_dim": -1}, "positive integer index_head_dim"),
        ({"dsa_layers": [4]}, "non-KDA"),
        ({"dsa_layers": [4, 4, 6]}, "non-KDA"),
        ({"dsa_layers": [3, 5]}, "non-KDA"),
        ({"dsa_layers": [4, 7]}, "non-KDA"),
    ],
)
def test_kimi_k31_config_rejects_unsupported_indexers(k3_modules, overrides, message):
    configuration = importlib.import_module(f"{k3_modules.bridge.__package__}.configuration")
    text_config = _text_config()
    text_config.dsa_config.update(overrides)
    with pytest.raises(ValueError, match=message):
        configuration.kimi_k31_dsa_config(text_config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_config": None}, "requires text_config.dsa_config"),
        ({"mla_use_nope": False}, "mla_use_nope"),
        ({"linear_attn_config": {"kda_layers": [1, 2, 2]}}, "unique one-based"),
        ({"linear_attn_config": {"kda_layers": [0, 1, 2, 3]}}, "unique one-based"),
        (
            {"linear_attn_config": {"kda_layers": [1, 2, 3, 5], "full_attn_layers": [4]}},
            "non-KDA",
        ),
    ],
)
def test_kimi_k31_config_rejects_inconsistent_attention_layout(k3_modules, overrides, message):
    configuration = importlib.import_module(f"{k3_modules.bridge.__package__}.configuration")
    text_config = _text_config()
    vars(text_config).update(overrides)
    with pytest.raises(ValueError, match=message):
        configuration.kimi_k31_dsa_config(text_config)
