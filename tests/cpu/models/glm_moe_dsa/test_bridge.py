# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def bridge_module(glm_modules, monkeypatch):
    pytest.importorskip("megatron.core", reason="Requires the Dockerfile-pinned MCore/Bridge source")
    bridge_spec = importlib.util.find_spec("megatron.bridge")
    if bridge_spec is None or not bridge_spec.submodule_search_locations:
        pytest.skip("Requires the Dockerfile-pinned Bridge source")
    bridge_root = Path(next(iter(bridge_spec.submodule_search_locations)))
    # Bypass unrelated model registrations. MappingRegistry and parameter
    # mappings are the real pinned Bridge implementations; HF model loading and
    # the generic provider constructor are external boundaries for this test.
    for suffix in ("", ".models", ".models.conversion", ".models.hf_pretrained", ".utils"):
        name = "megatron.bridge" + suffix
        package = ModuleType(name)
        package.__path__ = [str(bridge_root.joinpath(*suffix.strip(".").split("."))) if suffix else str(bridge_root)]
        monkeypatch.setitem(sys.modules, name, package)

    class ModelBridge:
        @classmethod
        def register_bridge(cls, **kwargs):
            return lambda bridge: bridge

        def provider_bridge(self, hf):
            return SimpleNamespace(num_layers=hf.config.num_hidden_layers)

    for name, attrs in {
        "megatron.bridge.models.conversion.model_bridge": {"MegatronModelBridge": ModelBridge},
        "megatron.bridge.models.hf_pretrained.causal_lm": {"PreTrainedCausalLM": object},
        "megatron.bridge.models.mla_provider": {"MLAModelProvider": object},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    transformers = importlib.import_module("transformers")
    monkeypatch.setitem(transformers.__dict__, "GlmMoeDsaForCausalLM", torch.nn.Module)
    return glm_modules("glm5_bridge")


def test_glm_bridge_transfers_hf_schedule_rope_and_mtp_mapping(bridge_module):
    bridge = bridge_module.GLM5Bridge()
    hf = SimpleNamespace(
        num_hidden_layers=6,
        first_k_dense_replace=3,
        moe_intermediate_size=16,
        n_shared_experts=1,
        rope_parameters={"rope_theta": 8000000},
        index_head_dim=4,
        index_n_heads=2,
        index_topk=8,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=["full"] * 3 + ["shared"] * 3,
        num_nextn_predict_layers=1,
    )
    provider = bridge.provider_bridge(SimpleNamespace(config=hf))
    assert provider.glm_dsa_index_sources == (1, 2, 3, 3, 3, 3)
    assert provider.rotary_base == 8000000
    assert provider.rope_type == "rope"
    assert provider.dsa_indexer_topk == 8
    assert callable(provider.mtp_layer_spec_transform)
    bridge.hf_config = hf
    registry = bridge.mapping_registry()
    mapping = registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.self_attention.wq_b.weight")
    assert mapping.hf_param == "model.layers.6.self_attn.indexer.wq_b.weight"
    mapping = registry.hf_to_megatron_lookup("model.layers.6.eh_proj.weight")
    assert mapping.megatron_param == "mtp.layers.0.eh_proj.weight"


def test_glm_bridge_loads_fp8_block_scales_before_mapping(bridge_module):
    bridge = bridge_module.GLM5Bridge()
    bridge.hf_config = SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    weight = torch.ones((130, 256)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.25, 2.0], [4.0, 0.5]])
    norm = torch.ones(130, dtype=torch.bfloat16)
    state = {"gate.weight": weight, "gate.weight_scale_inv": scales, "norm.weight": norm}
    loaded = bridge.maybe_modify_loaded_hf_weight({"gate": "gate.weight", "norm": "norm.weight"}, state)
    expected = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[:130].to(torch.bfloat16)
    torch.testing.assert_close(loaded["gate"], expected, rtol=0, atol=0)
    assert loaded["norm"] is norm


def test_glm_bridge_rejects_missing_or_invalid_fp8_scales(bridge_module):
    bridge = bridge_module.GLM5Bridge()
    bridge.hf_config = SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    state = {"proj.weight": torch.ones((128, 256)).to(torch.float8_e4m3fn)}
    with pytest.raises(ValueError, match="scale"):
        bridge.maybe_modify_loaded_hf_weight("proj.weight", state)
    state["proj.weight_scale_inv"] = torch.ones((1, 1))
    with pytest.raises(ValueError, match="shape"):
        bridge.maybe_modify_loaded_hf_weight("proj.weight", state)
