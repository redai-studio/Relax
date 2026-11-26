# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Synthetic shared-KV weights never enter the native synchronization
stream."""

from argparse import Namespace

import pytest
import torch


def test_gemma_shared_kv_export_uses_native_schema_without_changing_tensor_values(monkeypatch):
    pytest.importorskip("megatron.bridge.models.gemma_vl.gemma4_vl_bridge")
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
    from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4TextConfig

    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.models.gemma4.gemma4_bridge import Gemma4DenseBridge

    monkeypatch.setenv("GEMMA4_CONVERSION_MODE", "text")
    text = Gemma4TextConfig(
        num_hidden_layers=8,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        hidden_size_per_layer_input=4,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention"] * 5 + ["full_attention", "sliding_attention", "full_attention"],
    )
    config = Gemma4Config(text_config=text.to_dict(), architectures=["Gemma4ForConditionalGeneration"])
    adapter = AutoBridge.from_hf_config(config)._model_bridge
    assert isinstance(adapter, Gemma4DenseBridge)
    with torch.device("meta"):
        native = Gemma4ForCausalLM(text)
    hf_keys = {
        "model.language_model." + key[len("model.") :] for key in native.state_dict() if key.startswith("model.")
    }
    norms = [f"decoder.layers.{layer}.self_attention.k_layernorm.weight" for layer in range(8)]
    norms.append("decoder.layers.6.self_attention.q_layernorm.weight")
    registry = adapter.mapping_registry()
    adapter._validate_conversion_mappings(registry, norms, hf_keys)
    for required in (
        "model.language_model.layers.4.self_attn.k_norm.weight",
        "model.language_model.layers.6.self_attn.q_norm.weight",
    ):
        with pytest.raises(ValueError, match="missing mapped parameter"):
            adapter._validate_conversion_mappings(registry, norms, hf_keys - {required})
    before_rng = torch.random.get_rng_state()
    before_cuda = torch.cuda.is_initialized()
    export_filter = adapter._make_weight_sync_export_filter()
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() is before_cuda

    converter = BridgeConverter(Namespace(vocab_size=64), [], None)
    converter._bridge_task_map = {}
    converter._weight_sync_export_filter = export_filter
    tensor = torch.ones((2, 2), dtype=torch.bfloat16)
    for layer in (4, 5, 6, 7):
        for projection in ("q_proj", "k_proj", "v_proj", "k_norm"):
            name = f"decoder.layers.{layer}.self_attention.{projection}.weight"
            hf = f"model.language_model.layers.{layer}.self_attn.{projection}.weight"
            converter._bridge_task_map[name] = WeightConversionTask(
                param_name=name,
                global_param_name=name,
                mapping=ReplicatedMapping(name, hf),
                megatron_module=None,
                param_weight=None,
            )
            result = converter.convert(name, tensor)
            if layer >= 6 and projection != "q_proj":
                assert result == []
            else:
                assert len(result) == 1 and result[0][0] == hf and result[0][1] is tensor
