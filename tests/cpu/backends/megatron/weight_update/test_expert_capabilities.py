# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.core", reason="Bridge capability tests require Megatron Core", exc_type=ImportError)

from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter  # noqa: E402


@pytest.mark.parametrize("prefix", ["model.layers.2", "model.language_model.layers.2"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("transpose_on_export", [False, True])
def test_bridge_projection_capability_uses_existing_mapping(prefix, packed, transpose_on_export):
    source = "decoder.layers.2.mlp.experts.linear_fc1.weight3"
    target = f"{prefix}.mlp.experts"
    hf_param = (
        target + ".gate_up_proj"
        if packed
        else {"gate": target + ".3.gate_proj.weight", "up": target + ".3.up_proj.weight"}
    )
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._bridge_task_map = {
        source: SimpleNamespace(mapping=SimpleNamespace(hf_param=hf_param, transpose_on_export=transpose_on_export))
    }
    assert converter.supports_packed_experts([source]) is (not transpose_on_export)
    assert converter.supports_routed_expert(source)
    converter._bridge_task_map[source].mapping.hf_param = target + ".3.w1.weight"
    assert not converter.supports_packed_experts([source])
    assert not converter.supports_routed_expert(source)
    converter._bridge_task_map[source].mapping.hf_param = {"gate": target + ".3.gate_proj.weight"}
    assert not converter.supports_packed_experts([source])


def test_fp8_expert_routing_requires_checkpoint_scales(monkeypatch):
    from relax.backends.megatron import sglang

    monkeypatch.setattr(sglang, "should_deepgemm_weight_requant_ue8m0", lambda **kwargs: False)
    source = "decoder.layers.2.mlp.experts.linear_fc1.weight3"
    weight = "model.layers.2.ffn.experts.3"
    outputs = {"gate": f"{weight}.w1.weight", "up": f"{weight}.w3.weight"}
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._quantization_config = {"quant_method": "fp8", "weight_block_size": (128, 128)}
    converter._bridge_task_map = {source: SimpleNamespace(mapping=SimpleNamespace(hf_param=outputs))}
    converter._hf_fp8_scale_names = {name: name.removesuffix(".weight") + ".scale" for name in outputs.values()}

    assert converter.supports_routed_expert(source)
    del converter._hf_fp8_scale_names[outputs["up"]]
    assert not converter.supports_routed_expert(source)
