# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


pytest.importorskip("megatron.core", reason="Bridge capability tests require Megatron Core", exc_type=ImportError)

from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter  # noqa: E402


@pytest.mark.parametrize("prefix", ["model.layers.2", "model.language_model.layers.2"])
@pytest.mark.parametrize("packed", [False, True])
def test_bridge_projection_capability_uses_existing_mapping(prefix, packed):
    source = "decoder.layers.2.mlp.experts.linear_fc1.weight3"
    target = f"{prefix}.mlp.experts"
    hf_param = (
        target + ".gate_up_proj"
        if packed
        else {"gate": target + ".3.gate_proj.weight", "up": target + ".3.up_proj.weight"}
    )
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._bridge_task_map = {source: SimpleNamespace(mapping=SimpleNamespace(hf_param=hf_param))}
    assert converter.supports_packed_experts([source])
    converter._bridge_task_map[source].mapping.hf_param = target + ".3.w1.weight"
    assert not converter.supports_packed_experts([source])
    converter._bridge_task_map[source].mapping.hf_param = {"gate": target + ".3.gate_proj.weight"}
    assert not converter.supports_packed_experts([source])
