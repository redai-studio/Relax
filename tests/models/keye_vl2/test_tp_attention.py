# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


def test_keye_custom_output_projection_uses_explicit_row_mapping():
    from megatron.bridge.models.conversion.param_mapping import RowParallelMapping

    from relax.models.keye_vl2.bridge import KeyeBridge

    mapping = (
        KeyeBridge().mapping_registry().megatron_to_hf_lookup("decoder.layers.0.self_attention.linear_proj.weight")
    )
    assert isinstance(mapping, RowParallelMapping)
    assert mapping.hf_param == "model.layers.0.self_attn.o_proj.weight"


@pytest.mark.parametrize("projection", ["linear_fc1", "linear_fc2"])
def test_keye_transposed_experts_use_hf_weight_loading(projection):
    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter
    from relax.models.keye_vl2.bridge import KeyeBridge

    source = f"decoder.layers.0.mlp.experts.{projection}.weight0"
    mapping = KeyeBridge().mapping_registry().megatron_to_hf_lookup(source)
    converter = BridgeConverter.__new__(BridgeConverter)
    converter._bridge_task_map = {source: SimpleNamespace(mapping=mapping)}
    assert mapping.transpose_on_export
    assert not converter.supports_packed_experts([source])
