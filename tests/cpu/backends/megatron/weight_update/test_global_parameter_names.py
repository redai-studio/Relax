from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update import common


@pytest.mark.parametrize("prefix", ["", "module.", "module.module."])
@pytest.mark.parametrize("offset", [0, 18])
def test_pipeline_offsets_apply_to_bare_and_wrapped_parameters_and_buffers(monkeypatch, prefix, offset):
    parameter, buffer = torch.zeros(4), torch.ones(4)
    model = SimpleNamespace(
        config=SimpleNamespace(),
        named_parameters=lambda: [(prefix + "language_model.decoder.layers.0.mlp.weight", parameter)],
    )
    monkeypatch.setattr(common, "get_transformer_layer_offset", lambda config: offset)
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        common,
        "_named_buffers_for_weight_update",
        lambda *args: [(prefix + "language_model.decoder.layers.0.scale", buffer)],
    )
    output = dict(common.named_params_and_buffers(SimpleNamespace(num_experts=None), [model]))
    assert list(output) == [
        f"module.module.language_model.decoder.layers.{offset}.mlp.weight",
        f"module.module.language_model.decoder.layers.{offset}.scale",
    ]
    assert list(output.values()) == [parameter, buffer]
