# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def common(monkeypatch):
    module = pytest.importorskip("relax.backends.megatron.weight_update.common")
    monkeypatch.setattr(module.mpu, "get_expert_model_parallel_world_size", lambda: 8)
    # MTP indices must remain local to the MTP stack, even on the last PP stage.
    monkeypatch.setattr(module, "get_transformer_layer_offset", lambda config, vp_stage: 32)
    return module


@pytest.mark.parametrize("inner_path", ["transformer_layer", "mtp_model_layer.layers.0"])
@pytest.mark.parametrize("model_prefix", ["", "language_model."])
@pytest.mark.parametrize("projection", ["linear_fc1", "linear_fc2"])
def test_mtp_global_expert_names_preserve_all_ep_shards(common, monkeypatch, inner_path, model_prefix, projection):
    prefix = f"module.module.{model_prefix}mtp.layers.0.{inner_path}.mlp.experts.{projection}"
    exported = {}
    expected = {}
    for ep_rank in range(8):
        monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda rank=ep_rank: rank)
        params = []
        for local_id in range(64):
            global_id = ep_rank * 64 + local_id
            param = torch.nn.Parameter(torch.tensor(float(global_id)))
            params.append((f"{prefix}.weight{local_id}", param))
            expected[f"{prefix}.weight{global_id}"] = param
        model = SimpleNamespace(
            config=Namespace(), named_parameters=lambda: iter(params), named_buffers=lambda: iter(())
        )
        shard = dict(common.named_params_and_buffers(Namespace(num_experts=512), [model]))
        assert len(shard) == 64
        assert exported.keys().isdisjoint(shard), "EP shards would collide in Bridge's global-name buckets"
        exported.update(shard)

    assert exported.keys() == expected.keys()
    assert len(exported) == 512
    for name, param in exported.items():
        assert param is expected[name]


@pytest.mark.parametrize("inner_path", ["transformer_layer", "mtp_model_layer.layers.0"])
def test_mtp_non_expert_names_do_not_receive_ep_or_pp_offsets(common, monkeypatch, inner_path):
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda: 7)
    names = [
        f"module.module.language_model.mtp.layers.0.{inner_path}.mlp.router.weight",
        f"module.module.language_model.mtp.layers.0.{inner_path}.self_attention.linear_qkv.weight",
        "module.module.language_model.mtp.layers.0.enorm.weight",
    ]
    params = [(name, torch.nn.Parameter(torch.zeros(1))) for name in names]
    model = SimpleNamespace(config=Namespace(), named_parameters=lambda: iter(params), named_buffers=lambda: iter(()))
    result = list(common.named_params_and_buffers(Namespace(num_experts=512), [model]))
    assert [name for name, _ in result] == names
    assert all(actual is expected for (_, actual), (_, expected) in zip(result, params, strict=True))


@pytest.mark.parametrize("wrapper", ["", "module.", "module.module."])
def test_global_decoder_names_preserve_pp_layers_ep_experts_and_buffers(common, monkeypatch, wrapper):
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(common, "get_transformer_layer_offset", lambda config, vp_stage: config.offset + vp_stage * 8)
    stages = []
    for offset in (0, 4):
        chunks = []
        for vp in range(2):
            values = torch.nn.Module()
            decoder = values
            for fragment in (wrapper + "language_model.decoder").split("."):
                decoder.add_module(fragment, torch.nn.Module())
                decoder = getattr(decoder, fragment)
            layer = torch.nn.Module()
            layer.register_buffer("state", torch.tensor([offset + vp * 8]))
            decoder.layers = torch.nn.ModuleList([layer])
            layer.mlp = torch.nn.Module()
            experts = layer.mlp.experts = torch.nn.Module()
            experts.linear_fc1 = torch.nn.Module()
            experts.linear_fc1.bias0 = torch.nn.Parameter(torch.tensor([1.0]))
            experts.local_experts = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            values.config = Namespace(offset=offset)
            chunks.append(values)
        stages.append(
            dict(common.named_params_and_buffers(Namespace(num_experts=16), chunks, include_persistent_buffers=True))
        )
    assert stages[0].keys().isdisjoint(stages[1])
    for pp, stage in enumerate(stages):
        for vp in range(2):
            layer = pp * 4 + vp * 8
            prefix = f"module.module.language_model.decoder.layers.{layer}."
            assert stage[prefix + "state"].item() == layer
            assert prefix + "mlp.experts.linear_fc1.bias4" in stage
            assert prefix + "mlp.experts.local_experts.4.weight" in stage
