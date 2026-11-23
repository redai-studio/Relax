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


def test_hybrid_decoder_uses_constructed_layer_numbers_and_unique_expert_names(common, monkeypatch):
    layer = torch.nn.Module()
    layer.layer_number = 48
    layer.inner_layer = torch.nn.Module()
    layer.inner_layer.mlp = torch.nn.Module()
    layer.inner_layer.mlp.experts = torch.nn.Module()
    layer.inner_layer.mlp.experts.linear_fc1 = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(1))
    layer.inner_layer.mlp.experts.linear_fc1.register_parameter("weight0", weight)
    layer.register_buffer("expert_bias", torch.ones(1))
    model = torch.nn.Module()
    model.config = Namespace(hybrid_layer_pattern="KE")
    model.language_model = torch.nn.Module()
    model.language_model.decoder = torch.nn.Module()
    model.language_model.decoder.layers = torch.nn.ModuleList([layer])
    wrapper = torch.nn.Module()
    wrapper.module = model
    wrapper.config = model.config

    names = []
    for ep_rank in (0, 1):
        monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda rank=ep_rank: rank)
        exported = dict(common.named_params_and_buffers(Namespace(num_experts=288), [wrapper]))
        prefix = "module.module.language_model.decoder.layers.47"
        expert = f"{prefix}.inner_layer.mlp.experts.linear_fc1.weight{ep_rank * 36}"
        assert exported[expert] is weight
        assert exported[f"{prefix}.expert_bias"] is layer.expert_bias
        names.append(expert)
    assert len(set(names)) == 2
