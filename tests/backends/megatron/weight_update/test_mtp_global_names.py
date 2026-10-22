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
