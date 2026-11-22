# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Head rectangles preserve native gated QKV and replicated KV layouts."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.tp_shard_source import local_tp_regions
from relax.backends.sglang.tp_shard_layout import describe_tp_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import copy_rectangle, group_storage_views


@pytest.mark.parametrize("gate,source_tp,target_tp", [(True, 2, 4), (True, 4, 2), (False, 4, 2)])
@torch.no_grad()
def test_qkv_native_heads_group_splitting_and_kv_replication(monkeypatch, gate, source_tp, target_tp):
    from megatron.bridge.models.conversion import param_mapping as bridge
    from sglang.srt.layers.linear import QKVParallelLinear
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    config = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=2,
        attention_output_gate=gate,
    )
    names = {key: f"model.language_model.layers.0.self_attn.{key}_proj.weight" for key in ("q", "k", "v")}
    values = {
        key: (torch.arange(rows * 4).reshape(rows, 4) + index * 29).to(torch.bfloat16)
        for index, (key, rows) in enumerate(zip(names, (16 * (1 + gate), 4, 4), strict=True))
    }
    mapping = bridge.QKVMapping("linear_qkv.weight", **names)
    mapping._tp_group = SimpleNamespace(rank=0, size=source_tp)
    monkeypatch.setattr(bridge, "get_pg_rank", lambda group: group.rank)
    monkeypatch.setattr(bridge, "get_pg_size", lambda group: group.size)
    task = SimpleNamespace(
        mapping=mapping,
        megatron_module=SimpleNamespace(config=config),
        global_param_name="linear_qkv.weight",
    )
    packed = bridge.merge_qkv_weights(config, **values)
    records, sources = [], []
    for rank, shard in enumerate(packed.chunk(source_tp)):
        source = torch.cat((torch.zeros((1, 4), dtype=shard.dtype), shard))[1:]
        source.partition_dim, source.partition_stride = 0, 1
        mapping._tp_group.rank = rank
        regions = local_tp_regions("linear_qkv.weight", source, task, rank, source_tp)
        assert regions and all(region["group"] == sorted(names.values()) for region in regions)
        records.append(dict(rank=rank, regions=regions, groups={"linear_qkv.weight": regions[0]["group"]}))
        storage, _ = group_storage_views({"linear_qkv.weight": source})
        sources.append(next(iter(storage.values())))
    if source_tp == 4:
        assert {region["hf"] for region in records[0]["regions"]} == {names["q"]}
    description = dict(
        records=records,
        source_world_size=source_tp,
        native_specs={
            names[key]: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in values.items()
        },
    )
    for rank in range(target_tp):
        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        layer = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([layer])
        layer.qkv_proj = QKVParallelLinear(
            4,
            2,
            8 * (1 + gate),
            2,
            bias=False,
            params_dtype=torch.bfloat16,
            tp_rank=rank,
            tp_size=target_tp,
        )
        projection = layer.qkv_proj
        for key, value in values.items():
            projection.weight_loader(projection.weight, value, key)
        expected = projection.weight.clone()
        reply = describe_tp_targets(model, description, rank)
        assert reply["selected_hf"] == sorted(names.values())
        actual = torch.zeros_like(expected)
        for write in reply["writes"]:
            copy_rectangle(sources[write["source_rank"]], actual, write)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        projection.weight.data = projection.weight.data.float()
        rejected = describe_tp_targets(model, description, rank)
        assert not rejected["selected_hf"] and not rejected["writes"]
