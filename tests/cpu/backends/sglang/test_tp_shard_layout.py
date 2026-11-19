# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Direct TP rectangles agree with known BF16 linear weight loaders."""

import json
from math import prod
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.tp_shard_layout import describe_tp_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import copy_rectangle


class UnquantizedLinearMethod:
    pass


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, rows=4, width=4, *, bias=False):
        super().__init__()
        self.tp_rank, self.tp_size = 1, 2
        self.quant_method = UnquantizedLinearMethod()
        self.weight = torch.nn.Parameter(torch.zeros((rows, width), dtype=torch.bfloat16), requires_grad=False)
        self.weight.output_dim = 0
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(rows, dtype=torch.bfloat16), requires_grad=False)
            self.bias.output_dim = 0
            self.bias.weight_loader = self.weight_loader

    def weight_loader(self, param, value):
        param.copy_(value.narrow(0, self.tp_rank * param.shape[0], param.shape[0]))


class RowParallelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp_rank, self.tp_size = 1, 2
        self.quant_method = UnquantizedLinearMethod()
        self.weight = torch.nn.Parameter(torch.zeros((4, 4), dtype=torch.bfloat16), requires_grad=False)
        self.weight.input_dim = 1
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, value):
        param.copy_(value.narrow(1, self.tp_rank * param.shape[1], param.shape[1]))


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self):
        super().__init__(rows=8)
        self.output_sizes = [8, 8]
        self.output_partition_sizes = [4, 4]
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, value, shard):
        param.narrow(0, shard * 4, 4).copy_(value.narrow(0, self.tp_rank * 4, 4))


for cls in (ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear):
    cls.__module__ = "sglang.srt.layers.linear"


class Qwen3_5ForConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = MergedColumnParallelLinear()
        layer.mlp.down_proj = RowParallelLinear()
        self.visual = torch.nn.Module()
        self.visual.blocks = torch.nn.ModuleList([torch.nn.Module()])
        block = self.visual.blocks[0]
        block.mlp = torch.nn.Module()
        block.mlp.linear_fc1 = ColumnParallelLinear(bias=True)
        block.mlp.linear_fc2 = RowParallelLinear()
        block.attn = torch.nn.Module()
        block.attn.proj = RowParallelLinear()

    def load_weights(self, values):
        for name, value in values:
            if "gate_proj" in name:
                self.model.layers[0].mlp.gate_up_proj.weight_loader(
                    self.model.layers[0].mlp.gate_up_proj.weight, value, 0
                )
            elif "up_proj" in name:
                self.model.layers[0].mlp.gate_up_proj.weight_loader(
                    self.model.layers[0].mlp.gate_up_proj.weight, value, 1
                )
            elif "down_proj" in name:
                self.model.layers[0].mlp.down_proj.weight_loader(self.model.layers[0].mlp.down_proj.weight, value)
            elif "linear_fc1" in name:
                param = self.visual.blocks[0].mlp.linear_fc1
                param.weight_loader(getattr(param, name.rsplit(".", 1)[-1]), value)
            elif "linear_fc2" in name:
                param = self.visual.blocks[0].mlp.linear_fc2
                param.weight_loader(param.weight, value)
            elif "attn.proj" in name:
                param = self.visual.blocks[0].attn.proj
                param.weight_loader(param.weight, value)


Qwen3_5ForConditionalGeneration.__module__ = "sglang.srt.models.qwen3_5"


def _values():
    shapes = {
        "model.language_model.layers.0.mlp.gate_proj.weight": (8, 4),
        "model.language_model.layers.0.mlp.up_proj.weight": (8, 4),
        "model.language_model.layers.0.mlp.down_proj.weight": (4, 8),
        "model.visual.blocks.0.mlp.linear_fc1.weight": (8, 4),
        "model.visual.blocks.0.mlp.linear_fc1.bias": (8,),
        "model.visual.blocks.0.mlp.linear_fc2.weight": (4, 8),
        "model.visual.blocks.0.attn.proj.weight": (4, 8),
    }
    return {
        name: torch.arange(prod(shape), dtype=torch.bfloat16).reshape(shape) + index * 64
        for index, (name, shape) in enumerate(shapes.items())
    }


def _description(values):
    records = []
    for rank in range(2):
        regions, groups = [], {}
        for name, value in values.items():
            row_shard = "down_proj" not in name and "linear_fc2" not in name and "attn.proj" not in name
            shape = tuple(value.shape)
            rows, width = shape[0], value.numel() // shape[0]
            local_rows, local_width = (rows // 2, width) if row_shard else (rows, width // 2)
            regions.append(
                dict(
                    hf=name,
                    source=name,
                    global_shape=[rows, width],
                    offset=[rank * local_rows, 0] if row_shard else [0, rank * local_width],
                    shape=[local_rows, local_width],
                    storage_offset=0,
                    stride=local_width,
                    dtype=str(value.dtype),
                    itemsize=2,
                )
            )
            groups[name] = [name]
        records.append(dict(rank=rank, regions=regions, groups=groups))
    return dict(
        records=records,
        source_world_size=2,
        native_specs={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in values.items()},
    )


def test_known_linear_layout_matches_native_loader_without_mutating_model():
    model, values = Qwen3_5ForConditionalGeneration(), _values()
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    reply = describe_tp_targets(model, _description(values), 1)
    assert reply["selected_hf"] == sorted(values)
    assert len(reply["target_shapes"]) == 6
    actual = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
    for write in reply["writes"]:
        full = values[write["hf"]]
        axis = (
            0
            if write["hf"]
            not in (
                "model.language_model.layers.0.mlp.down_proj.weight",
                "model.visual.blocks.0.mlp.linear_fc2.weight",
                "model.visual.blocks.0.attn.proj.weight",
            )
            else 1
        )
        source = full.chunk(2, dim=axis)[write["source_rank"]].contiguous()
        copy_rectangle(source, actual[write["param"]], write)
    model.load_weights(values.items())
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, actual[name], rtol=0, atol=0)
        assert not torch.equal(value, original[name])


def test_alias_or_custom_loader_falls_back_as_complete_component():
    model, values = Qwen3_5ForConditionalGeneration(), _values()
    model.tied = model.model.layers[0].mlp.down_proj.weight
    model.visual.blocks[0].mlp.linear_fc1.weight.weight_loader = lambda param, value: None
    reply = describe_tp_targets(model, _description(values), 1)
    assert "model.language_model.layers.0.mlp.down_proj.weight" not in reply["selected_hf"]
    assert "model.visual.blocks.0.mlp.linear_fc1.weight" not in reply["selected_hf"]
    assert "model.language_model.layers.0.mlp.gate_proj.weight" in reply["selected_hf"]


def test_qwen3_5_moe_refresh_is_explicit():
    from relax.backends.sglang.tp_shard_adapters import needs_native_refresh

    model = Qwen3_5ForConditionalGeneration()
    assert not needs_native_refresh(model)
    moe = type(
        "Qwen3_5MoeForConditionalGeneration",
        (Qwen3_5ForConditionalGeneration,),
        {"__module__": "sglang.srt.models.qwen3_5"},
    )()
    moe._routed_experts_weights_of_layer = object()
    assert needs_native_refresh(moe)


def test_layout_reply_rejects_stale_or_misrouted_rpc():
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer

    sync = object.__new__(ShardP2PSynchronizer)
    sync.rank, sync.geometry = 0, {"tp_size": 2}
    requests = [
        [
            dict(session="s", engine=engine, version=3, nonce=f"{engine}-{tp}", description_hash="h", tp_rank=tp)
            for tp in range(2)
        ]
        for engine in range(2)
    ]
    replies = [
        [dict(command, selected_hf=[], components=[], target_shapes={}, target_hf={}, writes=[]) for command in pair]
        for pair in requests
    ]
    responses = [dict(success=True, message=json.dumps({"shard_p2p_tp_targets": pair})) for pair in replies]
    assert sync._decode_tp_replies(requests, responses) == replies[0]
    replies[1][0]["nonce"] = "stale"
    with pytest.raises(ValueError, match="Misrouted"):
        sync._decode_tp_replies(
            requests, [responses[0], dict(success=True, message=json.dumps({"shard_p2p_tp_targets": replies[1]}))]
        )


def test_ordinary_source_binds_fresh_storage_and_rejects_layout_change(monkeypatch):
    from relax.backends.megatron.weight_update import tp_shard_source as module

    source = object.__new__(module.OrdinaryTPSource)
    source.group, source.rank, source.replica_rank = "actor", 0, 0
    source.replica_ranks, source.tp_rank, source.tp_size = (0,), 0, 2
    source.native_specs, source.selected, source._schema = {}, frozenset(), None
    source.iterator = SimpleNamespace(_vanilla_key_map={"fused": "fused"})
    monkeypatch.setattr(module.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(module.mpu, "get_data_parallel_group", lambda **kwargs: "replicas")
    weights = {"fused": torch.arange(32, dtype=torch.bfloat16).view(8, 4)}
    region = dict(
        hf="gate",
        source="fused",
        global_shape=[16, 4],
        offset=[0, 0],
        shape=[8, 4],
        storage_offset=0,
        stride=4,
        dtype=str(torch.bfloat16),
    )
    source._views = lambda weights, *, selected_only: (weights, [region], {"fused": ["gate"]})
    source.catalogue(weights)
    source.selected = frozenset(("fused",))
    old = source.bind(weights, seed=True)["ordinary:storage-0"]
    fresh = {"fused": weights["fused"].clone() + 64}
    current = source.bind(fresh)["ordinary:storage-0"]
    assert current.data_ptr() == fresh["fused"].data_ptr() != old.data_ptr()
    with pytest.raises(ValueError, match="cold layout"):
        source.bind({"fused": fresh["fused"].t()})
