# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""GDN rectangles agree with the installed Bridge and SGLang conversions."""

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.tp_shard_source import local_tp_regions
from relax.backends.sglang.tp_shard_layout import describe_tp_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import copy_rectangle, group_storage_views


@pytest.mark.parametrize("source_tp,target_tp", [(2, 4), (4, 2)])
@torch.no_grad()
def test_gdn_native_conversion_and_complete_group_fallback(monkeypatch, source_tp, target_tp):
    from megatron.bridge.models.conversion import param_mapping as bridge
    from sglang.srt.layers.linear import MergedColumnParallelLinear
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet

    config = SimpleNamespace(
        hidden_size=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    names = {key: f"model.layers.0.linear_attn.in_proj_{key}.weight" for key in ("qkv", "z", "b", "a")}
    values = {
        key: (torch.arange(rows * 4).reshape(rows, 4) + index * 37).to(torch.bfloat16)
        for index, (key, rows) in enumerate(zip(names, (32, 16, 8, 8), strict=True))
    }
    mapping = bridge.GDNLinearMappingSeparate("in_proj.weight", **names)
    mapping._tp_group = SimpleNamespace(rank=0, size=source_tp)
    # Only process-group rank metadata is isolated; both numerical converters run natively.
    monkeypatch.setattr(bridge, "get_pg_rank", lambda group: group.rank)
    monkeypatch.setattr(bridge, "get_pg_size", lambda group: group.size)
    task = SimpleNamespace(
        mapping=mapping, megatron_module=SimpleNamespace(config=config), global_param_name="in_proj.weight"
    )
    fused = bridge._fuse_gdn_separate_to_grouped(config, **values)
    packed = bridge.merge_gdn_linear_weights(config, *fused, tp_size=source_tp)
    sources, records = [], []
    for rank, shard in enumerate(packed.chunk(source_tp)):
        # Preserve a nonzero physical storage offset in the real native shard.
        source = torch.cat((torch.zeros((1, 4), dtype=shard.dtype), shard))[1:]
        source.partition_dim = 0
        mapping.tp_group.rank = rank
        regions = local_tp_regions("in_proj.weight", source, task, rank, source_tp)
        assert len(regions) == 6
        records.append(dict(rank=rank, regions=regions, groups={"in_proj.weight": list(names.values())}))
        storage, _ = group_storage_views({"in_proj.weight": source})
        sources.append(next(iter(storage.values())))
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
        layer.linear_attn = torch.nn.Module()
        for key, sizes in (("qkvz", [8, 8, 16, 16]), ("ba", [8, 8])):
            module = MergedColumnParallelLinear(
                4,
                sizes,
                bias=False,
                params_dtype=torch.bfloat16,
                tp_rank=rank,
                tp_size=target_tp,
            )
            module.weight.weight_loader = Qwen3_5GatedDeltaNet._make_packed_weight_loader(module, module.weight_loader)
            setattr(layer.linear_attn, "in_proj_" + key, module)
        qkvz, ba = layer.linear_attn.in_proj_qkvz, layer.linear_attn.in_proj_ba
        qkvz.weight.weight_loader(qkvz.weight, values["qkv"], (0, 1, 2))
        qkvz.weight.weight_loader(qkvz.weight, values["z"], 3)
        ba.weight.weight_loader(ba.weight, values["b"], 0)
        ba.weight.weight_loader(ba.weight, values["a"], 1)
        expected = {name: value.clone() for name, value in model.named_parameters()}
        reply = describe_tp_targets(model, description, rank)
        assert reply["selected_hf"] == sorted(names.values())
        # Different receiver processes must emit the same dependency-group plan.
        assert list(dict.fromkeys(write["param"] for write in reply["writes"])) == sorted(reply["target_shapes"])
        actual = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
        for write in reply["writes"]:
            copy_rectangle(sources[write["source_rank"]], actual[write["param"]], write)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        # An unsupported member must reject both fused destinations and all four HF tensors.
        ba.weight.data = ba.weight.data.float()
        rejected = describe_tp_targets(model, description, rank)
        assert not rejected["selected_hf"] and not rejected["writes"]
