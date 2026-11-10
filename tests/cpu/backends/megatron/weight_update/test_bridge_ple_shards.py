# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def ple_converter(monkeypatch, request):
    pytest.importorskip("megatron.bridge.models.qwen.qwen4_exp_bridge")
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import mpu
    from megatron.bridge.models.qwen.qwen4_exp_bridge import PLENGramEmbeddingMapping

    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter

    owner, tp_rank = request.param
    tp_group = SimpleNamespace(size=lambda: 4, rank=lambda: tp_rank)
    pp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 0 if owner else 1)
    monkeypatch.setattr(mpu, "is_initialized", lambda: True)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: tp_group)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: pp_group)
    monkeypatch.setattr(mpu, "get_expert_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "get_expert_tensor_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.delenv("QWEN48_PLE_CPU_OFFLOAD", raising=False)
    name = "language_model.decoder.layers.1.per_layer_embedding.ple_embedding.ngram_embedding.weight"
    mapping = PLENGramEmbeddingMapping(
        name, "model.language_model.ple.ple_embedding.ngram_embedding.shard_{}.weight", 3
    )
    converter = BridgeConverter(Namespace(), [], None)
    converter._bridge_task_map = {name: SimpleNamespace(mapping=mapping, megatron_module=None)} if owner else {}
    converter._bridge_mapping_registry = MegatronMappingRegistry(mapping)
    return converter, name, mapping, tp_group, pp_group


@pytest.mark.parametrize(
    "ple_converter", [(owner, rank) for owner in (False, True) for rank in range(4)], indirect=True
)
def test_bridge_ple_shards_exports_on_owner_and_non_owner_pp_ranks(ple_converter, monkeypatch):
    converter, name, mapping, tp_group, pp_group = ple_converter
    full_table = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    rank_start = tp_group.rank() * 3
    local_rows = full_table[rank_start : rank_start + 3]
    gathered = []

    def all_reduce(shard, *, group):
        assert group is tp_group
        assert mapping.pp_group is None
        start = len(gathered) * 4
        expected_local = torch.zeros_like(shard)
        lo, hi = max(start, rank_start), min(start + 4, rank_start + 3)
        if hi > lo:
            expected_local[lo - start : hi - start] = full_table[lo:hi]
        torch.testing.assert_close(shard, expected_local)
        shard.copy_(full_table[start : start + 4])
        gathered.append(start)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    shards = converter.iter_ple_shards("module.module." + name, local_rows)
    assert gathered == []
    for index in range(3):
        hf_name, shard = next(shards)
        assert hf_name == f"model.language_model.ple.ple_embedding.ngram_embedding.shard_{index}.weight"
        torch.testing.assert_close(shard, full_table[index * 4 : (index + 1) * 4])
        assert len(gathered) == index + 1
    with pytest.raises(StopIteration):
        next(shards)
    assert mapping.pp_group is pp_group
    assert mapping.tp_group is tp_group


@pytest.mark.parametrize("ple_converter", [(False, 0), (True, 0)], indirect=True)
def test_bridge_ple_shards_restores_pp_group_on_early_close(ple_converter, monkeypatch):
    converter, name, mapping, tp_group, pp_group = ple_converter
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *, group: None)
    shards = converter.iter_ple_shards(name, torch.ones(3, 2))
    next(shards)
    assert mapping.pp_group is None
    shards.close()
    assert mapping.pp_group is pp_group
    assert mapping.tp_group is tp_group
