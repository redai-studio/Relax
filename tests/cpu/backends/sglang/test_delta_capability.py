# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual SG loader branches behind the optional dependency boundary."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.delta_capability import _trivial_expert_layout, full_loader_inputs, sparse_loader_identity
from relax.backends.sglang.delta_loader import load_delta_weights
from relax.distributed.weight_sync.codec import encode


def model_with_row():
    pytest.importorskip("sglang.srt.models.qwen3")
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.quant_config = None
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.pp_group = SimpleNamespace(is_last_rank=False)
    model.model = torch.nn.Module()
    row = RowParallelLinear.__new__(RowParallelLinear)
    torch.nn.Module.__init__(row)
    row.tp_rank, row.tp_size, row.use_presharded_weights = 0, 2, False
    row.quant_method = UnquantizedLinearMethod()
    row.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    row.weight.input_dim, row.weight.weight_loader = 1, row.weight_loader
    model.model.row = row
    return model, row


def expert_metadata():
    pytest.importorskip("sglang.srt.eplb.expert_location")
    from dataclasses import fields

    from sglang.srt.eplb.expert_location import ExpertLocationMetadata

    physical = torch.arange(4).repeat(2, 1)
    logical = torch.full((2, 4, 4), -1, dtype=torch.int64)
    logical[..., 0] = physical
    values = dict(
        physical_to_logical_map=physical.clone(),
        physical_to_logical_map_cpu=physical,
        logical_to_all_physical_map=logical.clone(),
        logical_to_all_physical_map_cpu=logical,
        logical_to_all_physical_map_num_valid=torch.ones(2, 4, dtype=torch.int64),
        logical_to_rank_dispatch_physical_map=None,
        ep_size=2,
    )
    metadata = ExpertLocationMetadata(**{field.name: values[field.name] for field in fields(ExpertLocationMetadata)})
    # The older local SG schema exposes ep_size through a distributed property;
    # model the new explicit metadata field without initializing any PG.
    vars(metadata)["ep_size"] = 2
    return metadata


def test_static_expert_layout_tracks_actual_loader_mapping_and_mutation():
    metadata = expert_metadata()
    identity = _trivial_expert_layout(metadata)
    assert identity is not None
    with torch.device("meta"):
        assert _trivial_expert_layout(metadata) == identity
    for layer in range(2):
        for expert in range(4):
            assert metadata.logical_to_all_physical(layer, expert) == [expert]
    metadata.logical_to_all_physical_map_cpu[0, 0, 0] = 1
    assert _trivial_expert_layout(metadata) is None
    metadata.logical_to_all_physical_map_cpu[0, 0, 0] = 0
    assert _trivial_expert_layout(metadata) != identity
    before = _trivial_expert_layout(metadata)
    metadata.logical_to_all_physical_map_cpu = metadata.logical_to_all_physical_map_cpu.clone()
    assert _trivial_expert_layout(metadata) != before


def test_static_expert_layout_rejects_duplicate_remap_and_unknown_helper():
    metadata = expert_metadata()
    metadata.logical_to_all_physical_map_cpu[0, 0, 1] = 1
    assert _trivial_expert_layout(metadata) is None
    metadata.logical_to_all_physical_map_cpu[0, 0, 1] = -1
    other = expert_metadata()
    other.logical_to_all_physical_map_cpu[0, 0, 0] = 1
    metadata.logical_to_all_physical = other.logical_to_all_physical
    assert metadata.logical_to_all_physical(0, 0) == [1]
    assert _trivial_expert_layout(metadata) is None
    metadata.logical_to_all_physical = lambda *args: [0]
    assert _trivial_expert_layout(metadata) is None
    assert _trivial_expert_layout(SimpleNamespace()) is None


def test_static_expert_layout_identity_tracks_class_helper_replacement(monkeypatch):
    metadata = expert_metadata()
    before = _trivial_expert_layout(metadata)
    monkeypatch.setattr(type(metadata), "logical_to_all_physical", lambda *args: [1])
    assert metadata.logical_to_all_physical(0, 0) == [1]
    assert _trivial_expert_layout(metadata) != before


def begin(model, version, *, seed=False, base=None):
    control = dict(
        op="begin", version=version, base=base, seed=seed, receiver_gpu_budget_bytes=4096, staging_budget_bytes=1024
    )
    load_delta_weights(model, [(json.dumps(control), torch.empty(0))])


def seed_row(model):
    begin(model, "0", seed=True)
    metadata, raw = encode(
        [("model.row.weight", (2, 4), torch.float32, None, torch.arange(8, dtype=torch.float32))], max_bytes=1024
    )
    status = torch.empty(1, dtype=torch.uint8)
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="consume", version="0", frame=0, metadata=metadata)), raw),
            ("__delta_capability__", status),
        ],
    )
    load_delta_weights(model, [(json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0))])
    return status


def test_actual_sg_tp_geometry_change_requires_full_before_any_sparse_write():
    model, row = model_with_row()
    assert sparse_loader_identity(model) is not None
    assert seed_row(model).tolist() == [0]
    before = row.weight.detach().clone()
    row.tp_rank = 1
    with pytest.raises(ValueError, match="acknowledged loader geometry"):
        begin(model, "1", base="0")
    assert torch.equal(before, row.weight)


@pytest.mark.parametrize("head", [False, True])
def test_actual_sg_vocab_shard_dataclass_is_admitted_and_geometry_is_tracked(head):
    model, _row = model_with_row()
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
        VocabParallelEmbeddingShardIndices,
    )

    cls = ParallelLMHead if head else VocabParallelEmbedding
    embedding = cls.__new__(cls)
    torch.nn.Module.__init__(embedding)
    embedding.tp_rank, embedding.tp_size, embedding.use_presharded_weights = 0, 2, False
    embedding.org_vocab_size = 8
    embedding.shard_indices = VocabParallelEmbeddingShardIndices(0, 4, 8, 8, 0, 4, 8, 8)
    embedding.weight = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
    embedding.weight.output_dim, embedding.weight.weight_loader = 0, embedding.weight_loader
    model.model.embedding = embedding
    before = sparse_loader_identity(model)
    assert before is not None
    embedding.shard_indices = VocabParallelEmbeddingShardIndices(4, 8, 8, 8, 4, 8, 8, 8)
    assert sparse_loader_identity(model) != before


def test_qkv_constructor_geometry_includes_separate_kv_partition():
    model, _row = model_with_row()
    from sglang.srt.layers.linear import QKVParallelLinear

    qkv = QKVParallelLinear(4, 1, 8, 4, bias=False, params_dtype=torch.bfloat16, tp_rank=0, tp_size=2)
    # Separate KV partitions exist in the actual deployment's SG release.
    # Older local releases inherit the query partition; make that explicit.
    qkv.kv_tp_rank, qkv.kv_tp_size = 0, 2
    model.model.qkv = qkv
    before = sparse_loader_identity(model)
    assert before is not None
    qkv.kv_tp_rank = 1
    assert sparse_loader_identity(model) != before


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("is_gguf_weight", True),
        ("is_gguf_weight_type", True),
        ("is_metadata", True),
        ("needs_scalar_to_array", True),
        ("use_bitsandbytes_4bit", True),
        ("packed_dim", 1),
        ("pack_factor", 2),
        ("packed_factor", 2),
        ("marlin_tile_size", 16),
    ],
)
def test_actual_sg_special_parameter_metadata_prevents_sparse_admission(attribute, value):
    model, row = model_with_row()
    assert sparse_loader_identity(model) is not None
    setattr(row.weight, attribute, value)
    assert sparse_loader_identity(model) is None


def test_real_qwen_distinct_embedding_alias_inputs_require_full_and_preserve_last_writer():
    model, _row = model_with_row()
    shared = torch.zeros(2, 2)
    model.model.embed_tokens = torch.nn.Module()
    model.model.embed_tokens.weight = torch.nn.Parameter(shared, requires_grad=False)
    model.lm_head = torch.nn.Module()
    model.lm_head.weight = torch.nn.Parameter(shared, requires_grad=False)
    names = ("model.embed_tokens.weight", "lm_head.weight")
    assert sparse_loader_identity(model) is not None
    assert full_loader_inputs(model) == frozenset(names)
    begin(model, "0", seed=True)
    metadata, raw = encode(
        [(name, (2, 2), torch.float32, None, torch.full((4,), float(i))) for i, name in enumerate(names)],
        max_bytes=1024,
    )
    status = torch.empty(2, dtype=torch.uint8)
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="consume", version="0", frame=0, metadata=metadata)), raw),
            ("__delta_capability__", status),
        ],
    )
    assert status.tolist() == [1, 1]
    assert torch.equal(shared, torch.ones(2, 2))
    load_delta_weights(model, [(json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0))])
    begin(model, "1", base="0")
    metadata, raw = encode(
        [(names[0], (2, 2), torch.float32, torch.tensor([0], dtype=torch.int32), torch.tensor([7.0]))],
        max_bytes=1024,
    )
    with pytest.raises(ValueError, match="complete conversion group as FULL"):
        load_delta_weights(
            model,
            [
                (json.dumps(dict(op="consume", version="1", frame=0, metadata=metadata)), raw),
                ("__delta_capability__", status[:1]),
            ],
        )
    assert torch.equal(shared, torch.ones(2, 2))


def test_duplicate_name_for_same_parameter_follows_sg_named_parameter_deduplication():
    model, _row = model_with_row()
    model.model.embed_tokens = torch.nn.Module()
    model.model.embed_tokens.weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    model.lm_head = model.model.embed_tokens
    assert full_loader_inputs(model) == frozenset()
    assert sparse_loader_identity(model) is not None
