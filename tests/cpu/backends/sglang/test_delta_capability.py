# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual SG loader branches behind the optional dependency boundary."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.delta_capability import sparse_loader_identity
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


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("is_gguf_weight", True),
        ("is_gguf_weight_type", True),
        ("is_metadata", True),
        ("needs_scalar_to_array", True),
        ("use_bitsandbytes_4bit", True),
        ("packed_dim", 1),
    ],
)
def test_actual_sg_special_parameter_metadata_prevents_sparse_admission(attribute, value):
    model, row = model_with_row()
    assert sparse_loader_identity(model) is not None
    setattr(row.weight, attribute, value)
    assert sparse_loader_identity(model) is None
