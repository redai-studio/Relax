# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual SG loader branches behind the optional dependency boundary."""

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.delta_capability import full_loader_inputs, sparse_loader_identity
from relax.backends.sglang.delta_loader import load_delta_weights
from relax.backends.sglang.delta_model_loaders import _trivial_expert_layout
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


def test_conditional_moe_sparse_scope_keeps_gated_attention_dense(monkeypatch):
    import sys

    from relax.backends.sglang.delta_capability import requires_full_input

    class ConditionalMoe(torch.nn.Module):
        def load_weights(self, weights):
            raise AssertionError("Admission must not execute a loader")

    monkeypatch.setitem(
        sys.modules, "sglang.srt.models.qwen3_5", SimpleNamespace(Qwen3_5MoeForConditionalGeneration=ConditionalMoe)
    )
    model = ConditionalMoe()
    model.config = SimpleNamespace(num_hidden_layers=40, num_experts=256)
    for name in (
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.39.mlp.experts.255.down_proj.weight",
        "model.language_model.layers.39.mlp.experts.gate_up_proj",
    ):
        assert not requires_full_input(model, name)
    for name in (
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.experts.256.up_proj.weight",
        "model.layers.40.mlp.experts.down_proj",
        "model.visual.layers.0.mlp.experts.down_proj",
    ):
        assert requires_full_input(model, name)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_real_model_dispatchers_reuse_observed_tp3_pp_copy_without_admission_override(family):
    """Only stage-local source columns are installed, preserving FP32 words."""
    pytest.importorskip("sglang.srt.layers.linear")
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.parameter import ModelWeightParameter

    if family == "llama":
        from sglang.srt.models.llama import LlamaForCausalLM as model_class
    else:
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM as model_class

    def make_model():
        model = model_class.__new__(model_class)
        torch.nn.Module.__init__(model)
        model.quant_config = None
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.model = torch.nn.Module()
        model.model.start_layer, model.model.end_layer = 2, 3
        row = RowParallelLinear(6, 2, bias=False, params_dtype=torch.float32, tp_rank=1, tp_size=3)
        if family == "llama":
            # Some SG checkpoints bind the v2 loader through a native Parameter
            # subclass. Exercise its real source slicing and physical copy.
            row.weight = ModelWeightParameter(
                data=torch.zeros_like(row.weight), input_dim=1, output_dim=0, weight_loader=row.weight_loader_v2
            )
        model.model.row = row
        return model

    actual, reference = make_model(), make_model()
    names = ("model.layers.0.missing.weight", "model.row.weight")
    words = torch.arange(12, dtype=torch.int32)
    positions = torch.tensor([0, 2, 3, 8, 10], dtype=torch.int32)
    for seed in (True, False):
        version = str(int(not seed))
        begin(actual, version, seed=seed, base=None if seed else "0")
        if not seed:
            words[positions.long()] = torch.tensor([-7, 2143289345, -2147483648, -43, 55], dtype=torch.int32)
        value = words.view(torch.float32)
        inputs = [
            (name, (2, 6), value.dtype, None if seed else positions, value if seed else value[positions.long()])
            for name in names
        ]
        meta, raw = encode(inputs, max_bytes=1024)
        status = torch.full((2,), 80, dtype=torch.uint8)
        load_delta_weights(
            actual,
            [
                (json.dumps(dict(op="consume", version=version, frame=0, metadata=meta)), raw),
                ("__delta_capability__", status),
            ],
        )
        assert not status.any()
        load_delta_weights(actual, [(json.dumps(dict(op="finalize", version=version, frames=1)), torch.empty(0))])
        reference.load_weights([(name, value.view(2, 6)) for name in names])
        assert torch.equal(actual.model.row.weight.view(torch.int32), reference.model.row.weight.view(torch.int32))
    metrics = actual._relax_delta_receiver.target_shard_metrics
    assert (metrics["entries"], metrics["ordinary_entries"], metrics["words"], metrics["local_words"]) == (2, 0, 10, 3)


@pytest.mark.parametrize(("dtype", "nan_word"), [(torch.bfloat16, 0x7FC1), (torch.float16, 0x7E01)])
def test_declared_dispatcher_keeps_full_derived_inputs_and_checks_epoch_before_sparse_writes(dtype, nan_word):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.zeros(4, dtype=dtype), requires_grad=False)
            self.register_buffer("derived", torch.zeros(4))
            self.invocations = 0
            self.epoch = 1

        def load_weights(self, items):
            for name, source in items:
                if name == "independent":
                    self.raw.copy_(source)
                else:
                    self.cached = source.clone()
                    self.derived.copy_(source.float() * 2)
            self.invocations += 1

        load_weights.relax_copy_layout = lambda model: (model.epoch, frozenset({"independent"}))

    model = Model()
    reference = Model()
    words = torch.tensor([-32768, 2, 3, 4], dtype=torch.int16)
    for seed in (True, False):
        version = str(int(not seed))
        begin(model, version, seed=seed, base=None if seed else "0")
        pos = None if seed else torch.tensor([1], dtype=torch.int32)
        if not seed:
            words[1] = nan_word
        val = words.view(dtype)
        full = torch.arange(4, dtype=torch.float32) + int(not seed)
        metadata, raw = encode(
            [
                ("independent", (4,), val.dtype, pos, val if seed else val[1:2]),
                ("cache", (4,), full.dtype, None, full),
            ],
            max_bytes=1024,
        )
        status = torch.empty(2, dtype=torch.uint8)
        load_delta_weights(
            model,
            [
                (json.dumps(dict(op="consume", version=version, frame=0, metadata=metadata)), raw),
                ("__delta_capability__", status),
            ],
        )
        assert status.tolist() == [0, 1]
        load_delta_weights(model, [(json.dumps(dict(op="finalize", version=version, frames=1)), torch.empty(0))])
        reference.load_weights([("independent", val), ("cache", full)])
        assert model.invocations == reference.invocations
        assert torch.equal(model.raw.view(torch.int16), reference.raw.view(torch.int16))
        assert torch.equal(model.derived, reference.derived) and torch.equal(model.cached, reference.cached)
    assert model._relax_delta_receiver.target_shard_metrics["local_words"] == 1
    model.epoch += 1
    with pytest.raises(ValueError, match="acknowledged loader geometry"):
        begin(model, "2", base="1")


def test_overlapping_seed_writes_require_full_even_for_transient_dlpack_destination():
    class Aliased(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def load_weights(self, items):
            for name, tensor in items:
                dst = self.weight if name == "first" else torch.from_dlpack(self.weight.detach())
                dst.copy_(tensor)

        load_weights.relax_copy_layout = lambda model: (1, frozenset({"first"}))

    model = Aliased()
    begin(model, "0", seed=True)
    meta, raw = encode(
        [(n, (4,), torch.float32, None, torch.full((4,), float(i))) for i, n in enumerate(("first", "alias"))],
        max_bytes=1024,
    )
    ack = torch.empty(2, dtype=torch.uint8)
    load_delta_weights(
        model,
        [(json.dumps(dict(op="consume", version="0", frame=0, metadata=meta)), raw), ("__delta_capability__", ack)],
    )
    load_delta_weights(
        model,
        [(json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0)), ("__delta_capability__", ack[:1])],
    )
    assert ack[0] == 1 and torch.equal(model.weight, torch.ones(4))


def test_declared_dispatcher_setup_cannot_redirect_a_seeded_write():
    class Rebinding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
            self.rebind = False

        def load_weights(self, weights):
            if self.rebind:
                self.weight = torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
            for _, value in weights:
                self.weight.copy_(value)

        load_weights.relax_copy_layout = lambda model: (1, frozenset({"weight"}))

    m = Rebinding()
    begin(m, "0", seed=True)

    def send(v, pos, values):
        meta, raw = encode([("weight", (4,), torch.float32, pos, values)], max_bytes=1024)
        ack = torch.empty(1, dtype=torch.uint8)
        load_delta_weights(
            m,
            [(json.dumps(dict(op="consume", version=v, frame=0, metadata=meta)), raw), ("__delta_capability__", ack)],
        )

    send("0", None, torch.arange(4.0))
    load_delta_weights(m, [(json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0))])
    old = m.weight
    begin(m, "1", base="0")
    m.rebind = True
    with pytest.raises(ValueError, match="setup changed"):
        send("1", torch.tensor([0], dtype=torch.int32), torch.tensor([17.0]))
    assert m._relax_delta_receiver.failed and m._relax_delta_receiver.version == "0"
    assert torch.equal(old, torch.arange(4.0)) and not m.weight.any()


def test_declared_loader_that_materializes_iterator_keeps_unobserved_input_full():
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def load_weights(self, weights):
            for _, value in list(weights):
                self.weight.copy_(value)

        load_weights.relax_copy_layout = lambda model: (1, frozenset({"weight"}))

    model = Buffered()
    begin(model, "0", seed=True)
    expected = torch.arange(4.0)
    directory, raw = encode([("weight", (4,), expected.dtype, None, expected)], max_bytes=1024)
    status = torch.zeros(1, dtype=torch.uint8)
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="consume", version="0", frame=0, metadata=directory)), raw),
            ("__delta_capability__", status),
        ],
    )
    assert torch.equal(model.weight, expected)
    assert status.tolist() == [1]
