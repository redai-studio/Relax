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
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0)),
            ("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)),
        ],
    )
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
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0)),
            ("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)),
        ],
    )
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
        load_delta_weights(
            actual,
            [
                (json.dumps(dict(op="finalize", version=version, frames=1)), torch.empty(0)),
                ("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)),
            ],
        )
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
        load_delta_weights(
            model,
            [
                (json.dumps(dict(op="finalize", version=version, frames=1)), torch.empty(0)),
                ("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)),
            ],
        )
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


def test_tensor_list_writer_preserves_unchanged_words_of_later_alias():
    from relax.distributed.weight_sync.codec import schema_digest

    class Batched(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def load_weights(self, items):
            for name, value in items:
                if name == "batched":
                    torch._foreach_copy_([self.weight], [value])
                else:
                    self.weight.copy_(value)

        load_weights.relax_copy_layout = lambda model: (1, frozenset({"last"}))

    model, reference = Batched(), Batched()
    schema = {name: ((4,), "float32") for name in ("batched", "last")}
    required = [True, True]
    for version in (0, 1):
        seed = version == 0
        begin(model, str(version), seed=seed, base=None if seed else "0")
        inputs = [("batched", torch.full((4,), 1.0 + version)), ("last", torch.tensor([3.0 + version, 3, 3, 3]))]
        reference.load_weights(inputs)
        entries = []
        for full, (name, value) in zip(required, inputs, strict=True):
            positions = None if full else torch.tensor([0], dtype=torch.int32)
            entries.append((name, (4,), value.dtype, positions, value if full else value[:1]))
        meta, raw = encode(entries, max_bytes=128)
        status = torch.empty(3, dtype=torch.uint8)
        control = dict(op="consume", version=str(version), frame=0, metadata=meta)
        load_delta_weights(model, [(json.dumps(control), raw), ("__delta_capability__", status[:2])])
        final = dict(op="finalize", version=str(version), frames=1)
        ack = [("__delta_capability__", status if seed else status[:1])]
        if seed:
            final.update(full_required_count=2, full_required_schema=schema_digest(schema))
        load_delta_weights(model, [(json.dumps(final), torch.empty(0)), *ack])
        if seed:
            required = status[1:].tolist()
        assert torch.equal(model.weight.view(torch.int32), reference.weight.view(torch.int32))


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
    load_delta_weights(
        m,
        [
            (json.dumps(dict(op="finalize", version="0", frames=1)), torch.empty(0)),
            ("__delta_capability__", torch.full((1,), 255, dtype=torch.uint8)),
        ],
    )
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


@pytest.mark.parametrize("tp", [2, 3])
def test_final_seed_mask_replays_unchanged_tied_writer_through_source_and_pipeline(monkeypatch, tp):
    import ray
    import torch.distributed as dist

    from relax.backends.megatron.weight_update.delta_export import ShardDeltaExporter
    from relax.backends.sglang.delta_pipeline import _install
    from relax.backends.sglang.weight_sync_proxy import WeightSyncProxy, _status_buffer
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    embed, head, row = "model.embed_tokens.weight", "lm_head.weight", "model.row.weight"
    values = {embed: torch.ones(2, 2), head: torch.full((2, 2), 2.0), row: torch.arange(4 * tp).float().view(2, -1)}
    iterator = SimpleNamespace(get_hf_weight_chunks=lambda shards: ([(name, shards[name])] for name in values))
    config = DeltaSyncConfig(stream="tied", anchor_interval=3)
    source = ShardDeltaExporter(iterator, SimpleNamespace(**vars(iterator)), snapshot_budget_bytes=1024, config=config)
    transfer = source.transfer = DeltaTransfer(config, "cpu")
    transfer.routes, transfer.connection = [], {}
    monkeypatch.setattr(transfer, "connect", lambda *args, **kwargs: False)
    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(dist, "all_gather_object", lambda out, obj, group: out.__setitem__(0, obj))
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *args, **kwargs: None)

    wire, emitted, replicas = {}, [], []

    def stage(directory, raw):
        wire[transfer.frame] = raw
        emitted.extend((e["name"], e["full"], e["count"]) for e in directory["tensors"])
        transfer._staged.append((transfer.frame, directory))
        transfer.frame += 1

    monkeypatch.setattr(transfer, "stage", stage)

    def engine(pp):
        models = []
        for stage_id in range(pp):
            for rank in range(tp):
                m, layer = model_with_row()
                layer.tp_size, layer.tp_rank = tp, rank
                m.config.tie_word_embeddings = pp == 2
                m.pp_group.is_last_rank = stage_id == pp - 1
                if stage_id == 0:
                    m.model.embed_tokens = torch.nn.Linear(2, 2, bias=False)
                if m.pp_group.is_last_rank:
                    m.lm_head = torch.nn.Linear(2, 2, bias=False)
                models.append(m)
        replicas.append(models)
        proxy = WeightSyncProxy(None)
        proxy.tp_size, proxy.pp_size = tp, pp
        proxy.children = [None] * len(models)
        proxy.identities = [dict(uuid="cpu")] * len(models)
        statuses = []
        monkeypatch.setattr(
            proxy,
            "_controls",
            lambda controls, **_: [
                load_delta_weights(m, [(json.dumps(c), torch.empty(0))]) for m, c in zip(models, controls, strict=True)
            ],
        )
        monkeypatch.setattr(
            proxy,
            "_shared_controls",
            lambda meta, _: (
                [dict(op="consume", metadata=meta, frame=proxy.batch, version=proxy.version)] * len(models),
                False,
                b"",
            ),
        )

        def commands(cmds, *, deadline):
            op = cmds[0]["op"]
            if op == "status":
                assert pp == 1 or all(status[-1].item() == 1 for status in statuses)
                return [dict(full_required=(s[:-1] if pp > 1 else s).tolist()) for s in statuses]
            if op != "export":
                return [dict(success=True)] * len(cmds)
            statuses.clear()
            for m, cmd in zip(models, cmds, strict=True):
                control = cmd["control"]
                raw = wire[cmd["round"]] if control["op"] == "consume" else torch.empty(0, dtype=torch.uint8)
                status = _status_buffer(cmd["count"], bytes(int(pp > 1)), "cpu")
                items = [(json.dumps(control), raw), ("__delta_capability__", status)]
                (_install if pp > 1 else load_delta_weights)(m, items)
                statuses.append(status)
                if control["op"] == "consume" and proxy.version in ("0", "3"):
                    assert not status[: cmd["count"]].any()  # Each early seed ACK is optimistic.
            return [dict(serialized=None)] * len(cmds)

        monkeypatch.setattr(proxy, "_commands", commands)
        monkeypatch.setattr(proxy, "_payload", lambda *args, **kwargs: {})
        monkeypatch.setattr(proxy, "_request", lambda *args, **kwargs: None)
        monkeypatch.setattr(proxy, "receive", lambda *args, **kwargs: None)
        monkeypatch.setattr(proxy, "ready", lambda *args, **kwargs: None)
        return SimpleNamespace(
            delta_weight_sync=SimpleNamespace(remote=lambda action, **kw: getattr(proxy, action)(**kw))
        )

    transfer.engines = (engine(1), engine(2))
    reference, ref_row = model_with_row()
    ref_row.tp_size = tp
    reference.config.tie_word_embeddings = reference.pp_group.is_last_rank = True
    reference.lm_head = torch.nn.Linear(2, 2, bias=False)
    for step in range(5):
        seed = step % 3 == 0
        emitted.clear()
        values[embed].add_(1)  # Later head remains unchanged, so its FULL replay is essential.
        values[row][0, 1] += 10
        result = source.synchronize(lambda: values, version=str(step), engines=[], gpu_counts=[], group=object())
        assert result["full"] == seed and not result["force_full_seed"] and source.full_names == {embed, head}
        assert emitted == [(embed, True, 4), (head, True, 4), (row, seed, 4 * tp if seed else 1)]
        assert all("full_required_mask" not in reply for reply in result["receivers"])
        with torch.no_grad():
            reference.load_weights(values.items())
        for rank, model in enumerate(replicas[1][tp:]):
            assert torch.equal(model.lm_head.weight, reference.lm_head.weight)
            assert torch.equal(model.model.row.weight, values[row][:, 2 * rank : 2 * rank + 2])
            if not seed:
                assert model._relax_delta_receiver.target_shard_metrics["local_words"] == int(rank == 0)
        source.commit()


def test_nested_chained_and_multi_region_writers_keep_disjoint_input_sparse():
    from relax.distributed.weight_sync.codec import schema_digest

    regions = {"a": (0, 8), "b": (1, 3), "c": (7, 11), "d": (10, 12), "e": (12, 13), "free": (13, 17)}

    class Slices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(17), requires_grad=False)
            self.calls = 0

        def load_weights(self, items):
            self.calls += 1
            for name, value in items:
                start, stop = regions[name]
                tensor = torch.from_dlpack(self.weight.detach()) if name == "b" else self.weight
                tensor[start:stop].copy_(value)
                if name == "e":
                    tensor[2:3].copy_(value)

        load_weights.relax_copy_layout = lambda model: (1, frozenset(regions))

    model, reference = Slices(), Slices()
    values = {name: torch.full((hi - lo,), float(i)) for i, (name, (lo, hi)) in enumerate(regions.items())}
    schema = {name: (tuple(v.shape), "float32") for name, v in values.items()}
    for step in range(2):
        begin(model, str(step), seed=step == 0, base="0" if step else None)
        values["a"].add_(1)
        values["free"][0] += 1
        for i, names in enumerate((["a", "b"], ["c", "d", "e", "free"])):
            entries = []
            for name in names:
                v = values[name]
                sparse = step == 1 and name == "free"
                pos = torch.tensor([0], dtype=torch.int32) if sparse else None
                entries.append((name, tuple(v.shape), v.dtype, pos, v[:1] if sparse else v))
            directory, raw = encode(entries, max_bytes=1024)
            directory["loader_ends"] = [1, len(entries)]
            ack = torch.empty(len(entries), dtype=torch.uint8)
            load_delta_weights(
                model,
                [
                    (json.dumps(dict(op="consume", version=str(step), frame=i, metadata=directory)), raw),
                    ("__delta_capability__", ack),
                ],
            )
            for group in (names[:1], names[1:]):
                reference.load_weights((name, values[name]) for name in group)
        options = {} if step else dict(full_required_count=len(schema), full_required_schema=schema_digest(schema))
        output = torch.full((1 if step else len(schema) + 1,), 255, dtype=torch.uint8)
        load_delta_weights(
            model,
            [
                (json.dumps(dict(op="finalize", version=str(step), frames=2, **options)), torch.empty(0)),
                ("__delta_capability__", output),
            ],
        )
        assert output.tolist() == ([0] if step else [0, 1, 1, 1, 1, 1, 0])
        assert torch.equal(model.weight, reference.weight) and model.calls == reference.calls
    assert model._relax_delta_receiver.target_shard_metrics["local_words"] == 1


@pytest.mark.parametrize("bad", ["bool", "negative", "short", "extra", "order", "alias", "frames", "missing"])
def test_bad_final_mask_keeps_old_version(bad):
    from relax.distributed.weight_sync.codec import schema_digest

    model, layer = model_with_row()
    begin(model, "0", seed=True)
    vals = torch.arange(8, dtype=torch.float32)
    names = ("model.row.weight", "row_mirror")
    directory, raw = encode([(n, (2, 4), vals.dtype, None, vals) for n in names], max_bytes=1024)
    ack = torch.empty(2, dtype=torch.uint8)
    load_delta_weights(
        model,
        [
            (json.dumps(dict(op="consume", version="0", frame=0, metadata=directory)), raw),
            ("__delta_capability__", ack),
        ],
    )
    before = layer.weight.clone()
    order = reversed(names) if bad == "order" else names
    digest = schema_digest({name: ((2, 4), "float32") for name in order})
    control = dict(
        op="finalize",
        version="0",
        frames=int(bad != "frames"),
        full_required_count={"negative": -1, "bool": True}.get(bad, 2),
        full_required_schema=digest,
    )
    status = torch.empty({"short": 2, "extra": 4}.get(bad, 3), dtype=torch.uint8)
    if bad == "alias":
        status = layer.weight.detach().view(torch.uint8).flatten()[:3]
    weights = [(json.dumps(control), torch.empty(0))]
    if bad != "missing":
        weights.append(("__delta_capability__", status))
    with pytest.raises(ValueError):
        load_delta_weights(model, weights)
    receiver = model._relax_delta_receiver
    assert receiver.failed and receiver.version is None and torch.equal(layer.weight, before)


def test_delayed_unattributed_overwrite_cannot_approve_a_partial_final_mask():
    from relax.distributed.weight_sync.codec import schema_digest

    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)

        def load_weights(self, weights):
            delayed = []
            for name, value in weights:
                if name == "direct":
                    self.weight.copy_(value)
                else:
                    delayed.append(value)
            for value in delayed:
                self.weight.copy_(value)

        load_weights.relax_copy_layout = lambda model: (1, frozenset({"direct"}))

    m = Buffered()
    begin(m, "0", seed=True)
    schema = {n: ((4,), "float32") for n in ("direct", "buffered")}
    directory, raw = encode(
        [(n, (4,), torch.float32, None, torch.full((4,), float(i))) for i, n in enumerate(schema)], max_bytes=128
    )
    status = torch.empty(3, dtype=torch.uint8)
    load_delta_weights(
        m,
        [
            (json.dumps(dict(op="consume", version="0", frame=0, metadata=directory)), raw),
            ("__delta_capability__", status[:2]),
        ],
    )
    assert status[:2].tolist() == [0, 1]
    final = dict(
        op="finalize", version="0", frames=1, full_required_count=2, full_required_schema=schema_digest(schema)
    )
    load_delta_weights(m, [(json.dumps(final), torch.empty(0)), ("__delta_capability__", status)])
    assert status.tolist() == [1, 1, 1] and torch.equal(m.weight, torch.ones(4))
