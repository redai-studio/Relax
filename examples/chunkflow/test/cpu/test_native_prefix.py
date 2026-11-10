# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numerical regression using the shipped native QSA/PLE source on CPU.

Only construction, Megatron QKV plumbing and RoPE are small CPU stand-ins.
Selection, sparse-mask attention, n-gram hashing, PLE gating and convolution
are loaded unchanged from the pinned patch. This is not a CUDA/TE validation.
"""

from __future__ import annotations

import ast
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.chunkflow.adapters import prefix, qsa
from examples.chunkflow.integration import install, uninstall
from examples.chunkflow.runtime import RECOMPUTE_ATTR, RUNTIME_ATTR, UNIT_ATTR, ChunkflowRuntime
from examples.chunkflow.scheduler import Chunk, Segment, build_chunks
from examples.chunkflow.test.cpu.test_integration import restore_rope_patch as restore_rope_patch
from examples.chunkflow.units import Unit, build_micro_batches, completed_groups, gen_unit_ops


PATCH = Path(__file__).resolve().parents[4] / "docker/patch/megatron/20261016-qwen38-models.patch"
QSA_MODULE = "megatron.core.transformer.experimental_attention_variant.qsa"
PLE_MODULE = "megatron.core.transformer.per_layer_embedding"


class Packed(SimpleNamespace):
    cu_seqlens_q_padded = None
    cu_seqlens_kv_padded = None


def packed(cu):
    bounds = torch.tensor(cu, dtype=torch.int32)
    return Packed(
        cu_seqlens_q=bounds,
        cu_seqlens_kv=bounds,
        qkv_format="thd",
        cu_seqlens_q_cpu=list(cu),
        max_seqlen_q=max(b - a for a, b in zip(cu[:-1], cu[1:])),
    )


def rotate(x, frequencies, config, cu_seqlens=None, **kwargs):
    if config.mrope_section is None and cu_seqlens is not None:
        positions = torch.cat([torch.arange(b - a) for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:])])
        frequencies = frequencies[positions]
    f = frequencies.reshape(x.shape[0], 1, -1)
    return x * f.cos() + rotate_half(x, False) * f.sin()


def rotate_half(x, interleaved):
    assert not interleaved
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Projection(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class TinySelfAttention(torch.nn.Module):
    def forward(self, hidden_states, attention_mask, *, rotary_pos_emb, packed_seq_params, **kwargs):
        q, k, v = self.projection(hidden_states).squeeze(1).split([8, 4, 4], dim=-1)
        q, k = (
            rotate(x.reshape(-1, heads, 4), rotary_pos_emb, self.config, cu_seqlens=packed_seq_params.cu_seqlens_q)
            for x, heads in ((q, 2), (k, 1))
        )
        out = self.core_attention(q, k, v.reshape(-1, 1, 4), attention_mask, packed_seq_params=packed_seq_params)
        return self.output(out).unsqueeze(1), self.bias


def native_source(relative):
    section = PATCH.read_text().split(f"diff --git a/{relative} b/{relative}\n", 1)[1].split("\ndiff --git ", 1)[0]
    return ast.parse(
        "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))
    )


def load_native():
    namespace = dict(
        torch=torch,
        Tensor=torch.Tensor,
        nn=torch.nn,
        F=F,
        math=math,
        dataclass=dataclass,
        replace=replace,
        nullcontext=nullcontext,
        __name__=__name__,
        SelfAttention=TinySelfAttention,
        MegatronModule=torch.nn.Module,
        get_pg_size=lambda group: 1,
        _rotate_half=rotate_half,
        apply_rotary_pos_emb=rotate,
        parallel_state=SimpleNamespace(get_context_parallel_world_size=lambda: 1),
    )
    nodes = []
    classes = {
        "QSASelection",
        "QSAIndexer",
        "QSACoreAttention",
        "QwenSparseSelfAttention",
        "NGramEmbedding",
        "PerLayerEmbedding",
    }
    functions = {
        "_sequence_layout",
        "build_qsa_dense_mask",
        "ngram_segment_positions",
        "shift_tokens_in_segment",
        "gated_residual_group_rmsnorm",
    }
    for relative in (
        QSA_MODULE.replace(".", "/") + ".py",
        PLE_MODULE.replace(".", "/") + ".py",
        "megatron/core/transformer/hyper_connection.py",
    ):
        for node in native_source(relative).body:
            if isinstance(node, ast.ClassDef) and node.name in classes:
                node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != "__init__"]
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in functions:
                nodes.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_QSA_SELECT_TILE_BYTES" for t in node.targets
            ):
                nodes.append(node)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(PATCH), "exec"), namespace)
    for name in classes:
        namespace[name].__module__ = PLE_MODULE if name in {"NGramEmbedding", "PerLayerEmbedding"} else QSA_MODULE
    return SimpleNamespace(**namespace)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", SimpleNamespace(PackedSeqParams=Packed))
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.ssm.mamba_context_parallel",
        SimpleNamespace(
            reconstruct_tensor_cp=lambda value, *args, **kwargs: value,
            split_tensor_cp=lambda value, *args, **kwargs: value,
        ),
    )
    for name in (
        "QSA_FUSED_SELECTOR",
        "QSA_FUSED_ATTENTION",
        "QSA_CPU_LAYOUT",
        "QSA_FIXED_POOL",
        "QSA_TP_SELECTOR",
        "RETAIN_QSA_ATTENTION",
        "PLE_STREAMED",
    ):
        monkeypatch.setenv("RELAX_QWEN38_" + name, "0")
    return load_native()


def model(native, *, absolute=False):
    config = SimpleNamespace(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mrope_section=[1, 1] if absolute else None,
        qsa_indexer_loss_coeff=0.0,
        qsa_force_sparse=True,
        ple_layer_ids=[1],
    )
    attention = native.QwenSparseSelfAttention()
    attention.config, attention.layer_number = config, 1
    attention.pg_collection = SimpleNamespace(tp=None, cp=None)
    attention.projection = torch.nn.Linear(8, 16, bias=False)
    attention.output = torch.nn.Linear(8, 8, bias=False)
    attention.bias = torch.nn.Parameter(torch.randn(8) * 0.1)
    indexer = native.QSAIndexer()
    indexer.config, indexer.pg_collection = config, attention.pg_collection
    indexer.n_heads, indexer.kv_heads, indexer.head_dim = 2, 1, 4
    indexer.compress_ratio, indexer.block_topk, indexer.rotary_interleaved = 4, 2, False
    indexer.index_qk_proj = Projection(8, 12, bias=False)
    indexer.q_layernorm, indexer.k_layernorm = torch.nn.RMSNorm(4), torch.nn.RMSNorm(4)
    attention.indexer = indexer
    core = native.QSACoreAttention()
    core.config, core.pg_collection, core.softmax_scale = config, attention.pg_collection, 0.5
    core.sparse_backend = "dense_masked"
    attention.core_attention = core

    layer = native.PerLayerEmbedding()
    layer.config, layer.layer_number = config, 1
    layer.n, layer.hidden_size, layer.norm_eps = 2, 4, 1e-6
    layer.tp_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    layer._cp_packed_seq_params = None
    layer.conv_kernel_size, layer.conv_dilation, layer.halo = 3, 3, 6
    layer.conv1d = torch.nn.Conv1d(8, 8, 3, groups=8, bias=False)
    layer.key_proj, layer.value_proj = torch.nn.Linear(4, 8, bias=False), torch.nn.Linear(4, 4, bias=False)
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = torch.nn.Module()
        norm.weight = torch.nn.Parameter(torch.randn(8) * 0.1)
        setattr(layer, name, norm)
    embedding = native.NGramEmbedding()
    embedding.ngram_size, embedding.heads_per_ngram, embedding.eos_token_id = 3, 1, 0
    embedding.layer_multipliers = torch.tensor([3, 5, 7])
    embedding.ngram_heads_vocab_sizes, embedding.ngram_heads_offsets = torch.tensor([17, 19]), torch.tensor([0, 17])
    embedding.ngram_embedding = torch.nn.Embedding(36, 2)
    embedding.ngram_embedding.reduce_scatter_embeddings = False
    layer.ple_embedding = embedding

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention, self.per_layer_embedding, self.config = attention, layer, config

        def forward(self, hidden_states, input_ids, packed_seq_params, rotary_pos_emb):
            self.per_layer_embedding.prepare(input_ids, cu_seqlens=packed_seq_params.cu_seqlens_q)
            hidden_states = hidden_states + self.per_layer_embedding(hidden_states)
            out, bias = self.self_attention(
                hidden_states, None, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params
            )
            return out + bias

    return Model()


@pytest.mark.parametrize("absolute,keep,pp", [(False, 1, 1), (True, 1, 1), (True, 99, 1), (True, 99, 4)])
def test_native_qsa_ple_prefix_matches_full_outputs_and_gradients(native, absolute, keep, pp):
    torch.manual_seed(41)
    full = model(native, absolute=absolute)
    split = model(native, absolute=absolute)
    full.config.pipeline_model_parallel_size = split.config.pipeline_model_parallel_size = pp
    split.load_state_dict(full.state_dict())
    lengths, cu = [19, 11, 2, 3], [0, 19, 30, 32, 35, 128]
    tokens = torch.randint(1, 16, (1, 128))
    tokens[0, [4, 21, 33]] = 0  # EOS inside chunks and near a boundary.
    x = torch.randn(128, 1, 8, requires_grad=True)
    xs = x.detach().clone().requires_grad_()
    upstream = torch.randn(128, 1, 8)
    upstream[35:] = 0
    freqs = torch.arange(128).float().view(-1, 1, 1, 1).expand(-1, 1, 1, 4) * 0.03
    if absolute:
        positions = torch.cat([torch.arange(b - a) for a, b in zip(cu[:-1], cu[1:])])
        freqs = freqs[positions]
    expected = full(x, tokens, packed(cu), freqs)
    expected.backward(upstream)
    names = tuple(split.state_dict())
    assert install(split, 5) == 2
    assert install(split, 5) == 0
    assert tuple(split.state_dict()) == names
    chunks = build_chunks(lengths, 5)
    # Interleave documents while preserving each group's causal order.
    chunks.sort(key=lambda c: (c.group_idx, c.group if c.group is not None else 99))
    units = build_micro_batches(chunks)
    saved = {}
    runtime = getattr(split, RUNTIME_ATTR)
    for operation, uid in gen_unit_ops(units, chunks, keep):
        unit = units[uid]
        rows = torch.cat(
            [
                torch.arange(cu[s.seq_id] + s.prefix, cu[s.seq_id] + s.prefix + s.length)
                for c in unit.chunks
                for s in c.segments
            ]
        )
        if operation == "B":
            saved.pop(uid).backward(upstream[rows])
            for group in completed_groups(unit):
                runtime.release_group(group)
            continue
        segments = [s for c in unit.chunks for s in c.segments]
        current_cu = [0]
        for s in segments:
            current_cu.append(current_cu[-1] + s.length)
        metadata = packed(current_cu)
        setattr(metadata, UNIT_ATTR, unit)
        setattr(metadata, RECOMPUTE_ATTR, operation == "Fr")
        setattr(metadata, prefix.PARENT_CU_ATTR, tuple(cu))
        setattr(metadata, prefix.MAX_LENGTH_ATTR, 93)
        prefix_rows = torch.cat([torch.arange(cu[s.seq_id], cu[s.seq_id] + s.prefix + s.length) for s in segments])
        setattr(metadata, prefix.TOKENS_ATTR, tokens[:, prefix_rows])
        # Reproduce the old overloaded RoPE-offset field. Native replay must discard it.
        metadata.cu_seqlens_q_padded = torch.tensor([segments[0].prefix, segments[0].prefix + unit.num_tokens])
        with torch.no_grad() if operation == "Fp" else nullcontext():
            h = prefix.pad(xs[rows], 128)
            current_tokens = F.pad(tokens[:, rows], (0, 128 - unit.num_tokens))
            rope = prefix.pad(freqs[rows], 128) if absolute else freqs
            actual = split(h, current_tokens, metadata, rope)[: unit.num_tokens]
            torch.testing.assert_close(actual, expected.detach()[rows], rtol=3e-5, atol=3e-6)
        if operation != "Fp":
            saved[uid] = actual
    runtime.assert_empty("numerical test")
    torch.testing.assert_close(xs.grad, x.grad, rtol=2e-4, atol=3e-6)
    for (name, param), (_, reference) in zip(split.named_parameters(), full.named_parameters()):
        if reference.grad is None:
            assert param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, reference.grad, rtol=2e-4, atol=3e-5, msg=name)
    assert uninstall(split) == 2
    assert "forward" not in split.__dict__
    assert "_pool_keys" not in split.self_attention.indexer.__dict__
    torch.testing.assert_close(split(xs.detach(), tokens, packed(cu), freqs), expected.detach())


def test_selector_ties_keep_parent_geometry(native):
    """Changing TopK width can select a different tied block and change
    output."""
    attention = model(native).self_attention
    indexer = attention.indexer
    # 515 blocks triggers a different CPU TopK algorithm than four blocks.
    q = torch.zeros(2060, 2, 4)
    q[..., 0] = 2
    keys = torch.zeros(2060, 4)
    keys[:4, 0], keys[4:8, 0], keys[8:12, 0] = 1, 1, 3
    docs, pos = torch.zeros(2060, dtype=torch.int32), torch.arange(2060, dtype=torch.int32)
    pools, _, valid = indexer._pool_keys(keys, docs, pos, 2060)
    expected, _ = indexer._select_blocks(q, pools, valid, docs, pos)
    short_pools, _, short_valid = indexer._pool_keys(keys[:16], docs[:16], pos[:16], 16)
    naive, _ = indexer._select_blocks(q[:16], short_pools, short_valid, docs[:16], pos[:16])
    assert not torch.equal(naive[15], expected[15, : naive.shape[1]])

    # Drive the real wrapper, replacing only the native projections with an
    # input fixture that produces the tied scores above.
    indexer.q_layernorm = torch.nn.Identity()
    indexer.k_layernorm = torch.nn.Identity()
    with torch.no_grad():
        indexer.index_qk_proj.weight.zero_()
        indexer.index_qk_proj.weight[0, 0] = 2
        indexer.index_qk_proj.weight[4, 0] = 2
        indexer.index_qk_proj.weight[8, 1] = 1
    h = torch.zeros(16, 1, 8)
    h[:, :, 0] = 1
    h[:, 0, 1] = keys[:16, 0]
    unit = Unit(0, [Chunk(0, [Segment(0, 0, 16)], 0, 0)])
    metadata = packed([0, 16])
    setattr(metadata, UNIT_ATTR, unit)
    setattr(metadata, prefix.PARENT_CU_ATTR, (0, 2060))
    setattr(metadata, prefix.MAX_LENGTH_ATTR, 2060)
    seen = []
    handle = indexer.register_forward_hook(lambda _m, _i, selection: seen.append(selection))
    undo = qsa.install(attention, 16, ChunkflowRuntime(16))
    attention(h, None, packed_seq_params=metadata, rotary_pos_emb=torch.zeros(2060, 1, 1, 4))
    selected = seen[0].selected_bits.view(128, -1)
    assert torch.equal(selected[:16], expected[:16])
    assert seen[0].all_selected is False
    handle.remove()
    undo()


def test_prefix_rejects_unsupported_modes_before_installing(native, monkeypatch):
    current = model(native)
    current.self_attention.config.qsa_indexer_loss_coeff = 0.1
    with pytest.raises(NotImplementedError, match="Stage-2"):
        install(current, 5)
    assert not hasattr(current, RUNTIME_ATTR)
    assert "forward" not in current.self_attention.__dict__
    current.config.qsa_indexer_loss_coeff = 0
    current.config.context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="context_parallel_size"):
        install(current, 5)
    current.config.context_parallel_size = 1
    monkeypatch.setenv("RELAX_QWEN38_PLE_STREAMED", "1")
    with pytest.raises(NotImplementedError, match="PLE_STREAMED"):
        install(current, 5)
    assert "forward" not in current.self_attention.__dict__
    monkeypatch.delenv("RELAX_QWEN38_PLE_STREAMED")
    indexer = current.self_attention.indexer
    original_select = indexer._select_blocks
    monkeypatch.setattr(indexer, "_select_blocks", original_select)
    with pytest.raises(NotImplementedError, match="Patched QSA"):
        install(current, 5)
    assert indexer._select_blocks == original_select
    assert not hasattr(current, RUNTIME_ATTR)


@pytest.mark.parametrize(
    "granularity,modules,core_checkpoint,allowed",
    [
        ("selective", ["moe"], False, True),
        ("selective", ["moe", "core_attn"], False, False),
        ("full", ["moe"], False, False),
        ("selective", ["moe"], True, False),
    ],
)
def test_prefix_native_recompute_boundary(granularity, modules, core_checkpoint, allowed):
    module = SimpleNamespace(
        config=SimpleNamespace(recompute_granularity=granularity, recompute_modules=modules),
        checkpoint_core_attention=core_checkpoint,
    )
    if allowed:
        prefix.validate(module)
    else:
        with pytest.raises(NotImplementedError, match="selective MoE"):
            prefix.validate(module)
