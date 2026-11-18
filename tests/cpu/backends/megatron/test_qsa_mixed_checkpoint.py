# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise both pinned checkpoint copies with mixed document metadata on
CPU."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType as Mask


ROOT = Path(__file__).resolve().parents[4]
QSA_MODULE = "megatron.core.transformer.experimental_attention_variant.qsa"
MIXED = "relax.models.qwen3_8.ops.qsa_mixed_ops"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint_modules(monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "grouped")
    from megatron.core.transformer.experimental_attention_variant import qsa

    module = SimpleNamespace(
        QSASelection=qsa.QSASelection,
        forward=qsa.QSACoreAttention.forward,
        _relax_native_forward=qsa.QSACoreAttention._relax_native_forward,
        _checkpointed_attention_forward=qsa.QwenSparseSelfAttention._checkpointed_attention_forward,
    )
    metadata = _load(monkeypatch, MIXED + ".selection", ROOT / "relax/models/qwen3_8/ops/qsa_mixed_ops/selection.py")
    _load(monkeypatch, MIXED + ".geometry", ROOT / "relax/models/qwen3_8/ops/qsa_mixed_ops/geometry.py")
    mixed = _load(monkeypatch, MIXED, ROOT / "relax/models/qwen3_8/ops/qsa_mixed_ops/__init__.py")
    return module, metadata, mixed


@pytest.mark.parametrize("differentiable_indexer", [False, True])
def test_mixed_bounds_survive_checkpoint_forward_and_replay(checkpoint_modules, monkeypatch, differentiable_indexer):
    qsa, metadata, mixed = checkpoint_modules
    native = qsa.QSASelection(
        doc_ids=torch.tensor([[0, 0, 1, 1, 1]]),
        positions=torch.tensor([[0, 1, 0, 1, 2]]),
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0),
        compress_ratio=4,
        all_selected=False,
        selected_ids=torch.full((5, 1), -1, dtype=torch.int32),
        index_query=torch.randn(5, 2, requires_grad=True) if differentiable_indexer else None,
        compressed_key=torch.randn(2, 2, requires_grad=True) if differentiable_indexer else None,
    )
    selection = metadata.bind_document_bounds(native, (0, 2, 5))
    assert isinstance(selection, qsa.QSASelection)
    selection._relax_qsa_query_positions = torch.tensor([0, 1, 8, 9, 10])
    selection._relax_qsa_key_reorder = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7])
    calls, replay, seen = [], [], []

    def checkpoint(fn, distributed, *args):
        replay.append((fn, args))
        return fn(*args)

    monkeypatch.setitem(
        sys.modules, "megatron.core", SimpleNamespace(tensor_parallel=SimpleNamespace(checkpoint=checkpoint))
    )
    monkeypatch.setitem(
        sys.modules,
        MIXED + ".mixed",
        SimpleNamespace(attention=lambda q, k, v, ids, scale: (calls.append(ids._qsa_bounds), q)[1]),
    )
    monkeypatch.setitem(
        sys.modules,
        MIXED + ".routes",
        SimpleNamespace(token_routes=lambda selection, batch, length: torch.full((batch, length, 3), -1)),
    )

    def sparse(q, k, v, selected, is_thd):
        seen.append(selected)
        assert selected is not selection
        assert selected._relax_qsa_query_positions is selection._relax_qsa_query_positions
        assert selected._relax_qsa_key_reorder is selection._relax_qsa_key_reorder
        assert selected.index_query is native.index_query
        assert selected.compressed_key is native.compressed_key
        return mixed.attention_for_selection(q, k, v, selected, 256**-0.5, is_thd)

    core = SimpleNamespace(
        _selection=selection,
        training=True,
        config=SimpleNamespace(qsa_indexer_loss_coeff=float(differentiable_indexer), qsa_force_sparse=False),
        pg_collection=SimpleNamespace(cp=None),
        sparse_backend="id_sparse",
        _id_sparse_forward=sparse,
        _attach_indexer_loss=lambda out, *args: out,
    )
    core._relax_native_forward = qsa._relax_native_forward.__get__(core)
    owner = SimpleNamespace(
        core_attention=core,
        attn_mask_type=Mask.causal,
        _run_core_attention=lambda *args, **kwargs: qsa.forward(core, *args, **kwargs),
    )
    query = torch.randn(1, 5, 12, 256)
    kv = torch.randn(1, 5, 1, 256)
    out = qsa._checkpointed_attention_forward(
        owner, query, kv, kv, None, packed_seq_params=SimpleNamespace(qkv_format="thd")
    )
    # A newer microbatch replaces module state before old activation recompute.
    core._selection = metadata.bind_document_bounds(native, (0, 1, 5))
    fn, args = replay[0]
    recomputed = fn(*args)
    torch.testing.assert_close(out, query)
    torch.testing.assert_close(recomputed, query)
    assert calls == [(0, 2, 5), (0, 2, 5)]
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert not hasattr(native, "_relax_qsa_mixed_bounds")


def test_mixed_producer_returns_replace_safe_selection(checkpoint_modules, monkeypatch):
    qsa, metadata, mixed = checkpoint_modules
    original = qsa.QSASelection(
        doc_ids=torch.tensor([[0, 0, 1, 1, 1]]),
        positions=torch.tensor([[0, 1, 0, 1, 2]]),
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0),
        compress_ratio=4,
        all_selected=True,
        selected_ids=torch.full((5, 1), -1, dtype=torch.int32),
    )
    monkeypatch.setitem(
        sys.modules,
        "relax.backends.megatron.qsa_layout_metadata",
        SimpleNamespace(valid_physical_layout=lambda packed: None),
    )
    monkeypatch.setitem(
        sys.modules, "relax.models.qwen3_8.qsa_dispatch", SimpleNamespace(selector_forward=lambda *args: original)
    )
    monkeypatch.setitem(
        sys.modules, "relax.utils.env", SimpleNamespace(Envs=SimpleNamespace(RELAX_QWEN38_QSA_FUSED_ATTENTION=False))
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1, qsa_indexer_loss_coeff=0), parameters=lambda: []
    )
    packed = SimpleNamespace(qkv_format="thd", qsa_select_offsets_cpu=(0, 2, 5))
    selected = mixed.forward(owner, None, None, packed)
    assert isinstance(selected, metadata.MixedQSASelection)
    assert replace(replace(selected, index_query=None), compressed_key=None)._relax_qsa_mixed_bounds == (0, 2, 5)
    assert not selected.all_selected and original.all_selected
    assert selected.selected_ids is original.selected_ids
