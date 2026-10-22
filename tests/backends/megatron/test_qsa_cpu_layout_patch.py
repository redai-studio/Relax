# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import sys
from copy import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3] / "relax/backends/megatron"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.setenv("RELAX_QWEN38_QSA_CPU_LAYOUT", "1")
    metadata = _load("layout_metadata_test", _ROOT / "qsa_layout_metadata.py")
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.qsa_layout_metadata", metadata)
    qsa = SimpleNamespace(
        _relax_native_sequence_layout=Mock(return_value="fallback"),
        QSAIndexer=SimpleNamespace(
            _relax_native_pool_keys=Mock(), _relax_native_select_blocks=Mock(), _relax_native_forward=Mock()
        ),
        _QSA_SELECT_TILE_BYTES=64 << 20,
    )
    package = ModuleType("megatron.core.transformer.experimental_attention_variant")
    package.qsa = qsa
    monkeypatch.setitem(sys.modules, package.__name__, package)
    patch = _load("cpu_layout_patch_test", _ROOT.parents[1] / "models/qwen3_8/qsa_cpu_layout.py")
    return metadata, patch


def _packed(metadata, lengths):
    boundaries = (0, *torch.tensor(lengths).cumsum(0).tolist())
    packed = SimpleNamespace(
        qkv_format="thd", cu_seqlens_q=torch.tensor(boundaries, dtype=torch.int32), cu_seqlens_q_padded=None
    )
    metadata.bind_physical_layout(packed, boundaries)
    return packed, boundaries


def test_binding_invalidates_mutation_and_replacement_and_rebinds_copy(modules):
    metadata, _ = modules
    packed, _ = _packed(metadata, (4, 8))
    assert metadata.get_cpu_layout(packed, 1, 12) == (2, 8, (0, 4, 12))
    record = metadata.valid_physical_layout(packed)
    mapped = copy(packed)
    mapped.cu_seqlens_q = packed.cu_seqlens_q.clone()
    assert metadata.get_cpu_layout(mapped, 1, 12) is None
    metadata.rebind_mapped_layout(mapped, record)
    assert metadata.get_cpu_layout(mapped, 1, 12) == (2, 8, (0, 4, 12))
    packed.cu_seqlens_q[1] = 5
    assert metadata.get_cpu_layout(packed, 1, 12) is None
    with torch.inference_mode():
        inference, _ = _packed(metadata, (4, 8))
    assert metadata.get_cpu_layout(inference, 1, 12) is None


@pytest.mark.parametrize("lengths", [(3,), (4, 8), (0, 9, 0, 17, 0), (128, 144)])
@pytest.mark.parametrize("topk", [1, 64])
def test_cpu_layout_pool_and_routes_match_reference_without_scalar_reads(modules, monkeypatch, lengths, topk):
    metadata, patch = modules
    packed, boundaries = _packed(metadata, lengths)
    model = SimpleNamespace(compress_ratio=4, block_topk=topk)
    patch._enabled.set(True)
    total = sum(lengths)
    # Nondegenerate data keeps TopK ordering deterministic across reference layouts.
    torch.manual_seed(25)
    raw = torch.randn(total, 4)
    queries = torch.randn(total, 4, 4)
    docs = torch.cat([torch.full((length,), i, dtype=torch.int32) for i, length in enumerate(lengths)])
    positions = torch.cat([torch.arange(length, dtype=torch.int32) for length in lengths])
    n_docs = max(i for i, length in enumerate(lengths) if length) + 1
    columns = max(lengths) // 4
    expected_pool = torch.zeros(n_docs, columns, 4)
    expected_valid = torch.zeros(n_docs, columns, dtype=torch.bool)
    for doc in range(n_docs):
        for block in range(lengths[doc] // 4):
            start = boundaries[doc] + block * 4
            expected_pool[doc, block] = raw[start : start + 4].mean(0)
            expected_valid[doc, block] = True
    expected_bits = torch.zeros(total, (columns + 8) // 8, dtype=torch.uint8)
    for row in range(total):
        doc = int(docs[row])
        visible = (int(positions[row]) + 1) // 4
        scores = (queries[row] @ expected_pool[doc].T).relu().sum(0) / 2
        scores[visible:] = -torch.inf
        for block in scores.topk(min(topk, columns)).indices.tolist():
            if block < visible:
                expected_bits[row, block // 8] |= 1 << (block % 8)
    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("unexpected scalar read")))
        guard.setattr(torch.Tensor, "tolist", Mock(side_effect=AssertionError("unexpected host transfer")))
        actual_docs, actual_positions, max_length = patch._sequence_layout(1, total, packed, "cpu")
        pooled, _, valid = patch._pool_keys(model, raw, actual_docs.flatten(), actual_positions.flatten(), max_length)
        bits, all_selected = patch._select_blocks(model, queries, pooled, valid, docs, positions)
    torch.testing.assert_close(actual_docs.flatten(), docs)
    torch.testing.assert_close(actual_positions.flatten(), positions)
    torch.testing.assert_close(pooled, expected_pool)
    torch.testing.assert_close(valid, expected_valid)
    torch.testing.assert_close(bits, expected_bits)
    assert all_selected == (max(lengths) // 4 <= topk)


def test_stage2_and_missing_metadata_fall_back_and_context_resets(modules):
    metadata, patch = modules
    packed, _ = _packed(metadata, (4, 8))
    model = SimpleNamespace(config=SimpleNamespace(qsa_indexer_loss_coeff=1))
    seen = []

    def forward(*args):
        seen.append(patch._enabled.get())
        patch._sequence_layout(1, 12, packed, "cpu")
        raise RuntimeError("forward failed")

    patch.qsa.QSAIndexer._relax_native_forward = forward
    with pytest.raises(RuntimeError, match="forward failed"):
        patch._indexer_forward(model, torch.empty(12, 1, 4), None, packed)
    assert seen == [False]
    assert patch._enabled.get() is False
    assert patch._layout.get() is None
    patch.qsa._relax_native_sequence_layout.assert_called_once()
    packed.cu_seqlens_q = packed.cu_seqlens_q.clone()
    patch._enabled.set(True)
    assert patch._sequence_layout(1, 12, packed, "cpu") == "fallback"
