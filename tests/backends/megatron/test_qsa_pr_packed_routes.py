# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Packed PR7553 route and per-block dK/dV ownership contracts on CPU."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def routes_module(monkeypatch):
    # Load the real adapter; only its optional Megatron type dependency is stubbed.
    name = "megatron.core.transformer.experimental_attention_variant"
    variant = ModuleType(name)
    variant.qsa = SimpleNamespace()
    monkeypatch.setitem(sys.modules, name, variant)
    spec = importlib.util.spec_from_file_location(
        "qsa_routes_under_test", ROOT / "relax/models/qwen3_8/qsa_community.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_for(lengths, compact):
    positions = torch.tensor([p for length in lengths for p in range(length)], dtype=torch.int32).view(1, -1)
    docs = torch.tensor([d for d, length in enumerate(lengths) for _ in range(length)], dtype=torch.int32)
    # Exercise sparse selection: one older complete block plus the causal tail.
    ids = torch.where(positions >= 3, (positions + 1) // 4 - 1, -1).reshape(-1, 1)
    prefix, starts = [0], []
    physical = 0
    for length in lengths:
        starts.extend(range(physical, physical + length - 3, 4))
        prefix.append(len(starts))
        physical += length
    return SimpleNamespace(
        positions=positions,
        doc_ids=docs.view(1, -1),
        selected_ids=ids,
        compress_ratio=4,
        compact_block_prefix=torch.tensor(prefix, dtype=torch.int32) if compact else None,
        # Real selectors may retain an upper-bound allocation with unused slots.
        compact_block_starts=torch.tensor(starts + [0] * 5, dtype=torch.int32) if compact else None,
    )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("lengths", [(1, 2, 3), (4, 4), (7, 9, 4, 1), (2052, 7)])
def test_packed_blocks_have_one_writer_and_document_local_routes(routes_module, lengths, compact):
    selection = selection_for(lengths, compact)
    length = sum(lengths)
    routes, counts, bases, ends = routes_module._pr_routes(selection, 1, length, True)
    owners = torch.zeros(length, dtype=torch.int32)
    for start, end in zip(bases.tolist(), ends.tolist()):
        assert 0 <= end - start <= 4
        owners[start:end] += 1
        if end > start:
            assert selection.doc_ids[0, start] == selection.doc_ids[0, end - 1]
    # The kernel writes even blocks with no selecting queries. A second writer
    # can overwrite a valid gradient with zero; empty descriptors must not write.
    torch.testing.assert_close(owners, torch.ones_like(owners))
    for row in range(length):
        position = int(selection.positions[0, row])
        document_start = row - position
        selected = int(selection.selected_ids[row, 0])
        expected = (
            list(range(document_start + selected * 4, document_start + selected * 4 + 4)) if selected >= 0 else []
        )
        expected += list(range(document_start + ((position + 1) // 4) * 4, row + 1))
        actual = []
        for block in routes[0, row, : counts[0, row]].tolist():
            actual.extend(range(int(bases[block]), min(int(ends[block]), row + 1)))
        assert actual == expected


@pytest.mark.parametrize("compact", [False, True])
def test_block_gradient_stores_match_independent_attention(routes_module, compact):
    """Emulate each kernel block's ordinary stores, including zero-query
    blocks."""
    selection = selection_for((7, 4, 2), compact)
    length = selection.positions.numel()
    routes, counts, bases, ends = routes_module._pr_routes(selection, 1, length, True)
    torch.manual_seed(42)
    q, k, v = [torch.randn(length, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    expected_mask = torch.zeros(length, length, dtype=torch.bool)
    actual_mask = torch.zeros_like(expected_mask)
    for row in range(length):
        pos = int(selection.positions[0, row])
        local = torch.arange(length) - (row - pos)
        selected = int(selection.selected_ids[row, 0])
        expected_mask[row] = (
            (local >= 0)
            & (local <= pos)
            & (((local // 4 == selected) & (selected >= 0)) | (local >= ((pos + 1) // 4) * 4))
        )
        for block in routes[0, row, : counts[0, row]].tolist():
            actual_mask[row, bases[block] : min(int(ends[block]), row + 1)] = True
    expected = (q @ k.T).masked_fill(~expected_mask, -torch.inf).softmax(-1) @ v
    actual = (q @ k.T).masked_fill(~actual_mask, -torch.inf).softmax(-1) @ v
    upstream = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream, retain_graph=True)
    dq, dk, dv = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
    stored_dk, stored_dv = torch.zeros_like(k), torch.zeros_like(v)
    # Each block owns its tokens and gathers only queries selecting that block.
    for block, (start, end) in enumerate(zip(bases.tolist(), ends.tolist())):
        selected_rows = ((routes[0] == block) & (torch.arange(routes.shape[-1]) < counts[0, :, None])).any(-1)
        partial = torch.autograd.grad(actual, (k, v), upstream * selected_rows[:, None], retain_graph=True)
        stored_dk[start:end], stored_dv[start:end] = partial[0][start:end], partial[1][start:end]
    torch.testing.assert_close(actual, expected)
    for got, want in zip((dq, stored_dk, stored_dv), expected_grads):
        torch.testing.assert_close(got, want)
    torch.testing.assert_close(stored_dk, dk)
    torch.testing.assert_close(stored_dv, dv)
