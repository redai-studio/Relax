# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts and opt-in-on-CUDA numerical checks for the PR1765 adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py"
SPEC = importlib.util.spec_from_file_location("qsa_mixed_cudnn_under_test", SOURCE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.mark.parametrize("length", [0, 1, 512, 1024, 2047, 2048, 2049, 2051, 2052, 65536])
def test_document_length_dispatch(monkeypatch, length):
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    short, long = adapter.document_groups((0, length), length)
    assert bool(short) == (0 < length <= 2051)
    assert bool(long) == (length >= 2052)


@pytest.mark.parametrize("limit", [2047, 2051])
@pytest.mark.parametrize("length", [2047, 2048, 2051, 2052])
def test_explicit_flash_boundary_matches_baseline(monkeypatch, limit, length):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(limit))
    short, long = adapter.document_groups((0, length), length)
    assert bool(short) == (length <= limit)
    assert bool(long) == (length > limit)


@pytest.mark.parametrize("limit", [-2, 2052])
def test_dispatch_rejects_non_dense_flash_boundary(monkeypatch, limit):
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", str(limit))
    with pytest.raises(ValueError, match="dense selector budget"):
        adapter.document_groups((0, 2052), 2052)


@pytest.mark.parametrize("bounds,total", [((1, 4), 4), ((0, 4), 5), ((0, 3, 2), 2), ((0, True), 1)])
def test_dispatch_rejects_invalid_boundaries(bounds, total):
    with pytest.raises(ValueError, match="producer CPU boundaries"):
        adapter.document_groups(bounds, total)


def test_dispatch_packed_offsets_and_empty_documents(monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    short, long = adapter.document_groups((0, 0, 7, 2055, 4107, 4107, 4110), 4110)
    assert short == ((0, 7), (7, 2055), (4107, 4110))
    assert long == ((2055, 4107),)


@pytest.mark.parametrize("length", [1, 3, 4, 5, 7, 2047, 2048, 2051, 2052])
def test_metadata_matches_complete_blocks_and_partial_tail(length):
    ids = torch.full((length, 512), -1, dtype=torch.int32)
    position = torch.arange(length)
    # Duplicates, invalid sentinels, current partial and future IDs must not add
    # a complete future block. The partial current block is added separately.
    ids[:, 0] = 0
    ids[:, 1] = 0
    ids[:, 2] = position // 4
    ids[:, 3] = (position + 1) // 4 - 1
    ids[:, 4] = (length + 3) // 4 + 10
    selected, lo, hi, blk_base, tok_base, first, last = adapter.block_metadata(ids)
    block = torch.arange((length + 3) // 4)[None, :]
    visible = ((position + 1) // 4)[:, None]
    wanted = ((block == 0) | (block == visible - 1)) & (block < visible)
    wanted |= ((position[:, None] + 1) % 4 != 0) & (block == position[:, None] // 4)
    assert torch.equal(selected.bool(), wanted)
    assert selected.dtype == torch.uint8
    assert torch.equal(hi, position.int())
    assert torch.equal(first, torch.arange(selected.shape[1], dtype=torch.int32) * 4)
    assert torch.equal(last, (first + 3).clamp_max(length - 1))
    for zero in (lo, blk_base, tok_base):
        assert torch.count_nonzero(zero) == 0


def test_packed_metadata_preserves_odd_document_boundaries_and_causal_keys(monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    bounds = (0, 3, 8, 15)
    ids = torch.full((15, 512), -1, dtype=torch.int32)
    ids[:, :5] = torch.tensor([0, 0, 1, 100, -1], dtype=torch.int32)
    selected, lo, hi, blk_base, tok_base, first, last = adapter.packed_block_metadata(ids, bounds)
    assert first.tolist() == [0, 3, 7, 8, 12]
    assert last.tolist() == [2, 6, 7, 11, 14]
    for start, end in zip(bounds, bounds[1:]):
        for query in range(start, end):
            position = query - start
            assert int(lo[query]) == int(tok_base[query]) == start
            assert int(hi[query]) == query
            actual, expected = set(), set()
            for key in range(15):
                if int(lo[query]) <= key <= int(hi[query]):
                    block = int(blk_base[query]) + (key - int(tok_base[query])) // 4
                    if selected[query, block]:
                        actual.add(key)
            complete = {block for block in ids[query].tolist() if 0 <= block < (position + 1) // 4}
            for key in range(start, query + 1):
                block = (key - start) // 4
                if block in complete or (position + 1) % 4 and block == position // 4:
                    expected.add(key)
            assert actual == expected


@pytest.mark.parametrize("maximum", [0, 8192, 16384])
def test_long_packs_keep_large_documents_independent(maximum):
    spans = ((0, 2053), (2053, 4108), (4108, 24108), (24108, 26160))
    packs = adapter.long_document_packs(spans, maximum)
    assert tuple(span for pack in packs for span in pack) == spans
    assert all(len(pack) == 1 or sum(end - start for start, end in pack) <= maximum for pack in packs)
    assert ((4108, 24108),) in packs
    assert len(packs) == (4 if maximum == 0 else 3)


@pytest.mark.parametrize("maximum", [-1, 16385])
def test_long_packs_reject_dispatch_changing_limits(maximum):
    with pytest.raises(ValueError, match="between 0 and 16384"):
        adapter.long_document_packs(((0, 2053),), maximum)


def _selection(total):
    return SimpleNamespace(compress_ratio=4, selected_ids=torch.full((total, 512), -1, dtype=torch.int32))


def test_mixed_pack_restores_order_and_gradients(monkeypatch):
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    bounds = (0, 7, 2055, 4107, 4110)
    total = bounds[-1]
    q = torch.zeros((1, total, 12, 256), dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros((1, total, 1, 256), dtype=torch.bfloat16)
    calls = []

    def flash(query, key, value, lengths, scale):
        calls.append(("flash", lengths))
        return query + 10

    def nv(query, key, value, ids, scale):
        calls.append(("nv", query.shape[0]))
        return query + 20

    monkeypatch.setattr(adapter, "_flash", flash)
    monkeypatch.setattr(adapter._CudnnAttention, "apply", nv)
    out = adapter.attention_for_selection(q, k, k, _selection(total), 0.0625, bounds)
    expected = torch.full_like(q, 20)
    expected[:, :2055] = 10
    expected[:, 4107:] = 10
    assert torch.equal(out, expected)
    assert calls == [("flash", (7, 2048, 3)), ("nv", 2052)]
    out.float().sum().backward()
    assert torch.equal(q.grad, torch.ones_like(q))


@pytest.mark.parametrize("bounds", [(0, 7, 2060, 2060, 4115, 4118), (0, 2053, 2053, 4108)])
@pytest.mark.parametrize("maximum", [0, 8192])
def test_multiple_packed_documents_preserve_all_qkv_gradients(monkeypatch, bounds, maximum):
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    monkeypatch.setenv("RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS", str(maximum))
    total = bounds[-1]
    q, k, v = [torch.ones((2, total, heads, 256), dtype=torch.bfloat16, requires_grad=True) for heads in (12, 1, 1)]
    factor = torch.empty((1, total, 1, 1), dtype=torch.bfloat16)
    for start, end in zip(bounds, bounds[1:]):
        factor[:, start:end] = 1 if end - start <= 2051 else 2

    def flash(query, key, value, lengths, scale):
        return query + 2 * key + 3 * value

    def nv(query, key, value, ids, scale):
        return 2 * query + 4 * key + 6 * value

    def packed(query, key, value, ids, scale, packed_bounds):
        assert packed_bounds == (0, 2053, 4108)
        return nv(query, key, value, ids, scale)

    monkeypatch.setattr(adapter, "_flash", flash)
    monkeypatch.setattr(adapter._CudnnAttention, "apply", nv)
    monkeypatch.setattr(adapter._PackedCudnnAttention, "apply", packed)
    output = adapter.attention_for_selection(q, k, v, _selection(2 * total), 0.0625, bounds)
    assert torch.equal(output, (6 * factor).expand_as(output))
    upstream = (torch.arange(total) % 4).to(torch.bfloat16).view(1, total, 1, 1).expand_as(q)
    output.backward(upstream)
    assert torch.equal(q.grad, upstream * factor)
    assert torch.equal(k.grad, upstream.sum(dim=2, keepdim=True) * (2 * factor))
    assert torch.equal(v.grad, upstream.sum(dim=2, keepdim=True) * (3 * factor))


def test_reject_tp8_and_nonproduction_selector_budget():
    q = torch.zeros((1, 7, 3, 256), dtype=torch.bfloat16)
    k = torch.zeros((1, 7, 1, 256), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="TP2 local 12Q/1KV"):
        adapter.attention_for_selection(q, k, k, _selection(7), 0.0625, (0, 7))
    q = torch.zeros((1, 7, 12, 256), dtype=torch.bfloat16)
    selection = SimpleNamespace(compress_ratio=4, selected_ids=torch.zeros((7, 8), dtype=torch.int32))
    with pytest.raises(ValueError, match="production 2048-token selector budget"):
        adapter.attention_for_selection(q, k, k, selection, 0.0625, (0, 7))


def test_entry_routes_nv_before_legacy_expansion(monkeypatch):
    entry_path = SOURCE.with_name("__init__.py")
    spec = importlib.util.spec_from_file_location("qsa_mixed_entry_under_test", entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    name = "relax.models.qwen3_8.ops.qsa_mixed_ops.cudnn"
    monkeypatch.setitem(sys.modules, name, adapter)
    seen = []

    def mixed(query, key, value, selection, scale, bounds):
        seen.append(bounds)
        return query

    monkeypatch.setattr(adapter, "attention_for_selection", mixed)
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "cudnn_pr1765")
    q = torch.empty((1, 2048, 12, 256), device="meta")
    k = torch.empty((1, 2048, 1, 256), device="meta")
    selection = SimpleNamespace(compress_ratio=4, selected_ids=object(), _relax_qsa_mixed_bounds=(0, 2048))
    assert entry.attention_for_selection(q, k, k, selection, 0.0625, True) is q
    assert seen == [(0, 2048)]
    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_LONG_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unknown mixed QSA long backend"):
        entry.attention_for_selection(q, k, k, selection, 0.0625, True)


@pytest.mark.parametrize("packed", [False, True])
def test_autograd_saves_fp32_and_invocation_local_metadata(monkeypatch, packed):
    api = ModuleType("cudnn.QSA.api")
    saved_masks, backwards = [], []

    def forward(q, k, v, *args):
        saved_masks.append(args[0])
        output = q.float() + k.float() + v.float()
        return output.to(q.dtype), output, torch.zeros((q.shape[1], q.shape[0]))

    def backward(grad, q, k, v, out_fp32, lse, *args):
        assert out_fp32.dtype == torch.float32
        assert args[0] is saved_masks[len(backwards)]
        backwards.append(args[0])
        return grad, grad.sum(1, keepdim=True), grad.sum(1, keepdim=True)

    api.qsa_forward, api.qsa_backward = forward, backward
    monkeypatch.setitem(sys.modules, "cudnn", ModuleType("cudnn"))
    monkeypatch.setitem(sys.modules, "cudnn.QSA", ModuleType("cudnn.QSA"))
    monkeypatch.setitem(sys.modules, "cudnn.QSA.api", api)
    inputs, outputs = [], []
    for length in (7, 11):
        values = [torch.zeros((length, heads, 256), dtype=torch.bfloat16, requires_grad=True) for heads in (12, 1, 1)]
        ids = torch.full((length, 512), -1, dtype=torch.int32)
        if packed:
            outputs.append(adapter._PackedCudnnAttention.apply(*values, ids, 0.0625, (0, 3, length)))
        else:
            outputs.append(adapter._CudnnAttention.apply(*values, ids, 0.0625))
        inputs.append(values)
    # The second forward must not replace state belonging to the first graph.
    for output, (q, k, v) in zip(outputs, inputs):
        output.float().sum().backward()
        assert torch.equal(q.grad, torch.ones_like(q))
        assert torch.equal(k.grad, torch.full_like(k, 12))
        assert torch.equal(v.grad, torch.full_like(v, 12))
    assert len(backwards) == 2 and saved_masks[0] is not saved_masks[1]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PR1765 numerics require Blackwell CUDA and optional cuDNN QSA"
)
@pytest.mark.parametrize("lengths", [(2051,), (2052,), (4097,), (7, 2053, 5, 2052)])
def test_cuda_output_and_qkv_gradients_against_fp32_reference(monkeypatch, lengths):
    """Synthetic numerical regression, not a TP/PP integration or performance
    test."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("This integration gate targets Blackwell SM100/SM103")
    pytest.importorskip("cudnn.QSA.api", reason="Install the pinned PR1765 optional dependency")
    monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN", raising=False)
    monkeypatch.setenv("RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS", "0")
    torch.manual_seed(1234)
    total = sum(lengths)
    values = [torch.randn((1, total, h, 256), device="cuda", dtype=torch.bfloat16) / 4 for h in (12, 1, 1)]
    actual_inputs = [v.detach().requires_grad_() for v in values]
    references = [v.float().detach().requires_grad_() for v in values]
    bounds, selections, expected = [0], [], []
    for length in lengths:
        start = bounds[-1]
        bounds.append(start + length)
        position = torch.arange(length, device="cuda")
        complete = (position + 1) // 4
        block_count = (length + 3) // 4
        scores = torch.rand((length, block_count), device="cuda")
        scores.masked_fill_(torch.arange(block_count, device="cuda")[None, :] >= complete[:, None], -torch.inf)
        selected_scores, blocks = scores.topk(min(512, block_count), dim=-1, sorted=False)
        blocks = blocks.to(torch.int32).masked_fill(~torch.isfinite(selected_scores), -1)
        blocks = torch.nn.functional.pad(blocks, (0, 512 - blocks.shape[1]), value=-1)
        selections.append(blocks)
        # Independent token-level oracle: selected complete blocks plus only the
        # query's partial current block, bounded by the current document.
        selected = torch.zeros((length, block_count), device="cuda", dtype=torch.bool)
        rows = position[:, None].expand_as(blocks)
        valid = blocks >= 0
        selected[rows[valid], blocks[valid].long()] = True
        key = position[None, :]
        mask = selected[:, position // 4] | (((position[:, None] + 1) % 4 != 0) & (key // 4 == position[:, None] // 4))
        mask &= key <= position[:, None]
        if length > 4096:
            # An always-dense or first-512-block implementation must not pass
            # merely because only one boundary row differs from Flash.
            assert selected[-1, 512:].any() and not selected[-1, :512].all()
            assert mask[-1, :-1].sum() == 2048
        q, k, v = [x[:, start : bounds[-1]].transpose(1, 2) for x in references]
        with sdpa_kernel(SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0, scale=256**-0.5, enable_gqa=True
            )
        expected.append(out.transpose(1, 2))
    selection = SimpleNamespace(compress_ratio=4, selected_ids=torch.cat(selections))
    actual = adapter.attention_for_selection(*actual_inputs, selection, 256**-0.5, tuple(bounds))
    expected = torch.cat(expected, dim=1)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)
    expected_grads = torch.autograd.grad(expected, references, upstream.float())
    for got, want in zip((actual, *actual_grads), (expected, *expected_grads)):
        assert torch.isfinite(got).all()
        error = torch.linalg.vector_norm(got.float() - want) / torch.linalg.vector_norm(want).clamp_min(1e-12)
        assert error < 0.02
