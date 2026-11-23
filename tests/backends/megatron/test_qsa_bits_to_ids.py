# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Lossless bitset conversion, including unaligned rows and reserved tail
slots."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from relax.models.qwen3_8.ops.qsa_mixed_ops.bits_to_ids import bits_to_ids
from tests.cpu.backends.megatron.test_qsa_flex_mask_replay import _load_functions
from tests.cpu.backends.megatron.test_qsa_mixed_checkpoint import ROOT, _load, checkpoint_modules  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Triton")
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_bits_to_ids_preserves_sparse_blocks_and_empty_rows(position_dtype):
    # Five-byte rows exercise byte strides that are not divisible by four.
    bits = torch.tensor(
        [[0, 0, 0, 0, 0], [3, 0, 0, 0, 0], [129, 0, 0, 0, 0], [2, 1, 0, 0, 0], [1, 0, 0, 128, 1]],
        dtype=torch.uint8,
        device="cuda",
    )
    positions = torch.tensor([0, 7, 31, 35, 131], dtype=position_dtype, device="cuda")
    expected = torch.tensor(
        [[-1, -1, -1, -1], [0, 1, -1, -1], [0, 7, -1, -1], [1, 8, -1, -1], [0, 31, 32, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(bits_to_ids(bits, positions, 4, 4), expected, rtol=0, atol=0)
    assert bits_to_ids(bits[:0], positions[:0], 4, 4).shape == (0, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Triton")
def test_preserved_bits_checkpoint_replays_each_microbatch(checkpoint_modules, monkeypatch):  # noqa: F811
    import relax.models.qwen3_8 as qwen38

    qsa, _, mixed = checkpoint_modules
    for name, value in {
        "FUSED_ATTENTION": "0",
        "FUSED_SELECTOR": "1",
        "RECOMPUTE_REPLAY": "1",
        "CPU_LAYOUT": "0",
        "TP_SELECTOR": "0",
        "MIXED_PRESERVE_BITS_SELECTOR": "1",
        "MIXED_FLASH_MAX_LEN": "0",
        "MIXED_FLEX_MAX_LEN": "0",
    }.items():
        monkeypatch.setenv("RELAX_QWEN38_QSA_" + name, value)
    replay = ModuleType("relax.models.qwen3_8.qsa_cp")
    monkeypatch.setitem(sys.modules, replay.__name__, replay)
    _load_functions("qsa_cp.py", {"_ReplayTape", "_cached", "_with_replay", "_indexer_forward"}, replay.__dict__)
    replay._active_tape = replay.ContextVar("test_preserved_bits_tape", default=None)
    replay.torch = torch
    dispatch = _load(monkeypatch, "relax.models.qwen3_8.qsa_dispatch", ROOT / "relax/models/qwen3_8/qsa_dispatch.py")
    monkeypatch.setattr(qwen38, "qsa_dispatch", dispatch, raising=False)
    replay.qsa_dispatch = dispatch
    _load(monkeypatch, "relax.models.qwen3_8.qsa_select", ROOT / "relax/models/qwen3_8/qsa_select.py")
    monkeypatch.setitem(
        sys.modules,
        "relax.backends.megatron.qsa_layout_metadata",
        SimpleNamespace(valid_physical_layout=lambda packed: None),
    )
    built, selected = [], [[], []]

    # Only the frozen indexer input is synthetic; dispatch, conversion, replay,
    # token routes, grouped attention and autograd execute production code.
    def native(hidden, rotary, packed, output_format):
        assert output_format == "bits"
        built.append(packed.microbatch)
        return packed.selection

    owner = SimpleNamespace(
        config=SimpleNamespace(context_parallel_size=1, qsa_indexer_loss_coeff=0),
        parameters=lambda: [],
        block_topk=1,
        _relax_native_forward=native,
    )
    batches = []
    torch.manual_seed(42)
    for microbatch, lengths in enumerate(((16,), (12, 4))):
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).int().unsqueeze(0)
        ids = torch.where(positions >= 3, 0, -1).int()
        if microbatch:
            ids = torch.where(positions >= 7, 1, ids)
        bits = torch.where(ids < 0, 0, 1 << ids.clamp_min(0)).to(torch.uint8)
        selection = qsa.QSASelection(
            doc_ids=torch.cat([torch.full((n,), i, device="cuda") for i, n in enumerate(lengths)]).unsqueeze(0),
            positions=positions,
            selected_bits=bits,
            bits_per_row=1,
            bits_per_row_t=torch.tensor(1, device="cuda"),
            compress_ratio=4,
            all_selected=False,
        )
        bounds = (0, 16) if microbatch == 0 else (0, 12, 16)
        packed = SimpleNamespace(
            qkv_format="thd", qsa_select_offsets_cpu=bounds, selection=selection, microbatch=microbatch
        )

        def forward(q, k, v, packed=packed, expected=ids.reshape(16, 1)):
            result = mixed.forward(owner, q, None, packed)
            torch.testing.assert_close(result.selected_ids, expected, rtol=0, atol=0)
            assert result._relax_qsa_mixed_bounds == packed.qsa_select_offsets_cpu
            selected[packed.microbatch].append(result.selected_ids)
            return mixed.attention_for_selection(q, k, v, result, 256**-0.5, True)

        inputs = [
            torch.randn(1, 16, heads, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for heads in (12, 1, 1)
        ]
        reference = forward(*inputs)
        upstream = torch.randn_like(reference)
        gradients = torch.autograd.grad(reference, inputs, upstream)
        actual_inputs = [x.detach().clone().requires_grad_() for x in inputs]
        actual = checkpoint(replay._with_replay(forward), *actual_inputs, use_reentrant=True)
        batches.append((actual, actual_inputs, reference, gradients, upstream))

    # Both forwards finish before backward; recompute the older microbatch last.
    for actual, inputs, reference, gradients, upstream in reversed(batches):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual.backward(upstream)
        for value, expected in zip(inputs, gradients):
            # Grouped KV gradients accumulate through FP32 atomics before BF16.
            torch.testing.assert_close(value.grad, expected, rtol=1e-2, atol=1e-3)
    assert built == [0, 0, 1, 1]  # No indexer execution during either replay.
    assert all(len(values) == 3 and values[1] is values[2] for values in selected)
    assert selected[0][1] is not selected[1][1]
    assert replay._active_tape.get() is None
