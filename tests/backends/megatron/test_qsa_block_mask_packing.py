# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed Q masks must match PyTorch's transpose for Flex backward."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA Flex block-mask kernels")


@pytest.fixture
def operators(monkeypatch):
    # These standalone CUDA operators do not need Megatron model registration.
    root = Path(__file__).resolve().parents[3] / "relax/models/qwen3_8/ops/qsa_mixed_ops"
    modules = []
    for name in ("block_mask", "flex"):
        spec = importlib.util.spec_from_file_location(f"qsa_packing_test_{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    monkeypatch.setitem(sys.modules, "relax.models.qwen3_8.ops.qsa_mixed_ops.block_mask", modules[0])
    return modules


def assert_ordered(actual_counts, actual_indices, expected_counts, expected_indices):
    torch.testing.assert_close(actual_counts, expected_counts, rtol=0, atol=0)
    columns = torch.arange(actual_indices.shape[-1], device="cuda")
    valid = columns < actual_counts[..., None]
    torch.testing.assert_close(actual_indices[valid], expected_indices[valid], rtol=0, atol=0)
    assert (actual_indices[~valid] == 0).all()
    if actual_indices.shape[-1] > 1:
        assert ((actual_indices[..., 1:] > actual_indices[..., :-1]) | ~valid[..., 1:]).all()


@pytest.mark.parametrize(
    "pattern,bounds",
    [(pattern, (0, 2055, 2062, 2192)) for pattern in ("empty", "dense", "sparse")]
    + [("sparse", tuple(8192 * i for i in range(17)) + (131075,))],
)
def test_qsa_block_mask_direct_q_pack_matches_ordered_transpose(pattern, bounds, operators):
    masks, flex = operators
    starts = flex.layout(bounds, torch.device("cuda"))[2]
    maximum = max(b - a for a, b in zip(bounds, bounds[1:]))
    words = (maximum // 4 + 32) // 32
    value = {"empty": 0, "dense": -1, "sparse": 0x55555555}[pattern]
    bits = torch.full((bounds[-1], words), value, dtype=torch.int32, device="cuda")

    def mask(batch, head, query, key):
        return query >= key

    actual = masks.create(bits, starts, mask)
    expected = BlockMask.from_kv_blocks(
        actual.kv_num_blocks,
        actual.kv_indices,
        actual.full_kv_num_blocks,
        actual.full_kv_indices,
        BLOCK_SIZE=128,
        mask_mod=mask,
        seq_lengths=(bounds[-1], bounds[-1]),
    )
    assert_ordered(actual.q_num_blocks, actual.q_indices, expected.q_num_blocks, expected.q_indices)
    assert_ordered(
        actual.full_q_num_blocks, actual.full_q_indices, expected.full_q_num_blocks, expected.full_q_indices
    )


def test_qsa_direct_q_pack_preserves_real_flex_outputs_and_all_qkv_gradients(monkeypatch, operators):
    masks, flex = operators
    bounds = (0, 2055, 2062, 2192)
    total = bounds[-1]
    torch.manual_seed(278)
    values = [torch.randn((total, heads, 256), device="cuda", dtype=torch.bfloat16) for heads in (6, 1, 1)]
    indices = torch.full((total, 2052), -1, device="cuda", dtype=torch.int32)
    for first, last in zip(bounds, bounds[1:]):
        count = min(512, (last - first) // 4)
        chosen = torch.arange(count, device="cuda", dtype=torch.int32)
        tokens = (chosen[:, None] * 4 + torch.arange(4, device="cuda"))[None].flatten(1)
        indices[first:last, : count * 4] = tokens + first
    indices._qsa_bounds = bounds
    gradient = torch.randn_like(values[0])
    create = masks.create

    def old_create(bits, starts, mask):
        packed = create(bits, starts, mask)
        return BlockMask.from_kv_blocks(
            packed.kv_num_blocks,
            packed.kv_indices,
            packed.full_kv_num_blocks,
            packed.full_kv_indices,
            BLOCK_SIZE=128,
            mask_mod=mask,
            seq_lengths=packed.seq_lengths,
        )

    def run():
        inputs = [value.detach().requires_grad_(True) for value in values]
        output = flex.attention(*inputs, indices, 256**-0.5)
        gradients = torch.autograd.grad(output, inputs, gradient)
        return output.detach(), gradients

    with monkeypatch.context() as old:
        old.setattr(masks, "create", old_create)
        expected = run()
    actual = run()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for got, want in zip(actual[1], expected[1], strict=True):
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        torch.testing.assert_close(got, want, rtol=0, atol=0)
