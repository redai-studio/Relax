# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU regression for the selected-ID QSA static dispatcher and its two
backends."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def _selection(lengths: tuple[int, ...], batch: int, budget: int, compact: bool):
    from megatron.core.transformer.experimental_attention_variant import qsa

    sequence_length = sum(lengths)
    docs = torch.tensor([doc for doc, length in enumerate(lengths) for _ in range(length)], device="cuda")
    positions = torch.tensor([pos for length in lengths for pos in range(length)], device="cuda")
    docs = docs.to(torch.int32).expand(batch, -1).contiguous()
    positions = positions.to(torch.int32).expand(batch, -1).contiguous()
    max_blocks = max(lengths) // 4
    torch.manual_seed(20260929)
    scores = torch.rand(batch, sequence_length, max_blocks, device="cuda")
    visible = (positions + 1) // 4
    scores.masked_fill_(torch.arange(max_blocks, device="cuda") >= visible[..., None], -torch.inf)
    top = scores.topk(min(max_blocks, budget), dim=-1)
    ids = top.indices.masked_fill(~torch.isfinite(top.values), -1).to(torch.int32)
    if ids.shape[-1] < budget:
        ids = torch.nn.functional.pad(ids, (0, budget - ids.shape[-1]), value=-1)

    prefix = None
    starts = None
    if compact:
        prefix_values = [0]
        start_values = []
        physical_start = 0
        for length in lengths:
            start_values.extend(physical_start + block * 4 for block in range(length // 4))
            prefix_values.append(len(start_values))
            physical_start += length
        prefix = torch.tensor(prefix_values, device="cuda", dtype=torch.int32)
        starts = torch.tensor(start_values, device="cuda", dtype=torch.int32)

    return qsa.QSASelection(
        doc_ids=docs,
        positions=positions,
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0, device="cuda"),
        compress_ratio=4,
        all_selected=False,
        selected_ids=ids.reshape(batch * sequence_length, budget).contiguous(),
        compact_block_prefix=prefix,
        compact_block_starts=starts,
    )


def _reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, selection, scale: float):
    """Independent FP32 attention over selected complete blocks and causal
    tail."""
    sequence_length, batch, heads, width = query.shape
    kv_heads = key.shape[2]
    ids = selection.selected_ids.view(batch, sequence_length, -1)
    outputs = []
    for row in range(batch):
        positions = selection.positions[row].long()
        docs = selection.doc_ids[row]
        blocks = sequence_length // 4 + 1
        present = torch.zeros(sequence_length, blocks, device=query.device, dtype=torch.int32)
        present.scatter_add_(1, ids[row].clamp_min(0).long(), (ids[row] >= 0).int())
        selected = present.gather(1, (positions[None, :] // 4).expand(sequence_length, -1)) > 0
        qp, kp = positions[:, None], positions[None, :]
        tail = kp >= ((qp + 1) // 4) * 4
        mask = (docs[:, None] == docs[None, :]) & (kp <= qp) & (tail | selected)

        q = query[:, row].float().transpose(0, 1)
        k = key[:, row].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        v = value[:, row].float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * scale
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        outputs.append((weights @ v).transpose(0, 1))
    return torch.stack(outputs, dim=1).reshape(sequence_length, batch, heads * width)


def _run_case(lengths: tuple[int, ...], batch: int, heads: int, kv_heads: int, budget: int, compact: bool):
    from megatron.core.transformer.experimental_attention_variant import qsa

    is_thd = len(lengths) > 1
    sequence_length = sum(lengths)
    selection = _selection(lengths, batch, budget, compact)
    torch.manual_seed(1000 + sequence_length + heads)
    values = [
        torch.randn(sequence_length, batch, count, 256, device="cuda", dtype=torch.bfloat16)
        for count in (heads, kv_heads, kv_heads)
    ]
    upstream = torch.randn(sequence_length, batch, heads * 256, device="cuda", dtype=torch.bfloat16)
    actual_inputs = [value.detach().clone().requires_grad_() for value in values]
    actual_args = [value.squeeze(1) if is_thd else value for value in actual_inputs]
    module = SimpleNamespace(softmax_scale=256**-0.5)
    module._relax_native_id_sparse_forward = qsa.QSACoreAttention._relax_native_id_sparse_forward.__get__(module)
    actual = qsa.QSACoreAttention._id_sparse_forward(module, *actual_args, selection, is_thd)
    actual = actual.unsqueeze(1) if is_thd else actual
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)

    reference_inputs = [value.detach().float().requires_grad_() for value in values]
    expected = _reference(*reference_inputs, selection, 256**-0.5)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream.float())
    for name, got, want in zip(("output", "dq", "dk", "dv"), (actual, *actual_grads), (expected, *expected_grads)):
        difference = (got.float() - want).abs()
        relative_l2 = (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(want).clamp_min(1e-8)).item()
        assert torch.isfinite(got).all(), name
        assert relative_l2 < 0.02, (name, lengths, relative_l2)
        torch.testing.assert_close(got.float(), want, atol=0.03, rtol=0.02, msg=lambda message: f"{name}: {message}")


def _worker(backend: str) -> None:
    os.environ["RELAX_QWEN38_QSA_COMMUNITY_BACKEND"] = backend
    os.environ["MCORE_QSA_SPARSE_BACKEND"] = "id_sparse"
    cases = (
        ((257,), 1, 3, 1, 8, False),
        ((259,), 2, 24, 2, 8, False),
        ((7, 129, 4, 116), 1, 3, 1, 8, False),
        ((7, 129, 4, 116), 1, 3, 1, 8, True),
        ((2307,), 1, 3, 1, 512, False),
    )
    for case in cases:
        _run_case(*case)
        print(f"PASS {backend} {case}", flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Community QSA kernels require CUDA")
@pytest.mark.parametrize("backend", ("pr7553", "tilelang7060"))
def test_qsa_community_patch_matches_fp32_reference(backend: str) -> None:
    environment = os.environ.copy()
    environment["RELAX_QWEN38_QSA_COMMUNITY_BACKEND"] = backend
    environment["MCORE_QSA_SPARSE_BACKEND"] = "id_sparse"
    subprocess.run([sys.executable, __file__, backend], env=environment, check=True, timeout=600)


if __name__ == "__main__":
    _worker(sys.argv[1])
