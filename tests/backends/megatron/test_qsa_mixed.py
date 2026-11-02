# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate mixed QSA against FP32 attention and pinned image kernels."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _routes(lengths: tuple[int, ...], budget: int):
    from megatron.core.transformer.experimental_attention_variant import qsa

    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths]).int().unsqueeze(0)
    docs = torch.cat([torch.full((length,), doc, device="cuda") for doc, length in enumerate(lengths)]).int()
    length = sum(lengths)
    columns = max(lengths) // 4
    scores = torch.rand((length, columns), device="cuda")
    scores.masked_fill_(torch.arange(columns, device="cuda") >= (positions[0, :, None] + 1) // 4, -torch.inf)
    top = scores.topk(min(columns, budget), dim=-1)
    ids = top.indices.masked_fill(~torch.isfinite(top.values), -1).int()
    ids = torch.nn.functional.pad(ids, (0, budget - ids.shape[1]), value=-1)
    bounds = [0]
    for value in lengths:
        bounds.append(bounds[-1] + value)
    selection = qsa.QSASelection(
        doc_ids=docs.unsqueeze(0),
        positions=positions,
        selected_bits=None,
        bits_per_row=0,
        bits_per_row_t=torch.tensor(0, device="cuda"),
        compress_ratio=4,
        all_selected=False,
        selected_ids=ids,
    )
    selection._relax_qsa_mixed_bounds = tuple(bounds)
    return selection


def _reference(inputs: list[torch.Tensor], indices: torch.Tensor, upstream: torch.Tensor):
    """Query chunks bound memory; no production attention/mask implementation
    used."""
    q, k, v = inputs
    k = k[:, 0].float().detach().requires_grad_()
    v = v[:, 0].float().detach().requires_grad_()
    output = torch.empty_like(q, dtype=torch.float32)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = torch.zeros_like(k), torch.zeros_like(v)
    for start in range(0, q.shape[0], 32):
        end = min(start + 32, q.shape[0])
        query = q[start:end].float().detach().requires_grad_()
        selected = indices[start:end].long()
        valid = selected >= 0
        selected = selected.clamp_min(0)
        keys, values = k[selected], v[selected]
        scores = torch.einsum("chd,csd->chs", query, keys) * (256**-0.5)
        weights = torch.softmax(scores.masked_fill(~valid[:, None], -torch.inf), dim=-1)
        result = torch.einsum("chs,csd->chd", weights, values)
        gradients = torch.autograd.grad(result, (query, k, v), upstream[start:end].float())
        output[start:end], dq[start:end] = result.detach(), gradients[0]
        dk.add_(gradients[1])
        dv.add_(gradients[2])
    return output, dq, dk.unsqueeze(1), dv.unsqueeze(1)


def _metrics(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, reference = got.float().flatten(), expected.float().flatten()
    diff = actual - reference
    return {
        "relative_l2": (diff.norm() / reference.norm().clamp_min(1e-12)).item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(actual, reference, dim=0).item(),
    }


def _run_case(lengths: tuple[int, ...], budget: int) -> None:
    from relax.models.qwen3_8 import qsa_community as patch
    from relax.models.qwen3_8.ops.qsa_mixed_ops.routes import token_routes

    torch.manual_seed(20261013)
    selection = _routes(lengths, budget)
    length = sum(lengths)
    values = [torch.randn(length, heads, 256, dtype=torch.bfloat16, device="cuda") for heads in (12, 1, 1)]
    actual_inputs = [value.detach().requires_grad_() for value in values]
    upstream = torch.randn_like(values[0])
    actual = patch._community_id_sparse_forward(
        SimpleNamespace(softmax_scale=256**-0.5), *actual_inputs, selection, True
    ).reshape_as(values[0])
    gradients = torch.autograd.grad(actual, actual_inputs, upstream)
    indices = torch.nn.functional.pad(token_routes(selection, 1, length)[0], (0, 1), value=-1)
    bases = torch.cat(
        [
            torch.full((n,), base, device="cuda", dtype=torch.int32)
            for n, base in zip(lengths, selection._relax_qsa_mixed_bounds)
        ]
    )
    selected = selection.selected_ids
    complete = selected[..., None] * 4 + torch.arange(4, device="cuda") + bases[:, None, None]
    complete = complete.masked_fill(selected[..., None] < 0, -1).flatten(1)
    positions = selection.positions[0]
    tail = bases[:, None] + ((positions[:, None] + 1) // 4) * 4 + torch.arange(4, device="cuda")
    tail = tail.masked_fill(tail > (bases + positions)[:, None], -1)
    independent_indices = torch.cat((complete, tail), dim=1).int()
    assert torch.equal(indices, independent_indices), "selected blocks/document offsets/causal tail differ"
    indices = independent_indices
    indices._qsa_bounds = selection._relax_qsa_mixed_bounds
    expected = _reference(values, indices, upstream)
    report = {"lengths": lengths, "block_budget": budget, "fp32": {}, "reference": {}}
    for name, got, want in zip(("output", "dq", "dk", "dv"), (actual, *gradients), expected):
        assert torch.isfinite(got).all(), name
        metrics = _metrics(got, want)
        report["fp32"][name] = metrics
        assert metrics["relative_l2"] < 0.02, (name, metrics)
        assert metrics["cosine"] > 0.9997, (name, metrics)
    source = os.environ.get("QSA_MIXED_REFERENCE_RUNTIME")
    # Empty-document handling is an intentional fix to the source mixed dispatcher.
    if source and all(lengths):
        sys.path.insert(0, source)
        from qsa_mixed_tp2 import attention as source_attention

        source_inputs = [value.detach().clone().requires_grad_() for value in values]
        source_output = source_attention(*source_inputs, indices, 256**-0.5)
        source_gradients = torch.autograd.grad(source_output, source_inputs, upstream)
        for name, got, want in zip(
            ("output", "dq", "dk", "dv"), (actual, *gradients), (source_output, *source_gradients)
        ):
            metrics = _metrics(got, want)
            report["reference"][name] = metrics
            assert metrics["relative_l2"] < 0.002, (name, metrics)
    print("QSA_MIXED_PRECISION " + json.dumps(report), flush=True)
    destination = os.environ.get("QSA_MIXED_PRECISION_REPORT")
    if destination:
        with Path(destination).open("a") as stream:
            stream.write(json.dumps(report) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mixed QSA precision requires CUDA BF16")
@pytest.mark.parametrize(
    "lengths,budget",
    [
        ((1, 3, 4, 5), 8),
        ((35,), 8),
        ((36,), 8),
        ((0, 7, 259, 0, 116), 8),
        ((2051,), 512),
        ((2052,), 512),
        ((16384,), 8),
        ((16385,), 512),
        ((7, 2052, 16385), 512),
    ],
)
def test_qsa_mixed_output_and_gradients(lengths: tuple[int, ...], budget: int) -> None:
    environment = os.environ.copy()
    environment.update(
        RELAX_QWEN38_QSA_COMMUNITY_BACKEND="mixed",
        MCORE_QSA_SPARSE_BACKEND="id_sparse",
        RELAX_QWEN38_QSA_FUSED_ATTENTION="0",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    subprocess.run(
        [sys.executable, __file__, json.dumps(lengths), str(budget)], env=environment, check=True, timeout=600
    )


def _check_layout_binding() -> None:

    from relax.models.qwen3_8 import qsa_dispatch
    from relax.models.qwen3_8.ops.qsa_mixed_ops import forward

    original = qsa_dispatch.selector_forward
    selection = replace(_routes((7, 5), 8), all_selected=True)
    qsa_dispatch.selector_forward = lambda *_args, **_kwargs: selection
    try:
        indexer = SimpleNamespace(
            config=SimpleNamespace(context_parallel_size=1, qsa_indexer_loss_coeff=0), parameters=lambda: iter(())
        )
        packed = SimpleNamespace(qkv_format="thd", qsa_select_offsets_cpu=(0, 7, 12))
        bound = forward(indexer, None, None, packed, "ids")
        assert bound._relax_qsa_mixed_bounds == (0, 7, 12)
        assert not bound.all_selected, "All-short batches must run the mixed Flash backend"
        assert selection.all_selected, "Keep checkpoint-replay selections immutable"
        packed.qsa_select_offsets_cpu = (0, 7, 11)
        with pytest.raises(ValueError, match="producer"):
            forward(indexer, None, None, packed, "ids")
    finally:
        qsa_dispatch.selector_forward = original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mixed QSA adapter validation requires CUDA")
def test_qsa_mixed_packed_binding_preserves_dispatch() -> None:
    environment = os.environ.copy()
    environment.update(
        RELAX_QWEN38_QSA_COMMUNITY_BACKEND="mixed",
        MCORE_QSA_SPARSE_BACKEND="id_sparse",
        RELAX_QWEN38_QSA_FUSED_ATTENTION="0",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    subprocess.run([sys.executable, __file__, "--layout-binding"], env=environment, check=True, timeout=120)


if __name__ == "__main__":
    if sys.argv[1] == "--layout-binding":
        _check_layout_binding()
    else:
        _run_case(tuple(json.loads(sys.argv[1])), int(sys.argv[2]))
