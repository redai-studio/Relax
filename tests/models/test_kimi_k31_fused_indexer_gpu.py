# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Blackwell fused scoring, partition merge, mask and tie regressions."""

import importlib.util
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a Blackwell CUDA GPU")


@pytest.fixture(scope="module")
def fused():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Fused K3.1 indexer is validated on Blackwell")
    path = Path(__file__).resolve().parents[2] / "relax/models/kimi_k3/dsa_fused_indexer.py"
    spec = importlib.util.spec_from_file_location("_fused_k31_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_indexer


@pytest.mark.parametrize("keys,zero_weights", [(4096, False), (8193, False), (131072, False), (8193, True)])
def test_fused_indexer_preserves_topk_scores_and_valid_support(fused, keys, zero_weights):
    from megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_fwd import indexer_fwd_interface

    torch.manual_seed(421)
    rows = 32
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn(keys, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.float32) * 0.1
    if zero_weights:
        weights.zero_()
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = ((torch.arange(rows, device="cuda") + 1) * keys // rows).int()
    starts[rows // 2 :] = keys // 2
    starts[0] = 256
    ends[0] = 256  # Empty row; candidates must all be invalid.
    logits = indexer_fwd_interface(q, k, weights, starts, ends)
    expected = logits.topk(2048, dim=-1, sorted=True).values
    actual, indices = fused(q, k, weights, starts, ends)
    valid = indices >= 0
    assert (((indices >= starts[:, None]) & (indices < ends[:, None])) | ~valid).all()
    ordered = indices.sort(dim=-1).values
    assert ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] < 0)).all()
    torch.testing.assert_close(actual.sort(dim=-1, descending=True).values, expected, rtol=5e-5, atol=2e-5)
    gathered = logits.gather(1, indices.clamp_min(0).long()).masked_fill(~valid, -torch.inf)
    torch.testing.assert_close(gathered, actual, rtol=5e-5, atol=2e-5)
