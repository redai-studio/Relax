# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-CUDA-GPU regressions for DoSA pruning, capacity and FP32 scoring."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((9, 0), (10, 3)),
    reason="QSA DoSA requires an SM90 or SM103 GPU",
)


@pytest.fixture
def operators(monkeypatch):
    # Isolate optional TE model registration, not selector computation.
    package = ModuleType("relax.models")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "relax/models")]
    monkeypatch.setitem(sys.modules, "relax.models", package)
    existing = set(sys.modules)
    selector = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute.selector")
    dosa = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute.seed_bound_selector")
    # The selector may already have been imported by a different test module.
    selector._interval_config.cache_clear()
    yield selector.select_ids, dosa.dosa_select_ids
    selector._interval_config.cache_clear()
    for name in set(sys.modules) - existing:
        if name.startswith("relax.models."):
            sys.modules.pop(name, None)


def _reference(query, keys, visible, budget):
    # Independent scalar reference. For the small dyadic test inputs each
    # product and partial dot sum is exactly representable in FP32, making
    # expected scores independent of the GPU kernel's reduction layout.
    scores = torch.zeros((len(query), len(keys)), device=query.device)
    for head in range(4):
        dots = torch.zeros_like(scores)
        for d in range(128):
            dots += query[:, head, d].float()[:, None] * keys[:, d].float()[None]
        scores += dots.relu()
    scores *= 128**-0.5
    valid = torch.arange(len(keys), device=query.device)[None] < visible[:, None]
    ids = scores.masked_fill(~valid, -torch.inf).argsort(dim=-1, descending=True, stable=True)[:, :budget]
    return torch.where(valid.gather(1, ids), ids, -1).int()


def test_qsa_dosa_prunes_without_losing_winners(operators):
    select, dosa = operators
    torch.manual_seed(53)
    # Aligned heads have zero projection residual, so this case must actually
    # prune. Random-head tests alone can pass while always taking fallback.
    q = (torch.randint(-8, 9, (5, 1, 128), device="cuda").bfloat16() / 8).expand(-1, 4, -1).contiguous()
    k = torch.randint(-8, 9, (4353, 128), device="cuda").bfloat16() / 8
    visible = torch.tensor([0, 17, 511, 3000, 4353], device="cuda", dtype=torch.int32)
    stats = {}
    actual = dosa(q, k, visible, 512, row_chunk=3, stats=stats)
    expected = _reference(q, k, visible, 512)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(select(q, k, visible * 4 - 1, 512, "triton"), expected, atol=0, rtol=0)
    assert (stats["survivors"][3:] < visible[3:]).all()
    assert not stats["overflow"].any()


def test_qsa_dosa_overflow_and_zero_score_ties(operators):
    _, dosa = operators
    q = torch.zeros(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(4097, 128, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([1, 512, 4097], device="cuda", dtype=torch.int32)
    stats = {}
    actual = dosa(q, k, visible, 512, candidate_capacity=512, stats=stats)
    ids = torch.arange(512, device="cuda").expand(3, -1)
    expected = torch.where(ids < visible[:, None], ids, -1).int()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert stats["overflow"].tolist() == [False, False, True]
    torch.testing.assert_close(stats["refined"], visible)


def test_qsa_dosa_preserves_fp32_score_order(operators):
    select, dosa = operators
    q = torch.zeros(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    q[:, 0, :2] = 1
    k = torch.zeros(4097, 128, device="cuda", dtype=torch.bfloat16)
    k[:, 0] = 1
    k[-1, 1] = 2.0**-10  # Distinct in FP32, tied after BF16 score rounding.
    visible = torch.full((2,), len(k), device="cuda", dtype=torch.int32)
    stats = {}
    actual = dosa(q, k, visible, 512, stats=stats)
    expected = torch.cat((torch.tensor([4096], device="cuda"), torch.arange(511, device="cuda"))).int()
    torch.testing.assert_close(actual, expected.expand(2, -1), atol=0, rtol=0)
    torch.testing.assert_close(select(q, k, visible * 4 - 1, 512, "dosa"), actual, atol=0, rtol=0)


def test_qsa_dosa_random_cancellation_matches_radix(operators):
    select, dosa = operators
    torch.manual_seed(67)
    q = (torch.randint(-8, 9, (7, 4, 256), device="cuda").bfloat16() / 8)[:, :, ::2]
    q[1, 1] = -q[1, 0]
    k = (torch.randint(-8, 9, (8201, 256), device="cuda").bfloat16() / 8)[:, ::2]
    positions = torch.tensor([0, 3, 2047, 8191, 16383, 32767, 32803], device="cuda")
    visible = ((positions + 1) // 4).clamp(max=len(k)).int()
    actual = dosa(q, k, visible, 512, row_chunk=4)
    expected = _reference(q, k, visible, 512)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(select(q, k, positions, 512, "triton"), expected, atol=0, rtol=0)


def test_qsa_dosa_certificate_encloses_scores_and_unsafe_values(operators):
    import triton

    from relax.models.qwen3_8.ops.qsa_cute import score_bound_kernels as kernels

    torch.manual_seed(91)
    q = torch.randn(7, 4, 128, device="cuda", dtype=torch.bfloat16)
    q[1] = q[0, :1].expand(4, -1)
    q[2, 1] = -q[2, 0]
    q[3] *= 2.0**-28
    q[4] *= 2.0**12
    q[5, 0, 0] = 2.0**-40  # Outside the certified arithmetic domain.
    q[6] = 0
    k = torch.randn(257, 128, device="cuda", dtype=torch.bfloat16)
    k[-1, 0] = 2.0**20
    rows, blocks = len(q), len(k)
    anchor = torch.empty((rows, 128), device="cuda", dtype=torch.bfloat16)
    fold = torch.empty(rows, device="cuda")
    error = torch.empty_like(fold)
    norms = torch.empty(blocks, device="cuda")
    visible = torch.full((rows,), blocks, device="cuda", dtype=torch.int32)
    tau = torch.ones(rows, device="cuda")
    flags = torch.empty((rows, blocks), device="cuda", dtype=torch.bool)
    counts = torch.empty((rows, triton.cdiv(blocks, 128)), device="cuda", dtype=torch.int32)
    bounds = torch.empty((rows, blocks), device="cuda")
    launch = {"num_warps": 4, "enable_fp_fusion": False}
    kernels.prepare[(rows,)](q, anchor, fold, error, **launch)
    kernels.key_norm[(triton.cdiv(blocks, 32),)](k, norms, blocks, **launch)
    kernels.screen[(triton.cdiv(rows, 16), triton.cdiv(blocks, 128))](
        k,
        norms,
        anchor,
        fold,
        error,
        visible,
        tau,
        flags,
        counts,
        rows,
        blocks,
        BOUNDS=bounds,
        DUMP=True,
        **launch,
    )
    exact = torch.einsum("qhd,kd->qhk", q.double(), k.double()).relu().sum(1) * (128**-0.5)
    assert (bounds.double() >= exact).all()
    assert bounds[5].isinf().all()
    assert bounds[:, -1].isinf().all()
    assert flags[5].all() and flags[:, -1].all()


def test_qsa_dosa_short_input_uses_existing_selector(operators):
    select, dosa = operators
    q = torch.zeros(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(17, 128, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
    assert dosa(q, k, visible, 512) is None
    torch.testing.assert_close(
        select(q, k, visible * 4 - 1, 512, "dosa"), select(q, k, visible * 4 - 1, 512, "triton"), atol=0, rtol=0
    )


def test_qsa_dosa_512k_radix_accumulation_layout(operators):
    select, dosa = operators
    # Regression: 16-row dot lowering changes FP32 accumulation order.
    # At this fixed seed/shape, its selected order differed in 20 entries.
    # This protects radix compatibility; the smaller dyadic tests separately
    # verify the scoring definition with an independent reference.
    torch.manual_seed(20261120)
    q = torch.randn(128, 4, 128, device="cuda", dtype=torch.bfloat16)
    q = q[:, :1].expand(-1, 4, -1).contiguous()
    k = torch.randn(131072, 128, device="cuda", dtype=torch.bfloat16)
    visible = torch.full((len(q),), len(k), device="cuda", dtype=torch.int32)
    stats = {}
    actual = dosa(q, k, visible, 512, stats=stats)
    torch.testing.assert_close(actual, select(q, k, visible * 4 - 1, 512, "triton"), atol=0, rtol=0)
    assert (stats["survivors"] < visible // 4).all()
    assert not stats["overflow"].any()


@pytest.mark.parametrize("capacity", [256, 768], ids=["smaller-than-topk", "non-power-of-two"])
def test_qsa_dosa_small_topk_requires_compatible_capacity(operators, capacity):
    select, _ = operators
    from relax.models.qwen3_8.ops.qsa_cute.score_interval_selector import IntervalConfig, select_interval_ids

    query = torch.zeros(1, 4, 128, dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros(4097, 128, dtype=torch.bfloat16, device="cuda")
    visible = torch.tensor([17], dtype=torch.int32, device="cuda")
    config = IntervalConfig(capacity=capacity, prefix_threshold=True, small_topk=True)

    stats = {}
    result = select_interval_ids(query, keys, visible, 512, stats=stats, config=config)
    radix = select(query, keys, visible * 4 - 1, 512, impl="triton")
    torch.testing.assert_close(result, radix, atol=0, rtol=0)
    assert stats["refined"].tolist() == [17]
    assert stats["fallback"].tolist() == [0]


@pytest.mark.parametrize("boundary_only", [False, True], ids=["interval", "boundary"])
def test_qsa_dosa_interval_preserves_fp32_topk_membership(operators, monkeypatch, boundary_only):
    select, _ = operators
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import _has_cudnn_topk

    if not _has_cudnn_topk():
        pytest.skip("DoSA interval and boundary selection require cuDNN DSA indexer TopK")

    score_interval_selector = importlib.import_module("relax.models.qwen3_8.ops.qsa_cute.score_interval_selector")
    query = torch.zeros(1, 4, 128, dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros(8192, 128, dtype=torch.bfloat16, device="cuda")
    query[:, :, 0] = 1
    query[:, :, 1] = 2.0**-12
    keys[:511, 0] = 2
    keys[511:513, 0] = 1
    keys[512, 1] = 1
    positions = torch.tensor([len(keys) * 4 - 1], device="cuda")
    near_scores = torch.tensor([1.0, 1.0 + 2.0**-12], dtype=torch.float32) * (4 * 128**-0.5)
    assert near_scores[1] > near_scores[0]
    assert near_scores.bfloat16()[1] == near_scores.bfloat16()[0]

    config = (
        {"capacity": 256, "half_scores": True, "boundary_reuse_topk": True} if boundary_only else {"capacity": 1024}
    )
    monkeypatch.setenv("QWEN38_DOSA_VARIANT", "boundary" if boundary_only else "interval")
    monkeypatch.setenv("QWEN38_DOSA_INTERVAL_CONFIG", json.dumps(config))

    stats = {}
    select_interval_ids = score_interval_selector.select_interval_ids

    def record_stats(*args, **kwargs):
        assert kwargs["config"] == score_interval_selector.IntervalConfig(**config)
        assert kwargs["boundary_only"] is boundary_only
        return select_interval_ids(*args, **kwargs, stats=stats)

    monkeypatch.setattr(score_interval_selector, "select_interval_ids", record_stats)
    result = select(query, keys, positions, 512, impl="dosa-set" if boundary_only else "dosa")
    radix = select(query, keys, positions, 512, impl="triton")
    expected = torch.arange(512, dtype=torch.int32, device="cuda").unsqueeze(0)
    expected[0, -1] = 512
    torch.testing.assert_close(radix, expected, atol=0, rtol=0)
    torch.testing.assert_close(result.sort(dim=-1).values if boundary_only else result, radix, atol=0, rtol=0)
    assert stats["fallback"].tolist() == [0]
    assert stats["refined"].tolist() == ([2] if boundary_only else [513])
    if boundary_only:
        assert stats["certified"].tolist() == [511]


@pytest.mark.parametrize("threshold", ["histogram", "cudnn"])
def test_qsa_dosa_fp16_boundary_keeps_radix_membership_and_fallback(operators, threshold):
    select, _ = operators
    from relax.models.qwen3_8.ops.qsa_cute.score_interval_selector import IntervalConfig, select_interval_ids
    from relax.models.qwen3_8.ops.qsa_cute.triton_qsa_select import _has_cudnn_topk

    if threshold == "cudnn" and not _has_cudnn_topk():
        pytest.skip("FP16 DoSA boundary selection requires cuDNN DSA indexer TopK")

    query = torch.zeros(2, 4, 128, dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros(8192, 128, dtype=torch.bfloat16, device="cuda")
    query[0, :, 0] = query[1, :, 1] = 1
    keys[2048:2528, 0] = 4
    keys[1024:1120, 0] = 2
    keys[:, 1] = 1
    visible = torch.full((2,), len(keys), dtype=torch.int32, device="cuda")
    config = IntervalConfig(
        coarse_rows=128,
        chunk_rows=1024,
        refine_tile=64,
        capacity=256,
        half_scores=True,
        boundary_reuse_topk=threshold == "cudnn",
        boundary_bins=2048 if threshold == "histogram" else 0,
    )

    stats = {}
    result = select_interval_ids(query, keys, visible, 512, stats=stats, config=config, boundary_only=True)
    assert result is not None
    radix = select(query, keys, visible * 4 - 1, 512, impl="triton")
    torch.testing.assert_close(result.sort(dim=-1).values, radix.sort(dim=-1).values, atol=0, rtol=0)
    # The first row keeps 480 winners and selects the lowest 32 IDs at the tie.
    # The second row has 8192 tied scores and must use the dense fallback.
    assert stats["certified"].tolist() == [480, 0]
    assert stats["refined"].tolist() == [96, len(keys)]
    assert stats["fallback"].tolist() == [0, len(keys)]
