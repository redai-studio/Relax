# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 decode requires CUDA")
@pytest.mark.parametrize("width", [73, 2113])
def test_batched_decode_matches_separate_requests_with_poisoned_padding(width):
    from relax.models.keye_vl2.indexer import indexer_scores
    from relax.models.keye_vl2.sglang.model import decode_attention

    torch.manual_seed(27)
    lengths = [width - 15, width - 7, width]
    q = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, width, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(3, 16, 64, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(3, width, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    expected = []
    for row, length in enumerate(lengths):
        scores = indexer_scores(iq[row : row + 1], ik[row, :length, None], weights[row : row + 1])
        support = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            1, scores.topk(min(2048, length), -1).indices, True
        )
        result = F.scaled_dot_product_attention(
            q[row : row + 1, :, None],
            k[row : row + 1, :length].repeat_interleave(4, 2).transpose(1, 2),
            v[row : row + 1, :length].repeat_interleave(4, 2).transpose(1, 2),
            attn_mask=support[:, None],
        )
        expected.append(result.flatten(1))
        k[row, length:] = float("nan")
        v[row, length:] = float("nan")
        ik[row, length:] = float("nan")
    actual = decode_attention(q, k, v, iq, ik, weights, torch.tensor(lengths, device="cuda"))
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye paged decode requires CUDA")
@pytest.mark.parametrize("dim", [64, 128])
def test_fused_rope_matches_text_and_image_axes(dim):
    from relax.models.keye_vl2.indexer import text_rope
    from relax.models.keye_vl2.sglang.ops import rope

    x = torch.randn(4, 16, dim, device="cuda", dtype=torch.bfloat16)
    for positions in [
        torch.tensor([0, 73, 8001, 32767], device="cuda"),
        torch.tensor([[0, 0, 0], [3, 7, 11], [71, 9, 0], [8001, 8002, 8003]], device="cuda"),
    ]:
        torch.testing.assert_close(rope(x, positions, 1e7), text_rope(x, positions, 1e7), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye paged decode requires CUDA")
def test_paged_decode_preserves_bf16_topk_and_replays_changed_lengths():
    from relax.models.keye_vl2.indexer import indexer_scores
    from relax.models.keye_vl2.sglang.model import decode_attention
    from relax.models.keye_vl2.sglang.ops import paged_decode, paged_scores

    torch.manual_seed(91)
    batch, width = 3, 2176
    q = torch.randn(batch, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * width, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(batch, 16, 64, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(batch * width, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, 16, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(batch * width, device="cuda").int().reshape(batch, width)
    requests = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([73, 2100, width], device="cuda", dtype=torch.int32)

    def run():
        return paged_decode(q, k, v, iq, ik, weights, table, requests, lengths, width)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for values in [[73, 2100, width], [width, 125, 2051]]:
        lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        requests.copy_(requests.roll(1))
        graph.replay()
        slots = table[requests.long()].long()
        scores = paged_scores(iq, ik, weights, table, requests, lengths, width)
        for row, n in enumerate(values):
            expected_scores = indexer_scores(iq[row : row + 1], ik[slots[row, :n], None], weights[row : row + 1])[0]
            torch.testing.assert_close(scores[row, :n], expected_scores, atol=0, rtol=0)
        expected = decode_attention(q, k[slots], v[slots], iq, ik[slots], weights, lengths)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SGLang model imports require the CUDA runtime")
@pytest.mark.parametrize("multimodal", [False, True])
def test_language_decode_ignores_unused_graph_mrope_buffer(monkeypatch, multimodal):
    from types import SimpleNamespace

    from relax.models.keye_vl2.sglang import model

    attention = model.KeyeAttention.__new__(model.KeyeAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_proj = lambda hidden: (hidden, None)
    attention.q_size, attention.kv_size = 256, 128
    attention.num_heads, attention.num_kv_heads, attention.head_dim = 2, 1, 128
    attention.q_norm, attention.k_norm = torch.nn.Identity(), torch.nn.Identity()
    attention.rope_theta = 1e7
    attention.keye_multimodal_enabled = multimodal
    positions = torch.tensor([83, 129, 261])
    batch = SimpleNamespace(mrope_positions=torch.zeros(3, 3, dtype=torch.int64))

    class CapturedPosition(Exception):
        pass

    def capture(x, actual_positions, theta):
        expected = batch.mrope_positions.transpose(0, 1) if multimodal else positions
        torch.testing.assert_close(actual_positions, expected)
        raise CapturedPosition

    monkeypatch.setattr(model, "rope", capture)
    with pytest.raises(CapturedPosition):
        attention(positions, torch.randn(3, 512), batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 alignment requires CUDA")
def test_train_rope_matches_rollout_rounding_at_long_positions():
    from types import SimpleNamespace

    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.models.keye_vl2.provider import KeyeModelProvider
    from relax.models.keye_vl2.sglang.ops import rope

    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, apply_rope_fusion=True)
    KeyeModelProvider.configure_runtime(config, SimpleNamespace(multimodal_keys=None))
    x = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 73, 8001, 32767], device="cuda")
    frequencies = 1 / (1e7 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128))
    angles = positions[:, None].float() * frequencies
    angles = torch.cat((angles, angles), -1)[:, None, None]
    actual = apply_rotary_pos_emb(x[:, None], angles, config, cp_group=SimpleNamespace(size=lambda: 1))[:, 0]
    torch.testing.assert_close(actual, rope(x, positions, 1e7), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 router alignment requires CUDA")
def test_router_preserves_expert_order_lost_by_bf16_logits():
    from relax.models.keye_vl2.sglang.model import KeyeRouter

    router = object.__new__(KeyeRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(torch.tensor([[1, 0.001], [1, 0.002]], device="cuda", dtype=torch.bfloat16))
    hidden = torch.ones(1, 2, device="cuda", dtype=torch.bfloat16)
    rounded = F.linear(hidden, router.weight)
    assert rounded[0, 0] == rounded[0, 1]
    logits, bias = router(hidden)
    assert logits.dtype == torch.float32
    assert bias is None
    assert logits.argmax(-1).tolist() == [1]
    with torch.no_grad():
        router.weight.copy_(router.weight.flip(0))
    updated, _ = router(hidden)
    assert updated.argmax(-1).tolist() == [0]
    # Captured inference must read new BF16 weights after online updates too.
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            captured, _ = router(hidden)
        router.weight.copy_(router.weight.flip(0))
        graph.replay()
    assert captured.argmax(-1).tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye router dot requires CUDA")
def test_router_dot_is_exact_across_token_counts_and_has_correct_gradients():
    from relax.models.keye_vl2.provider import KeyeTopKRouter

    router = object.__new__(KeyeTopKRouter)
    torch.nn.Module.__init__(router)
    torch.manual_seed(8192)
    router.weight = torch.nn.Parameter(torch.randn(128, 2048, device="cuda", dtype=torch.bfloat16) * 0.02)
    router.bias = None
    x = torch.randn(8192, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        whole = router.gating(x)
        for length in (1, 8, 127, 512, 2113):
            torch.testing.assert_close(router.gating(x[:length]), whole[:length], atol=0, rtol=0)
    small = x[:3].detach().requires_grad_()
    weight = router.weight.detach().clone().requires_grad_()
    reference = F.linear(small.double(), weight.double()).float()
    actual = router.gating(small)
    dy = torch.randn_like(actual)
    expected_dx, expected_dw = torch.autograd.grad(reference, (small, weight), dy)
    dx, dw = torch.autograd.grad(actual, (small, router.weight), dy)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(dx, expected_dx, atol=0, rtol=0)
    torch.testing.assert_close(dw, expected_dw, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye residual alignment requires CUDA")
def test_residual_norm_uses_rounded_bf16_sum():
    from sglang.srt.layers.layernorm import RMSNorm

    from relax.models.keye_vl2.norm import RMSNorm as StableRMSNorm
    from relax.models.keye_vl2.sglang.model import KeyeRMSNorm

    reference = StableRMSNorm(2048, eps=1e-6, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(2048, eps=1e-6).cuda().bfloat16()
    # Match model initialization, including the existing dispatch binding.
    original_weight = norm.weight
    KeyeRMSNorm.convert(norm)
    assert norm.weight is original_weight
    norm.weight.data.copy_(reference.weight)
    x = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    with torch.no_grad():
        expected_residual = x + residual
        expected = reference(expected_residual)
        actual, updated = norm(x, residual)
    torch.testing.assert_close(updated, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            captured, captured_residual = norm(x, residual)
        residual.neg_()
        graph.replay()
        torch.testing.assert_close(captured_residual, x + residual, atol=0, rtol=0)
        torch.testing.assert_close(captured, reference(x + residual), atol=0, rtol=0)


def test_routing_order_is_identical_with_and_without_gradients():
    from relax.models.keye_vl2.provider import _StableTopK

    selector = _StableTopK(None)
    logits = torch.tensor([[0.11, 0.9, -0.3, 0.25, 0.44, 0.12, 0.8, 0.33, 0.01]], requires_grad=True)
    with torch.no_grad():
        scores, ids = selector.get_replay_topk(logits, 8, None, None, None)
        reference = scores.softmax(-1)
    scores, actual_ids = selector.get_replay_topk(logits, 8, None, None, None)
    actual = scores.softmax(-1)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert torch.equal(ids, actual_ids)
    actual[0, 0].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0

    class Replay:
        def get_replay_topk(self, scores, topk, num_groups, group_topk, select):
            values, ids = select(scores, topk)
            return values.flip(-1), ids.flip(-1)

    delegated = _StableTopK(Replay())
    values, replay_ids = delegated.get_replay_topk(logits, 8, None, None, None)
    assert torch.equal(replay_ids, actual_ids.flip(-1))
    torch.testing.assert_close(values, scores.flip(-1), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye expert merge requires CUDA")
def test_expert_merge_matches_te_rounding_at_bf16_midpoint():
    from relax.models.keye_vl2.sglang.ops import merge_experts

    # The tiny last expert moves an exact BF16 midpoint. A tree reduction or
    # ascending accumulation rounds up; TE's descending accumulation rounds down.
    values = [
        -0.006317138671875,
        -0.01226806640625,
        -0.00323486328125,
        0.0361328125,
        0.005767822265625,
        -0.0032958984375,
        0.0069580078125,
        2.0954757928848267e-09,
    ]
    routed = torch.tensor(values, device="cuda", dtype=torch.bfloat16).reshape(1, 8, 1)
    ids = torch.tensor([[21, 33, 43, 46, 51, 67, 70, 127]], device="cuda", dtype=torch.int32)
    expected = torch.tensor([[0.023681640625]], device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(merge_experts(routed, ids), expected, atol=0, rtol=0)
    order = torch.tensor([3, 5, 0, 7, 1, 2, 6, 4], device="cuda")
    torch.testing.assert_close(merge_experts(routed[:, order], ids[:, order]), expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye routed experts require CUDA")
def test_experts_match_stepwise_bf16_weighting_before_down_projection(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    from relax.models.keye_vl2.sglang.ops import expert_forward

    monkeypatch.setattr(
        fused_moe_triton_config,
        "get_exec",
        lambda: SimpleNamespace(deterministic=SimpleNamespace(enable_deterministic_inference=False)),
    )
    torch.manual_seed(53)
    hidden = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    down = torch.randn(4, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    ids = torch.tensor([[0, 2], [1, 3], [0, 1]], device="cuda", dtype=torch.int32)
    probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.8, 0.2]], device="cuda")

    def reference():
        rows = []
        for row, experts in enumerate(((0, 2), (1, 3), (0, 1))):
            outputs = []
            for slot, expert in enumerate(experts):
                gate, up = F.linear(hidden[row], gate_up[expert]).chunk(2, -1)
                activated = (F.silu(gate) * up * probabilities[row, slot : slot + 1]).to(hidden.dtype)
                outputs.append(F.linear(activated, down[expert]))
            rows.append(torch.stack(outputs).float().sum(0).to(hidden.dtype))
        return torch.stack(rows)

    actual = expert_forward(hidden, gate_up, down, probabilities, ids)
    torch.testing.assert_close(actual, reference(), atol=2e-5, rtol=0.008)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = expert_forward(hidden, gate_up, down, probabilities, ids)
    probabilities.copy_(probabilities.flip(-1))
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=2e-5, rtol=0.008)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye sparse GQA requires CUDA")
def test_shared_sparse_forward_is_bitexact_across_batching_and_graph_padding():
    from relax.models.keye_vl2.sparse_attention import compact_indices, sparse_attention

    torch.manual_seed(73)
    q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2176, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Slot zero represents an unused, potentially uninitialized graph cache entry.
    k[0] = float("nan")
    v[0] = float("nan")
    selected = torch.stack([torch.randperm(2175, device="cuda")[:2048] + 1 for _ in range(3)])
    valid = torch.arange(2048, device="cuda")[None] < torch.tensor([73, 2048, 0], device="cuda")[:, None]
    slots, counts = compact_indices(selected, valid)

    def run():
        return sparse_attention(q, k, v, slots, counts)

    with torch.no_grad():
        whole = run()
        separate = torch.cat(
            [sparse_attention(q[i : i + 1], k, v, slots[i : i + 1], counts[i : i + 1]) for i in range(3)]
        )
        torch.testing.assert_close(whole, separate, atol=0, rtol=0)
        assert whole.isfinite().all()
        assert torch.count_nonzero(whole[2]) == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        q.neg_()
        counts[1] = 127
        graph.replay()
        torch.testing.assert_close(captured, run(), atol=0, rtol=0)
