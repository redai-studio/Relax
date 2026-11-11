# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Keye BF16 Hadamard requires CUDA")


def _attention():
    from relax.models.keye_vl2.attention import KeyeCoreAttention

    config = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        hidden_size=32,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
        keye_indexer_loss_coeff=0.01,
    )
    return KeyeCoreAttention(config, 1).cuda().bfloat16()


def _inputs(length: int):
    return (
        torch.randn(length, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(length, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    )


def test_keye_sparse_recompute_has_indexer_gradients_and_no_future_leak():
    torch.manual_seed(19)
    attention = _attention()
    q, k, v, hidden = _inputs(2113)
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, 2113], num_valid_tokens=2113)

    def forward(q, k, v, hidden):
        return attention(q, k, v, packed_seq_params=packed, hidden_states=hidden)

    actual = checkpoint(forward, q, k, v, hidden, use_reentrant=True)
    actual.float().square().mean().backward()
    assert hidden.grad is None  # The auxiliary objective must not modify the backbone input.
    for parameter in attention.sa_indexer.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0
    assert all(tensor.grad.isfinite().all() for tensor in (q, k, v))
    with torch.no_grad():
        k_changed, v_changed, hidden_changed = k.clone(), v.clone(), hidden.clone()
        k_changed[-1].add_(5)
        v_changed[-1].add_(5)
        hidden_changed[-1].add_(5)
        changed = forward(q, k_changed, v_changed, hidden_changed)
    torch.testing.assert_close(actual[:-1], changed[:-1], atol=0, rtol=0)


def test_keye_frozen_indexer_skips_auxiliary_and_preserves_attention(monkeypatch):
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler

    attention = _attention()
    attention.sa_indexer.requires_grad_(False)
    q, k, v, hidden = _inputs(2113)
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, 2113], num_valid_tokens=2113)
    before = {name: parameter.detach().clone() for name, parameter in attention.sa_indexer.named_parameters()}
    with torch.no_grad():
        expected = attention(q, k, v, packed_seq_params=packed, hidden_states=hidden)
    monkeypatch.setattr(
        DSAIndexerLossAutoScaler, "apply", lambda *args: pytest.fail("Frozen indexer ran auxiliary loss")
    )
    monkeypatch.setattr(F, "kl_div", lambda *args, **kwargs: pytest.fail("Frozen indexer computed teacher KL"))
    actual = checkpoint(
        lambda q, k, v, hidden: attention(q, k, v, packed_seq_params=packed, hidden_states=hidden),
        q,
        k,
        v,
        hidden,
        use_reentrant=True,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    assert hidden.grad is None
    assert all(tensor.grad.isfinite().all() and torch.count_nonzero(tensor.grad) > 0 for tensor in (q, k, v))
    for name, parameter in attention.sa_indexer.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_causal_prefix_matches_full_matrix_outputs_and_auxiliary_gradients():
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler

    from relax.models.keye_vl2.indexer import indexer_scores

    torch.manual_seed(57)
    length = 2176
    attention = _attention()
    reference = deepcopy(attention)
    inputs = _inputs(length)
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    valid = torch.ones(length, device="cuda", dtype=torch.bool)
    valid[-5:] = False
    valid[33] = False
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], keye_valid_mask=valid)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda") / length)
    actual = attention(*inputs[:3], hidden_states=inputs[3], packed_seq_params=packed)

    q, k, v, hidden = reference_inputs
    positions = torch.arange(length, device="cuda")
    iq, ik, iw = reference.sa_indexer(hidden[:, 0], positions)
    logits = indexer_scores(iq, ik, iw)
    causal = (positions[:, None] >= positions[None]) & valid[None]
    logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
    support = torch.zeros_like(causal).scatter_(1, logits.detach().topk(2048, -1).indices, True) & causal
    keys = k.repeat_interleave(8, 1).transpose(0, 1)
    values = v.repeat_interleave(8, 1).transpose(0, 1)
    query = q.transpose(0, 1)
    expected = F.scaled_dot_product_attention(query[None], keys[None], values[None], attn_mask=support[None, None])
    expected = expected[0].transpose(0, 1).reshape(length, -1)
    with torch.no_grad():
        teacher = (query.float() @ keys.float().transpose(-1, -2)) * 128**-0.5
        target = teacher.masked_fill(~support[None], float("-inf")).softmax(-1).mean(0).nan_to_num()
    log_probs = logits.float().masked_fill(~support, torch.finfo(torch.float32).min).log_softmax(-1)
    auxiliary = (F.kl_div(log_probs, target, reduction="none").sum(-1) * valid).sum()
    expected = DSAIndexerLossAutoScaler.apply(expected, auxiliary * 0.01)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    weight = torch.randn_like(actual) / length
    (actual * weight).float().sum().backward()
    (expected * weight).float().sum().backward()
    for x, rx in zip(inputs[:3], reference_inputs[:3]):
        torch.testing.assert_close(x.grad, rx.grad, atol=2e-5, rtol=0.08)
    assert inputs[3].grad is None
    for (_, parameter), (_, rp) in zip(attention.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, rp.grad, atol=2e-5, rtol=0.08)


def test_recorded_rows_skip_index_scores_but_missing_last_row_keeps_exact_fallback(monkeypatch):
    import relax.models.keye_vl2.attention as module
    from relax.distributed.replay_manager.backend.megatron import IndexerReplay
    from relax.models.keye_vl2.replay import capture_rows

    torch.manual_seed(97)
    length = 2113
    attention = _attention()
    attention.sa_indexer.requires_grad_(False)
    packed = SimpleNamespace(cu_seqlens_q_cpu=[0, length], num_valid_tokens=length)
    inputs = _inputs(length)
    captured = []
    topk = module.dsa_topk_indices

    def record(scores):
        selected = topk(scores)
        start = sum(len(chunk) for chunk in captured)
        positions = torch.arange(start, start + len(selected), device="cuda")
        captured.append(capture_rows(selected, selected <= positions[:, None]))
        return selected

    monkeypatch.setattr(module, "dsa_topk_indices", record)
    expected = attention(*inputs[:3], hidden_states=inputs[3], packed_seq_params=packed)
    choices = torch.cat(captured)
    choices[-1] = -1
    choices._relax_indexer_missing_rows = (length - 1,)
    valid = torch.arange(length, device="cuda") < length - 1
    monkeypatch.setattr(module, "dsa_topk_indices", topk)
    scored_rows = []
    scores = module.indexer_scores

    def count_scores(query, key, weights):
        scored_rows.append(len(query))
        return scores(query, key, weights)

    monkeypatch.setattr(module, "indexer_scores", count_scores)
    replay_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    IndexerReplay.begin_step([1])
    IndexerReplay.record(1, choices, valid)
    IndexerReplay.set_stage("replay_forward")
    attention.config.relax_use_indexer_replay = True
    try:
        actual = attention(*replay_inputs[:3], hidden_states=replay_inputs[3], packed_seq_params=packed)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        weight = torch.randn_like(actual) / length
        (expected * weight).float().sum().backward()
        (actual * weight).float().sum().backward()
        for reference, replayed in zip(inputs[:3], replay_inputs[:3]):
            torch.testing.assert_close(replayed.grad, reference.grad, atol=2e-5, rtol=0.08)
        assert scored_rows == [1]
        IndexerReplay.finish_step(expect_backward=False)
    finally:
        IndexerReplay.clear()
