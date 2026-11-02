# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.backends.megatron.test_selective_mtp_loss import Head, ce, patch


@pytest.mark.parametrize("fused_ce", [False, True])
@pytest.mark.parametrize("all_excluded", [False, True])
@pytest.mark.parametrize("is_training", [False, True])
def test_acceptance_switch_preserves_loss_and_gradients(monkeypatch, fused_ce, all_excluded, is_training):
    from relax.backends.megatron import fused_linear_ce as fused

    torch.manual_seed(38)
    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", "1")
    monkeypatch.setattr(patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(patch, "_main_update_token_normalization_enabled", lambda: False)
    monkeypatch.setattr(patch, "_resolve_chunk_size", lambda: 3)
    mask = torch.zeros(1, 8) if all_excluded else torch.tensor([[0, 1, 1, 1, 0, 1, 1, 0]], dtype=torch.float)
    head = Head()
    head.tp_group = object()
    head.gradient_accumulation_fusion = False
    hidden = torch.randn(16, 1, 5, requires_grad=True)
    labels = torch.randint(11, (1, 8))
    counts, logits_flags, scaled, tracked = [], [], [], []

    def acceptance(logits, labels, mask, *_args):
        counts.append(True)
        return ((logits[:, 0].argmax(-1) == labels[0]) & mask[0].bool()).sum(), mask.bool().sum()

    def linear(hidden, weight, labels, group, **kwargs):
        logits_flags.append(kwargs["return_logits"])
        logits = F.linear(hidden, weight)
        losses = F.cross_entropy(logits[:, 0], labels[:, 0], reduction="none").unsqueeze(1)
        return losses, logits[:, 0].detach() if kwargs["return_logits"] else None

    def roll(tensor, **_kwargs):
        out = torch.zeros_like(tensor)
        out[:, :-1] = tensor[:, 1:]
        return out, out.sum()

    def scale(hidden, loss):
        scaled.append(loss)
        return hidden

    def track(loss, num_tokens, depth, config, correct, total, **_kwargs):
        tracked.append((loss.detach().clone(), num_tokens.detach().clone(), correct, total))

    monkeypatch.setattr(fused, "supported", lambda *_args: fused_ce)
    monkeypatch.setattr(fused, "linear_cross_entropy", linear)
    monkeypatch.setattr(patch, "_COMPUTE_MTP_ACCEPTANCE_COUNTS", acceptance)
    monkeypatch.setattr(patch, "roll_tensor", lambda tensors, **kwargs: [roll(t, **kwargs)[0] for t in tensors])
    monkeypatch.setattr(patch, "MTPLossAutoScaler", SimpleNamespace(apply=scale))
    monkeypatch.setattr(patch, "_save_mtp_loss_to_tracker", track)
    rows = roll(mask)[0][0].nonzero().flatten()
    params = SimpleNamespace(qkv_format="thd", relax_sft_selective_mtp_rows=(rows,), relax_sft_selective_full_rows=8)
    config = SimpleNamespace(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        mtp_detach_heads=False,
        mtp_detach_lm_head=False,
    )
    outputs = []
    for enabled in (None, "1", "0"):
        if enabled is None:
            monkeypatch.delenv("RELAX_SFT_MTP_ACCEPTANCE_METRICS", raising=False)
        else:
            monkeypatch.setenv("RELAX_SFT_MTP_ACCEPTANCE_METRICS", enabled)
        counts.clear()
        logits_flags.clear()
        scaled.clear()
        tracked.clear()
        patch.process_mtp_loss(
            hidden,
            labels,
            mask,
            head,
            None,
            None,
            is_training,
            ce,
            config,
            cp_group=SimpleNamespace(size=lambda: 1),
            packed_seq_params=params,
        )
        loss = sum(x.sum() for x in scaled)
        gradients = torch.autograd.grad(loss, (hidden, head.weight))
        outputs.append((loss.detach(), gradients, [(x[0], x[1]) for x in tracked]))
        expected_metrics = is_training and enabled != "0"
        assert bool(counts) == expected_metrics
        if fused_ce:
            assert logits_flags and all(flag == expected_metrics for flag in logits_flags)
        if not expected_metrics:
            assert all(x[2] is None and x[3] is None for x in tracked)
    for actual in outputs[1:]:
        torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Requires Blackwell CuTe runtime",
)
@pytest.mark.parametrize("rows,hidden_size,local_vocab", [(23, 64, 520), (23, 2560, 124160), (2048, 2560, 124160)])
def test_real_fused_ce_metrics_switch_preserves_loss_and_gradients(rows, hidden_size, local_vocab):
    from relax.backends.megatron.fused_linear_ce import linear_cross_entropy

    torch.manual_seed(39)
    hidden = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(local_vocab, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_()
    labels = torch.randint(local_vocab, (rows,), device="cuda")
    labels[3] = -100
    upstream = torch.rand(rows, device="cuda")
    upstream[::3] = 0
    outputs = []
    for enabled in (True, False):
        loss, logits = linear_cross_entropy(hidden, weight, labels, None, return_logits=enabled)
        assert (logits is not None) == enabled
        gradients = torch.autograd.grad((loss * upstream).sum(), (hidden, weight))
        outputs.append((loss.detach(), gradients))
        del logits
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
