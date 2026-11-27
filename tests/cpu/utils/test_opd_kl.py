# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for OPD KL computation, masking, and advantage clipping."""

from argparse import Namespace

import pytest
import torch

from relax.utils.opd.opd_utils import (
    apply_opd_to_advantages,
    compute_opd_kl,
    compute_opd_kl_topk,
    compute_policy_opd_loss,
)


def _exact_tail_distribution(lp: torch.Tensor) -> torch.Tensor:
    probs = lp.exp()
    tail = (1.0 - probs.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    return torch.cat([probs, tail], dim=-1)


@pytest.mark.parametrize("kl_type", ["reverse_kl", "forward_kl", "jsd"])
def test_opd_kl_tail_mode_with_mask_preserves_tail_bin_and_gradients(kl_type: str) -> None:
    """Masked union rows in ``norm_mode='tail'`` must keep the tail bin.

    Without the tail bin, KL can become negative (violating Gibbs' inequality)
    and student gradients point in the wrong direction.
    """
    s_raw = torch.tensor([[-1.0, -2.0, -9.0], [-0.5, -8.0, -9.0]], dtype=torch.float32)
    t_raw = torch.tensor([[-1.2, -1.8, -9.0], [-0.7, -8.0, -9.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    s_masked = s_raw.clone().requires_grad_(True)
    actual = compute_opd_kl_topk(s_masked, t_raw, kl_type=kl_type, jsd_alpha=0.3, norm_mode="tail", mask=mask)
    actual.sum().backward()

    # Exact per-row reference on the unpadded valid prefix with tail bin
    expected_vals = []
    expected_grads = torch.zeros_like(s_raw)
    for row in range(s_raw.size(0)):
        k_valid = int(mask[row].sum().item())
        s_row = s_raw[row, :k_valid].clone().requires_grad_(True)
        t_row = t_raw[row, :k_valid]
        p_s = _exact_tail_distribution(s_row)
        p_t = _exact_tail_distribution(t_row)
        if kl_type == "reverse_kl":
            ref = (p_s * (p_s.log() - p_t.log())).sum()
        elif kl_type == "forward_kl":
            ref = (p_t * (p_t.log() - p_s.log())).sum()
        else:
            alpha = 0.3
            p_m = (1.0 - alpha) * p_s + alpha * p_t
            ref = (1.0 - alpha) * (p_s * (p_s.log() - p_m.log())).sum() + alpha * (p_t * (p_t.log() - p_m.log())).sum()
        ref.backward()
        expected_vals.append(ref.detach())
        expected_grads[row, :k_valid] = s_row.grad

    expected = torch.stack(expected_vals)
    assert (actual >= -1e-6).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_masked.grad, expected_grads, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kl_type", ["reverse_kl", "forward_kl", "jsd"])
def test_opd_kl_tail_all_true_mask_matches_unmasked(kl_type: str) -> None:
    s = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=torch.float32)
    t = torch.tensor([[-1.2, -1.8, -2.5], [-0.7, -1.3, -2.0]], dtype=torch.float32)
    all_true = torch.ones_like(s, dtype=torch.bool)

    unmasked = compute_opd_kl_topk(s, t, kl_type=kl_type, jsd_alpha=0.5, norm_mode="tail", mask=None)
    masked = compute_opd_kl_topk(s, t, kl_type=kl_type, jsd_alpha=0.5, norm_mode="tail", mask=all_true)
    torch.testing.assert_close(masked, unmasked, atol=1e-6, rtol=1e-6)


def test_opd_kl_topk_low_var_kl_returns_per_token_vector() -> None:
    """``compute_opd_kl_topk`` with ``low_var_kl`` must return shape ``[R]``, not a scalar."""
    s = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], dtype=torch.float32, requires_grad=True)
    t = torch.tensor([[-1.2, -1.8, -2.5], [-0.7, -1.3, -2.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    actual = compute_opd_kl_topk(s, t, kl_type="low_var_kl", mask=mask)
    assert actual.shape == (2,)

    log_ratio = s.detach() - t
    per_elem = torch.exp(-log_ratio) - 1.0 + log_ratio
    expected = per_elem.masked_fill(~mask, 0.0).sum(dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    # Also verify end-to-end compatibility with compute_policy_opd_loss (cat along dim=0).
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="low_var_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_log_prob_min_clamp=None,
        opd_token_selection="union",
        opd_fkl_entropy_gate=False,
        opd_per_token_clip=None,
        opd_is_clip=None,
    )
    batch = {
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "opd_topk_teacher_log_probs": [t],
        "opd_topk_ksz": [torch.tensor([2, 3])],
    }
    loss, _ = compute_policy_opd_loss(
        args=args,
        batch=batch,
        metric_reducer=lambda v: v.mean(),
        log_probs=torch.zeros(2),
        old_log_probs=torch.zeros(2),
        log_probs_and_entropy={"topk_log_probs": [s]},
    )
    assert loss is not None and loss.ndim == 0
    torch.testing.assert_close(loss, expected.mean(), atol=1e-6, rtol=1e-6)


def test_apply_opd_to_advantages_clips_sampled_token_kl() -> None:
    """``--opd-per-token-clip`` must apply on the 1D sampled-token advantage path too."""
    args = Namespace(
        opd_kl_coef=0.5,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_log_prob_min_clamp=None,
        opd_token_selection="student_sampled",
        opd_fkl_entropy_gate=False,
        opd_per_token_clip=2.0,
    )
    rollout_data = {
        "rollout_log_probs": [torch.tensor([-0.5, -1.0, -2.0])],
        "teacher_log_probs": [torch.tensor([-10.5, -2.0, -1.0])],  # raw KL = [10.0, 1.0, -1.0]
    }
    advantages = [torch.zeros(3)]

    apply_opd_to_advantages(args, rollout_data, advantages)
    expected = -0.5 * torch.tensor([2.0, 1.0, -1.0])
    torch.testing.assert_close(advantages[0], expected, atol=1e-6, rtol=1e-6)


def test_opd_kl_masked_all_false_row_stays_finite() -> None:
    s = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]], dtype=torch.float32, requires_grad=True)
    t = torch.tensor([[-1.2, -1.8], [-1.2, -1.8]], dtype=torch.float32)
    mask = torch.tensor([[True, True], [False, False]])

    for norm_mode in ("tail", "norm", "trunc"):
        s_in = s.detach().clone().requires_grad_(True)
        out = compute_opd_kl(s_in, t, kl_type="jsd", jsd_alpha=0.5, norm_mode=norm_mode, mask=mask)
        out.sum().backward()
        assert torch.isfinite(out).all()
        assert out[1].item() == pytest.approx(0.0, abs=1e-6)
        assert torch.isfinite(s_in.grad).all()
        assert torch.count_nonzero(s_in.grad[1]) == 0
