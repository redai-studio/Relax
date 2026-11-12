# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SC gradients against a dense distribution, independent of the head
reduction."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import compute_policy_loss_for  # noqa: E402
from relax.algorithms.score_centering import compute_score_centering_loss  # noqa: E402
from relax.algorithms.spec import get_algorithm  # noqa: E402


def _weight(ratio: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "tis":
        return ratio.clamp_max(2.0)
    if mode == "mis":
        return torch.where((ratio >= 0.5) & (ratio <= 5.0), ratio, 0.0)
    return torch.ones_like(ratio)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize("sample_head", [True, False])
def test_score_centering_gradient_matches_dense_tail_oracle(mode: str, k: int, sample_head: bool) -> None:
    generator = torch.Generator().manual_seed(27)
    logits = torch.randn(4, 7, generator=generator, dtype=torch.float64).requires_grad_()
    q = (2 * torch.randn(4, 7, generator=generator, dtype=torch.float64)).softmax(-1)
    q_head, ids = q.topk(k, dim=-1)
    sampled = q.argmax(-1) if sample_head else q.argmin(-1)
    advantage = torch.tensor([1.7, -0.8, 0.0, 0.2], dtype=torch.float64, requires_grad=True)
    sampled_q = q.gather(-1, sampled[:, None]).squeeze(-1).log().requires_grad_()
    head_q = q_head.log().requires_grad_()
    logp = logits.log_softmax(-1)
    actual, metrics = compute_score_centering_loss(
        log_probs=logp.gather(-1, sampled[:, None]).squeeze(-1),
        advantages=advantage,
        rollout_log_probs=sampled_q,
        train_topk_log_probs=logp.gather(-1, ids),
        rollout_topk_log_probs=head_q,
        loss_mask=torch.ones(4),
        mode=mode,
    )
    actual.sum().backward()

    oracle_logits = logits.detach().clone().requires_grad_()
    oracle_logp = oracle_logits.log_softmax(-1)
    p = oracle_logp.detach().exp()
    rho = (1 - q_head.sum(-1)).clamp_min(1e-6) / (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
    q_hat = p * rho[:, None]
    q_hat.scatter_(-1, ids, q_head)
    head_weights = _weight(p / q_hat, mode)
    sample_weights = _weight(p.gather(-1, sampled[:, None]).squeeze(-1) / sampled_q.detach().exp(), mode)
    oracle = -advantage.detach() * (
        sample_weights * oracle_logp.gather(-1, sampled[:, None]).squeeze(-1)
        - (q_hat * head_weights * oracle_logp).sum(-1)
    )
    oracle.sum().backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad, atol=1e-12, rtol=1e-12)
    assert advantage.grad is None and sampled_q.grad is None and head_q.grad is None
    assert all(not metric.requires_grad for metric in metrics.values())


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_score_centering_full_distribution_centers_constant_advantage(mode: str) -> None:
    logits = torch.tensor([0.7, -1.2, 2.3, 0.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.1, 0.5, 0.15, 0.25], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    loss, _ = compute_score_centering_loss(
        log_probs=logp,
        advantages=torch.ones(4),
        rollout_log_probs=q.log(),
        train_topk_log_probs=logp.expand(4, 4),
        rollout_topk_log_probs=q.log().expand(4, 4),
        loss_mask=torch.ones(4),
        mode=mode,
    )
    (q * loss).sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-12, rtol=0)


@pytest.mark.parametrize("matched", [True, False])
def test_score_centering_matched_policy_or_unclipped_is_has_no_correction(matched: bool) -> None:
    p = torch.tensor([[0.55, 0.3, 0.15]], dtype=torch.float64)
    q = p if matched else torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    logp = p.log().requires_grad_()
    loss, metrics = compute_score_centering_loss(
        log_probs=logp[:, 2],
        advantages=torch.tensor([2.0]),
        rollout_log_probs=q[:, 2].log(),
        train_topk_log_probs=logp[:, :2],
        rollout_topk_log_probs=q[:, :2].log(),
        loss_mask=torch.ones(1),
        mode="none" if matched else "tis",
        tis_clip=10,
    )
    weight = 1.0 if matched else p[0, 2] / q[0, 2]
    torch.testing.assert_close(metrics["sc_correction"], torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(loss, -2 * weight * logp[:, 2])


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_score_centering_masked_rows_and_candidate_padding_have_zero_gradient(mode: str) -> None:
    logp = torch.tensor([-0.7, float("nan")], requires_grad=True)
    train_head = torch.tensor([[-0.8, -1.1, float("nan")], [float("nan")] * 3], requires_grad=True)
    loss, metrics = compute_score_centering_loss(
        log_probs=logp,
        advantages=torch.tensor([1.0, float("nan")]),
        rollout_log_probs=torch.tensor([-0.6, float("nan")]),
        train_topk_log_probs=train_head,
        rollout_topk_log_probs=torch.tensor([[-0.7, -1.2, -torch.inf], [float("nan")] * 3]),
        loss_mask=torch.tensor([1.0, 0.0]),
        mode=mode,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert loss[1] == 0 and logp.grad[1] == 0
    assert torch.isfinite(train_head.grad).all()
    assert torch.count_nonzero(train_head.grad[1]) == 0
    assert train_head.grad[0, 2] == 0
    for metric in metrics.values():
        assert torch.isfinite(metric).all() and metric[1] == 0


def test_score_centering_mis_keeps_correction_when_sample_weight_is_zero() -> None:
    logits = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64).log().requires_grad_()
    logp = logits.log_softmax(-1)
    loss, metrics = compute_score_centering_loss(
        log_probs=logp[:, 0],
        advantages=torch.ones(1),
        rollout_log_probs=torch.tensor([0.1], dtype=torch.float64).log(),
        train_topk_log_probs=logp[:, 1:],
        rollout_topk_log_probs=torch.tensor([[0.2, 0.7]], dtype=torch.float64).log(),
        loss_mask=torch.ones(1),
        mode="mis",
    )
    expected_gradient = torch.autograd.grad(0.3 * logp[0, 1], logits, retain_graph=True)[0]
    loss.sum().backward()
    assert metrics["sc_importance_weight"].item() == 0
    assert metrics["sc_correction"].item() != 0
    torch.testing.assert_close(logits.grad, expected_gradient)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_score_centering_tiny_tail_and_extreme_ratios_are_finite(mode: str) -> None:
    logp = torch.tensor([0.0], requires_grad=True)
    train_head = torch.tensor([[0.0, -10000.0]], requires_grad=True)
    loss, metrics = compute_score_centering_loss(
        log_probs=logp,
        advantages=torch.ones(1),
        rollout_log_probs=torch.tensor([-10000.0]),
        train_topk_log_probs=train_head,
        rollout_topk_log_probs=torch.tensor([[-10000.0, 0.0]]),
        loss_mask=torch.ones(1),
        mode=mode,
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logp.grad).all()
    assert torch.isfinite(train_head.grad).all()
    assert all(torch.isfinite(metric).all() for metric in metrics.values())
    assert metrics["sc_tail_floor_fraction"].item() == 1


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator="score_centering",
        score_centering_is="none",
        score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5,
        score_centering_mis_high=5.0,
    )


@pytest.mark.parametrize("n", [0, 3])
def test_score_centering_dispatch_handles_empty_and_all_masked_rows(n: int) -> None:
    logp = torch.zeros(n, requires_grad=True)
    head = torch.zeros(n, 2, requires_grad=True)
    loss, clipfrac, metrics = compute_policy_loss_for(
        _args(),
        log_probs=logp,
        ppo_kl=torch.zeros(n),
        advantages=torch.ones(n),
        score_centering_context={
            "rollout_log_probs": torch.zeros(n),
            "train_topk_log_probs": head,
            "rollout_topk_log_probs": torch.full((n, 2), -torch.inf),
            "loss_mask": torch.zeros(n),
        },
    )
    loss.sum().backward()
    assert logp.grad is not None and head.grad is not None
    assert torch.count_nonzero(logp.grad) == 0 and torch.count_nonzero(head.grad) == 0
    assert torch.count_nonzero(loss) == 0 and torch.count_nonzero(clipfrac) == 0
    assert tuple(metrics) == get_algorithm("score_centering").policy_scalar_metric_names
    assert all(metric == 0 and not metric.requires_grad for metric in metrics.values())


def test_score_centering_dispatch_requires_generation_context() -> None:
    with pytest.raises(ValueError, match="requires rollout"):
        compute_policy_loss_for(_args(), log_probs=torch.zeros(1), ppo_kl=torch.zeros(1), advantages=torch.ones(1))


def test_score_centering_preset_uses_group_mean_and_broadcast_without_on_policy_restriction() -> None:
    spec = get_algorithm("score_centering")
    assert (spec.reward_normalizer, spec.advantage_fn, spec.policy_loss_fn) == (
        "group_mean",
        "grpo_broadcast",
        "score_centering",
    )
    assert spec.requires_complete_reward_groups and spec.requires_rewards_normalization
    assert spec.min_group_size == 2
    assert spec.forbids_normalize_advantages and spec.forbids_reward_side_kl
    assert spec.requires_global_token_loss
    assert spec.supports_context_parallel and not spec.requires_on_policy_updates
