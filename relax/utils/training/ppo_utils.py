# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

import contextlib
import math
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def validate_ppo_config(config: Namespace) -> None:
    if getattr(config, "advantage_estimator", None) != "ppo":
        return

    resource = getattr(config, "resource", None) or {}
    if "critic" not in resource:
        raise ValueError(
            "--advantage-estimator ppo requires a 'critic' entry in --resource. "
            "Fix: add a critic entry such as 'critic': [1, 1]."
        )

    if getattr(config, "fully_async", False) or getattr(config, "hybrid", False):
        raise ValueError("PPO does not currently support --fully-async or --hybrid.")

    is_sync_colocate = getattr(config, "colocate", False) and not getattr(config, "fully_async", False)
    if is_sync_colocate and getattr(config, "max_staleness", 0) != 0:
        raise ValueError("Synchronous colocate PPO requires --max-staleness 0.")

    _validate_actor_critic_resume_consistency(config)


def _validate_actor_critic_resume_consistency(config: Namespace) -> None:
    """Keep actor and critic checkpoints in lockstep.

    If both sides are cold-started or both resume from the same iteration, we
    do nothing. If only one side has a tracker (typical when actor's ckpt was
    wiped after a crash but critic's stale ckpt lingered), fall back to cold-
    start on both by routing the surviving side at a path without a tracker —
    Megatron then re-inits from ``--hf-checkpoint``. Only raise when both sides
    have trackers at different iterations, which is a genuine inconsistency the
    user must resolve.
    """
    actor_load = getattr(config, "load", None)
    critic_load = getattr(config, "critic_load", None) or actor_load
    actor_iter = _read_latest_iter(actor_load)
    critic_iter = _read_latest_iter(critic_load)

    if actor_iter == critic_iter:
        return

    if actor_iter is None and critic_iter is not None and actor_load is not None:
        logger.warning(
            f"PPO resume: actor has no tracker at {actor_load!r} while critic has "
            f"iter={critic_iter} at {critic_load!r}. Cold-starting both from "
            f"--hf-checkpoint to keep them in sync; the stale critic ckpt is left "
            f"on disk — delete it manually if you want cleanup."
        )
        config.critic_load = actor_load
        return

    if critic_iter is None and actor_iter is not None and critic_load is not None:
        logger.warning(
            f"PPO resume: critic has no tracker at {critic_load!r} while actor has "
            f"iter={actor_iter} at {actor_load!r}. Cold-starting both from "
            f"--hf-checkpoint to keep them in sync; the stale actor ckpt is left "
            f"on disk — delete it manually if you want cleanup."
        )
        config.load = critic_load
        return

    raise ValueError(
        "PPO resume requires actor and critic checkpoints to be in the same state, "
        f"but got actor iter={actor_iter} at --load={actor_load!r} and "
        f"critic iter={critic_iter} at --critic-load={critic_load!r}. "
        "Either provide both Megatron checkpoints at the same iteration, or remove both "
        "so training cold-starts from --hf-checkpoint for actor and critic."
    )


def _read_latest_iter(load_path: str | None) -> int | None:
    """Return the iteration recorded in a Megatron checkpoint tracker."""
    if not load_path:
        return None
    tracker = Path(load_path) / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    try:
        return int(tracker.read_text().strip())
    except (ValueError, OSError):
        return None


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html.

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_sapo_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the Soft Adaptive Policy Optimization (SAPO) loss.

    SAPO replaces hard clipping with a smooth, temperature-controlled gate:
    f(r) = (4/τ) * σ(τ(r-1))

    where r = π_θ(y|x) / π_θ_old(y|x) is the importance ratio.

    Args:
        ppo_kl: KL divergence approximation (old_log_probs - new_log_probs).
            Shape: [batch_size * seq_len] or [total_tokens]
            Note: ratio = exp(-ppo_kl)
        advantages: Advantage values, shape matches ppo_kl.
        tau_pos: Temperature for positive advantages (typically ~1.0).
        tau_neg: Temperature for negative advantages (typically ~1.05, higher to
            dampen unstable negative gradients).

    Returns:
        pg_loss: Element-wise SAPO loss tensor (negative objective for minimization).
            Shape matches input. NO reduction applied (caller handles masking).
        clipfrac: Element-wise indicator of off-policy behavior (weight < 0.9).
            Shape matches input.
    """
    # Calculate importance ratio: r(θ) = π_new / π_old = exp(-KL)
    ratios = torch.exp(-ppo_kl)

    # Select temperature based on advantage sign
    # Asymmetric: higher tau_neg dampens negative token gradients more aggressively
    # This is critical for training stability
    tau = torch.where(
        advantages > 0,
        torch.tensor(tau_pos, device=advantages.device, dtype=advantages.dtype),
        torch.tensor(tau_neg, device=advantages.device, dtype=advantages.dtype),
    )

    # Compute Soft Gate: f(r) = (4/τ) * σ(τ(r - 1))
    # This implements a smooth trust region centered at r=1 (on-policy)
    # - At r=1: f(1) = (4/τ) * σ(0) = (4/τ) * 0.5 = 2/τ
    # - As r deviates: sigmoid saturates, f(r) → 0 or 4/τ smoothly
    scaled_diff = tau * (ratios - 1.0)
    sigmoid_val = torch.sigmoid(scaled_diff)
    soft_gate_val = (4.0 / tau) * sigmoid_val

    # Compute SAPO Objective: J(θ) = f(r) * A
    # Paper maximizes this, but we return negative for PyTorch minimization
    pg_loss = -(soft_gate_val * advantages)

    # Compute clipfrac proxy (for monitoring off-policy behavior)
    # Gradient weight: w(r) = 4p(1-p) where p = σ(τ(r-1))
    weight = 4.0 * sigmoid_val * (1.0 - sigmoid_val)
    clipfrac = (weight < 0.9).float()

    return pg_loss, clipfrac


@torch.compile(dynamic=True)
def compute_cispo_loss(
    log_probs: torch.Tensor,
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the CISPO (Clipped Importance-ratio Soft Policy Optimization)
    loss.

    Unlike PPO which zeros out gradients when the importance ratio exceeds
    the clipping threshold, CISPO preserves the gradient direction but caps
    the gradient magnitude via a stop-gradient'd coefficient.

    The key formulation:
        ratio = π_new / π_old = exp(-ppo_kl)
        coef  = clamp(ratio, ...) based on advantage sign
        loss  = -stop_grad(coef) * stop_grad(A) * log_probs

    Gradients flow ONLY through `log_probs`, not through the ratio or advantages.
    Reference: MiniMax-M1 (arXiv:2501.15116).

    Args:
        log_probs: Current policy log-probabilities (gradient source).
            Shape: [total_tokens]
        ppo_kl: KL divergence approximation (old_log_probs - new_log_probs).
            Shape: [total_tokens]. Used only for ratio computation (detached).
        advantages: Advantage values, shape matches log_probs.
        eps_clip: Lower clipping margin (ratio lower bound = 1 - eps_clip).
        eps_clip_high: Upper clipping margin (ratio upper bound = 1 + eps_clip_high).

    Returns:
        pg_loss: Element-wise CISPO loss tensor (negative objective for minimization).
            Shape matches input. NO reduction applied (caller handles masking).
        clipfrac: Element-wise indicator of tokens where ratio exceeds bounds.
            Shape matches input.
    """
    # Importance ratio: r(θ) = π_new / π_old = exp(-ppo_kl)
    ratio = torch.exp(-ppo_kl)

    upper = 1.0 + eps_clip_high
    lower = 1.0 - eps_clip

    # Asymmetric clipping based on advantage sign:
    #   A >= 0: cap ratio from above → coef = min(ratio, 1 + eps_clip_high)
    #   A <  0: cap ratio from below → coef = max(ratio, 1 - eps_clip)
    coef_pos = torch.clamp(ratio, max=upper)
    coef_neg = torch.clamp(ratio, min=lower)
    coef = torch.where(advantages >= 0, coef_pos, coef_neg)
    del coef_pos, coef_neg

    # CISPO objective: gradient flows ONLY through log_probs.
    # coef and advantages are detached to prevent extra gradient paths.
    pg_loss = -(coef.detach() * advantages.detach() * log_probs)

    # Clipfrac: fraction of tokens where ratio exceeds trust-region bounds
    clipfrac = ((advantages >= 0) & (ratio > upper)).float() + ((advantages < 0) & (ratio < lower)).float()
    del ratio, coef

    return pg_loss, clipfrac


@torch.compile(dynamic=True)
def compute_rloo_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the RLOO (REINFORCE Leave-One-Out) loss.

    RLOO uses a dedicated *unclipped* REINFORCE objective. The leave-one-out
    baseline is computed upstream (see ``compute_rloo_leave_one_out_rewards``),
    so here we only apply the policy-gradient term:

        pg_loss = -stop_gradient(A) * log π_θ(y)

    Gradients flow ONLY through ``log_probs``; ``advantages`` is detached.
    This is mathematically equivalent to CISPO with the importance ratio fixed
    to 1 (on-policy). ``clipfrac`` is always zero — it serves as a wiring
    self-check that the unclipped path is active.

    Reference: Ahmadian et al. 2024 (arXiv:2402.14740); Kool et al. 2019.

    Args:
        log_probs: Current policy log-probabilities (gradient source).
            Shape: [total_tokens]
        advantages: Leave-one-out advantage values, shape matches log_probs.

    Returns:
        pg_loss: Element-wise RLOO loss (negative objective for minimization).
            Shape matches input. NO reduction applied (caller handles masking).
        clipfrac: Element-wise zero tensor (unclipped — wiring self-check).
            Shape matches input.
    """
    pg_loss = -(advantages.detach() * log_probs)
    clipfrac = torch.zeros_like(log_probs)
    return pg_loss, clipfrac


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert eps_clip_c > 1.0, (
            f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        )
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


# ── M2PO: Second-Moment Trust Policy Optimization ──────────────────────────
# Adapted from https://github.com/Infini-AI-Lab/M2PO (Apache 2.0)
# Paper: "Prosperity before Collapse" (NeurIPS 2025), https://arxiv.org/abs/2510.01161


def _solve_tau_from_sorted_delta2(sorted_delta2: torch.Tensor, target_sum: float) -> tuple[float, float]:
    n = sorted_delta2.numel()
    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12:
        return 100000.0, total / n
    if target_sum <= 1e-12:
        return 0.0, 0.0
    csum = torch.cumsum(sorted_delta2, dim=0)
    for k in range(n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0, float(csum[-1].item()) / n
            M2_after = (sorted_delta2[k - 1].item() * (rest + 1) + float(csum[k - 1].item())) / n
            return max(sorted_delta2[k - 1].item() - 1e-12, 0.0) ** 0.5, M2_after
    return 100000.0, total / n


def _get_trust_region_delta_sq(ppo_kl: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    ratio = (-ppo_kl).exp()
    pos_harmful = (advantages > 1e-12) & (ratio > 1.0 + 1e-12)
    neg_harmful = (advantages < -1e-12) & (ratio < 1.0 - 1e-12)
    return ppo_kl[pos_harmful | neg_harmful].pow(2)


def kpo_clip_harmful_tokens(
    ppo_kl: torch.Tensor, advantages: torch.Tensor, kl2_budget: float
) -> tuple[float, float, float, float]:
    tr_delta_sq = _get_trust_region_delta_sq(ppo_kl, advantages)
    n = tr_delta_sq.numel()
    if n == 0:
        return 0.0, 100000.0, 0.0, 0.0
    M2_now = float(tr_delta_sq.sum().detach().item() / n)
    if M2_now <= kl2_budget + 1e-12:
        return 0.0, 100000.0, M2_now, M2_now
    sorted_delta2, _ = torch.sort(tr_delta_sq)
    tau, M2_after = _solve_tau_from_sorted_delta2(sorted_delta2, kl2_budget * float(n))
    return math.exp(-tau), math.exp(tau), M2_now, M2_after


def compute_m2po_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    kl2_budget: float,
    miniclip_low: float = 0.3,
    miniclip_high: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, float]:
    clip_low, clip_high, M2_now, M2_after = kpo_clip_harmful_tokens(ppo_kl, advantages, kl2_budget)
    eps_low = max(1.0 - clip_low, miniclip_low)
    eps_high = max(clip_high - 1.0, miniclip_high)
    ratio = (-ppo_kl).exp()
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    pg_loss = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = (pg_losses2 > pg_losses1).float()
    return pg_loss, clipfrac, M2_now, M2_after, eps_low, eps_high


# ───────────────────────────────────────────────────────────────────────────


def _maybe_all_reduce(tensor: torch.Tensor, op: dist.ReduceOp, process_group) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op, group=process_group)


def _get_vocab_parallel_rank_size(process_group) -> tuple[int, int]:
    if process_group is not None and hasattr(process_group, "rank") and hasattr(process_group, "size"):
        return process_group.rank(), process_group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)
    return 0, 1


class _VocabParallelLogProbEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        log_prob_keep_mask: torch.Tensor | None,
        process_group,
        with_entropy: bool,
        with_entropy_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with_entropy_grad = with_entropy and with_entropy_grad
        vocab_parallel_logits = vocab_parallel_logits.float()
        seq_len, vocab_parallel_size = vocab_parallel_logits.shape
        rank, _world_size = _get_vocab_parallel_rank_size(process_group)
        vocab_start_index = rank * vocab_parallel_size
        vocab_end_index = vocab_start_index + vocab_parallel_size

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target_1d = (target - vocab_start_index).clone()
        masked_target_1d[target_mask] = 0
        arange_1d = torch.arange(seq_len, device=vocab_parallel_logits.device)

        def vocab_parallel_softmax(
            logits: torch.Tensor,
            inplace: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits_max = logits.max(dim=-1, keepdim=True).values
            _maybe_all_reduce(logits_max, dist.ReduceOp.MAX, process_group)
            # Subtract the max for numerical stability. When ``inplace`` is set, the
            # caller passed a scratch buffer it owns, so overwrite it instead of
            # allocating another [seq_len, vocab] tensor.
            normalized_logits = logits.sub_(logits_max) if inplace else logits - logits_max
            # The normalized logit at the target position is the log-prob numerator;
            # gather it (a small copy) before the in-place ``exp_`` destroys it.
            predicted_logits = normalized_logits.view(-1, vocab_parallel_size)[arange_1d, masked_target_1d]
            # Reuse the ``normalized_logits`` storage for exp and softmax so the whole
            # softmax costs a single [seq_len, vocab] buffer instead of three.
            exp_logits = normalized_logits.exp_()
            sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
            _maybe_all_reduce(sum_exp_logits, dist.ReduceOp.SUM, process_group)
            softmax = exp_logits.div_(sum_exp_logits)
            return predicted_logits, sum_exp_logits, softmax, logits_max

        entropy = vocab_parallel_logits.new_zeros((0,))
        entropy_softmax = vocab_parallel_logits.new_empty((0,))
        sum_softmax_times_logits = vocab_parallel_logits.new_empty((0,))

        def sum_softmax_logits(softmax: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            if softmax.is_cuda:
                # Avoid materializing the full [seq_len, vocab] product buffer.
                return torch.einsum("ij,ij->i", softmax, logits).unsqueeze(-1)
            return (softmax * logits).sum(dim=-1, keepdim=True)

        if log_prob_keep_mask is None:
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, log_prob_logits_max = vocab_parallel_softmax(
                vocab_parallel_logits
            )
            if with_entropy:
                entropy_softmax = log_prob_softmax
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = log_prob_logits_max + log_prob_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)
        else:
            if with_entropy:
                _entropy_predicted_logits, entropy_sum_exp_logits, entropy_softmax, entropy_logits_max = (
                    vocab_parallel_softmax(vocab_parallel_logits)
                )
                sum_softmax_times_logits = sum_softmax_logits(entropy_softmax, vocab_parallel_logits)
                _maybe_all_reduce(sum_softmax_times_logits, dist.ReduceOp.SUM, process_group)
                entropy = entropy_logits_max + entropy_sum_exp_logits.log() - sum_softmax_times_logits
                entropy = entropy.squeeze(dim=-1)

            local_target_rows = torch.nonzero(~target_mask, as_tuple=False).squeeze(-1)
            log_prob_logits = vocab_parallel_logits.masked_fill(~log_prob_keep_mask, float("-inf"))
            if local_target_rows.numel() > 0:
                log_prob_logits[local_target_rows, masked_target_1d[local_target_rows]] = vocab_parallel_logits[
                    local_target_rows, masked_target_1d[local_target_rows]
                ]
            # ``log_prob_logits`` is an owned scratch buffer here, so let the softmax
            # consume it in place rather than allocating another copy.
            predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, _log_prob_logits_max = vocab_parallel_softmax(
                log_prob_logits, inplace=True
            )

        predicted_logits = predicted_logits.masked_fill_(target_mask, 0.0).unsqueeze(-1)
        _maybe_all_reduce(predicted_logits, dist.ReduceOp.SUM, process_group)
        log_prob = predicted_logits - log_prob_sum_exp_logits.log()

        if not with_entropy_grad:
            ctx.mark_non_differentiable(entropy)

        ctx.with_entropy_grad = with_entropy_grad
        # Metric-only entropy still returns values, but does not need the
        # full-vocab entropy tensors kept alive for backward.
        saved_entropy_softmax = entropy_softmax if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        saved_sum_softmax_times_logits = (
            sum_softmax_times_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        )
        saved_logits = vocab_parallel_logits if with_entropy_grad else vocab_parallel_logits.new_empty((0,))
        ctx.save_for_backward(
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            saved_entropy_softmax,
            saved_sum_softmax_times_logits,
            saved_logits,
        )
        return log_prob, entropy

    @staticmethod
    def backward(
        ctx, grad_log_prob: torch.Tensor | None, grad_entropy: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        (
            log_prob_softmax,
            target_mask,
            masked_target_1d,
            entropy_softmax,
            sum_softmax_times_logits,
            vocab_parallel_logits,
        ) = ctx.saved_tensors

        if grad_log_prob is None:
            raise RuntimeError(
                "_VocabParallelLogProbEntropy expected a materialized grad_log_prob. "
                "Do not call ctx.set_materialize_grads(False)."
            )

        grad_entropy_input = None
        if ctx.with_entropy_grad and grad_entropy is not None and grad_entropy.numel() > 0:
            # In the unmasked path, entropy_softmax aliases log_prob_softmax.
            # Build entropy grad before mutating log_prob_softmax below.
            grad_entropy_input = sum_softmax_times_logits - vocab_parallel_logits
            grad_entropy_input.mul_(entropy_softmax)
            grad_entropy_input.mul_(grad_entropy.reshape(-1, 1))

        vocab_parallel_size = log_prob_softmax.size(-1)
        grad_input = log_prob_softmax.neg_()
        grad_2d = grad_input.view(-1, vocab_parallel_size)
        arange_1d = torch.arange(grad_2d.size(0), device=grad_2d.device)
        target_update = (~target_mask).to(dtype=grad_2d.dtype)
        grad_2d[arange_1d, masked_target_1d] += target_update
        grad_input.mul_(grad_log_prob.reshape(-1, 1))

        if grad_entropy_input is not None:
            grad_input.add_(grad_entropy_input)

        return grad_input, None, None, None, None, None


def _calculate_log_probs_and_entropy_chunk(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group,
    *,
    with_entropy: bool,
    with_entropy_grad: bool = True,
    log_prob_keep_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    log_prob, entropy = _VocabParallelLogProbEntropy.apply(
        logits,
        tokens,
        log_prob_keep_mask,
        tp_group,
        with_entropy,
        with_entropy_grad,
    )
    if not with_entropy:
        entropy = None
    return log_prob, entropy


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def compute_rloo_leave_one_out_rewards(group_rewards: torch.Tensor) -> torch.Tensor:
    """Compute RLOO leave-one-out advantages for a single reward group.

    For G samples with rewards r_1..r_G, the leave-one-out baseline for
    sample i is the mean of all *other* samples:

        baseline_i = sum_{j != i}(r_j) / (G - 1)
        A_i = r_i - baseline_i = G/(G-1) * (r_i - mean(r))

    The implementation uses the scaled identity form (right-hand side) which is
    numerically equivalent but vectorised. Unlike GRPO, RLOO does NOT divide by
    the group standard deviation — the advantage preserves the reward scale and
    is unbiased because baseline_i is independent of r_i.

    Args:
        group_rewards: 1-D tensor of per-sample rewards for one group.
            Length G must be >= 2 (G-1=0 is undefined).

    Returns:
        1-D tensor of leave-one-out advantages, same length as input.

    Raises:
        ValueError: If input is not 1-D, has fewer than 2 elements, or
            contains non-finite values.
    """
    if group_rewards.dim() != 1:
        raise ValueError(
            f"compute_rloo_leave_one_out_rewards expects a 1-D tensor (one group), "
            f"got shape {tuple(group_rewards.shape)}. A 2-D input would be silently "
            f"treated as a single oversized group and produce wrong results."
        )
    group_size = group_rewards.shape[0]
    if group_size < 2:
        raise ValueError(
            f"RLOO requires n_samples_per_prompt >= 2 (group size {group_size} "
            f"makes the leave-one-out baseline undefined: division by G-1=0)."
        )
    if not torch.isfinite(group_rewards).all():
        raise ValueError(f"RLOO group contains non-finite reward(s): {group_rewards}. All rewards must be finite.")
    mean_reward = group_rewards.mean()
    scale = group_size / (group_size - 1)
    return scale * (group_rewards - mean_reward)


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """Calculates discounted returns for REINFORCE++
    (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    num_responses = len(rewards)
    if not (
        len(kl) == num_responses
        and len(loss_masks) == num_responses
        and len(response_lengths) == num_responses
        and len(total_lengths) == num_responses
    ):
        raise ValueError(
            "rewards, kl, loss_masks, response_lengths, and total_lengths must contain the same number of responses."
        )

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from relax.backends.megatron.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        full_mask = loss_masks[i]
        if full_kl_response.shape != full_mask.shape:
            raise ValueError(
                f"KL and loss mask for response {i} must have the same shape, "
                f"got {full_kl_response.shape} and {full_mask.shape}."
            )
        valid_mask = full_mask != 0
        if not torch.any(valid_mask):
            returns_for_seq = torch.zeros_like(full_kl_response)
            if cp_size > 1:
                from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

                returns_for_seq = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
            final_returns_chunks.append(returns_for_seq)
            continue

        # Multiplication is insufficient here because NaN * 0 is still NaN.
        # Select valid values before any return arithmetic so masked padding can
        # never contaminate a valid token.
        masked_kl = torch.where(valid_mask, full_kl_response, torch.zeros_like(full_kl_response))
        token_level_rewards = -kl_coef * masked_kl
        last_idx = valid_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return
        returns_for_seq = torch.where(valid_mask, returns_for_seq, torch.zeros_like(returns_for_seq))

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Calculates the unwhitened advantages for the REINFORCE++-baseline
    algorithm.

    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token tensors used only to determine
            the local response shapes.
        loss_masks (list[Tensor]): A list of per-token loss masks.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    if not (len(rewards) == len(kl) == len(loss_masks)):
        raise ValueError("rewards, token shapes, and loss_masks must contain the same number of responses.")

    # Token KL is intentionally not part of this advantage. The baseline
    # variant applies reference regularization as a separate k2 loss.
    unwhitened_advantages = []
    for response_index, (kl_tensor, reward_val, loss_mask) in enumerate(zip(kl, rewards, loss_masks, strict=True)):
        if kl_tensor.shape != loss_mask.shape:
            raise ValueError(
                f"Token shape and loss mask for response {response_index} must match, "
                f"got {kl_tensor.shape} and {loss_mask.shape}."
            )
        valid_mask = loss_mask != 0
        broadcast_reward = torch.ones_like(kl_tensor) * reward_val
        unwhitened_advantages.append(torch.where(valid_mask, broadcast_reward, torch.zeros_like(kl_tensor)))

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from relax.backends.megatron.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
    padded_total_lengths=None,
):
    """Batched GAE with CP support.

    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from relax.backends.megatron.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for idx, (total_len, resp_len, v, r) in enumerate(
                zip(total_lengths, response_lengths, values_list, rewards_list, strict=False)
            ):
                ptl = padded_total_lengths[idx] if padded_total_lengths is not None else None
                full_v = all_gather_with_cp(v, total_len, resp_len, padded_total_length=ptl)
                full_r = all_gather_with_cp(r, total_len, resp_len, padded_total_length=ptl)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from relax.backends.megatron.cp_utils import slice_log_prob_with_cp

            for idx, (total_len, resp_len, adv_row, ret_row) in enumerate(
                zip(
                    total_lengths,
                    response_lengths,
                    full_advantages,
                    full_returns,
                    strict=False,
                )
            ):
                adv_sliced = slice_log_prob_with_cp(
                    adv_row[:resp_len],
                    total_len,
                    resp_len,
                    padded_total_length=padded_total_lengths[idx] if padded_total_lengths is not None else None,
                )
                ret_sliced = slice_log_prob_with_cp(
                    ret_row[:resp_len],
                    total_len,
                    resp_len,
                    padded_total_length=padded_total_lengths[idx] if padded_total_lengths is not None else None,
                )

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """Compute Generalized Advantage Estimation (GAE) using a
    FlashLinearAttention- inspired algorithm: parallel prefix scan within
    chunks and recurrent state propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    log_prob_keep_mask=None,
    with_entropy_grad: bool = True,
):
    logits = logits.contiguous()
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            logits_chunks = logits.chunk(num_chunks, dim=0)
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            mask_chunks = (
                log_prob_keep_mask.chunk(num_chunks, dim=0) if log_prob_keep_mask is not None else [None] * num_chunks
            )

            log_probs = []
            entropy_chunks = []
            for tokens_chunk, logits_chunk, mask_chunk in zip(tokens_chunks, logits_chunks, mask_chunks, strict=True):
                log_prob, entropy_chunk = _calculate_log_probs_and_entropy_chunk(
                    logits_chunk,
                    tokens_chunk,
                    tp_group,
                    with_entropy=with_entropy,
                    with_entropy_grad=with_entropy_grad,
                    log_prob_keep_mask=mask_chunk,
                )
                log_probs.append(log_prob)
                if entropy_chunk is not None:
                    entropy_chunks.append(entropy_chunk)
            log_prob = torch.cat(log_probs, dim=0)
            if entropy_chunks:
                entropy = torch.cat(entropy_chunks, dim=0)
        else:
            log_prob, entropy = _calculate_log_probs_and_entropy_chunk(
                logits,
                tokens,
                tp_group,
                with_entropy=with_entropy,
                with_entropy_grad=with_entropy_grad,
                log_prob_keep_mask=log_prob_keep_mask,
            )
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy


# ============================================================================
# PPO critic value head plumbing (Megatron backend)
#
# All helpers below manipulate a scalar value head that replaces the vocab-
# sized LM head on the critic model. They must be co-located because Bridge /
# DDP / optimizer construction order matters (see design doc
# docs/superpowers/specs/2026-07-22-critic-value-head-registration-design.md).
#
# Megatron imports are lazy so this module stays importable in FSDP-only
# / non-Megatron setups (``ppo_utils`` is loaded by cross-backend callers).
# ============================================================================


_RELAX_HF_OUTPUT_LAYER_ATTR = "_relax_hf_output_layer"
_RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR = "_relax_seq_cls_hf_output_layer"

_CRITIC_VH_INIT_STATS_ATTR = "_critic_value_head_init_stats"
_CRITIC_VH_VERIFIED_ATTR = "_critic_value_head_verified"
_CRITIC_VH_CHECK_COUNT_ATTR = "_critic_value_head_check_count"
_CRITIC_VH_WARN_AFTER_STEPS = 5


class LinearForLastLayer(torch.nn.Linear):
    """Scalar value head that swaps in for Megatron's
    ``GPTModel.output_layer``.

    Sequence-parallel-aware: if ``config.sequence_parallel`` is set, the
    forward gathers the SP output so downstream slicing sees the full
    sequence dimension.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,  # megatron.core.transformer.TransformerConfig — forward ref to avoid top-level megatron dep
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True
            if bias:
                self.bias.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            from megatron.core import tensor_parallel

            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def _find_output_layer_owner(model: torch.nn.Module) -> torch.nn.Module | None:
    """Walk DDP / Float16Module / bridge-VL wrappers to the module that owns
    ``output_layer``.

    Handles the common Bridge convention where VL/Omni models nest the inner
    GPTModel under ``.language_model``. Returns None on non-last PP stages
    (where ``output_layer`` is ``nn.Identity``).
    """
    current = model
    for _ in range(6):
        for candidate in (current, getattr(current, "language_model", None)):
            if candidate is None:
                continue
            output_layer = getattr(candidate, "output_layer", None)
            if output_layer is not None and not isinstance(output_layer, torch.nn.Identity):
                return candidate
        current = getattr(current, "module", None)
        if current is None:
            break
    return None


def install_critic_value_head_in_provider(
    model: torch.nn.Module,
    role: str,
    post_process: bool,
    *,
    stash_lm_head: bool = False,
) -> None:
    """Swap ``model.output_layer`` for a ``LinearForLastLayer(hidden, 1)``.

    Called from ``get_model_provider_func`` after the underlying provider
    (Bridge / custom / default) builds a ``GPTModel``. Running before
    ``get_model(..., wrap_with_ddp=True)`` and ``get_megatron_optimizer(...)``
    is what lets DDP and the optimizer own the value head from the start;
    doing this after wrap orphans the new params.

    ``stash_lm_head=True`` (Bridge path only) preserves the original LM head
    as an unregistered attribute so ``use_critic_lm_head_for_hf_load`` can
    restore it during HF weight conversion.
    """
    if role != "critic" or not post_process:
        return

    owner = _find_output_layer_owner(model)
    if owner is None:
        return

    output_layer = owner.output_layer
    if isinstance(output_layer, LinearForLastLayer) and output_layer.out_features == 1:
        return

    if stash_lm_head:
        object.__setattr__(owner, _RELAX_HF_OUTPUT_LAYER_ATTR, output_layer)
    owner.output_layer = LinearForLastLayer(
        input_size=owner.config.hidden_size,
        output_size=1,
        config=owner.config,
    )


def install_sequence_classification_head_in_provider(
    model: torch.nn.Module,
    args,
    role: str,
    post_process: bool,
    *,
    stash_lm_head: bool = False,
) -> None:
    """Install a replicated ``hidden_size -> num_labels`` head before DDP.

    Only the actor's final PP/VPP chunk owns the classification head. Bridge
    models stash their vocabulary head under an unregistered attribute so HF
    CausalLM loading can temporarily restore it.
    """
    if getattr(args, "task_type", "causal_lm") != "seq_cls" or role != "actor" or not post_process:
        return

    owner = _find_output_layer_owner(model)
    if owner is None:
        return

    num_labels = int(args.num_labels)
    output_layer = owner.output_layer
    if isinstance(output_layer, LinearForLastLayer) and output_layer.out_features == num_labels:
        return

    if stash_lm_head:
        object.__setattr__(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR, output_layer)
    owner.output_layer = LinearForLastLayer(
        input_size=owner.config.hidden_size,
        output_size=num_labels,
        config=owner.config,
        bias=False,
    )


def ensure_sequence_classification_head_trainable(
    model: torch.nn.Module,
    args,
    role: str,
    post_process: bool,
) -> None:
    """Undo PEFT/freeze wrappers that would otherwise freeze the task head."""
    if getattr(args, "task_type", "causal_lm") != "seq_cls" or role != "actor" or not post_process:
        return
    owner = _find_output_layer_owner(model)
    if owner is None:
        return
    head = owner.output_layer
    if not isinstance(head, LinearForLastLayer) or head.out_features != int(args.num_labels):
        raise TypeError(f"sequence classification output layer is not installed: got {type(head).__name__}")
    for param in head.parameters():
        param.requires_grad = True


@contextlib.contextmanager
def use_sequence_classification_lm_head_for_hf_load(model):
    """Temporarily restore Bridge vocabulary heads while loading HF weights."""
    restored_heads = []
    try:
        for model_chunk in model:
            owner = _find_output_layer_owner(model_chunk)
            if owner is None or not hasattr(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR):
                continue

            classification_head = owner.output_layer
            classification_param_ids = tuple(id(param) for param in classification_head.parameters())
            lm_head = getattr(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR)
            restored_heads.append((owner, classification_head, classification_param_ids))

            classification_param = next(classification_head.parameters(), None)
            if classification_param is not None:
                lm_head.to(device=classification_param.device, dtype=classification_param.dtype)
            owner.output_layer = lm_head
        yield
    finally:
        for owner, classification_head, classification_param_ids in reversed(restored_heads):
            owner.output_layer = classification_head
            object.__delattr__(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR)
            assert owner.output_layer is classification_head, (
                "sequence classification head object changed during HF checkpoint loading"
            )
            assert tuple(id(param) for param in classification_head.parameters()) == classification_param_ids, (
                "sequence classification head parameters changed during HF checkpoint loading"
            )


def release_sequence_classification_lm_heads(model) -> None:
    """Drop any unregistered Bridge LM-head references after checkpoint
    load."""
    for model_chunk in model:
        owner = _find_output_layer_owner(model_chunk)
        if owner is not None and hasattr(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR):
            object.__delattr__(owner, _RELAX_SEQ_CLS_HF_OUTPUT_LAYER_ATTR)


@contextlib.contextmanager
def use_critic_lm_head_for_hf_load(model):
    """Temporarily restore the stashed LM head for HF Bridge weight loading.

    Bridge can only convert HF weights against a vocab-sized ``output_layer``;
    the scalar value head is put back in ``finally`` (asserting the exact same
    object survives) so DDP / optimizer references remain valid.
    """
    restored_heads = []
    try:
        for model_chunk in model:
            owner = _find_output_layer_owner(model_chunk)
            if owner is None or not hasattr(owner, _RELAX_HF_OUTPUT_LAYER_ATTR):
                continue

            value_head = owner.output_layer
            value_param_ids = tuple(id(param) for param in value_head.parameters())
            lm_head = getattr(owner, _RELAX_HF_OUTPUT_LAYER_ATTR)
            restored_heads.append((owner, value_head, value_param_ids))

            value_param = next(value_head.parameters(), None)
            if value_param is not None:
                lm_head.to(device=value_param.device, dtype=value_param.dtype)
            owner.output_layer = lm_head
        yield
    finally:
        for owner, value_head, value_param_ids in reversed(restored_heads):
            owner.output_layer = value_head
            object.__delattr__(owner, _RELAX_HF_OUTPUT_LAYER_ATTR)
            assert owner.output_layer is value_head, "critic value head object changed during HF checkpoint loading"
            assert tuple(id(param) for param in value_head.parameters()) == value_param_ids, (
                "critic value head parameters changed during HF checkpoint loading"
            )


def release_critic_lm_heads(model) -> None:
    """Drop the stashed LM-head reference after checkpoint load finishes."""
    for model_chunk in model:
        owner = _find_output_layer_owner(model_chunk)
        if owner is not None and hasattr(owner, _RELAX_HF_OUTPUT_LAYER_ATTR):
            object.__delattr__(owner, _RELAX_HF_OUTPUT_LAYER_ATTR)


def _ddp_owns_param(model_chunk: torch.nn.Module, param: torch.nn.Parameter) -> bool:
    if getattr(param, "main_grad", None) is not None:
        return True
    for module in (model_chunk, getattr(model_chunk, "module", None)):
        if module is None:
            continue
        for attr in ("param_to_buffer", "param_to_bucket", "param_to_bucket_group"):
            mapping = getattr(module, attr, None)
            if mapping is not None and param in mapping:
                return True
    return False


def validate_sequence_classification_head_registration(model, optimizer, args) -> tuple[int, ...]:
    """Verify shape, trainability, live-model registration, and DDP
    ownership."""
    del optimizer  # DistributedOptimizer shards ownership; DDP is the stable registration contract.
    classification_head_param_ids = []

    for model_chunk in model:
        owner = _find_output_layer_owner(model_chunk)
        if owner is None:
            continue
        classification_head = owner.output_layer
        assert isinstance(classification_head, LinearForLastLayer), (
            "sequence classification output layer must be LinearForLastLayer, "
            f"got {type(classification_head).__name__}"
        )
        expected_shape = (int(args.num_labels), owner.config.hidden_size)
        assert tuple(classification_head.weight.shape) == expected_shape, (
            f"sequence classification head weight must have shape {expected_shape}, "
            f"got {tuple(classification_head.weight.shape)}"
        )
        assert classification_head.bias is None, "sequence classification head must use bias=False"

        registered_param_ids = {id(param) for param in model_chunk.parameters()}
        for name, param in classification_head.named_parameters(recurse=False):
            param_name = f"output_layer.{name}"
            assert param.requires_grad, f"sequence classification head parameter {param_name} is frozen"
            assert id(param) in registered_param_ids, (
                f"sequence classification head parameter {param_name} is not registered in the live model"
            )
            assert _ddp_owns_param(model_chunk, param), (
                f"DDP does not own sequence classification head parameter {param_name}"
            )
            classification_head_param_ids.append(id(param))

    from megatron.core import mpu

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        assert classification_head_param_ids, "sequence classification head was not found on the final pipeline stage"
    else:
        assert not classification_head_param_ids, "sequence classification head exists on a non-final pipeline stage"
    return tuple(classification_head_param_ids)


def validate_critic_value_head_registration(model, optimizer) -> tuple[int, ...]:
    """Fail-fast structural checks for the critic value head.

    We verify shape and DDP ownership. DDP ownership is the load-bearing
    contract: once ``main_grad`` is allocated for a param, Megatron guarantees
    that exactly one DP rank's optimizer shard owns it and will update it
    during ``optimizer.step()``. We deliberately do NOT assert per-rank
    optimizer ownership because ``DistributedOptimizer`` shards params across
    DP ranks — only the shard-owning rank will have ``.main_param`` set or the
    param id present in its ``opt_group_ranges``.
    """
    del optimizer  # kept in signature for future use; ownership is DDP-only
    value_head_param_ids = []

    for model_chunk in model:
        owner = _find_output_layer_owner(model_chunk)
        if owner is None:
            continue
        value_head = owner.output_layer
        assert isinstance(value_head, LinearForLastLayer), (
            f"critic output layer must be LinearForLastLayer, got {type(value_head).__name__}"
        )
        assert tuple(value_head.weight.shape) == (1, owner.config.hidden_size), (
            "critic value head weight must have shape "
            f"(1, {owner.config.hidden_size}), got {tuple(value_head.weight.shape)}"
        )

        registered_param_ids = {id(param) for param in model_chunk.parameters()}
        for name, param in value_head.named_parameters(recurse=False):
            param_name = f"output_layer.{name}"
            assert id(param) in registered_param_ids, (
                f"critic value head parameter {param_name} is not registered in the live model"
            )
            assert _ddp_owns_param(model_chunk, param), f"DDP does not own critic value head parameter {param_name}"
            value_head_param_ids.append(id(param))

    return tuple(value_head_param_ids)


def snapshot_critic_value_head_state(model) -> dict:
    """One-scalar-per-param snapshot of value head weights, keyed by
    chunk+name.

    Uses ``.item()`` so this is a GPU→CPU sync — call once at initialization,
    never in the hot path. Returns an empty dict when this rank has no post-
    process value head (e.g., non-last PP stage).
    """
    stats: dict[str, float] = {}
    for chunk_idx, model_chunk in enumerate(model):
        owner = _find_output_layer_owner(model_chunk)
        if owner is None:
            continue
        for name, param in owner.output_layer.named_parameters(recurse=False):
            stats[f"chunk{chunk_idx}.output_layer.{name}"] = param.detach().float().abs().sum().item()
    return stats


def has_critic_value_head_moved(model, init_stats: dict) -> bool | None:
    """Return True if any value head param differs from its init snapshot.

    Returns ``None`` when this rank has no value head to check (empty
    ``init_stats``) so callers can skip logging for non-post-process ranks.
    Uses ``.item()``; caller should gate invocation to avoid per-step syncs.
    """
    if not init_stats:
        return None
    for chunk_idx, model_chunk in enumerate(model):
        owner = _find_output_layer_owner(model_chunk)
        if owner is None:
            continue
        for name, param in owner.output_layer.named_parameters(recurse=False):
            key = f"chunk{chunk_idx}.output_layer.{name}"
            init_val = init_stats.get(key)
            if init_val is None:
                continue
            current = param.detach().float().abs().sum().item()
            if current != init_val:
                return True
    return False


def install_critic_value_head_runtime_check(model) -> None:
    """Arm the resident value-head-movement check on ``model[0]``.

    Records a scalar snapshot of the value head weights and initializes the
    per-run state that ``maybe_verify_critic_value_head_movement`` consumes.
    Safe to call for non-post-process PP ranks — the snapshot is empty and the
    check will short-circuit as ``verified``.
    """
    init_stats = snapshot_critic_value_head_state(model)
    setattr(model[0], _CRITIC_VH_INIT_STATS_ATTR, init_stats)
    setattr(model[0], _CRITIC_VH_VERIFIED_ATTR, not init_stats)
    setattr(model[0], _CRITIC_VH_CHECK_COUNT_ATTR, 0)


def maybe_verify_critic_value_head_movement(model, optimizer, update_successful: bool) -> None:
    """Resident check that the critic value head is actually being updated.

    Costs one ``.item()`` per value head param per call; short-circuits after
    first observed movement so long-run overhead is zero. Emits a
    ``logger.warning`` (never asserts) after ``_CRITIC_VH_WARN_AFTER_STEPS``
    eligible steps still show no movement. Skips silently on ineligible steps
    (warmup ``lr==0``, grad-overflow ``update_successful=False``, actor role,
    non-post-process PP rank) so it never false-alarms.
    """
    if getattr(model[0], "role", "actor") != "critic":
        return
    if getattr(model[0], _CRITIC_VH_VERIFIED_ATTR, True):
        return
    if not update_successful:
        return
    if not any(pg.get("lr", 0.0) > 0 for pg in optimizer.param_groups):
        return

    init_stats = getattr(model[0], _CRITIC_VH_INIT_STATS_ATTR)
    moved = has_critic_value_head_moved(model, init_stats)
    if moved is True:
        setattr(model[0], _CRITIC_VH_VERIFIED_ATTR, True)
        return
    if moved is False:
        count = getattr(model[0], _CRITIC_VH_CHECK_COUNT_ATTR) + 1
        setattr(model[0], _CRITIC_VH_CHECK_COUNT_ATTR, count)
        if count >= _CRITIC_VH_WARN_AFTER_STEPS:
            logger.warning(
                "Critic value head weights unchanged after %d successful optimizer "
                "steps with lr>0 — suspect the head is not registered in DDP/optimizer. "
                "Verify get_model_provider_func installs the head before setup_model_and_optimizer.",
                count,
            )
            setattr(model[0], _CRITIC_VH_VERIFIED_ATTR, True)
