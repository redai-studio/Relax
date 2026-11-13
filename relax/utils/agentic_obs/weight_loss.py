# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Observe policy objective membership without changing the objective."""

from __future__ import annotations

from typing import Any

from relax.utils.agentic_obs.train_hooks import capture_effective_loss_mask
from relax.utils.agentic_obs.writers import best_effort


@best_effort
def observe_policy_tokens(
    args: Any,
    batch: Any,
    *,
    reducer_masks: Any,
    policy_gate: Any = None,
    tis_weights: Any = None,
    direct_opd: bool = False,
    score_centering_mask: Any = None,
    sequence_level_ratio: bool = False,
) -> None:
    """Union supported token objectives; unsupported coverage stays unknown."""
    if "_weight_audit_capture" not in batch:
        return
    from megatron.core import mpu

    reason = None
    if batch.get("dynamic_cp_size", mpu.get_context_parallel_world_size()) != 1:
        reason = "cp_shards_unavailable"
    elif getattr(args, "loss_type", None) == "custom_loss":
        # A custom objective may nest `policy_loss_function` and add further
        # terms; the nested call would certify only its own slice as complete.
        reason = "custom_loss"
    elif getattr(args, "enable_mtp_training", False) or getattr(args, "mtp_only_training", False):
        reason = "mtp_targets_unavailable"
    elif getattr(args, "custom_pg_loss_reducer_function_path", None):
        reason = "custom_reducer"
    elif getattr(args, "custom_tis_function_path", None) not in (
        None,
        "relax.backends.megatron.loss.icepop_function",
        "relax.backends.megatron.loss.vanilla_tis_function",
    ):
        reason = "custom_tis"
    elif direct_opd:
        reason = "direct_opd_targets_unavailable"
    if reason:
        capture_effective_loss_mask(batch, None, status=reason)
        return
    import torch

    lengths = [mask.numel() for mask in batch["loss_masks"]]
    base = torch.cat(batch["loss_masks"]).detach().bool()
    reducer = torch.cat(reducer_masks).detach().bool() & base
    policy = reducer.clone()
    if score_centering_mask is not None:
        policy &= score_centering_mask.detach().bool()
        reducer &= score_centering_mask.detach().bool()
    if policy_gate is not None:
        policy &= policy_gate.detach() != 0
    if tis_weights is not None:
        policy &= tis_weights.detach() != 0
    auxiliary = bool(getattr(args, "entropy_coef", 0)) or (
        getattr(args, "use_kl_loss", False) and bool(getattr(args, "kl_loss_coef", 0))
    )
    if sequence_level_ratio and not auxiliary and (tis_weights is not None or policy_gate is not None):
        # Sequence-level objectives (e.g. GSPO) build a differentiable sequence
        # ratio from every log-prob inside `loss_masks` before per-token gating
        # (TIS/OPSM) zeroes outer terms. Rejected tokens still contribute through
        # the ratio, so a per-position membership mask cannot be exact. Whether
        # any token was actually filtered lives on the training device; reading
        # it here would synchronize the loss hot path, so the combination is
        # conservatively reported unknown whenever a gate could reject.
        capture_effective_loss_mask(batch, None, status="sequence_ratio_unresolved")
        return
    effective = policy | reducer if auxiliary else policy
    capture_effective_loss_mask(batch, list(effective.split(lengths)))
