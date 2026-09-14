# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from numbers import Real
from typing import Any


def filter_zero_advantage_rows(
    args: Namespace,
    rollout_data: dict[str, Any],
    *,
    reference_matches_actor: bool,
) -> tuple[dict[str, Any], int]:
    """Prune zero-gradient GRPO trajectories before forward passes.

    Preserve logical batch and trajectory denominators. An empty local DP keeps
    one zero-contribution row so collectives and the optimizer/scheduler still
    run. Unsupported loss semantics and batch layouts use the original data
    unchanged.
    """
    disabled_flags = (
        "normalize_advantages",
        "use_opd",
        "partial_rollout",
        "use_dynamic_global_batch_size",
        "calculate_per_token_loss",
        "use_rollout_routing_replay",
        "use_routing_replay",
        "num_experts",
        "mtp_num_layers",
    )
    custom_hooks = (
        "custom_loss_function_path",
        "custom_pg_loss_reducer_function_path",
        "custom_tis_function_path",
        "custom_megatron_before_log_prob_hook_path",
        "custom_megatron_before_train_step_hook_path",
        "rollout_data_postprocess_path",
    )
    if (
        getattr(args, "advantage_estimator", None) != "grpo"
        or getattr(args, "loss_type", "policy_loss") != "policy_loss"
        or not getattr(args, "compute_advantages_and_returns", False)
        or not getattr(args, "use_agentic_rollout", False)
        or not getattr(args, "use_dynamic_batch_size", False)
        or getattr(args, "num_iters_per_train_update", 1) != 1
        or getattr(args, "entropy_coef", 0.0) != 0.0
        or getattr(args, "kl_coef", 0.0) != 0.0
        or getattr(args, "attention_dropout", 0.0) != 0.0
        or getattr(args, "hidden_dropout", 0.0) != 0.0
        or any(getattr(args, flag, False) for flag in disabled_flags)
        or any(getattr(args, hook, None) is not None for hook in custom_hooks)
    ):
        return rollout_data, 0
    if getattr(args, "use_kl_loss", False) and getattr(args, "kl_loss_coef", 0.0) != 0.0:
        if not reference_matches_actor or getattr(args, "kl_loss_type", None) not in ("k2", "k3", "low_var_kl"):
            return rollout_data, 0

    lengths = rollout_data.get("total_lengths")
    rewards = rollout_data.get("rewards")
    identities = rollout_data.get("sample_indices")
    denominators = rollout_data.get("sample_index_mask_sums")
    if not isinstance(lengths, list) or not lengths:
        return rollout_data, 0
    count = len(lengths)
    # Explicit boundaries prove this is one update, even after filtering makes the
    # physical row count smaller than the logical global batch size.
    if rollout_data.get("rollout_mini_local_sample_counts") != [count]:
        return rollout_data, 0
    if not all(
        isinstance(values, (list, tuple)) and len(values) == count for values in (rewards, identities, denominators)
    ):
        return rollout_data, 0
    # CPU scalars only: never synchronize GPU rewards to decide which rows to keep.
    if not all(isinstance(reward, Real) for reward in rewards):
        return rollout_data, 0
    if not all(isinstance(identity, (int, str)) for identity in identities):
        return rollout_data, 0

    active_identities = {identity for identity, reward in zip(identities, rewards, strict=True) if reward != 0.0}
    keep = [i for i, identity in enumerate(identities) if identity in active_identities]
    if len(keep) == count:
        return rollout_data, 0
    if not keep:
        keep = [min(range(count), key=lengths.__getitem__)]

    import torch

    filtered: dict[str, Any] = {}
    for key, value in rollout_data.items():
        if key.startswith("rollout_mini_"):
            filtered[key] = value
        elif isinstance(value, (list, tuple)):
            if len(value) != count:
                return rollout_data, 0
            selected = [value[i] for i in keep]
            filtered[key] = tuple(selected) if isinstance(value, tuple) else selected
        elif isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.size(0) != count:
                return rollout_data, 0
            indices = torch.tensor(keep, dtype=torch.long, device=value.device)
            filtered[key] = value.index_select(0, indices)
        else:
            filtered[key] = value
    filtered["rollout_mini_local_sample_counts"] = [len(keep)]
    return filtered, count - len(keep)
