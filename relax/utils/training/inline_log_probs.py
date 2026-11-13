# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Automatic eligibility for reusing a synchronous GRPO training forward."""

from argparse import Namespace
from collections.abc import Callable, Iterable, Sequence


def can_reuse_colocate_log_probs(
    args: Namespace,
    num_microbatches: Sequence[int],
    *,
    backup_tags: Iterable[str],
    rollout_data_postprocess: Callable | None,
) -> bool:
    """Check actual per-rollout state without changing the algorithm or
    arguments."""
    # These conditions make the current training forward identical to the skipped
    # old-policy forward. Unsupported cases retain the framework's normal forward.
    required = {
        "colocate": True,
        "true_on_policy_mode": True,
        "fully_async": False,
        "hybrid": False,
        "advantage_estimator": "grpo",
        "loss_type": "policy_loss",
        "compute_advantages_and_returns": True,
        "num_iters_per_train_update": 1,
        "max_staleness": 0,
        "update_weights_interval": 1,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "dynamic_context_parallel": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "bf16": True,
        "fp16": False,
        "fp8": None,
        "fp4": None,
        "fp8_param_gather": False,
        "moe_input_jitter_eps": None,
        "moe_router_enable_expert_bias": False,
        "moe_router_force_load_balancing": False,
        "moe_router_force_biased": None,
        "moe_expert_capacity_factor": None,
        "moe_aux_loss_coeff": 0.0,
        "kl_coef": 0.0,
        "use_kl_loss": False,
        "use_opd": False,
        "use_critic": False,
        "keep_old_actor": False,
        "use_rollout_logprobs": False,
        "use_routing_replay": True,
        "use_rollout_routing_replay": True,
        "use_rollout_indexer_replay": False,
        "enable_mtp_training": False,
        "use_opsm": False,
        "use_tis": True,
        "calculate_per_token_loss": True,
        "entropy_coef": 0.0,
        "recompute_loss_function": False,
        "custom_tis_function_path": None,
        "custom_pg_loss_reducer_function_path": None,
        "custom_loss_function_path": None,
        "custom_model_provider_path": None,
        "custom_megatron_before_log_prob_hook_path": None,
        "custom_megatron_before_train_step_hook_path": None,
    }
    invalid = [name for name, expected in required.items() if getattr(args, name, None) != expected]
    if invalid:
        return False
    if args.rollout_batch_size * args.n_samples_per_prompt != args.global_batch_size:
        invalid.append("rollout size must equal global_batch_size")
    if len(num_microbatches) != 1:
        invalid.append("exactly one optimizer step per rollout is required")
    if getattr(args, "num_steps_per_rollout", None) not in (None, 1):
        invalid.append("num_steps_per_rollout")
    if args.moe_router_load_balancing_type not in ("none", ["none"]):
        invalid.append("moe_router_load_balancing_type")
    if set(backup_tags) - {"actor"}:
        invalid.append("reference, teacher, or old-actor weights")
    if rollout_data_postprocess is not None:
        invalid.append("rollout_data_postprocess")
    return not invalid
