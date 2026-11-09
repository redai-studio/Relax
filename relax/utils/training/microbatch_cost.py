# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Producer-side CPU costs for global FLOPs scheduling."""

import math
from argparse import Namespace
from typing import Any


def _validate_flops_configuration(args: Namespace) -> None:
    if getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("FLOPs balancing requires --train-backend megatron")
    if not getattr(args, "use_dynamic_batch_size", False):
        raise ValueError("FLOPs balancing requires --use-dynamic-batch-size")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("FLOPs scheduling requires THD packed sequences")
    if (
        (getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) != 1
        or getattr(args, "num_layers_per_virtual_pipeline_stage", None) is not None
        or getattr(args, "num_virtual_stages_per_pipeline_rank", None) not in (None, 1)
        or getattr(args, "dynamic_context_parallel", False)
    ):
        raise ValueError("FLOPs queue integration currently requires VPP1 and static context parallelism")
    if not getattr(args, "polling_mode", True):
        raise ValueError("FLOPs variable-row DP assignment requires TransferQueue polling mode")
    if getattr(args, "debug_train_only", False):
        raise ValueError("FLOPs global scheduling requires TransferQueue; --debug-train-only bypasses its planner")
    if getattr(args, "sft_packing_strategy", "balanced") != "balanced":
        raise ValueError("FLOPs balancing requires balanced packing, not binpack")
    if getattr(args, "sft_wholepack", False):
        raise ValueError("FLOPs balancing does not support wholepack")
    if getattr(args, "fully_async", False) or getattr(args, "hybrid", False):
        raise ValueError("FLOPs balancing does not support fully-async streaming/drain scheduling")
    if getattr(args, "advantage_estimator", None) == "m2po":
        raise ValueError("FLOPs balancing does not support microbatch-dependent M2PO clipping")
    if getattr(args, "loss_type", None) in ("dpo", "rm"):
        raise ValueError("FLOPs balancing does not support preference pairs")
    auxiliary = getattr(args, "moe_aux_loss_coeff", 0.0)
    auxiliary = auxiliary if isinstance(auxiliary, (list, tuple)) else [auxiliary]
    if any(value is not None and value != 0 for value in auxiliary):
        raise ValueError("FLOPs balancing requires zero MoE auxiliary-loss coefficients to preserve the objective")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError("FLOPs balancing does not support microbatch-dependent MoE token dropping")
    routing = getattr(args, "moe_router_load_balancing_type", "none")
    routing = routing if isinstance(routing, (list, tuple)) else [routing]
    if any(mode not in ("none", "aux_loss", "seq_aux_loss", "global_aux_loss") for mode in routing):
        raise ValueError("FLOPs balancing requires token-independent MoE routing; batch-coupled modes are unsupported")


def validate_flops_balance(args: Namespace) -> bool:
    """Gate both scheduling layers; OFF bypasses every FLOPs restriction."""
    enabled = getattr(args, "balance_by_flops", False)
    if not isinstance(enabled, bool):
        raise ValueError("--balance-by-flops must be a boolean")
    if not enabled:
        return False
    _validate_flops_configuration(args)
    if getattr(args, "loss_type", None) == "sft":
        from relax.engine.sft.runtime import sft_tq_num_shards

        if sft_tq_num_shards(args) != 1:
            raise ValueError("FLOPs global-step scheduling requires one SFT TransferQueue shard")
        if getattr(args, "n_samples_per_prompt", 1) != 1:
            raise ValueError("FLOPs DP balancing for SFT requires n_samples_per_prompt=1")
        if getattr(args, "sft_image_preprocess_on_rank", False):
            raise ValueError("FLOPs DP balancing requires CPU media metadata before queue publication")
    return True


def sample_costs(args: Namespace, rollout_data: dict[str, Any]) -> list[float] | None:
    """Estimate each sample once, before publication to the queue."""
    if not validate_flops_balance(args):
        return None
    from relax.utils.misc import get_hf_config
    from relax.utils.training.qwen35_cost import estimate_sample_costs

    config = get_hf_config(args.hf_checkpoint)
    if config.model_type not in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError(f"FLOPs scheduling does not support model_type={config.model_type!r}")
    costs = [float(value) for value in estimate_sample_costs(config, rollout_data, args)]
    if len(costs) != len(rollout_data["total_lengths"]):
        raise ValueError("Sample cost count must match total_lengths")
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        raise ValueError("Sample costs must be positive and finite")
    return costs


def add_flops_cost_metadata(
    args: Namespace | None, batch: dict[str, Any], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach CPU costs before payloads enter the queue's sampling layer."""
    costs = sample_costs(args, batch) if args is not None else None
    if costs is None:
        return metadata
    return [{**row, "flops_cost": cost} for row, cost in zip(metadata, costs, strict=True)]
