# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Configuration boundary for the opt-in ordinary text SFT scheduler."""

from argparse import Namespace
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from relax.utils.training.flops_counter import SampleFlopsCost


def validate_flops_balancing(args: Namespace, *, check_prepack: bool = True) -> None:
    """Reject independent schedulers without changing existing
    configuration."""
    if not getattr(args, "balance_by_flops", False):
        return
    prefix = "--balance-by-flops"
    if getattr(args, "loss_type", None) != "sft":
        raise ValueError(f"{prefix} requires ordinary text SFT (--loss-type sft).")
    if getattr(args, "train_backend", None) != "megatron":
        raise ValueError(f"{prefix} requires the Megatron backend.")
    if not args.use_dynamic_batch_size:
        raise ValueError(f"{prefix} requires --use-dynamic-batch-size.")
    if getattr(args, "task_type", "causal_lm") != "causal_lm":
        raise ValueError(f"{prefix} requires causal-LM SFT.")
    if args.n_samples_per_prompt != 1:
        raise ValueError(f"{prefix} requires --n-samples-per-prompt 1.")
    if getattr(args, "multimodal_keys", None) or getattr(args, "generation_task", None) == "i2i":
        raise ValueError(f"{prefix} does not support multimodal inputs.")
    for name in (
        "fully_async",
        "hybrid",
        "sft_wholepack",
        "dynamic_context_parallel",
        "enable_mtp_training",
        "mtp_only_training",
        "use_routing_replay",
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "freeze_language_model",
        "freeze_params_name_list",
        "only_train_params_name_list",
    ):
        if getattr(args, name, False):
            raise ValueError(f"{prefix} does not support {name}.")
    if getattr(args, "lora_rank", 0):
        raise ValueError(f"{prefix} does not yet estimate LoRA training workloads.")
    aux_coeff = getattr(args, "moe_aux_loss_coeff", None)
    aux_coeffs = aux_coeff if isinstance(aux_coeff, (list, tuple)) else (aux_coeff,)
    if any(coeff is not None and coeff != 0 for coeff in aux_coeffs):
        raise ValueError(
            f"{prefix} requires zero --moe-aux-loss-coeff; auxiliary loss depends on microbatch grouping."
        )
    if getattr(args, "moe_z_loss_coeff", None) not in (None, 0):
        raise ValueError(f"{prefix} requires zero --moe-z-loss-coeff to preserve dummy zero-gradient semantics.")
    if getattr(args, "moe_expert_capacity_factor", None) is not None:
        raise ValueError(
            f"{prefix} requires unlimited MoE expert capacity; token dropping depends on microbatch grouping."
        )
    router_type = getattr(args, "moe_router_load_balancing_type", None)
    router_types = router_type if isinstance(router_type, (list, tuple)) else (router_type,)
    if any(kind in ("sinkhorn", "quantile_balancing") for kind in router_types):
        raise ValueError(f"{prefix} does not support microbatch-dependent sinkhorn/quantile_balancing routing.")
    if (
        getattr(args, "moe_router_enable_expert_bias", False)
        and getattr(args, "moe_router_bias_update_rate", 1e-3) != 0
    ):
        raise ValueError(f"{prefix} requires disabled expert-bias updates to avoid counting dummy tokens.")
    if check_prepack and getattr(args, "sft_async_prepack", False):
        raise ValueError(
            f"{prefix} requires the ordinary SFT data path; automatic async prepack is enabled. "
            "Select it explicitly with --sft-max-in-flight-steps 1."
        )


def get_sample_flops_cost(args: Namespace) -> "SampleFlopsCost":
    """Build model constants once per configuration, without loading
    weights."""
    cost = getattr(args, "_sample_flops_cost", None)
    if cost is None:
        from relax.utils.misc import get_hf_config
        from relax.utils.training.flops_counter import SampleFlopsCost

        cost = SampleFlopsCost.from_config(get_hf_config(args.hf_checkpoint))
        args._sample_flops_cost = cost
    return cost


def get_flops_capacity_settings(args: Namespace) -> tuple[int, int]:
    """Controller guard settings for both possible worker CP document
    layouts."""
    from relax.utils.data.packing import get_microbatch_token_capacity

    cp_size = getattr(args, "context_parallel_size", 1)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    capacity = get_microbatch_token_capacity(
        args.max_tokens_per_gpu,
        cp_size,
        tp_size=tp_size,
        allgather_cp=getattr(args, "allgather_cp", False),
        data_pad_size_multiplier=getattr(args, "data_pad_size_multiplier", 1),
    )
    alignment = tp_size * cp_size * 2 if cp_size > 1 and getattr(args, "qkv_format", "thd") == "thd" else 1
    return capacity, alignment
