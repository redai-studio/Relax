# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compatibility for Relax consumers of native Megatron configuration."""

from inspect import signature
from typing import Any, Callable, Collection, Optional


def supports_dynamic_context_parallel(initializer: Callable[..., Any]) -> bool:
    """Check whether native Megatron accepts dynamic CP."""
    return "dynamic_context_parallel" in signature(initializer).parameters


def dynamic_context_parallel_enabled(args: Any) -> bool:
    """Older Megatron builds omit this switch and cannot enable dynamic CP."""
    return bool(getattr(args, "dynamic_context_parallel", False))


def get_context_parallel_size(args: Any) -> int:
    """Non-Megatron parsers may omit the SDK's context parallel size."""
    return getattr(args, "context_parallel_size", 1)


def get_recompute_layout(args: Any) -> tuple[str | None, int | None]:
    """Read native profiling settings, allowing the caller to diagnose
    omissions."""
    return getattr(args, "recompute_granularity", None), getattr(args, "recompute_num_layers", None)


def resolve_kv_channels(args: Any) -> int:
    """Recover the native head width from older conversion inputs that omit
    it."""
    kv_channels = getattr(args, "kv_channels", None)
    if kv_channels is None:
        return args.hidden_size // args.num_attention_heads
    return kv_channels


def legacy_te_op_fuser_enabled(config: Any) -> bool:
    """The supported core_v0.16.1 schema has no TE op-fuser option."""
    return bool(getattr(config, "use_transformer_engine_op_fuser", False))


def legacy_single_grouped_weight_enabled(config: Any) -> bool:
    """The supported core_v0.16.1 schema lacks single grouped weights."""
    return bool(getattr(config, "moe_single_grouped_weight", False))


def legacy_cuda_graphs_enabled(config: Any) -> bool:
    """Read the legacy switch before native translation and removal."""
    return bool(getattr(config, "enable_cuda_graph", False))


def get_profile_ranks(args: Any) -> Optional[Collection[int]]:
    """Return the SDK rank filter when the backend declares one."""
    return getattr(args, "profile_ranks", None)


def normalize_gloo_process_groups(args: Any) -> None:
    """Map the core_v0.16.1 field to the current Megatron name."""
    if not hasattr(args, "use_gloo_process_groups"):
        args.use_gloo_process_groups = getattr(args, "enable_gloo_process_groups", False)
