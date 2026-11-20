# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dispatch model-specific LoRA setup at its two lifecycle boundaries."""

from argparse import Namespace
from collections.abc import Sequence

import torch


def _uses_hc_ple_lora(model: torch.nn.Module) -> bool:
    # Shared configuration also identifies pipeline stages without HC/PLE owners.
    return getattr(getattr(model, "config", None), "mhc_variant", None) == "gated_residual"


def resolve_lora_targets(model: torch.nn.Module, targets: list[str], args: Namespace) -> list[str]:
    """Resolve model-specific targets before scope/freeze filtering and
    injection."""
    if _uses_hc_ple_lora(model):
        from relax.models.qwen3_8.lora import qwen38_lora_targets

        return qwen38_lora_targets(model, targets, args)
    return targets


def finalize_lora_models(models: Sequence[torch.nn.Module], args: Namespace) -> None:
    """Validate and initialize LoRA after freezing/DDP, before checkpoint
    restore.

    Call once per rank with all unwrapped local VP chunks. Dispatch relies on
    consistent model configuration across PP stages, including empty stages;
    model-specific collectives must not be invoked separately for each chunk.
    """
    if any(_uses_hc_ple_lora(model) for model in models):
        from relax.models.qwen3_8.lora import validate_qwen38_lora

        validate_qwen38_lora(models, args)
