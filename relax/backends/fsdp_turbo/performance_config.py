# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in AdamW execution options; model fusion adapters are independent."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PerformanceConfig:
    adamw_fused: bool = False
    rmsnorm_implementation: str = "native"
    grouped_rmsnorm_implementation: str = "native"
    qhc_implementation: str = "native"

    @classmethod
    def from_dict(cls, payload):
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("relax_performance must be a mapping")
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown relax_performance options: {sorted(unknown)}")
        options = cls(**payload)
        if type(options.adamw_fused) is not bool:
            raise ValueError("adamw_fused must be a boolean")
        if options.rmsnorm_implementation not in ("native", "triton"):
            raise ValueError("rmsnorm_implementation must be one of ('native', 'triton')")
        if options.grouped_rmsnorm_implementation not in ("native", "triton"):
            raise ValueError("grouped_rmsnorm_implementation must be one of ('native', 'triton')")
        if options.qhc_implementation not in ("native", "ascendc"):
            raise ValueError("qhc_implementation must be one of ('native', 'ascendc')")
        return options


def optimizer_options(config):
    options = config.relax_performance
    if not options.adamw_fused:
        return {}
    if config.optimizer.optimizer_type.lower() != "adamw":
        raise ValueError("relax_performance optimizer options require AdamW")
    # Torch 2.10 supports fused AdamW on CPU as well as accelerator devices.
    # Keep this opt-in: the old Torch 2.9 NPU environment is not validated.
    return {"fused": True}


def validate_model_options(options, *, is_qwen4_exp: bool) -> None:
    """Reject model-specific selections instead of silently running native
    code."""
    if is_qwen4_exp:
        return
    defaults = type(options)()
    selected = [
        field.name
        for field in fields(options)
        if field.name != "adamw_fused" and getattr(options, field.name) != getattr(defaults, field.name)
    ]
    if selected:
        raise ValueError(f"Qwen4-Exp model required for relax_performance options: {', '.join(selected)}")
