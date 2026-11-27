# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in AdamW execution options; model fusion adapters are independent."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PerformanceConfig:
    adamw_fused: bool = False
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
        if options.qhc_implementation not in ("native", "triton"):
            raise ValueError("qhc_implementation must be one of ('native', 'triton')")
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
