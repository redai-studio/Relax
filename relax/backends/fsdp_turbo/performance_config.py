# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in AdamW execution options; model fusion adapters are independent."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PerformanceConfig:
    adamw_fused: bool = False
    moe_implementation: str = "native"
    token_chunk_size: int = 0
    gdn_implementation: str = "auto"
    causal_conv1d_implementation: str = "auto"
    gdn_grouped_qk: bool = False
    skip_gdn_recompute: bool | None = None

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
        if options.gdn_implementation not in ("auto", "triton", "ascendc"):
            raise ValueError("gdn_implementation must be one of ('auto', 'triton', 'ascendc')")
        if options.causal_conv1d_implementation not in ("auto", "triton", "ascendc"):
            raise ValueError("causal_conv1d_implementation must be one of ('auto', 'triton', 'ascendc')")
        if type(options.gdn_grouped_qk) is not bool:
            raise ValueError("gdn_grouped_qk must be a boolean")
        if options.skip_gdn_recompute is not None and type(options.skip_gdn_recompute) is not bool:
            raise ValueError("skip_gdn_recompute must be null or a boolean")
        if options.gdn_grouped_qk and options.gdn_implementation != "ascendc":
            raise ValueError("Grouped QK requires AscendC GDN")
        if options.moe_implementation not in ("native", "megamoe"):
            raise ValueError("moe_implementation must be one of ('native', 'megamoe')")
        if type(options.token_chunk_size) is not int or options.token_chunk_size < 0:
            raise ValueError("token_chunk_size must be a nonnegative integer")
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
