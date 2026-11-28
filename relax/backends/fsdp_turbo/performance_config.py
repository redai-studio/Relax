# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in AdamW execution options; model fusion adapters are independent."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class PerformanceConfig:
    adamw_fused: bool = False
    qsa_indexer_implementation: str = "triton"
    qsa_id2mask_implementation: str = "eager"
    attention_implementation: str = "native"
    skip_flash_attn_recompute: bool = False
    qsa_triton_validate_prefix_mask: bool = True
    qsa_triton_query_chunk_size: int = 0

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
        if options.qsa_indexer_implementation not in ("triton", "ascendc"):
            raise ValueError("qsa_indexer_implementation must be one of ('triton', 'ascendc')")
        if options.qsa_id2mask_implementation not in ("eager", "triton"):
            raise ValueError("qsa_id2mask_implementation must be one of ('eager', 'triton')")
        if options.attention_implementation not in ("native", "npu", "ascendc_grouped"):
            raise ValueError("attention_implementation must be one of ('native', 'npu', 'ascendc_grouped')")
        if type(options.skip_flash_attn_recompute) is not bool:
            raise ValueError("skip_flash_attn_recompute must be a boolean")
        if type(options.qsa_triton_validate_prefix_mask) is not bool:
            raise ValueError("qsa_triton_validate_prefix_mask must be a boolean")
        if type(options.qsa_triton_query_chunk_size) is not int or options.qsa_triton_query_chunk_size < 0:
            raise ValueError("qsa_triton_query_chunk_size must be a nonnegative integer")

        if options.qsa_id2mask_implementation == "triton" and options.qsa_indexer_implementation != "ascendc":
            raise ValueError("Triton id2mask requires the AscendC LightningIndexer")
        if options.skip_flash_attn_recompute and options.attention_implementation != "npu":
            raise ValueError("Selective attention recompute requires NPU attention")
        if options.attention_implementation == "ascendc_grouped" and options.qsa_indexer_implementation != "ascendc":
            raise ValueError("Grouped QSA requires the AscendC LightningIndexer")
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
