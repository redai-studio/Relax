# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare model settings after the same native Transformers normalization."""

from .store import canonical, digest


def configured_experts(config: dict) -> int | None:
    """Read original and Transformers-normalized Qwen expert counts."""
    text = config.get("text_config", config)
    values = [text[name] for name in ("num_experts", "num_local_experts") if name in text]
    if not values:
        return None
    if any(type(value) is not int or value < 1 for value in values) or len(set(values)) != 1:
        raise ValueError("Invalid or conflicting expert counts in model configuration")
    return values[0]


def unpacked_expert_schema(schema: dict, config: dict) -> dict:
    """Express published Qwen3-VL packed experts as individual HF matrices.

    Only the inventory changes here. Installation still checks every live
    worker's shape, dtype and bytes against the supplied checkpoint.
    """
    if config.get("model_type") != "qwen3_vl_moe":
        return dict(schema)
    text = config.get("text_config", config)
    result = {}
    for name, (shape, dtype) in schema.items():
        shape = tuple(shape)
        suffix = next((part for part in ("gate_up_proj", "down_proj") if name.endswith(".experts." + part)), None)
        if suffix is None:
            entries = {name: (shape, dtype)}
        else:
            experts = configured_experts(text)
            if len(shape) != 3 or type(experts) is not int or shape[0] != experts or min(shape) < 1:
                raise ValueError(f"Invalid packed expert inventory: {name}, {shape}")
            hidden, intermediate = text.get("hidden_size"), text.get("moe_intermediate_size")
            if hidden is not None and intermediate is not None:
                expected = (hidden, 2 * intermediate) if suffix == "gate_up_proj" else (intermediate, hidden)
                if shape[1:] != expected:
                    raise ValueError(f"Packed expert shape differs from model configuration: {name}, {shape}")
            prefix = name.rsplit(".", 1)[0]
            entries = {}
            if suffix == "gate_up_proj":
                if shape[2] % 2:
                    raise ValueError(f"Invalid packed gate/up dimension: {name}")
                for expert in range(experts):
                    for part in ("gate_proj", "up_proj"):
                        entries[f"{prefix}.{expert}.{part}.weight"] = ((shape[2] // 2, shape[1]), dtype)
            else:
                for expert in range(experts):
                    entries[f"{prefix}.{expert}.down_proj.weight"] = ((shape[2], shape[1]), dtype)
        if result.keys() & entries.keys():
            raise ValueError("Overlapping packed and individual expert inventory")
        result.update(entries)
    return result


def configuration_signature(config: dict) -> str:
    from transformers import AutoConfig

    values = dict(config)
    model_type = values.pop("model_type")
    if model_type not in ("qwen3", "qwen3_moe", "qwen3_vl", "qwen3_vl_moe"):
        raise ValueError("Register a configuration adapter for this model type")
    normalized = AutoConfig.for_model(model_type, **values).to_dict()

    def settings(value):
        if isinstance(value, dict):
            return {
                str(key): settings(item)
                for key, item in value.items()
                if key not in ("_name_or_path", "_commit_hash", "transformers_version")
            }
        if isinstance(value, list):
            return [settings(item) for item in value]
        return value

    return digest(canonical(settings(normalized)))
