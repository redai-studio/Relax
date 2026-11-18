# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Utilities for quantized checkpoint casting and metadata handling."""

import json
import os
from argparse import Namespace
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def resolve_receiver_quantization_config(
    args: Namespace, hf_config: Any, recipients: list[tuple[str, str, str | None, str | None]]
) -> dict | None:
    """Resolve a compatible sender format for already selected receivers."""
    from relax.utils.model_source import is_model_source_alias

    configs = []
    checkpoint_keys = {}
    for name, model_path, override_json, quantization in recipients:
        # SGLang can load the original URI, but this process already owns the
        # local copy needed for metadata and compressed-tensor shard scans.
        if is_model_source_alias(args, model_path):
            model_path = args.hf_checkpoint
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config_values = json.load(f)
        elif model_path == args.hf_checkpoint:
            config_values = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        else:
            from transformers import PretrainedConfig

            config_values, _ = PretrainedConfig.get_config_dict(model_path)
        # SGLang updates an existing text sub-config field by field, but
        # replaces top-level quantization dictionaries and explicit None.
        overrides = json.loads(override_json or "{}")
        if isinstance(overrides.get("text_config"), dict) and isinstance(config_values.get("text_config"), dict):
            overrides["text_config"] = {**config_values["text_config"], **overrides["text_config"]}
        config_values = {**config_values, **overrides}
        quant_config = _get_checkpoint_quantization_config(config_values) or None
        # None here is the effective format, not permission to reread the
        # original checkpoint and undo an explicit override.
        if quant_config:
            quant_config = dict(quant_config)
            if quantization is not None:
                quant_config.setdefault("quant_method", quantization)
            if quant_config.get("quant_method") == "compressed-tensors":
                local_path = os.path.realpath(model_path)
                if local_path not in checkpoint_keys:
                    checkpoint_keys[local_path] = _list_all_keys(model_path)
                quant_config = augment_compressed_tensors_ignore(
                    quant_config, model_path, checkpoint_keys=checkpoint_keys[local_path]
                )
        configs.append((name, quant_config))
    if not configs:
        raise ValueError("Weight publication has no rollout receiver groups")
    first_name, first_config = configs[0]
    for name, quant_config in configs[1:]:
        if _normalize_quantization_format(quant_config) != _normalize_quantization_format(first_config):
            raise ValueError(f"Rollout receivers require different quantization configs: {first_name} and {name}")
    return first_config


def _get_checkpoint_quantization_config(config: dict) -> dict | None:
    quant_config = config.get("quantization_config")
    if quant_config is None:
        text_config = config.get("text_config") or {}
        quant_config = text_config.get("quantization_config")
    if quant_config is None:
        quant_config = config.get("compression_config")
    return quant_config


def _normalize_quantization_format(config: dict | None) -> dict | None:
    if not config:
        return None
    normalized = dict(config)
    if normalized.get("quant_method") == "fp8":
        normalized.setdefault("fmt", "e4m3")
        normalized.setdefault("weight_block_size", None)
        normalized.setdefault("scale_fmt", None)
        normalized["modules_to_not_convert"] = sorted(set(normalized.get("modules_to_not_convert") or []))
    return normalized


def read_quantization_config(hf_dir: str | os.PathLike) -> dict | None:
    """Read receiver quantization metadata, including compression fallback.

    VLM checkpoints like Kimi K3 nest the compressed-tensors config under the
    text backbone, leaving the top-level block null — so callers that read
    ``hf_config.quantization_config`` see ``None`` even though the release is
    quantized.
    """
    config_path = os.path.join(hf_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    return _get_checkpoint_quantization_config(cfg)


def derive_extra_ignore_namespaces(
    hf_dir: str | os.PathLike, *, checkpoint_keys: list[str] | None = None
) -> list[str]:
    """Return top-level module names whose subtree has plain ``.weight`` keys
    but zero ``.weight_packed`` triplets in the source checkpoint.

    Used to augment ``quantization_config["ignore"]`` at runtime when the
    checkpoint's own ignore list misses namespaces that aren't quantized.
    K2.6 INT4 release omits ``vision_tower`` / ``mm_projector`` from its
    ignore list; without this augmentation the bridge tries to INT4-repack
    those plain ``.weight`` tensors and SGLang errors with
    ``ValueError: Weight X.weight_packed not found in params_dict``.

    Returns sorted top-level namespaces. Reads safetensors headers only
    (no tensor data), so a 64-shard scan takes ~1-3s.
    """
    from collections import defaultdict

    namespaces: dict[str, dict[str, bool]] = defaultdict(lambda: {"plain": False, "packed": False})
    if checkpoint_keys is None:
        checkpoint_keys = _list_all_keys(hf_dir)
    for key in checkpoint_keys:
        top = key.split(".", 1)[0]
        if key.endswith(".weight_packed"):
            namespaces[top]["packed"] = True
        elif key.endswith(".weight"):
            namespaces[top]["plain"] = True
    return sorted(top for top, info in namespaces.items() if info["plain"] and not info["packed"])


def augment_compressed_tensors_ignore(
    quantization_config: dict | None, hf_dir: str | os.PathLike, *, checkpoint_keys: list[str] | None = None
) -> dict | None:
    """Return a copy of ``quantization_config`` whose ``ignore`` list is
    augmented with any source-checkpoint top-level namespaces that have no
    ``.weight_packed`` triplets (i.e. they aren't actually quantized).

    For K2.6-style models where only expert MLP weights are quantized, also
    add patterns to ignore non-expert gate weights (e.g., model.layers.*.mlp.gate),
    embedding layers, and head layers.

    Pass-through if ``quantization_config`` is None or not compressed-tensors.
    When it is None (Kimi K3-style VLM nesting), resolve it from config.json
    first.
    """
    if quantization_config is None:
        quantization_config = read_quantization_config(hf_dir)
    if not quantization_config or quantization_config.get("quant_method") != "compressed-tensors":
        return quantization_config
    if checkpoint_keys is None:
        checkpoint_keys = _list_all_keys(hf_dir)
    extra = derive_extra_ignore_namespaces(hf_dir, checkpoint_keys=checkpoint_keys)

    existing = list(quantization_config.get("ignore", []))
    added = []

    # Add auto-derived namespaces
    for ns in extra:
        pattern = f"re:.*{ns}.*"
        if pattern not in existing and ns not in existing:
            existing.append(pattern)
            added.append(pattern)

    # For K2.6-style MoE models: ignore non-expert gate weights (only quantize expert gates)
    # Pattern: any weight path with .mlp.gate (non-expert) should be ignored
    # Experts are quantized, non-expert regular MLPs should not be re-quantized
    moe_gate_pattern = "re:.*\\.mlp\\.gate\\.weight$"
    if moe_gate_pattern not in existing:
        # Check if there are expert weights being quantized
        has_expert_quantized = any(".mlp.experts." in k for k in checkpoint_keys if k.endswith(".weight_packed"))
        if has_expert_quantized:
            existing.append(moe_gate_pattern)
            added.append(moe_gate_pattern)

    # For VLM models: ignore embedding, output layers, and vision components
    # These layers are often left in original precision (embed_tokens, lm_head, vision_tower, mm_projector)
    unquantized_layers = ["embed_tokens", "lm_head", "vision_tower", "mm_projector"]
    all_keys = checkpoint_keys
    for layer_name in unquantized_layers:
        pattern = f"re:.*\\.{layer_name}\\.weight$"
        if pattern not in existing:
            # Check if this layer exists in the checkpoint but is NOT quantized
            has_plain_weight = any(f".{layer_name}.weight" in k for k in all_keys)
            has_packed_weight = any(f".{layer_name}.weight_packed" in k for k in all_keys)
            if has_plain_weight and not has_packed_weight:
                existing.append(pattern)
                added.append(pattern)

    if not added:
        return quantization_config
    logger.info(f"augmented quantization_config.ignore with auto-derived namespaces: {added}")
    return {**quantization_config, "ignore": existing}


def _list_all_keys(hf_dir: str | os.PathLike) -> list[str]:
    """Return all parameter keys from all safetensors files in hf_dir."""
    from safetensors import safe_open

    all_keys = []
    for fname in os.listdir(hf_dir):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(hf_dir, fname), framework="pt") as reader:
            all_keys.extend(reader.keys())
    return all_keys
