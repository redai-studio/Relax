# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Utilities for quantized checkpoint casting and metadata handling."""

import json
import os

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def read_quantization_config(hf_dir: str | os.PathLike) -> dict | None:
    """Read ``quantization_config`` from config.json, falling back to
    ``text_config``.

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
    return cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or None


def derive_extra_ignore_namespaces(hf_dir: str | os.PathLike) -> list[str]:
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

    from safetensors import safe_open

    namespaces: dict[str, dict[str, bool]] = defaultdict(lambda: {"plain": False, "packed": False})
    for fname in os.listdir(hf_dir):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(hf_dir, fname), framework="pt") as reader:
            for key in reader.keys():
                top = key.split(".", 1)[0]
                if key.endswith(".weight_packed"):
                    namespaces[top]["packed"] = True
                elif key.endswith(".weight"):
                    namespaces[top]["plain"] = True
    return sorted(top for top, info in namespaces.items() if info["plain"] and not info["packed"])


def augment_compressed_tensors_ignore(quantization_config: dict | None, hf_dir: str | os.PathLike) -> dict | None:
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
    extra = derive_extra_ignore_namespaces(hf_dir)

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
        has_expert_quantized = any(
            ".mlp.experts." in k for k in _list_all_keys(hf_dir) if k.endswith(".weight_packed")
        )
        if has_expert_quantized:
            existing.append(moe_gate_pattern)
            added.append(moe_gate_pattern)

    # For VLM models: ignore embedding, output layers, and vision components
    # These layers are often left in original precision (embed_tokens, lm_head, vision_tower, mm_projector)
    unquantized_layers = ["embed_tokens", "lm_head", "vision_tower", "mm_projector"]
    all_keys = _list_all_keys(hf_dir)
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
        try:
            with safe_open(os.path.join(hf_dir, fname), framework="pt") as reader:
                all_keys.extend(reader.keys())
        except Exception:
            pass
    return all_keys
