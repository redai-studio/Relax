# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict adapter reconstruction and base-preserving DCP overlay for K3
export."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any


_LANGUAGE_PREFIXES = ("decoder.", "embedding.", "output_layer.")


def normalize_key(key: str) -> str:
    """Compare the language-only Bridge model with Relax's multimodal keys."""
    return key.removeprefix("language_model.")


def _is_language(key: str) -> bool:
    return key.startswith("language_model.") or key.startswith(_LANGUAGE_PREFIXES)


def _union_keys(keys: set[str]) -> set[str]:
    import torch.distributed as dist

    if not dist.is_initialized():
        return keys
    group = dist.group.WORLD
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, keys, group=group)
    return set().union(*gathered)


def build_spec(input_dir: str, patch: Any) -> dict[str, Any] | None:
    metadata = copy.copy(patch._read_checkpoint_metadata(input_dir))
    # Model-space optimizer tensors reuse the model's adapter suffixes.
    # Exclude them before both namespace validation and Bridge reconstruction,
    # without modifying the metadata used to load the actual checkpoint.
    metadata.state_dict_metadata = {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if not re.match(r"^(?:chained_\d+\.)?optimizer(?:\.|$)", key)
    }
    unknown = [
        key
        for key, value in metadata.state_dict_metadata.items()
        if hasattr(value, "size")
        and not _is_language(key)
        and not key.startswith(("vision_tower.", "mm_projector.", "optimizer.", "rng_state."))
    ]
    if unknown:
        raise ValueError(f"Unsupported saved tensor namespaces in K3 LoRA checkpoint: {unknown[:5]}")
    adapter_keys = [key for key in metadata.state_dict_metadata if ".adapter." in key]
    unsupported = [key for key in adapter_keys if not _is_language(key)]
    if unsupported:
        raise ValueError(f"K3 LoRA export currently supports language adapters only: {unsupported[:5]}")
    spec = patch._build_lora_checkpoint_spec(metadata)
    if adapter_keys and spec is None:
        raise ValueError("Adapter metadata exists but contains no supported LoRA tensor pairs")
    return spec


def configure_patch(patch: Any, spec: dict[str, Any]) -> None:
    """Reuse Bridge LoRA reconstruction, normalizing names and global
    coverage."""
    normalized = dict(spec)
    normalized["target_modules"] = [normalize_key(key) for key in spec["target_modules"]]
    normalized["adapter_keys"] = {normalize_key(key) for key in spec["adapter_keys"]}
    original_assert = patch._assert_adapter_coverage

    def assert_coverage(expected: Iterable[str], actual: Iterable[str]) -> None:
        original_assert(
            {normalize_key(key) for key in expected},
            _union_keys({normalize_key(key) for key in actual}),
        )

    patch._assert_adapter_coverage = assert_coverage
    patch._lora_checkpoint_spec = normalized


def prepare_overlay(state: dict[str, Any], metadata_keys: set[str]) -> dict[str, Any]:
    """Load saved tensors and retain initialized base tensors under strict
    load.

    Factories stay factories: their expanded children are rewritten too, so
    factory merge functions still restore fused model parameters after loading.
    Unexpected language tensors are checked across all export ranks.
    """
    from megatron.core.dist_checkpointing.mapping import (
        LocalNonpersistentObject,
        ShardedObject,
        ShardedTensor,
        ShardedTensorFactory,
    )

    used: set[str] = set()
    missing_adapters: set[str] = set()

    def resolve(key: str) -> str | None:
        plain = normalize_key(key)
        candidates = [key, "language_model." + plain, plain]
        if ".adapter." not in key:
            candidates += [candidate.replace(".to_wrap.", ".") for candidate in candidates]
        matches = set(candidates) & metadata_keys
        if len(matches) > 1:
            raise ValueError(f"Ambiguous checkpoint aliases for {key}: {sorted(matches)}")
        return next(iter(matches)) if matches else None

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if isinstance(value, ShardedTensorFactory):
            # Inspect expanded names, which may differ from the factory key.
            original_build = value.build_fn
            expanded = rewrite(value.build())
            if all_local(expanded):
                return LocalNonpersistentObject(value.data)

            def build(key: str, data: Any, replica_id: Any, flattened_range: Any) -> Any:
                return rewrite(original_build(key, data, replica_id, flattened_range))

            value.build_fn = build
            return value
        if isinstance(value, (ShardedTensor, ShardedObject)):
            resolved = resolve(value.key)
            if resolved is not None:
                value.key = resolved
                used.add(resolved)
                return value
            if ".adapter." in value.key:
                missing_adapters.add(value.key)
            if isinstance(value, ShardedTensor) and value.data is None:
                raise ValueError(f"Missing checkpoint tensor has no initialized base value: {value.key}")
            return LocalNonpersistentObject(value.data)
        return value

    def all_local(value: Any) -> bool:
        if isinstance(value, dict):
            return all(all_local(child) for child in value.values())
        if isinstance(value, list):
            return all(all_local(child) for child in value)
        return isinstance(value, LocalNonpersistentObject)

    result = rewrite(state)
    missing = _union_keys(missing_adapters)
    if missing:
        raise ValueError(f"Missing reconstructed adapter tensors: {sorted(missing)[:5]}")
    global_used = _union_keys(used)
    unexpected = {key for key in metadata_keys if _is_language(key)} - global_used
    if unexpected:
        raise ValueError(f"Saved language tensors not covered by model: {sorted(unexpected)[:5]}")
    return result
