# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Resolve training model routes from the selected provider's capabilities."""

from pathlib import Path


def resolve_is_vl_model(provider: object | None, checkpoint: str | None) -> bool:
    """Prefer an explicit provider declaration over checkpoint metadata.

    ``relax_is_multimodal`` describes the selected model, independent of the
    current batch's contents. Providers without a declaration retain the
    processor-config heuristic, including custom and raw model providers.
    """
    declared = getattr(provider, "relax_is_multimodal", None)
    if declared is not None:
        if not isinstance(declared, bool):
            raise TypeError("provider.relax_is_multimodal must be bool or None")
        return declared
    if checkpoint is None:
        return False
    return any((Path(checkpoint) / name).exists() for name in ("processor_config.json", "preprocessor_config.json"))
