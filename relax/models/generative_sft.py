# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model contract for image-edit supervision, independent of RL
trajectories."""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from relax.engine.sft.condition_cache import ConditionFieldSpec, ConditionRequest


class GenerativeSFTAdapter(Protocol):
    family: str
    supported_tasks: tuple[str, ...]
    lora_target_modules: tuple[str, ...]

    def load_train_model(self, args: Any) -> torch.nn.Module:
        """Return the complete forward root: model(sample) yields scalar
        loss."""
        ...

    def prepare_for_sharding(self, model: torch.nn.Module) -> None:
        """Finalize model-specific precision after optional LoRA injection."""
        ...

    def on_device_ready(self, model: torch.nn.Module, device: torch.device) -> None:
        """Update non-parameter device metadata after FSDP wrapping."""
        ...

    def prepare_sample(self, row: dict[str, Any], root: Path, config: dict[str, Any], index: int = 0) -> Any:
        """Load ordered conditions, target and prompt for the forward root.

        ``index`` is the dataset row id; adapters may use it to derive a per-
        sample deterministic random stream (e.g. fixed-pixel tier sampling)
        that survives checkpoint resume.
        """
        ...


@runtime_checkable
class ConditionCacheAdapter(Protocol):
    """Adapter methods required to cache frozen condition outputs.

    With ``condition_cache_enabled``, the SFT trainer rejects adapters that do
    not implement this protocol. The model forward reads reused values from,
    and inserts online outputs into, the binding passed to
    ``bind_condition_cache``.
    """

    def condition_field_specs(self, model: torch.nn.Module) -> tuple[ConditionFieldSpec, ...]:
        """Declare the cacheable condition fields and their dependencies."""
        ...

    def prepare_condition_request(self, model: torch.nn.Module, data: Any) -> dict[str, ConditionRequest]:
        """Build one request per declared field for a prepared sample."""
        ...

    def bind_condition_cache(self, model: torch.nn.Module, cache: Any | None) -> None:
        """Attach the cache binding to ``model``; ``None`` detaches it."""
        ...
