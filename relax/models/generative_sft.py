# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model contract for image-edit supervision, independent of RL
trajectories."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class ConditioningSpec:
    """Adapter-declared dependencies of one reusable frozen computation.

    ``dependencies`` contains JSON-compatible input/processing metadata. The
    runtime adds its immutable encoder-weight fingerprint before building the
    cache key. ``cacheable`` is checked against the actual parameter boundary,
    not just a training-mode name.
    """

    name: str
    dependencies: dict[str, Any]
    cacheable: bool


class ConditioningResolver(Protocol):
    """Resolve features inside the model's FSDP forward boundary.

    A distributed runtime must make the compute/skip choice unanimous across
    the parameter-sharding group unless its backend has validated the adapter's
    ``_conditioning_collective_free_modules`` declaration. That declaration
    promises every compute/restore callback is collective-free and uses only
    the listed frozen replica roots. The runtime agrees the execution policy
    once, never based on a rank-local cache hit or changing frozen-state guard.
    It must call ``restore(compute())`` without detaching or storing the result
    when ``spec.cacheable`` is false, because those features may carry
    conditioning gradients.
    """

    enabled: bool

    def resolve(
        self,
        spec: ConditioningSpec,
        compute: Callable[[], dict[str, torch.Tensor]],
        restore: Callable[[dict[str, torch.Tensor]], dict[str, Any]],
    ) -> dict[str, Any]: ...

    def record_input_transfer(self, nbytes: int) -> None:
        """Record condition-input H2D bytes from the online encoder path."""
        ...


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
