# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Metadata checks used when selecting a checkpoint resume path."""

import re
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

from relax.utils.distributed_utils import get_gloo_group


def _collective_checkpoint_probe(probe: Callable[[], Any]) -> Any:
    """Read local checkpoint metadata, then agree before collective loading.

    All ranks enter this function even when a mount or metadata read fails.
    Inconsistent views fail together instead of selecting different loaders.
    """
    import torch.distributed as dist

    try:
        result = (None, probe())
    except Exception as exc:
        result = (f"{type(exc).__name__}: {exc}", None)
    results = [result]
    if dist.is_initialized():
        group = get_gloo_group()
        results = [None] * dist.get_world_size(group)
        dist.all_gather_object(results, result, group=group)
    errors = [error for error, _ in results if error is not None]
    if errors:
        raise RuntimeError(f"Checkpoint metadata probe failed: {errors}")
    if any(value != results[0][1] for _, value in results[1:]):
        raise RuntimeError("Checkpoint metadata differs across ranks; check shared storage before resuming.")
    return results[0][1]


def _checkpoint_has_optimizer_state(checkpoint_dir: Path, args: Namespace) -> bool:
    """Whether a DCP checkpoint directory carries optimizer state.

    Megatron omits the ``optimizer`` / ``opt_param_scheduler`` entries when
    saving with ``--no-save-optim``. For DCP-based formats the shard index
    (``.metadata``) records the stored shard keys, including the
    ``chained_<index>.optimizer`` namespaces used for dense/expert optimizers.
    Inspect those namespaces without reading any tensor data. Legacy ``torch``
    checkpoints have no ``.metadata`` and are assumed loadable, keeping the
    previous fail-loudly behavior.
    """
    if not (Path(checkpoint_dir) / ".metadata").is_file():
        return True

    metadata = _validate_dcp_storage(checkpoint_dir)
    return _has_optimizer_keys(metadata)


def _has_optimizer_keys(metadata: Any) -> bool:
    return any(
        re.match(r"^(?:chained_\d+\.)*optimizer(?:[./]|$)", str(key)) is not None
        for key in metadata.state_dict_metadata
    )


def _validate_dcp_storage(checkpoint_dir: Path) -> Any:
    """Check indexed shard extents without loading tensor payloads."""
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint_dir = Path(checkpoint_dir)
    metadata = FileSystemReader(str(checkpoint_dir)).read_metadata()
    if not metadata.storage_data:
        raise RuntimeError(f"Native checkpoint has no tensor storage metadata: {checkpoint_dir}")
    # Many tensor entries share a shard. Stat each file once, rather than once
    # per tensor, which is costly on shared storage.
    sizes: dict[Path, int] = {}
    root = checkpoint_dir.resolve()
    for entry in metadata.storage_data.values():
        relative = Path(entry.relative_path)
        if relative not in sizes:
            path = (checkpoint_dir / relative).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f"Checkpoint shard is outside the checkpoint directory: {path}")
            if not path.is_file():
                raise RuntimeError(f"Checkpoint shard is missing: {path}")
            sizes[relative] = path.stat().st_size
        if entry.offset < 0 or entry.length < 0 or sizes[relative] < entry.offset + entry.length:
            raise RuntimeError(f"Checkpoint shard is truncated or has an invalid extent: {checkpoint_dir / relative}")
    return metadata


def validate_native_checkpoint(
    checkpoint_dir: str | Path,
    *,
    require_optimizer: bool = True,
    require_rng: bool = True,
    common_state: dict | None = None,
) -> dict:
    """Preflight a Megatron DCP iteration for resume or weights-only loading.

    Recipes retain ownership of their dataset/agent state. Passing both
    requirements as False permits an intentional model-only checkpoint; it
    never permits missing or truncated tensor shards.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / ".metadata").is_file():
        raise RuntimeError(f"Expected a native DCP checkpoint: {checkpoint_dir}")
    metadata = _validate_dcp_storage(checkpoint_dir)
    if common_state is None:
        from megatron.core.dist_checkpointing import load_common_state_dict

        common_state = load_common_state_dict(checkpoint_dir)
    has_scheduler = any(key in common_state for key in ("opt_param_scheduler", "lr_scheduler"))
    if require_optimizer and (not _has_optimizer_keys(metadata) or not has_scheduler):
        raise RuntimeError("Native resume requires optimizer and scheduler state; use explicit weights-only loading")
    keys = metadata.state_dict_metadata
    if (
        require_rng
        and "rng_state" not in common_state
        and not any(re.match(r"^rng_state(?:[./]|$)", str(key)) is not None for key in keys)
    ):
        raise RuntimeError("Native resume requires RNG state; use --no-load-rng for an intentional RNG reset")
    iteration = re.fullmatch(r"iter_(\d{7})", checkpoint_dir.name)
    if iteration and common_state.get("iteration") != int(iteration[1]):
        raise RuntimeError(f"Native checkpoint iteration does not match its directory: {checkpoint_dir}")
    return common_state
