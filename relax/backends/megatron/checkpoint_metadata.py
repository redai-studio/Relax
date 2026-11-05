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
    if getattr(args, "ckpt_format", None) not in ("torch_dist", "torch_dcp", "fsdp_dtensor"):
        return True
    if not (Path(checkpoint_dir) / ".metadata").is_file():
        return True
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(str(checkpoint_dir)).read_metadata()
    return any(
        re.match(r"^(?:chained_\d+\.)*optimizer(?:[./]|$)", str(key)) is not None
        for key in metadata.state_dict_metadata
    )
