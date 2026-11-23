# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Role-aware validation for Megatron-FSDP pipeline checkpoints."""

import re
from pathlib import Path
from typing import Optional, Union


def _is_megatron_checkpoint_path(path: Optional[Union[str, Path]]) -> bool:
    if not path:
        return False
    path = Path(path)
    return (path / "latest_checkpointed_iteration.txt").is_file() or bool(re.fullmatch(r"iter_\d{7}", path.name))


def check_fsdp_pp_checkpoint_support(args, role: str = "actor") -> None:
    """Reject VPP checkpointing not covered by the image-built FSDP PP fix."""
    if not getattr(args, "use_megatron_fsdp", False):
        return
    if getattr(args, "pipeline_model_parallel_size", 1) <= 1:
        return
    if getattr(args, "ckpt_format", None) != "fsdp_dtensor":
        return
    if role == "critic":
        load_path = getattr(args, "critic_load", None)
        save_path = getattr(args, "critic_save", None)
    elif role in ("reference", "actor_fwd"):
        load_path = getattr(args, "ref_load", None)
        save_path = None
    else:
        load_path = getattr(args, "load", None)
        save_path = getattr(args, "save", None)
    if not save_path and not _is_megatron_checkpoint_path(load_path):
        return
    if (getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) > 1:
        raise RuntimeError(
            "[Relax] --ckpt-format fsdp_dtensor with --use-megatron-fsdp, "
            "--pipeline-model-parallel-size > 1, and virtual pipeline parallelism "
            "(--num-virtual-stages-per-pipeline-rank / --virtual-pipeline-model-parallel-size) "
            "is not supported: the global-layer-numbering patch only handles one model chunk "
            "per rank. Drop VPP, or disable native Megatron checkpoint load/save for a smoke test."
        )
