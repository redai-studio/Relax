# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from typing import Any


def configure_runtime(provider: Any, args: Namespace) -> None:
    """Configure the Flash Bridge input layout without changing CLI
    defaults."""
    is_sft = getattr(args, "loss_type", None) == "sft"
    packed_cp = provider.context_parallel_size > 1
    packed = getattr(args, "qkv_format", "thd") == "thd"
    args.needs_cpu_image_grid = (
        getattr(provider, "vision_dp_when_tp", False) and provider.tensor_model_parallel_size > 1
    )
    if (packed_cp or is_sft) and not packed:
        raise ValueError("GLM-5.3-Flash SFT and CP require packed THD batches.")
    if getattr(args, "dynamic_context_parallel", False):
        raise ValueError("GLM-5.3-Flash Bridge requires static context parallelism.")
    if packed_cp:
        if getattr(args, "allgather_cp", False) or getattr(args, "cp_partition_mode", "zigzag") != "zigzag":
            raise ValueError("GLM-5.3-Flash Bridge requires per-document zigzag CP packing.")
        if provider.cp_comm_type != "allgather":
            raise ValueError("GLM-5.3-Flash DSA requires --cp-comm-type allgather with CP > 1.")
        if getattr(args, "enable_mtp_training", False):
            raise ValueError("GLM-5.3-Flash Bridge CP currently requires MTP training to be disabled.")
        args.global_packed_thd_cp = True
        args.uses_unsplit_forward = True

    if is_sft and not getattr(args, "enable_mtp_training", False):
        # The HF checkpoint contains a draft block, but ordinary SFT
        # consumes only the main model logits and supplies no MTP labels.
        provider.mtp_num_layers = None
        provider.mtp_hybrid_override_pattern = None
        provider.mtp_use_repeated_layer = False
