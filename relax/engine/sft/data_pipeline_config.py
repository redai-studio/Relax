# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Choose SFT data preparation after backend configuration is finalized."""

from argparse import Namespace

from relax.engine.sft.image_preprocessing_config import configure_image_preprocessing
from relax.engine.sft.runtime import is_offline_mode


def configure_sft_data_pipeline(args: Namespace) -> None:
    is_sft = getattr(args, "loss_type", None) in ("sft", "sft_loss", "sft-loss")
    if is_sft or is_offline_mode(args):
        args.per_rank_fetch = args.train_backend == "megatron"
        lookahead = args.per_rank_fetch and args.max_staleness >= 1
        # Image references use the raw prefetch worker to reconstruct pixels.
        # NCCL describes the training workers, unlike the driver's local device.
        args.sft_async_prepack = bool(
            is_sft
            and lookahead
            and "image" not in (getattr(args, "multimodal_keys", None) or {})
            and args.distributed_backend == "nccl"
            and args.qkv_format == "thd"
            and args.pipeline_model_parallel_size == 1
            and args.context_parallel_size == 1
            and (getattr(args, "virtual_pipeline_model_parallel_size", None) or 1) == 1
            and not getattr(args, "dynamic_context_parallel", False)
            and not getattr(args, "use_routing_replay", False)
            and not getattr(args, "use_rollout_routing_replay", False)
            and not getattr(args, "use_rollout_indexer_replay", False)
        )
        args.sft_train_data_prefetch = lookahead and not args.sft_async_prepack
    else:
        args.sft_train_data_prefetch = False
    configure_image_preprocessing(args, is_sft)
