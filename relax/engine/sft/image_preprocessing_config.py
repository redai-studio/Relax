# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Derive rank-side image reconstruction from the existing SFT data path."""

from argparse import Namespace


def configure_image_preprocessing(args: Namespace, is_sft: bool) -> None:
    # File references are shared across training nodes. A rank that cannot
    # read a reference must fail; only inline/unsupported images keep pixels.
    args.sft_image_preprocess_on_rank = bool(
        is_sft
        and "image" in (args.multimodal_keys or {})
        and args.sft_train_data_prefetch
        and args.per_rank_fetch
        and args.max_staleness >= 1
        and not args.sft_async_prepack
    )
