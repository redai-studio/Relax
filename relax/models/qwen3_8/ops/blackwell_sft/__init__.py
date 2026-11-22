# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen3.8 Flash Next training hooks for packed 128K SFT on Blackwell SM103."""

from argparse import Namespace


def initialize_direct(args: Namespace) -> None:
    """Use compressed four-token block IDs directly for the long-document
    suffix."""
    from . import qsa_direct

    if (args.tensor_model_parallel_size, args.context_parallel_size) != (2, 1):
        raise ValueError("Direct QSA requires TP2 and CP1")
    qsa_direct.install()


def initialize_paired(args: Namespace) -> None:
    """Share KV loads between adjacent queries while preserving each query's
    causal mask."""
    from . import qsa_paired

    initialize_direct(args)
    qsa_paired.install()


def initialize_mfsdp(args: Namespace) -> None:
    """Install the PP2 Blackwell SM103 recipe once per train actor, after
    process-group initialization."""
    from relax.backends.megatron.fused_linear_ce import enable_mfsdp_persistent_head

    from . import (
        decoder_checkpoint,
        expanded_dispatch,
        expert_wgrad,
        fsdp_gather_groups,
        fsdp_residency,
        l2norm_cache,
        microbatch_order,
        moe_tail,
        packed_gdn,
        pooled_head,
        qsa_documents,
        ragged_gemm,
    )

    if not (
        args.use_megatron_fsdp
        and args.bf16
        and args.moe_flex_dispatcher_backend == "deepepv2"
        and not args.gradient_accumulation_fusion
        and args.enable_mtp_training
        and args.mtp_num_layers == 1
    ):
        raise ValueError(
            "The Blackwell SM103 MFSDP recipe requires BF16, DeepEPv2, one MTP layer, and unfused dense gradients"
        )
    enable_mfsdp_persistent_head()
    fsdp_gather_groups.initialize(args)
    decoder_checkpoint.install(args)
    fsdp_residency.install(args)
    l2norm_cache.install()
    pooled_head.install(args)
    microbatch_order.install(args)
    packed_gdn.install()
    ragged_gemm.install()
    expanded_dispatch.install(args)
    moe_tail.install(args)
    initialize_paired(args)
    qsa_documents.install()
    expert_wgrad.initialize(args)
