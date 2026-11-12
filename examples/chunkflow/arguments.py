# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from argparse import ArgumentParser, Namespace


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add the example's chunked-training options."""
    parser.add_argument(
        "--use-chunkflow",
        action="store_true",
        default=False,
        help=(
            "Execute sequence chunks through the model with cross-chunk states and gradients. "
            "Requires --chunkflow-chunk-size and --context-parallel-size 1."
        ),
    )
    parser.add_argument(
        "--chunkflow-chunk-size",
        type=int,
        default=0,
        help=(
            "Tokens per chunk for --use-chunkflow. Must be > 0 when chunkflow is enabled; a value "
            "at or above the sequence length degenerates to a single chunk (no chunking)."
        ),
    )
    parser.add_argument(
        "--chunkflow-pack-policy",
        type=str,
        default="staggered",
        choices=["staggered", "lockstep"],
        help=(
            "'staggered' executes one chunk per unit. 'lockstep' groups equal-depth chunks; "
            "multiple dependent segments are currently rejected by the position/GDN adapters."
        ),
    )
    parser.add_argument(
        "--chunkflow-keep-last-k",
        type=int,
        default=0,
        help=(
            "How many of a dependency group's last chunks keep their activation; the earlier ones are "
            "recomputed before their backward. 0 (default) keeps every chunk."
        ),
    )
    return parser


def validate_args(args: Namespace) -> None:
    """Reject unsupported execution and loss contracts before training
    starts."""
    if not getattr(args, "use_chunkflow", False):
        return

    chunk_size = getattr(args, "chunkflow_chunk_size", 0)
    if chunk_size <= 0:
        raise ValueError(
            f"--use-chunkflow requires --chunkflow-chunk-size > 0 (got {chunk_size}); "
            "without it chunkflow would silently run unchunked."
        )
    if getattr(args, "context_parallel_size", 1) > 1:
        raise ValueError(
            "--use-chunkflow is incompatible with --context-parallel-size > 1: context parallelism "
            "splits the very sequence chunkflow caches in its per-layer StateStore. "
            "Remove one of the two."
        )
    if getattr(args, "num_distributed_optimizer_instances", 1) != 1:
        raise ValueError(
            "--use-chunkflow requires --num-distributed-optimizer-instances 1 for gradient synchronization"
        )
    if not getattr(args, "calculate_per_token_loss", False):
        raise ValueError(
            "--use-chunkflow requires --calculate-per-token-loss: sample-mean loss would treat "
            "each chunk as a separate sample and change the training objective."
        )
    if getattr(args, "enable_mtp_training", False):
        raise ValueError(
            "--use-chunkflow does not support --enable-mtp-training: "
            "chunk-local MTP labels omit supervision across chunk boundaries."
        )
    from relax.algorithms import ALGORITHM_SPECS

    algorithm = ALGORITHM_SPECS.get(getattr(args, "advantage_estimator", None))
    if getattr(args, "loss_type", None) == "policy_loss" and (
        getattr(args, "use_opsm", False)
        or (
            algorithm is not None
            and (
                algorithm.kl_level == "sequence"
                or algorithm.needs_full_log_probs
                or algorithm.policy_loss_fn == "m2po"
            )
        )
    ):
        raise ValueError(
            "--use-chunkflow does not support sequence/batch-level objectives (GSPO/OPSM/M2PO): "
            "chunk-local statistics cannot replace full-response or batch reductions."
        )
    if getattr(args, "fully_async", False) and getattr(args, "use_dynamic_batch_size", False):
        raise ValueError(
            "--use-chunkflow does not support --fully-async with --use-dynamic-batch-size: "
            "streaming iterators have an unknown micro-batch count, but ChunkFlow requires a fixed plan. "
            "Disable --fully-async or --use-dynamic-batch-size."
        )
    if getattr(args, "chunkflow_keep_last_k", 0) < 0:
        raise ValueError(
            f"--chunkflow-keep-last-k must be >= 0 (0 keeps every chunk), got {args.chunkflow_keep_last_k}"
        )
