# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Delay the chunked SFT head's reduction until all backward chunks finish."""

from types import MethodType
from typing import Any


def defer_head_bucket_grad_sync(bucket_group: Any) -> None:
    """Protect one MCore bucket without changing its accumulation or DDP
    config.

    Reentrant head checkpoints invoke the same parameter hook once per chunk.
    Dynamic packing changes that count between steps, so a first-step hook-
    count threshold cannot establish when this bucket's gradients are complete.
    Megatron finalization runs after every microbatch backward and before the
    optimizer; dispatch here while preserving MCore's scaling/wait/copyback.
    """
    if getattr(bucket_group, "_relax_chunked_sync_installed", False):
        return
    if not bucket_group.ddp_config.overlap_grad_reduce:
        raise ValueError("Deferred head bucket requires gradient overlap")
    if bucket_group.grad_reduce_handle is not None:
        raise RuntimeError("Install deferred head sync before the first backward")
    original_start = bucket_group.start_grad_sync
    original_finish = bucket_group.finish_grad_sync

    def deferred_start(self: Any, force_all_reduce: bool = False) -> None:
        # Both ready hooks and an explicit schedule request reach this method.
        # Finalization supplies the collective choice after all accumulation.
        return None

    def deferred_finish(self: Any, force_all_reduce: bool = False) -> None:
        if self.grad_reduce_finished:
            return
        if self.grad_reduce_handle is not None:
            raise RuntimeError("Chunked head bucket dispatched before finalization")
        original_start(force_all_reduce=force_all_reduce)
        # On the first step original_finish calls self.start_grad_sync again;
        # deferred_start makes that request a no-op, avoiding a double dispatch.
        original_finish(force_all_reduce=force_all_reduce)

    bucket_group.start_grad_sync = MethodType(deferred_start, bucket_group)
    bucket_group.finish_grad_sync = MethodType(deferred_finish, bucket_group)
    bucket_group._relax_chunked_sync_installed = True
