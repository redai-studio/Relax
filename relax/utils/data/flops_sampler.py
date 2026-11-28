# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in sampler adapters; admission/cache/state remain owned by legacy
samplers."""

from __future__ import annotations

import inspect
import time
from typing import Any

from transfer_queue.sampler import BaseSampler

from relax.utils.data.flops_scheduling import RowPlacement, runtime_context
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FlopsSftSampler(BaseSampler):
    """Use the pinned TQ pre-publication extension, not post-cache
    rewriting."""

    def __init__(self, dp_size: int, placement: RowPlacement) -> None:
        from transfer_queue import SeqlenBalancedSampler

        self.base = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=dp_size)
        if "assignment_transform" not in inspect.signature(self.base.sample).parameters:
            raise RuntimeError("FLOPs SFT placement requires the existing TransferQueue assignment_transform patch")
        self.placement = placement

    def has_cached_result(self, partition_id: str, task_name: str, sampling_config: Any = None) -> bool:
        return self.base.has_cached_result(partition_id, task_name, sampling_config)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = "",
        partition_id: str = "",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        if "eval" in task_name or "eval" in partition_id:
            return self.base.sample(ready_indexes, batch_size, task_name, partition_id, *args, **kwargs)

        def transform(original: list[list[int]]) -> list[list[int]]:
            started = time.perf_counter()
            rows = [i for group in original for i in group]
            # Stable input order independent of the preceding token KK ordering.
            rows.sort()
            metadata = kwargs["partition"].get_custom_meta(rows)
            groups = self.placement(rows, metadata, self.base.dp_size)
            logger.info(
                "FLOPs DP placement: dp=%s rows=%s cpu_seconds=%s",
                self.base.dp_size,
                len(rows),
                time.perf_counter() - started,
            )
            return groups

        return self.base.sample(
            ready_indexes, batch_size, task_name, partition_id, *args, assignment_transform=transform, **kwargs
        )

    def clear_cache(self, partition_id: str) -> None:
        self.base.clear_cache(partition_id)

    def save_checkpoint(self) -> dict:
        return {
            "base": self.base.save_checkpoint(),
            "signature": self.placement.signature,
            "dp_size": self.base.dp_size,
        }

    def load_checkpoint(self, state: dict) -> None:
        if state.get("signature") != self.placement.signature or state.get("dp_size") != self.base.dp_size:
            raise ValueError("Checkpoint FLOPs SFT model/mode/DP geometry mismatch")
        self.base.load_checkpoint(state["base"])


def configure_sampler(args: Any, original: Any, dp_size: int) -> Any:
    order = getattr(args, "flops_scheduling_order", "dp-first")
    if order not in ("dp-first", "pack-first") or (
        order == "pack-first" and getattr(args, "flops_scheduling", "off") != "both"
    ):
        raise ValueError("pack-first requires --flops-scheduling=both; order must be dp-first or pack-first")
    if (
        getattr(args, "flops_scheduling", "off") in ("dp", "both")
        and getattr(args, "loss_type", None) != "sft"
        and not getattr(args, "balance_data", False)
    ):
        raise ValueError(
            "RL FLOPs-aware DP requires balance_data=True (--balance-data) and the original Balanced sampler; "
            "Sequential admission is not supported"
        )
    if order == "pack-first":
        from relax.utils.data.global_flops_pack_protocol import validate_pack_runtime

        validate_pack_runtime(args)
    context = runtime_context(args)
    if order == "pack-first":
        from relax.utils.data.global_flops_pack_sampler import GlobalPackSampler

        return GlobalPackSampler(args, context, dp_size, original)
    if context is None or context.mode == "microbatch":
        return original
    placement = RowPlacement(context)
    if getattr(args, "loss_type", None) == "sft":
        if getattr(args, "n_samples_per_prompt", 1) != 1:
            raise ValueError("FLOPs SFT DP placement currently supports n_samples_per_prompt=1 only")
        return FlopsSftSampler(dp_size, placement)
    from relax.utils.data.identity_window_sampler import IdentityWindowSampler

    return IdentityWindowSampler(dp_size=dp_size, placement="balanced", row_placement=placement)
