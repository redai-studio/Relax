# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch
    import torch.distributed as dist


class PackingMetrics:
    """Accumulate host metadata once per microbatch on the final virtual PP
    stage.

    Dummy batches count because they execute computation. Pack sums are
    weighted by inverse *actual* CP size, so dynamic CP subdivisions count as
    distinct packs. Microbatch count is the mean number of forwards per static
    DP replica.
    """

    def __init__(self) -> None:
        self.microbatches = 0
        self.packs = 0.0
        self.tokens = 0.0
        self.max_tokens = 0

    def add(self, batch: dict[str, Any], static_cp_size: int) -> None:
        cp_size = batch.get("dynamic_cp_size", static_cp_size)
        tokens = batch["tokens"]
        # Bridge VL models repack unsplit inputs with different alignment from
        # the loss-side token tensor. Host boundaries describe that physical pack.
        vlm_params = batch.get("vlm_packed_seq_params")
        pack_tokens = vlm_params.cu_seqlens_q_cpu[-1] if vlm_params is not None else tokens.shape[-1] * cp_size
        num_packs = 1 if vlm_params is not None else tokens.shape[0]
        self.microbatches += 1
        self.packs += num_packs / cp_size
        self.tokens += num_packs * pack_tokens / cp_size
        self.max_tokens = max(self.max_tokens, pack_tokens)

    def reduce(self, process_group: dist.ProcessGroup, device: torch.device) -> dict[str, float]:
        """Called by all ranks in the final physical PP stage's DP+CP group."""
        import torch
        import torch.distributed as dist

        totals = torch.tensor([self.microbatches, self.packs, self.tokens], dtype=torch.float64, device=device)
        maximum = torch.tensor(self.max_tokens, dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=process_group)
        microbatches, packs, tokens = totals.tolist()
        return {
            "num_microbatches_mean": microbatches / dist.get_world_size(process_group),
            "pack_tokens_mean": tokens / packs if packs else 0.0,
            "pack_tokens_max": maximum.item(),
        }
