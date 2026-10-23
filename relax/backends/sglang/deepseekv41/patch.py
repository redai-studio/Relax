# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Handle V4.1 DP-attention padding without changing real-token computation."""

from functools import wraps
from typing import TYPE_CHECKING, Any

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    import torch


logger = get_logger(__name__)


def apply_engram_idle_patch() -> None:
    import torch
    from sglang.srt.layers.engram import EngramHasher

    original = EngramHasher.forward
    if getattr(original, "_relax_v41_engram_idle", False):
        return

    @wraps(original)
    def forward(self: Any, input_ids: "torch.Tensor", forward_batch: Any) -> "torch.Tensor":
        if forward_batch.forward_mode.is_idle():
            assert self.history is not None, "EngramHasher.init_history was not called"
            # DP padding can give an idle rank nonempty dummy rows. Keep those
            # rows for the sharded embedding collective, but never hash or
            # commit them into a live request's history.
            return input_ids.new_zeros(
                (input_ids.shape[0], self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
            )
        return original(self, input_ids, forward_batch)

    forward._relax_v41_engram_idle = True
    EngramHasher.forward = forward
    logger.info("Applied V4.1 DP-attention padded-idle Engram fix")


def apply_dp_prefill_padding_patch() -> None:
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    original = backend.DeepseekV4AttnBackend.forward_low_ratio_sources
    if getattr(original, "_relax_v41_dp_padding", False):
        return

    @wraps(original)
    def forward_low_ratio_sources(
        self: Any,
        *,
        layer: Any,
        x: "torch.Tensor",
        q_lora: "torch.Tensor | None",
        positions: "torch.Tensor",
        forward_batch: Any,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and backend.get_parallel().attn_dp_size > 1
            and not backend.dsa_use_prefill_cp(forward_batch)
            and not self.low_ratio_prefill_graph
        ):
            # DP attention pads tensors to attention-TP alignment, but the eager
            # request/compressor/indexer metadata describes only real tokens.
            # extend_num_tokens is already padded here; use the CPU lengths.
            num_tokens = sum(forward_batch.extend_seq_lens_cpu)
            if num_tokens < positions.shape[0]:
                x = x[:num_tokens]
                q_lora = q_lora[:num_tokens] if q_lora is not None else None
                positions = positions[:num_tokens]
        return original(
            self,
            layer=layer,
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            run_compressor=run_compressor,
            run_indexer=run_indexer,
        )

    forward_low_ratio_sources._relax_v41_dp_padding = True
    backend.DeepseekV4AttnBackend.forward_low_ratio_sources = forward_low_ratio_sources
    logger.info("Applied V4.1 eager DP-attention compressor/indexer padding fix")
