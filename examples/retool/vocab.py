# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Match SGLang's legal-token distribution in native old/new logprobs."""

import functools
from typing import Any


TOKENIZER_LIMIT = 151665
MODEL_VOCAB = 152064


def mask_shard(logits: Any, *, limit: int, rank: int, world_size: int) -> Any:
    import torch

    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise ValueError("Expected FP32 response logits [R,V/TP]")
    width = logits.shape[-1]
    if not 0 <= rank < world_size or not 0 < limit <= width * world_size:
        raise ValueError("Invalid tensor-parallel vocabulary layout")
    first_invalid = max(0, min(width, limit - rank * width))
    if first_invalid == width:
        return logits
    return logits.masked_fill(torch.arange(width, device=logits.device) >= first_invalid, -1e30)


def install(args: Any) -> None:
    from megatron.core import mpu

    from relax.backends.megatron import loss

    original = loss.get_responses
    if getattr(original, "_retool_legal_vocab", False):
        return

    @functools.wraps(original)
    def responses(*positional: Any, **keywords: Any) -> Any:
        for logits, tokens in original(*positional, **keywords):
            yield (
                mask_shard(
                    logits,
                    limit=TOKENIZER_LIMIT,
                    rank=mpu.get_tensor_model_parallel_rank(),
                    world_size=mpu.get_tensor_model_parallel_world_size(),
                ),
                tokens,
            )

    responses._retool_legal_vocab = True
    loss.get_responses = responses


def before_log_prob(args: Any, model: Any, store_prefix: str) -> None:
    install(args)


def before_train_step(
    args: Any, rollout_id: int, step_id: int, model: Any, optimizer: Any, opt_param_scheduler: Any
) -> None:
    install(args)
