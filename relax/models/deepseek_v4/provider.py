# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""DeepSeek-V4 mixed MXFP4/FP8 QAT model provider."""

from __future__ import annotations

from typing import Any


def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None) -> Any:
    from megatron.training.global_vars import get_args

    from relax.models.deepseek_v4.fp8_weights import install_native_fp8_weights
    from relax.models.deepseek_v4.indexer import install_indexer_qat, select_process_mode
    from relax.models.deepseek_v4.qat import _fp4_mode
    from relax.models.deepseek_v4.qat import model_provider as routed_provider
    from relax.utils.env import Envs
    from relax.utils.logging_utils import get_logger

    mode = _fp4_mode()
    enabled = mode == "native" and Envs.RELAX_DSV4_FP4_INDEXER
    scores = Envs.RELAX_DSV4_FP4_BF16_SCORES
    select_process_mode(enabled=enabled, bf16_scores=scores)
    model = routed_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    fp8 = install_native_fp8_weights(model, hf_checkpoint=get_args().hf_checkpoint) if mode == "native" else ()
    indexer = ()
    if enabled:
        indexer = install_indexer_qat(model, bf16_scores=scores)
    get_logger(__name__).info(
        "[DSV4-MXFP4] mode=%s; native P2 FP8 projections=%d; indexer Q/K MXFP4 modules=%d; "
        "BF16 final score simulation=%s. H800 indexer uses BF16 kernels. Uncovered FP8: %s",
        mode,
        len(fp8),
        len(indexer),
        bool(indexer) and scores,
        getattr(model, "_relax_native_fp8_uncovered", ()),
    )
    return model
