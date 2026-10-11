# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A real GenRM inference reward for the unified-service training smoke run."""

import asyncio
import math
from typing import Any

import httpx

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample
from relax.utils.utils import get_serve_url


logger = get_logger(__name__)


async def score(args: Any, sample: Sample | list[Sample], **kwargs: Any) -> dict[str, float] | list[dict[str, float]]:
    if isinstance(sample, list):
        semaphore = asyncio.Semaphore(args.reward_max_concurrency)

        async def score_one(item: Sample) -> dict[str, float]:
            async with semaphore:
                return await _score_sample(item)

        return await asyncio.gather(*(score_one(item) for item in sample))

    return await _score_sample(sample)


async def _score_sample(sample: Sample) -> dict[str, float]:
    if len(sample.teacher_log_probs or []) != sample.response_length:
        raise RuntimeError("Training smoke reward requires completed Teacher writeback")
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{get_serve_url('genrm')}/generate",
            json={
                "text": f"Evaluate this answer: {sample.response}",
                "sampling_params": {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True},
                "return_logprob": True,
            },
        )
        response.raise_for_status()
        rows = response.json()["meta_info"]["output_token_logprobs"]
    if len(rows) != 2 or not all(math.isfinite(row[0]) for row in rows):
        raise RuntimeError("GenRM did not return two finite token log probabilities")
    reward = sum(row[0] for row in rows) / len(rows)
    logger.info(
        "UNIFIED_TRAINING_GENRM_OK index=%s teacher_tokens=%s score=%s",
        sample.index,
        len(sample.teacher_log_probs),
        reward,
    )
    return {"score": reward}
