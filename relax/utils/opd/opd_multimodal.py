# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from typing import Any


def image_inputs(multimodal_inputs: dict | None) -> list:
    """Native image OPD deliberately excludes audio/video processor
    fallbacks."""
    media = multimodal_inputs or {}
    if media.get("videos") or media.get("audio"):
        raise ValueError("Native SGLang OPD supports images only; audio/video inputs are unsupported.")
    return media.get("images") or []


def get_image_token_id(processor: Any, tokenizer: Any) -> int:
    token_id = getattr(processor, "image_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "image_token_id", None)
    if token_id is None:
        token = getattr(processor, "image_token", None)
        if token is not None:
            token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Native image OPD requires a processor with a single image placeholder token.")
    return int(token_id)


def validate_image_input_ids(
    input_ids: list[int],
    expanded_ids: list[int],
    response_length: int,
    image_count: int,
    image_token_id: int,
) -> None:
    """Check the two prompt coordinates without decoding the sampled
    response."""
    if response_length < 0 or min(len(input_ids), len(expanded_ids)) <= response_length:
        raise ValueError("Native image OPD requires nonempty collapsed and expanded prompts.")
    if response_length:
        response_ids = expanded_ids[-response_length:]
        if input_ids[-response_length:] != response_ids:
            raise ValueError("Native image OPD response IDs differ between collapsed and expanded inputs.")
        if image_token_id in response_ids:
            raise ValueError("Native image OPD cannot expand an image placeholder sampled in the response.")
        collapsed_prompt = input_ids[:-response_length]
        expanded_prompt = expanded_ids[:-response_length]
    else:
        collapsed_prompt, expanded_prompt = input_ids, expanded_ids
    if collapsed_prompt.count(image_token_id) != image_count:
        raise ValueError("Native image OPD requires exactly one collapsed placeholder per image.")
    if expanded_prompt.count(image_token_id) < image_count or (
        [t for t in collapsed_prompt if t != image_token_id] != [t for t in expanded_prompt if t != image_token_id]
    ):
        raise ValueError("Native image OPD processor must only expand image placeholders and preserve text IDs.")


def validate_prompt_token_ids(response: dict, expected_ids: list[int]) -> None:
    actual_ids = response.get("prompt_token_ids")
    if actual_ids is None:
        raise ValueError("Native image OPD requires SGLang v0.5.17 return_prompt_token_ids through the router.")
    if actual_ids != expected_ids:
        mismatch = next((i for i, (a, b) in enumerate(zip(actual_ids, expected_ids)) if a != b), None)
        raise ValueError(
            "Native image OPD expanded input IDs differ from the client processor: "
            f"actual_length={len(actual_ids)}, expected_length={len(expected_ids)}, first_mismatch={mismatch}. "
            "Check tokenizer/processor compatibility and SGLANG_MM_AVOID_RETOKENIZE=true."
        )


async def encode_image_inputs(multimodal_inputs: dict | None) -> dict[str, list[str]] | None:
    images = image_inputs(multimodal_inputs)
    if not images:
        return None
    # Samples may be copied or serialized while another coroutine encodes the
    # shared media. Only cache completed strings; Futures are not serializable.
    key = "_opd_image_b64_cache"
    cached = multimodal_inputs.get(key)
    if cached is None:
        from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

        cached = await asyncio.gather(*(async_encode_image_for_rollout_engine(img) for img in images))
        multimodal_inputs[key] = cached
    return {"image_data": list(cached)}
