# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The GenRM messages contract, shared by HTTP and managed lookaside
callers."""

import asyncio
from typing import Any


async def genrm_payload(
    tokenizer: Any, sampling_config: dict, messages: list, sampling_params: dict | None = None
) -> dict:
    # Tokenization is CPU work; do not block the gateway/rollout event loop.
    input_ids = await asyncio.to_thread(
        tokenizer.apply_chat_template,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **(sampling_config.get("chat_template_kwargs") or {}),
    )
    if not isinstance(input_ids, list):
        input_ids = (
            input_ids["input_ids"] if hasattr(input_ids, "keys") and "input_ids" in input_ids else list(input_ids)
        )
    sampling = {
        "temperature": sampling_config.get("temperature", 0.2),
        "top_p": sampling_config.get("top_p", 1.0),
        "top_k": sampling_config.get("top_k", -1),
        "max_new_tokens": sampling_config.get("max_response_len", 1024),
    }
    sampling.update(sampling_params or {})
    return {"input_ids": input_ids, "sampling_params": sampling}
