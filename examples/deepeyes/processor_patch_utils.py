# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Helpers for the DeepEyes SGLang processor registration."""

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from packaging.version import Version


SUPPORTED_SGLANG_VERSION = Version("0.5.12.post1")


def collapse_consecutive_image_tokens(input_ids: Any, image_token_id: int) -> Any:
    """Collapse each consecutive image-token run to one placeholder."""
    if not isinstance(input_ids, list):
        return input_ids

    collapsed = []
    previous_was_image = False
    for token_id in input_ids:
        is_image = token_id == image_token_id
        if not (is_image and previous_was_image):
            collapsed.append(token_id)
        previous_was_image = is_image
    return collapsed


def validate_sglang_contract(processor_cls: type, sglang_version: str) -> None:
    """Fail before server startup when the pinned upstream contract changed."""
    if Version(sglang_version) != SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            "DeepEyes processor registration supports SGLang "
            f"{SUPPORTED_SGLANG_VERSION}, but found {sglang_version}. "
            "Review the upstream QwenVLImageProcessor contract before updating SUPPORTED_SGLANG_VERSION."
        )

    expected_signatures = {
        "process_mm_data_async": ("self", "image_data", "input_text", "request_obj"),
        "compute_mrope_positions": ("self", "input_ids", "mm_items"),
    }
    for method_name, expected_names in expected_signatures.items():
        method = getattr(processor_cls, method_name, None)
        if method is None:
            raise RuntimeError(f"SGLang QwenVLImageProcessor.{method_name} is missing.")
        signature = inspect.signature(method)
        if tuple(signature.parameters)[: len(expected_names)] != expected_names:
            raise RuntimeError(f"SGLang QwenVLImageProcessor.{method_name} has an incompatible signature: {signature}")


def merge_mrope_grid_items(mm_items: Any) -> Any:
    """Present per-image SGLang items as one grid collection for mRoPE."""
    grid_values: dict[str, list[Any]] = {"image_grid_thw": [], "video_grid_thw": []}
    for item in mm_items or []:
        model_specific_data = getattr(item, "model_specific_data", {})
        for key in grid_values:
            value = model_specific_data.get(key)
            if value is not None:
                grid_values[key].append(value)

    merged = {}
    for key, values in grid_values.items():
        if not values:
            continue
        if len(values) == 1:
            merged[key] = values[0]
        elif all(isinstance(value, (list, tuple)) for value in values):
            merged[key] = [row for value in values for row in value]
        else:
            import torch

            merged[key] = torch.cat(values, dim=0)
    if not merged:
        return mm_items
    return [SimpleNamespace(model_specific_data=merged)]


async def process_preexpanded_inputs(
    processor: Any,
    upstream_process: Callable[..., Awaitable[Any]],
    image_data: Any,
    input_text: Any,
    request_obj: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Delegate to SGLang while preserving DeepEyes pre-expanded token IDs."""
    if not (isinstance(input_text, list) and input_text and isinstance(input_text[0], int)):
        return await upstream_process(image_data, input_text, request_obj, *args, **kwargs)

    original_input_ids = list(input_text)
    patched_input = collapse_consecutive_image_tokens(input_text, processor.mm_tokens.image_token_id)
    output = await upstream_process(image_data, patched_input, request_obj, *args, **kwargs)
    if output is None:
        return None
    if not dataclasses.is_dataclass(output):
        raise RuntimeError(
            "DeepEyes processor registration received an unsupported SGLang output type "
            f"{type(output).__name__}; review the processor contract for this SGLang release."
        )

    mrope_items = merge_mrope_grid_items(output.mm_items)
    mrope_positions, mrope_position_delta = processor.compute_mrope_positions(original_input_ids, mrope_items)
    return dataclasses.replace(
        output,
        input_ids=original_input_ids,
        mrope_positions=mrope_positions,
        mrope_position_delta=mrope_position_delta,
    )
