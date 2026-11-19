# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from typing import Any

import numpy as np
import torch


_DEFAULT_SPATIAL_MERGE_SIZE = 2


def _to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _iter_train_inputs(multimodal_train_inputs: Any):
    if multimodal_train_inputs is None:
        return
    if isinstance(multimodal_train_inputs, dict):
        yield multimodal_train_inputs
        return
    for item in multimodal_train_inputs:
        if isinstance(item, dict):
            yield item


def _grid_token_counts(grid_thw: Any, spatial_merge_size: int = _DEFAULT_SPATIAL_MERGE_SIZE) -> list[int]:
    grid_thw = _to_list(grid_thw)
    if grid_thw is None:
        return []
    merge_area = spatial_merge_size**2
    return [int(t) * int(h) * int(w) // merge_area for t, h, w in grid_thw]


def _audio_token_counts(mm_input: dict[str, Any]) -> list[int]:
    audio_seqlens = mm_input.get("audio_seqlens")
    if audio_seqlens is not None:
        return [int(v) for v in _to_list(audio_seqlens)]

    feature_attention_mask = mm_input.get("feature_attention_mask")
    if feature_attention_mask is None:
        return []
    if isinstance(feature_attention_mask, torch.Tensor):
        return [int(v) for v in feature_attention_mask.sum(-1).tolist()]
    return [int(v) for v in torch.tensor(feature_attention_mask).sum(-1).tolist()]


def get_multimodal_token_counts(multimodal_train_inputs: Any) -> dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0}
    for mm_input in _iter_train_inputs(multimodal_train_inputs) or ():
        spans = _to_list(mm_input.get("image_spans"))
        if spans is not None:
            counts["image"] += sum(length for _, length in spans)
        else:
            counts["image"] += sum(_grid_token_counts(mm_input.get("image_grid_thw")))
        counts["video"] += sum(_grid_token_counts(mm_input.get("video_grid_thw")))
        counts["audio"] += sum(_audio_token_counts(mm_input))
    counts["total"] = counts["image"] + counts["video"] + counts["audio"]
    return counts


def count_images(multimodal_inputs: dict[str, Any] | None, multimodal_train_inputs: Any = None) -> int:
    if multimodal_inputs is not None:
        for key in ("images", "image"):
            images = multimodal_inputs.get(key)
            if images:
                return len(images) if isinstance(images, list) else 1

    image_count = 0
    for mm_input in _iter_train_inputs(multimodal_train_inputs) or ():
        grid_thw = _to_list(mm_input.get("image_grid_thw", mm_input.get("image_grid_hws")))
        if grid_thw is not None:
            image_count += len(grid_thw)
    return image_count


def get_sample_multimodal_stats(sample: Any) -> dict[str, int]:
    token_counts = get_multimodal_token_counts(getattr(sample, "multimodal_train_inputs", None))
    return {
        "image_count": count_images(
            getattr(sample, "multimodal_inputs", None),
            getattr(sample, "multimodal_train_inputs", None),
        ),
        "image_token_count": token_counts["image"],
        "video_token_count": token_counts["video"],
        "audio_token_count": token_counts["audio"],
        "multimodal_token_count": token_counts["total"],
    }
