# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pixel-only reconstruction on rank-side processor workers."""

from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from relax.utils.data.image_refs import (
    HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
    KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
    validate_image_ref_descriptor,
)
from relax.utils.data.kimi_k3 import build_kimi_k3_image_features
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.image_utils import load_image


if TYPE_CHECKING:
    from relax.utils.data.processor_pool import ProcessorPool

logger = get_logger(__name__)
_worker_threads_limited = False


def _limit_worker_thread_counts() -> None:
    """Cap intra-op threads in rank-side rebuild workers.

    A rebuild pool runs one worker process per vision rank next to the training
    process; without this cap its torch ops would oversubscribe the node's CPUs
    together with the DataLoader and the training threads.
    """
    global _worker_threads_limited
    if _worker_threads_limited:
        return
    try:
        torch.set_num_threads(1)
    except RuntimeError as exc:
        logger.debug(f"Could not cap torch intra-op threads in rebuild worker: {exc}")
    _worker_threads_limited = True


def build_hf_image_features(processor: Any, images: list[Image.Image]) -> dict[str, Any]:
    """Run the generic HF pixel pipeline only: image_processor, no text.

    Mirrors the producer's ``adapt_processor_kwargs`` + full-processor call for
    models whose processor delegates images to its own ``image_processor`` with
    config defaults (Qwen-VL family and friends). Pixel-only on purpose: the
    consumer must never re-tokenize — the producer's tokens and masks are what
    trains. The descriptor's kind must travel through ``_REBUILD_BY_KIND``.
    """
    import torch

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError(
            "Processor has no image_processor; the generic image-ref rebuild cannot produce pixels "
            "without re-tokenizing the text."
        )
    features = image_processor(images=images, return_tensors="pt")
    if "pixel_values" not in features or "image_grid_thw" not in features:
        raise ValueError(
            f"Generic image rebuild needs pixel_values and image_grid_thw from the image processor; "
            f"got {sorted(features.keys())}."
        )
    pixel_values = features["pixel_values"]
    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)
    if pixel_values.dtype == torch.float32:
        # Same rule as _BF16_DOWNCAST_KEYS and the K3 rebuild: downstream casts
        # the encoder input to the vision tower's weight dtype.
        pixel_values = pixel_values.to(torch.bfloat16)
    return {
        "pixel_values": pixel_values.contiguous(),
        "image_grid_thw": features["image_grid_thw"],
    }


# Descriptor kind -> pixel-only rebuild. A new model family registers its
# builder here and stamps the matching kind on the producer side
# (relax/engine/sft/dataset/streaming.py._build_processed).
_REBUILD_BY_KIND = {
    KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND: build_kimi_k3_image_features,
    HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND: build_hf_image_features,
}


def build_image_features_in_worker(descriptor: dict) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Rebuild image pixels on a rank-side worker from an image-ref descriptor.

    Runs inside a ProcessorPool worker for ``--sft-image-preprocess-on-rank``:
    reads the referenced image files from shared storage and applies the pixel
    pipeline named by the descriptor's ``kind``, so the rebuild input matches
    the producer exactly (same numpy normalization, same resize limits, same
    BF16 rule). The producer's ``pixel_values`` are never shipped in this mode,
    so both sides must normalize identically.

    Returns:
        (features, timings) where features holds ``pixel_values`` (bf16, shared
        memory) and ``image_grid_thw``, and timings splits file read from
        processor work for the ``sft_rank_image_*`` metrics.

    Raises:
        RuntimeError: If descriptor validation, file reading, or the processor
            fails, with the original error details preserved.
    """
    from relax.utils.data.processor_pool import get_worker_processor, prepare_worker_images

    try:
        validate_image_ref_descriptor(descriptor)
        processor = get_worker_processor()
        _limit_worker_thread_counts()

        read_start = perf_counter()
        # Match the producer's PIL -> numpy IPC round-trip before applying
        # their shared normalization and processor-specific resize rules.
        images = prepare_worker_images([np.asarray(load_image(ref)) for ref in descriptor["image_refs"]])
        read_s = perf_counter() - read_start

        process_start = perf_counter()
        rebuild = _REBUILD_BY_KIND.get(descriptor["kind"])
        if rebuild is None:  # unreachable: validate_image_ref_descriptor rejects unknown kinds
            raise ValueError(f"unsupported image-ref descriptor kind {descriptor['kind']!r}")
        features = rebuild(processor, images)
        pixel_values = features["pixel_values"]
        grid = features["image_grid_thw"]
        process_s = perf_counter() - process_start

        # Runtime contract checks: count, order (grid rows), geometry and pixel
        # shape must match what the producer described. Failing here means the
        # failure lands in the prefetch future, which the cross-rank prefetch
        # agreement surfaces on every rank — no rank-divergent training state.
        if grid.dim() != 2 or grid.shape[0] != len(images):
            raise ValueError(f"rebuilt image grid has {tuple(grid.shape)} rows but {len(images)} image references.")
        producer_grid = descriptor.get("image_grid_thw")
        if producer_grid is not None and grid.tolist() != [list(row) for row in producer_grid]:
            raise ValueError(
                f"rebuilt image grid {grid.tolist()} does not match the producer descriptor "
                f"grid {producer_grid}; the consumer processor config diverged from the producer."
            )
        expected_shape = tuple(descriptor.get("pixel_shape") or ())
        if expected_shape and tuple(pixel_values.shape) != expected_shape:
            raise ValueError(
                f"rebuilt pixel shape {tuple(pixel_values.shape)} does not match the producer "
                f"descriptor shape {expected_shape}."
            )
        return (
            {"pixel_values": pixel_values.share_memory_(), "image_grid_thw": grid},
            {"read_s": read_s, "process_s": process_s},
        )

    except Exception as e:
        import traceback

        error_msg = f"Rank-side image rebuild failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from None


def build_batch_image_features(
    pool: "ProcessorPool", descriptors: list[Any]
) -> tuple[list[dict[str, torch.Tensor] | None], dict[str, float]]:
    """Rebuild pixels for every sample descriptor on a rank-side pool.

    Text-only samples carry ``None``. Each rebuild runs in the pool's worker
    process, so file I/O and processor work never hold the caller's GIL, and
    the returned tensors are CPU tensors backed by shared memory. Every sample
    is submitted before the first result is collected, so a pool sized for
    parallel rebuilds actually runs them in parallel; the per-sample wall clock
    split is returned for the training thread to record (summed over samples,
    so these exceed the caller's wall clock when the rebuilds overlap).
    Background workers must not mutate the training thread's Timer.
    """
    features: list[dict[str, torch.Tensor] | None] = [None] * len(descriptors)
    batch_timings = {"sft_rank_image_read": 0.0, "sft_rank_image_process": 0.0}
    # One processor call per sample: submitting them all up front is what makes
    # the pool parallel — a submit/result pair per sample serializes the batch,
    # and with ~20 images per sample this call is the prefetch's tail.
    submitted = [
        (index, pool.executor.submit(build_image_features_in_worker, descriptor))
        for index, descriptor in enumerate(descriptors)
        if descriptor is not None
    ]
    for index, future in submitted:
        try:
            sample_features, timings = future.result()
        except Exception as exc:
            raise RuntimeError(f"rank-side image preprocessing failed for sample {index}: {exc}") from exc
        batch_timings["sft_rank_image_read"] += timings.get("read_s", 0.0)
        batch_timings["sft_rank_image_process"] += timings.get("process_s", 0.0)
        features[index] = sample_features
    return features, batch_timings
