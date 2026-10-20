# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT-side multimodal feature extraction wrapper.

Bridges CanonicalSample.{images,videos,audios} -> processor-ready tensors via
the existing `relax.utils.multimodal` and `relax.utils.data.processor_pool`
infrastructure (no duplication).
"""

import asyncio
import threading
import time
from typing import Any

from relax.engine.sft.dataset.sample import CanonicalSample
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Producer-side preprocessing cost (worker-side media load + processor call). The
# producer's per-step batch is what the training step ends up waiting on, so it
# has to be visible without a profiler; the dataset drains it per batch.
_PREPROCESS_STATS = {"samples": 0, "pool_s": 0.0}
_PREPROCESS_STATS_LOCK = threading.Lock()


def take_preprocess_stats() -> dict[str, float]:
    """Read and reset the accumulated preprocessing split."""
    with _PREPROCESS_STATS_LOCK:
        stats = dict(_PREPROCESS_STATS)
        _PREPROCESS_STATS.update(samples=0, pool_s=0.0)
    return stats


class SFTMultimodalMediaLoadError(RuntimeError):
    """A sample media source could not be loaded or decoded."""


def has_multimodal_content(sample: CanonicalSample) -> bool:
    return bool(sample.images or sample.videos or sample.audios)


def sample_media_paths(sample: CanonicalSample) -> dict[str, list[Any]]:
    """Ordered media sources per kind, as the worker loaders expect them.

    Only references travel to the worker; it decodes them there. Shipping
    decoded media through the pool cost an ~86 MiB pickle per 20-image sample
    in the producer's main process, which capped the pool's throughput.
    """
    return {
        "image": list(sample.images or []),
        "video": list(sample.videos or []),
        "audio": list(sample.audios or []),
    }


def _media_load_failure(sample: CanonicalSample, exc: BaseException) -> SFTMultimodalMediaLoadError:
    """Re-raise a worker-side load failure under the dataset's skip policy."""
    return SFTMultimodalMediaLoadError(f"{exc} (sample row_index={sample.metadata.get('row_index')})")


def preprocess_multimodal(
    sample: CanonicalSample,
    *,
    processor_pool,
    rendered_text: str = "",
    processor_kwargs: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Run the HF processor on a multimodal sample.

    Returns ``(prompt_ids, mm_train_inputs)``:

    - ``prompt_ids`` is the processor-expanded ``input_ids`` (each
      ``<|image_pad|>`` / ``<|video_pad|>`` / ``<|audio_pad|>`` placeholder is
      replaced with N copies based on the corresponding ``image_grid_thw`` /
      ``video_grid_thw`` / audio length). The model expects this expanded
      form when scattering visual / audio embeddings.
    - ``mm_train_inputs`` are the processor's pixel/grid/audio tensors.

    Both are ``None`` for text-only samples.

    Args:
        sample: CanonicalSample.
        processor_pool: Required when sample has any media; raises otherwise.
        rendered_text: The full chat-template-rendered text (the processor
            needs the text alongside the media to expand placeholders).
        processor_kwargs: Extra kwargs for the underlying HF processor.
    """
    if not has_multimodal_content(sample):
        return None, None
    if processor_pool is None:
        raise ValueError(
            "preprocess_multimodal: sample has multimodal content but "
            "processor_pool is None. Pass an instance of "
            "`relax.utils.data.processor_pool.ProcessorPool`."
        )
    from relax.utils.data.processor_pool import MediaLoadError, process_sample_from_paths_in_worker

    pool_started = time.perf_counter()
    future = processor_pool.executor.submit(
        process_sample_from_paths_in_worker, rendered_text, sample_media_paths(sample), processor_kwargs or {}
    )
    try:
        result = future.result()
    except MediaLoadError as exc:
        raise _media_load_failure(sample, exc) from exc
    with _PREPROCESS_STATS_LOCK:
        _PREPROCESS_STATS["samples"] += 1
        _PREPROCESS_STATS["pool_s"] += time.perf_counter() - pool_started
    return result


async def preprocess_multimodal_async(
    sample: CanonicalSample,
    *,
    processor_pool,
    rendered_text: str = "",
    processor_kwargs: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Async variant of `preprocess_multimodal`: dispatches the HF processor
    call to `processor_pool.executor` via `loop.run_in_executor` so the calling
    coroutine yields control while the work runs in another process.

    Returns the same ``(prompt_ids, mm_train_inputs)`` tuple as the sync
    variant; both are ``None`` for text-only samples.
    """
    if not has_multimodal_content(sample):
        return None, None
    if processor_pool is None:
        raise ValueError(
            "preprocess_multimodal_async: sample has multimodal content but "
            "processor_pool is None. Pass an instance of "
            "`relax.utils.data.processor_pool.ProcessorPool`."
        )
    from relax.utils.data.processor_pool import MediaLoadError, process_sample_from_paths_in_worker

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            processor_pool.executor,
            process_sample_from_paths_in_worker,
            rendered_text,
            sample_media_paths(sample),
            processor_kwargs or {},
        )
    except MediaLoadError as exc:
        raise _media_load_failure(sample, exc) from exc
