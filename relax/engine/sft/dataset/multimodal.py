# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT-side multimodal feature extraction wrapper.

Bridges CanonicalSample.{images,videos,audios} -> processor-ready tensors via
the existing `relax.utils.multimodal` and `relax.utils.data.processor_pool`
infrastructure (no duplication).
"""

import asyncio
from typing import Any

from relax.engine.sft.dataset.sample import CanonicalSample
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def has_multimodal_content(sample: CanonicalSample) -> bool:
    return bool(sample.images or sample.videos or sample.audios)


def _fetch_media(sample: CanonicalSample, rendered_text: str) -> tuple[dict[str, Any], str]:
    """Load image/video/audio bytes from sample paths.

    Returns (mm_inputs_dict, text). The text is unchanged here -- it has
    already been processed by chat_template.py to contain the model's media
    token placeholders. The mm_inputs_dict shape is what
    `processor_pool.process_sample_in_worker` expects.
    """
    from relax.utils.multimodal.audio_utils import load_audio
    from relax.utils.multimodal.image_utils import load_image
    from relax.utils.multimodal.video_utils import load_video

    mm_inputs: dict[str, list[Any]] = {}
    if sample.images:
        mm_inputs["images"] = [load_image(p) for p in sample.images]
    if sample.videos:
        mm_inputs["videos"] = [load_video(p, use_audio_in_video=False)[0] for p in sample.videos]
    if sample.audios:
        mm_inputs["audios"] = [load_audio(p) for p in sample.audios]
    return mm_inputs, rendered_text


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
    from relax.utils.data.processor_pool import prepare_mm_inputs_for_ipc, process_sample_in_worker

    mm_inputs, text = _fetch_media(sample, rendered_text)
    mm_inputs_ipc = prepare_mm_inputs_for_ipc(mm_inputs)
    future = processor_pool.executor.submit(process_sample_in_worker, text, mm_inputs_ipc, processor_kwargs or {})
    return future.result()


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
    from relax.utils.data.processor_pool import prepare_mm_inputs_for_ipc, process_sample_in_worker

    mm_inputs, text = _fetch_media(sample, rendered_text)
    mm_inputs_ipc = prepare_mm_inputs_for_ipc(mm_inputs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        processor_pool.executor,
        process_sample_in_worker,
        text,
        mm_inputs_ipc,
        processor_kwargs or {},
    )
