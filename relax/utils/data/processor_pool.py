# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Process pool for HuggingFace processor execution without GIL contention.

Problem: HuggingFace ProcessorMixin.__call__() involves significant Python-level CPU work
(image resize/normalize loops, padding/stacking, multimodal feature assembly) that holds the
GIL. Running these in a ThreadPoolExecutor causes GIL contention under high concurrency.

Solution: Run processor calls in separate processes via ProcessPoolExecutor. Use
torch.multiprocessing shared memory for zero-copy tensor transfer on the return path:
  - Worker calls tensor.share_memory_() on output tensors → backed by /dev/shm
  - Return through mp Queue uses fd-based passing (no data copy)
  - Main process receives tensors backed by same shm segment → true zero-copy
  - GC-based lifecycle: shm is freed when all references are dropped

Input path optimization:
  - PIL Images → numpy arrays (faster pickle than PIL serialization)
  - Video torch.Tensors → share_memory_() for zero-copy transfer
  - Audio numpy arrays → pass as-is (small, pickle is acceptable)
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from relax.utils.data.processing_utils import (
    adapt_processor_kwargs,
    expand_kimi_k25_placeholders,
    load_processor,
    remap_mm_train_inputs,
)
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.image_utils import resize_qwen_vl_extreme_aspect_ratio


logger = get_logger(__name__)

# Worker-local processor instance, initialized once per worker via pool initializer.
_worker_processor = None


def _init_worker(model_path: str, trust_remote_code: bool) -> None:
    """Initialize the HuggingFace processor in each worker process (called
    once)."""
    global _worker_processor
    _worker_processor = load_processor(model_path, trust_remote_code=trust_remote_code)
    logger.info(f"ProcessorPool worker initialized (pid={os.getpid()})")


def _is_qwen_vl_processor(processor: object) -> bool:
    """Return whether ``processor`` uses a Transformers Qwen-VL image
    processor."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return False
    for cls in type(image_processor).__mro__:
        if cls.__module__.startswith(
            ("transformers.models.qwen2_vl.", "transformers.models.qwen3_vl.")
        ) and cls.__name__.startswith(("Qwen2VLImageProcessor", "Qwen3VLImageProcessor")):
            return True
    return False


def prepare_mm_inputs_for_ipc(multimodal_inputs: dict) -> dict:
    """Prepare multimodal inputs for efficient cross-process transfer.

    - PIL Images → numpy uint8 arrays (faster pickle than PIL serialization)
    - Video torch.Tensors → share_memory_() for zero-copy fd-based transfer
    - Audio numpy arrays → unchanged (small, standard pickle is fine)

    Note: share_memory_() on video tensors modifies storage in-place (moves to /dev/shm).
    The data content is preserved, so downstream reads (e.g. encoding for sglang server)
    remain valid.
    """
    result = {}
    for k, v in multimodal_inputs.items():
        if k == "images" and v:
            result[k] = [np.asarray(img) for img in v]
        elif k == "videos" and v:
            result[k] = [vid.contiguous().share_memory_() if isinstance(vid, torch.Tensor) else vid for vid in v]
        else:
            result[k] = v
    return result


def process_sample_in_worker(
    text: str,
    multimodal_inputs: dict,
    processor_kwargs: dict[str, Any],
) -> tuple[list[int], dict[str, torch.Tensor] | None]:
    """Run the HuggingFace processor in a worker process.

    Restores input types (numpy → PIL for images), runs the processor, then places
    output tensors in shared memory for zero-copy return to the main process.

    Args:
        text: The prompt text to process.
        multimodal_inputs: Dict prepared by prepare_mm_inputs_for_ipc().
        processor_kwargs: Extra kwargs forwarded to the processor (e.g. use_audio_in_video).

    Returns:
        (prompt_ids, mm_train_inputs) where mm_train_inputs tensors are in shared memory.

    Raises:
        RuntimeError: If processor execution fails, with original error details preserved.
    """
    global _worker_processor

    try:
        if _worker_processor is None:
            raise RuntimeError("Processor not initialized in worker process")

        # Restore PIL Images from numpy arrays (HF processor expects PIL for some code paths)
        restored = dict(multimodal_inputs)
        if images := restored.get("images"):
            restored["images"] = [Image.fromarray(arr) for arr in images]
            if _is_qwen_vl_processor(_worker_processor):
                resized_images = []
                for image in restored["images"]:
                    resized = resize_qwen_vl_extreme_aspect_ratio(image)
                    if resized is not image:
                        logger.warning(
                            f"Qwen-VL image aspect ratio exceeded 200; resized from {image.size} to {resized.size}."
                        )
                    resized_images.append(resized)
                restored["images"] = resized_images
        # Videos arrive as shared-memory torch.Tensors — usable directly by the processor.
        # Audio arrives as numpy arrays — usable directly by the processor.

        # Translate to the processor's native call shape (no-op for Qwen-VL etc.;
        # rewrites images→medias and drops conflicting return_tensors for Kimi K2.x).
        adapted = adapt_processor_kwargs(_worker_processor, restored, processor_kwargs)
        processor_output = _worker_processor(text=text, **adapted)

        prompt_ids = processor_output["input_ids"][0]
        # K2.x adapt_processor_kwargs forces return_tensors="pt", so
        # input_ids is a 1D Tensor; downstream sample.tokens contract is list[int].
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()

        mm_train_inputs = {}
        for k, v in processor_output.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                # Place in shared memory for zero-copy return via fd-based IPC.
                # contiguous() is required: share_memory_() does not support non-contiguous storage.
                mm_train_inputs[k] = v.contiguous().share_memory_()

        train_inputs = remap_mm_train_inputs(_worker_processor, mm_train_inputs or None)
        prompt_ids = expand_kimi_k25_placeholders(_worker_processor, prompt_ids, train_inputs)
        return prompt_ids, train_inputs

    except Exception as e:
        import traceback

        error_msg = f"Processor execution failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        if "No space left on device" in error_msg:
            logger.error(
                f"{error_msg}\n"
                "HINT: Shared memory (/dev/shm) is exhausted. "
                "Increase its size with: mount -o remount,size=64G /dev/shm"
            )
        else:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from None


class ProcessorPool:
    """Process pool for running HuggingFace processors with true parallelism.

    Bypasses GIL by running processor calls in separate processes spawned via
    torch.multiprocessing (spawn context). Each worker loads the processor once
    at initialization and reuses it for all subsequent calls.

    Output tensors are placed in shared memory (torch.Tensor.share_memory_()) and
    transferred via fd-based IPC — no data copy on the return path. Shared memory
    is automatically freed when all tensor references are garbage collected.

    Usage with asyncio::

        pool = ProcessorPool(model_path)
        loop = asyncio.get_running_loop()
        prompt_ids, mm_inputs = await loop.run_in_executor(
            pool.executor,
            process_sample_in_worker,
            text,
            prepared_inputs,
            kwargs,
        )
    """

    def __init__(
        self,
        model_path: str,
        pool_size: int | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        if pool_size is None:
            pool_size = min(16, os.cpu_count() or 8)

        ctx = mp.get_context("spawn")
        self._pool = ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(model_path, trust_remote_code),
        )
        self._pool_size = pool_size
        logger.info(f"ProcessorPool created with {pool_size} workers (spawn context)")

    @property
    def executor(self) -> ProcessPoolExecutor:
        """The underlying ProcessPoolExecutor, for use with
        loop.run_in_executor()."""
        return self._pool

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool, optionally waiting for pending tasks."""
        self._pool.shutdown(wait=wait)
        logger.info("ProcessorPool shut down")
