# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import atexit
import base64
import io
import json
import os
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import imageio.v2 as imageio
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Shared thread pool for offloading CPU-intensive media encoding from the asyncio event loop.
# PNG compression (libpng), H.264 encoding (libx264), and base64 encoding are all C-level
# operations that release the GIL, so a thread pool achieves true parallelism without the
# serialization overhead of a process pool.
#
# Built lazily (not at import) so --encode-max-workers, parsed after import, can size the pool.
_ENCODE_MAX_WORKERS_ENV = "RELAX_ENCODE_MAX_WORKERS"
_encode_executor: ThreadPoolExecutor | None = None
_encode_executor_workers: int | None = None
_encode_executor_lock = threading.Lock()


def _usable_cpu_count() -> int:
    """CPUs this process may actually run on, respecting cgroup/cpuset
    affinity.

    ``os.sched_getaffinity`` reflects Ray/container CPU pinning;
    ``os.cpu_count`` reports every host core and would oversubscribe on a
    shared or limited node.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux platforms
        return os.cpu_count() or 8


def _validate_encode_max_workers(max_workers) -> int:
    # bool is rejected explicitly since bool is a subclass of int.
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise TypeError(f"encode_max_workers must be an integer, got {max_workers!r}")
    if max_workers <= 0:
        raise ValueError(f"encode_max_workers must be a positive integer, got {max_workers}")
    return max_workers


def resolve_encode_max_workers(max_workers: int | None) -> tuple[int, str]:
    """Resolve the worker count and its source, in priority order: explicit
    ``--encode-max-workers`` > ``$RELAX_ENCODE_MAX_WORKERS`` > affinity-aware
    default."""
    if max_workers is not None:
        return _validate_encode_max_workers(max_workers), "flag"
    env_value = os.environ.get(_ENCODE_MAX_WORKERS_ENV)
    if env_value is not None:
        return _validate_encode_max_workers(int(env_value)), f"${_ENCODE_MAX_WORKERS_ENV}"
    return min(32, _usable_cpu_count()), "default"


def configure_encode_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Create the shared media-encoding thread pool once and return it.

    Called from the rollout worker init. Idempotent for a matching size; raises
    if asked to reconfigure a live pool to a different size, since a
    ThreadPoolExecutor cannot be resized.
    """
    global _encode_executor, _encode_executor_workers
    workers, source = resolve_encode_max_workers(max_workers)
    with _encode_executor_lock:
        if _encode_executor is None:
            _encode_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="relax-encode")
            _encode_executor_workers = workers
            atexit.register(shutdown_encode_executor)
            logger.info("Initialized encode executor with %d workers (source: %s)", workers, source)
        elif _encode_executor_workers != workers:
            raise RuntimeError(
                f"Encode executor already initialized with {_encode_executor_workers} workers; "
                f"cannot reconfigure to {workers}."
            )
        return _encode_executor


def get_encode_executor() -> ThreadPoolExecutor:
    """Return the shared encode pool, lazily creating it with defaults if
    needed."""
    executor = _encode_executor
    if executor is not None:
        return executor
    return configure_encode_executor()


def shutdown_encode_executor() -> None:
    """Tear down the shared encode pool (registered with atexit on
    creation)."""
    global _encode_executor, _encode_executor_workers
    with _encode_executor_lock:
        executor = _encode_executor
        _encode_executor = None
        _encode_executor_workers = None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # Multimodal models like Qwen3-Omni ship the chat template in a standalone
    # chat_template.json (loaded by AutoProcessor) rather than tokenizer_config.json,
    # so AutoTokenizer leaves chat_template unset. Backfill from the sidecar file.
    if getattr(tokenizer, "chat_template", None) is None and os.path.isdir(name_or_path):
        chat_template_path = os.path.join(name_or_path, "chat_template.json")
        if os.path.isfile(chat_template_path):
            with open(chat_template_path) as f:
                chat_template = json.load(f).get("chat_template")
            if chat_template:
                tokenizer.chat_template = chat_template
                logger.info(f"Loaded chat_template from {chat_template_path}")
    return tokenizer


def _is_kimi_k25_style_processor(processor: object) -> bool:
    """Detect Kimi-K2.x VLM processors whose ``__call__`` takes
    ``medias=[{...}]`` instead of the standard HF
    ``images=``/``videos=``/``audio=`` triple.

    Duck-typing on ``media_processor`` (K2.x's bespoke attribute) plus a class
    name prefix fallback so future K2.* renames don't silently fall through.
    """
    if hasattr(processor, "media_processor"):
        return True
    name = type(processor).__name__
    return name.startswith("KimiK2") and name.endswith("Processor")


def adapt_processor_kwargs(
    processor: object,
    multimodal_inputs: dict | None,
    extra_kwargs: dict | None = None,
) -> dict:
    """Translate Relax's ``{images, videos, audio}`` shape into kwargs that the
    given HF processor's ``__call__`` actually accepts, then merge
    ``extra_kwargs``.

    Default path: returns ``{**(multimodal_inputs or {}), **(extra_kwargs or {})}``
    unchanged — covers Qwen-VL / Qwen-Omni and the rest of the HF zoo.

    Kimi-K2.x VLM path: ``KimiK25Processor.__call__`` requires either ``messages``
    or ``(medias, text)`` and rejects the standard ``images=`` keyword via its
    ``if messages is None and (medias is None or text is None)`` guard. We
    translate ``images=[PIL]`` to ``medias=[{"type":"image","image":PIL}]`` and
    drop the now-unused ``videos``/``audio`` (warning if non-empty so callers
    notice when video/audio inputs reach this codepath unconverted). Also forces
    ``return_tensors="pt"`` and discards the per-modality ``*_kwargs`` from
    ``build_processor_kwargs`` because K2.x's signature ignores them — and would
    explode if both this function and ``extra_kwargs`` named the same key.
    """
    mm = multimodal_inputs or {}
    extra = extra_kwargs or {}

    if not _is_kimi_k25_style_processor(processor):
        return {**mm, **extra}

    medias: list[dict] = []
    if images := mm.get("images"):
        medias.extend({"type": "image", "image": img} for img in images)
    if mm.get("videos"):
        logger.warning(
            "K2.x processor adapter received videos but only image translation is implemented; "
            "video inputs are being dropped."
        )
    if mm.get("audio"):
        logger.warning(
            "K2.x processor adapter received audio but K2.x VLMs do not consume audio; audio inputs are being dropped."
        )

    adapted: dict = {}
    if medias:
        adapted["medias"] = medias
    adapted["return_tensors"] = "pt"
    if extra:
        dropped = sorted(k for k in extra if k not in adapted)
        if dropped:
            _warn_dropped_kimi_k25_kwargs(tuple(dropped))
    return adapted


_WARNED_DROPPED_KIMI_KWARGS: set[tuple[str, ...]] = set()


def _warn_dropped_kimi_k25_kwargs(dropped: tuple[str, ...]) -> None:
    """Warn once per unique dropped-key set; K2.x's ``__call__`` ignores most
    of the standard HF processor kwargs and we don't want per-sample log
    spam."""
    if dropped in _WARNED_DROPPED_KIMI_KWARGS:
        return
    _WARNED_DROPPED_KIMI_KWARGS.add(dropped)
    logger.warning(f"K2.x processor adapter dropped unsupported kwargs: {list(dropped)}")


# K2.x HF processor output → KimiK25VLModel.forward kwargs.
# The bridge model in megatron-bridge uses `image_grid_thw` (matching the rest
# of HF VLM convention), but the K2.x HF processor returns the field as
# `grid_thws`.  Rename here so kwargs unpacked into the bridge model match.
_KIMI_K25_OUTPUT_RENAME = {"grid_thws": "image_grid_thw"}


def remap_mm_train_inputs(processor: object, train_inputs: dict | None) -> dict | None:
    """Rename HF-processor output keys to match the bridge model's forward
    kwargs.

    No-op for non-K2.x processors. For K2.x, applies
    ``_KIMI_K25_OUTPUT_RENAME``.
    """
    if not train_inputs:
        return train_inputs
    if not _is_kimi_k25_style_processor(processor):
        return train_inputs
    return {_KIMI_K25_OUTPUT_RENAME.get(k, k): v for k, v in train_inputs.items()}


# Per-process cache of the ``<|media_pad|>`` token id keyed by processor identity.
# convert_tokens_to_ids on TikToken is cheap but we hit this once per sample on the
# rollout-side hot path. Use weakref.finalize to evict on GC so id() reuse can't
# return a stale value.
_KIMI_K25_PLACEHOLDER_ID_CACHE: dict[int, int] = {}


def _kimi_k25_placeholder_id(processor: object) -> int:
    key = id(processor)
    cached = _KIMI_K25_PLACEHOLDER_ID_CACHE.get(key)
    if cached is None:
        cached = int(processor.tokenizer.convert_tokens_to_ids("<|media_pad|>"))
        _KIMI_K25_PLACEHOLDER_ID_CACHE[key] = cached
        weakref.finalize(processor, _KIMI_K25_PLACEHOLDER_ID_CACHE.pop, key, None)
    return cached


def _kimi_k25_image_feature_lengths(processor: object, grid_thws) -> list[int]:
    """Per-image post-merger token count.

    ``MoonViT3d.tpool_patch_merger`` collapses temporal frames via mean and reshapes
    the spatial grid by ``merge_kernel_size``, so per image the projector emits
    ``(h_patches // mh) * (w_patches // mw)`` tokens regardless of T.

    Note: the image processor stores ``merge_kernel_size`` as a single int (square
    kernel) in ``media_proc_cfg``, while the model's vision_config stores it as a
    ``(mh, mw)`` tuple. Accept both shapes.
    """
    mks = processor.media_processor.media_proc_cfg["merge_kernel_size"]
    if isinstance(mks, (int, float)):
        mh = mw = int(mks)
    else:
        mh, mw = int(mks[0]), int(mks[1])
    if isinstance(grid_thws, torch.Tensor):
        grid_thws = grid_thws.tolist()
    return [int((h // mh) * (w // mw)) for (_, h, w) in grid_thws]


def sanitize_kimi_k25_response_tokens(
    processor: object,
    response_tokens: list[int],
    *,
    replacement_id: int | None = None,
) -> list[int]:
    """Replace stray ``<|media_pad|>`` tokens hallucinated by the model in
    rollout responses.

    K2.x VLMs reserve ``<|media_pad|>`` for vision-input slots. The model is
    not supposed to emit it as part of generation, but freshly-cast or
    early-step checkpoints occasionally do. Each stray placeholder inflates
    ``num_placeholders`` past ``sum(feature_lengths)`` in the bridge's
    ``_merge_input_ids_with_image_features``, which falls into the dynamic-
    expansion path and broadcasts a single ``feature_lengths[0]=N`` across all
    pre-expanded N positions, producing an ``N²`` allocation that OOMs.

    Replace (don't strip) so positional accounting in
    ``sample.tokens``/``rollout_tokens``/``loss_mask`` stays consistent with
    sglang's per-token logprobs.
    """
    if not _is_kimi_k25_style_processor(processor) or not response_tokens:
        return response_tokens
    placeholder_id = _kimi_k25_placeholder_id(processor)
    if replacement_id is None:
        replacement_id = int(getattr(processor.tokenizer, "pad_token_id", 0) or 0)
    return [replacement_id if t == placeholder_id else t for t in response_tokens]


def expand_kimi_k25_placeholders(
    processor: object,
    prompt_ids: list[int],
    train_inputs: dict | None,
) -> list[int]:
    """Pre-expand ``<|media_pad|>`` tokens for K2.x VLMs.

    The K2.x HF processor emits exactly one ``<|media_pad|>`` per image, but the
    bridge model's vision tower produces N tokens per image where
    ``N = (h_patches // mh) * (w_patches // mw)``. Without pre-expansion the bridge
    falls into its dynamic-expansion path, which grows the sequence length mid-forward
    and invalidates the ``cu_seqlens`` Megatron's THD-format rotary embedding has
    already split on. Pre-expanding here keeps ``num_placeholders == sum(feature_lengths)``
    so the bridge takes the 1:1 pre-expanded branch and ``packed_seq_params`` stays
    consistent with the actual sequence length.
    """
    if not _is_kimi_k25_style_processor(processor) or not train_inputs:
        return prompt_ids
    grid_thws = train_inputs.get("image_grid_thw")
    if grid_thws is None or len(grid_thws) == 0:
        return prompt_ids

    placeholder_id = _kimi_k25_placeholder_id(processor)
    feature_lengths = _kimi_k25_image_feature_lengths(processor, grid_thws)

    expanded: list[int] = []
    feat_idx = 0
    for tid in prompt_ids:
        if tid == placeholder_id:
            if feat_idx >= len(feature_lengths):
                raise RuntimeError(
                    "K2.x prompt has more <|media_pad|> tokens than images in grid_thws "
                    f"({feat_idx + 1} vs {len(feature_lengths)}); "
                    f"len(prompt_ids)={len(prompt_ids)}, first_tokens={prompt_ids[:16]}"
                )
            expanded.extend([placeholder_id] * feature_lengths[feat_idx])
            feat_idx += 1
        else:
            expanded.append(tid)
    if feat_idx != len(feature_lengths):
        raise RuntimeError(
            "K2.x prompt has fewer <|media_pad|> tokens than images in grid_thws "
            f"({feat_idx} vs {len(feature_lengths)}); "
            f"len(prompt_ids)={len(prompt_ids)}, first_tokens={prompt_ids[:16]}"
        )
    return expanded


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:
    forced = {
        # force return_tensors to None for input_ids
        "return_tensors": None,
    }
    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    result.update(forced)

    # set return_tensors="pt" for modality-specific outputs
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def process_vision_info(prompt, processor, use_audio_in_video, config=None):
    # temporary solution, will write image utils for slime later
    from relax.utils.multimodal.process import process_multimodal_info

    if hasattr(processor.image_processor, "patch_size"):
        image_patch_size = processor.image_processor.patch_size
    else:
        logger.info(f"Using default patch size: {DEFAULT_PATCH_SIZE}")
        image_patch_size = DEFAULT_PATCH_SIZE
    images, videos, audios = process_multimodal_info(
        prompt, image_patch_size=image_patch_size, use_audio_in_video=use_audio_in_video, config=config
    )
    multimodal_inputs = {"images": images, "videos": videos, "audio": audios}
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_video_tensor_for_rollout_engine(video: torch.Tensor) -> str:
    """
    video: Tensor[T, C, H, W], RGB, uint8
    return: base64 encoded mp4
    """
    if video.dtype != torch.uint8:
        video = video.clamp(0, 255).to(torch.uint8)

    video_np = video.permute(0, 2, 3, 1).cpu().numpy()  # T,H,W,C

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        writer = imageio.get_writer(
            f.name,
            fps=4,
            codec="libx264",
        )
        for frame in video_np:
            writer.append_data(frame)
        writer.close()

        with open(f.name, "rb") as rf:
            return base64.b64encode(rf.read()).decode("utf-8")


def encode_audio_for_rollout_engine(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    """Encode audio waveform into WAV base64 string for sglang rollout.

    Args:
        audio: np.ndarray, shape (N,) or (N, C), float32
        sample_rate: audio sampling rate

    Returns:
        base64 encoded wav string
    """
    if audio is None:
        return None

    if not isinstance(audio, np.ndarray):
        audio = np.asarray(audio)

    audio = audio.astype(np.float32)

    buffer = io.BytesIO()
    sf.write(
        buffer,
        audio,
        samplerate=sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    # For sglang, it needs base64 format audio file starting with "data:,"
    return "data:," + base64.b64encode(buffer.getvalue()).decode("utf-8")


async def async_encode_image_for_rollout_engine(image) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_encode_executor(), encode_image_for_rollout_engine, image)


async def async_encode_video_tensor_for_rollout_engine(video: torch.Tensor) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_encode_executor(), encode_video_tensor_for_rollout_engine, video)


async def async_encode_audio_for_rollout_engine(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_encode_executor(), encode_audio_for_rollout_engine, audio, sample_rate)
