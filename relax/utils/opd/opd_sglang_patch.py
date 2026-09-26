# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey patch for SGLang Qwen-VL processor: accept PRE-EXPANDED input_ids
with raw image bytes.

Why
---
SGLang's multimodal ``/generate`` path decodes the request ``input_ids`` back
to text and RE-TOKENIZES it. For OPD this breaks alignment: the student-sampled
response tokens are NOT canonical BPE, so ``tokenize(detokenize(x)) != x`` ->
the teacher prefill sees a different token sequence/length than the student ->
"Teacher log-prob length mismatch", every sample falls back and the OPD signal
is lost.

What this patch does
--------------------
When the request carries::

    image_data = [
        {
            "format": "opd_preexpanded_raw",
            "images_b64": ["<base64 PNG>", ...],  # raw images (compressed)
            "image_grid_thw": [[t, h, w], ...],  # client-precomputed grid
        }
    ]

the patched method:
  * uses ``request_obj.input_ids`` AS-IS (already expanded with N
    ``<|image_pad|>`` tokens per image) -> NO decode, NO re-tokenize ->
    ZERO token drift;
  * decodes the raw images and runs the *server-side* HF image processor to
    get fresh ``pixel_values`` -> attaches them as ``feature`` so the
    teacher's own vision tower runs on them;
  * derives offsets / M-RoPE from the expanded input_ids + image_grid_thw,
    mirroring SGLang's own ``get_mm_data`` / ``process_mm_data_async`` logic.

Wire payload is ~10-100x smaller than shipping post-processed pixel_values
directly (raw PNG bytes vs. a [N_patches, C*ps²] float tensor), and there is
no client/server dedup state to maintain.

Any request WITHOUT the ``opd_preexpanded_raw`` marker falls through to the
original implementation unchanged, so text-only and normal raw-image requests
are not affected.

Usage
-----
Applied by ``_launch_server_with_patches()`` in ``sglang_engine.py`` when the
env flag ``RELAX_OPD_PREEXPANDED_PATCH=1`` is set. The patch is applied at
runtime via ``apply_opd_preexpanded_patch()``; otherwise it is a complete
no-op and sglang is left untouched. Idempotent: applying twice is a no-op.
"""

from __future__ import annotations

import base64
import io
import logging

import torch


logger = logging.getLogger("opd_sglang_patch")

PREEXPANDED_FORMAT = "opd_preexpanded_raw"
_PATCH_FLAG = "_opd_preexpanded_patched"


def _extract_preexpanded(image_data):
    """Return the single pre-expanded dict item, or None if not this fast
    path."""
    if not image_data or not isinstance(image_data, list):
        return None
    item = image_data[0]
    if isinstance(item, dict) and item.get("format") == PREEXPANDED_FORMAT:
        return item
    return None


def _decode_raw_images(item) -> list:
    """Decode base64-encoded raw images from the payload."""
    from PIL import Image

    images_b64 = item.get("images_b64")
    if not images_b64:
        raise ValueError("opd_preexpanded_raw payload missing 'images_b64'")
    if isinstance(images_b64, str):
        images_b64 = [images_b64]
    out = []
    for b64 in images_b64:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        out.append(img)
    return out


def apply_patch() -> bool:
    """Monkey patch QwenVLImageProcessor.process_mm_data_async.

    Idempotent.
    """
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
    )
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    if getattr(QwenVLImageProcessor, _PATCH_FLAG, False):
        logger.info("[opd-patch] already applied, skip")
        return False

    _orig = QwenVLImageProcessor.process_mm_data_async

    async def _patched(self, image_data, input_text, request_obj, *args, **kwargs):
        item = _extract_preexpanded(image_data)
        if item is None:
            # Not our fast path -> original behavior (text / raw-image unchanged)
            return await _orig(self, image_data, input_text, request_obj, *args, **kwargs)

        try:
            # --- 1) expanded input_ids, used AS-IS (no decode -> no re-tokenize) ---
            input_ids = getattr(request_obj, "input_ids", None)
            if input_ids is None and isinstance(input_text, list):
                input_ids = input_text
            if input_ids is None:
                raise ValueError(
                    "opd_preexpanded_raw requires request input_ids (the expanded token sequence); got None."
                )
            input_ids_t = torch.tensor(input_ids, dtype=torch.long)

            # --- 2) decode raw images and run server-side HF image processor ---
            images = _decode_raw_images(item)
            img_processor = self._processor.image_processor
            img_out = img_processor(images=images, return_tensors="pt")
            pixel_values = img_out["pixel_values"]
            grid = torch.as_tensor(img_out["image_grid_thw"], dtype=torch.long)
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)

            image_token_id = self.mm_tokens.image_token_id

            # --- 3) offsets: scan contiguous <|image_pad|> runs in expanded ids ---
            offsets = self.get_mm_items_offset(input_ids_t, image_token_id)

            mm_item = MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=offsets,
                feature=pixel_values,
            )
            mm_item.set("image_grid_thw", grid)

            # --- 4) M-RoPE from EXPANDED ids + grid (same as upstream get_mm_data) ---
            merge = self.hf_config.vision_config.spatial_merge_size
            mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=merge,
                image_token_id=image_token_id,
                video_token_id=self.mm_tokens.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                input_ids=input_ids_t.unsqueeze(0),
                image_grid_thw=grid,
                tokens_per_second=getattr(self.hf_config.vision_config, "tokens_per_second", None),
            )
            mrope_positions = mrope_positions.squeeze(1)

            logger.debug(
                "[opd-patch] fast path: len(input_ids)=%d n_images=%d grid=%s pixel_values=%s (NO re-tokenize)",
                len(input_ids),
                len(images),
                grid.tolist(),
                tuple(pixel_values.shape),
            )

            ret = {
                "input_ids": list(input_ids),
                "mm_items": [mm_item],
                "im_start_id": self.vision_start_token_id,
                "im_end_id": self.vision_end_token_id,
                "im_token_id": image_token_id,
                "video_token_id": self.mm_tokens.video_token_id,
                "audio_token_id": self.mm_tokens.audio_token_id,
                "mrope_positions": mrope_positions,
                "mrope_position_delta": mrope_position_delta,
            }
            # sglang >= 0.5.12 replaced the plain-dict return value of
            # ``process_mm_data_async`` with a typed ``MultimodalProcessorOutput``
            # dataclass; the TokenizerManager now consumes it via attribute
            # access (``mm_inputs.input_ids`` / ``.mm_items`` / ...). Wrap our
            # dict into that object when the type exists, and fall back to the
            # raw dict on older sglang so this patch stays version-agnostic.
            try:
                from sglang.srt.managers.schedule_batch import (
                    MultimodalProcessorOutput,
                )

                return MultimodalProcessorOutput.from_dict(ret)
            except Exception:
                return ret
        except Exception as exc:
            logger.exception("[opd-patch] fast path FAILED: %r", exc)
            raise

    QwenVLImageProcessor.process_mm_data_async = _patched
    setattr(QwenVLImageProcessor, _PATCH_FLAG, True)
    logger.info("[opd-patch] applied to QwenVLImageProcessor.process_mm_data_async (raw-image mode)")
    return True


def apply_opd_preexpanded_patch() -> None:
    """Apply the OPD pre-expanded multimodal patch in the current (server)
    process so the student SGLang TokenizerManager accepts
    ``image_data=[{"format":"opd_preexpanded_raw", ...}]`` requests and skips
    the decode->re-tokenize round-trip."""
    try:
        apply_patch()
    except Exception as e:  # never block server startup on the patch
        logger.warning("Failed to apply OPD pre-expanded patch: %r", e)


# No apply-on-import hook. The sole importer,
# ``sglang_engine.py::_launch_server_with_patches``, already checks
# ``RELAX_OPD_PREEXPANDED_PATCH`` before importing this module and then calls
# ``apply_opd_preexpanded_patch()`` explicitly, so an import-time gate would
# only apply the (idempotent) patch a second time — and would read the flag at
# import time, before a Ray worker's runtime_env is necessarily in play.
# Importing this module is therefore a complete no-op: sglang and all its other
# modules stay untouched until a caller opts in.
