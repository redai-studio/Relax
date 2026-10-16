# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Producer-side policy for transporting image references instead of pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from relax.utils.data.image_refs import (
    HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND,
    KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND,
    SFT_HF_IMAGE_REBUILD_SUPPORTED,
    extract_stable_image_refs,
    split_pixel_values_from_mm_inputs,
)
from relax.utils.data.kimi_k3 import KIMI_K3_SFT_REQUEST
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from relax.engine.sft.dataset.streaming import _RenderedSample

logger = get_logger(__name__)


class ImageReferenceTransport:
    def __init__(self, image_preprocess_on_rank: bool) -> None:
        self.enabled = image_preprocess_on_rank

    def capture(self, images: list[Any] | None, idx: int, source_name: str) -> tuple[list[str] | None, bool]:
        image_refs = None
        image_fallback = False
        if self.enabled and images:
            # Captured before any decode: the references must keep the sample's
            # placeholder order; inline payloads keep the existing pixel path.
            try:
                image_refs = extract_stable_image_refs(images, idx=idx, source_name=source_name)
            except ValueError as exc:
                # Per-sample fallback to the old pixel path; the rest of the
                # dataset keeps the reference mode (mixed batches are fine:
                # rows without a descriptor simply ship their pixels).
                logger.warning(
                    f"SFTStreamingDataset[image-preprocess-on-rank fallback]: sample idx={idx} in "
                    f"{source_name!r} has no usable file references; shipping its pixel payload. "
                    f"{exc}"
                )
                image_fallback = True
        return image_refs, image_fallback

    def split(
        self, mm_inputs: dict[str, Any] | None, rendered: _RenderedSample, source_name: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        image_refs_descriptor = None
        hf_rebuild_supported = mm_inputs.pop(SFT_HF_IMAGE_REBUILD_SUPPORTED, False) if mm_inputs else False
        if (
            self.enabled
            and mm_inputs
            and not rendered.image_fallback
            and rendered.image_refs
            and mm_inputs.get("pixel_values") is not None
            and mm_inputs.get("image_grid_thw") is not None
        ):
            # Stamp the descriptor with the pixel pipeline that actually ran:
            # the rank side rebuilds by dispatching on this kind, so an
            # unrecognized encoding path keeps pixels instead of shipping a
            # descriptor nobody can rebuild.
            if rendered.processor_kwargs and KIMI_K3_SFT_REQUEST in rendered.processor_kwargs:
                descriptor_kind = KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND
            elif rendered.rendered_text is not None and hf_rebuild_supported:
                descriptor_kind = HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND
            else:
                return mm_inputs, None
            mm_inputs, image_refs_descriptor = split_pixel_values_from_mm_inputs(
                mm_inputs,
                rendered.image_refs,
                idx=rendered.idx,
                source_name=source_name,
                kind=descriptor_kind,
            )
        return mm_inputs, image_refs_descriptor
