# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image-reference descriptors for rank-side SFT image preprocessing.

``--sft-image-preprocess-on-rank`` keeps large pixel tensors out of
TransferQueue: the producer still runs the full processor (tokens, loss masks,
grid, oversize filtering) but ships an ordered list of image file references
instead of ``pixel_values``. Every vision rank rebuilds the pixels on its own
CPU worker during the next-step prefetch.

The descriptor is a small JSON-able dict so it survives the TransferQueue
round-trip as plain metadata (``dict_to_tensordict`` stores the field as-is
and the consumer unwraps it like ``multimodal_train_inputs``).
"""

from argparse import Namespace
from typing import Any

import torch


# TransferQueue field carrying per-sample image-ref descriptors (dict | None).
SFT_IMAGE_REFS_FIELD = "multimodal_image_refs"
# Worker-to-producer capability marker; removed before transport to training.
SFT_HF_IMAGE_REBUILD_SUPPORTED = "_sft_hf_image_rebuild_supported"

# Both fields contain per-sample Python payloads, not tensors to concatenate.
MULTIMODAL_PAYLOAD_FIELDS = ("multimodal_train_inputs", SFT_IMAGE_REFS_FIELD)


# Bumped whenever the rebuild contract changes (processor config, dtype rule,
# descriptor layout) so stale partitions cannot be consumed silently.
SFT_IMAGE_REF_DESCRIPTOR_VERSION = 1

# Kimi K3 SFT image path: the rank-side rebuild runs
# ``build_kimi_k3_image_features``, mirroring the producer's
# ``process_kimi_k3_sft_images`` pixel pipeline (preprocess_medias +
# media_processor.preprocess + the shared BF16 downcast rule).
KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND = "kimi_k3_sft_image_v1"

# Generic HF processor path (Qwen-VL family and friends): the rank-side
# rebuild runs ``build_hf_image_features``, mirroring the producer's
# ``adapt_processor_kwargs`` + full-processor call — pixel-only, via the
# processor's own ``image_processor`` with its config defaults.
HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND = "hf_processor_image_v1"

SUPPORTED_IMAGE_REF_DESCRIPTOR_KINDS = frozenset(
    {KIMI_K3_SFT_IMAGE_DESCRIPTOR_KIND, HF_PROCESSOR_IMAGE_DESCRIPTOR_KIND}
)


def sft_multimodal_data_fields(args: Namespace) -> list[str]:
    """Use the same multimodal transport schema for SFT train and eval."""
    if getattr(args, "multimodal_keys", None) is None:
        return []
    if getattr(args, "sft_image_preprocess_on_rank", False):
        return list(MULTIMODAL_PAYLOAD_FIELDS)
    return ["multimodal_train_inputs"]


def _as_image_list(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def _require_file_reference(value: Any, *, position: int, idx: int, source_name: str) -> str:
    """Validate one image source as a node-stable file reference."""
    path: str | None = None
    if isinstance(value, str):
        path = value
    elif (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and "bytes" not in value
        and "base64" not in value
    ):
        path = value["path"]
    if path is not None and not path.startswith("data:"):
        return path
    raise ValueError(
        f"Rank-side image reconstruction requires file-path image sources that every training node can read, "
        f"but sample idx={idx} in {source_name!r} carries {type(value).__name__} at image position {position}. "
        "Materialize inline payloads (base64 / data URIs / raw bytes) into shared-storage files "
        "to use reference transport for this sample."
    )


def extract_stable_image_refs(images: Any, *, idx: int, source_name: str) -> list[str]:
    """Return the ordered file references for a sample's images.

    The order matches the sample's image placeholders, so the rebuilt
    ``pixel_values`` rows line up with the producer's grid.
    """
    return [
        _require_file_reference(image, position=position, idx=idx, source_name=source_name)
        for position, image in enumerate(_as_image_list(images))
    ]


def build_image_ref_descriptor(
    image_refs: list[str], pixel_values: torch.Tensor, grid: Any, kind: str
) -> dict[str, Any]:
    """Describe the dropped pixels so the consumer can rebuild and check them.

    ``kind`` names the producer pixel pipeline that produced these pixels; the
    rank-side rebuild dispatches on it and validates its own grid against the
    descriptor's inside the prefetch worker, where failures are covered by the
    cross-rank prefetch agreement instead of diverging a single training rank.
    """
    if kind not in SUPPORTED_IMAGE_REF_DESCRIPTOR_KINDS:
        raise ValueError(f"unsupported image-ref descriptor kind {kind!r}")
    return {
        "version": SFT_IMAGE_REF_DESCRIPTOR_VERSION,
        "kind": kind,
        "image_refs": list(image_refs),
        "pixel_shape": [int(dim) for dim in pixel_values.shape],
        "image_grid_thw": [[int(t), int(h), int(w)] for t, h, w in torch.as_tensor(grid).tolist()],
    }


def validate_image_ref_descriptor(descriptor: Any) -> dict[str, Any]:
    """Check descriptor version/kind before rebuilding pixels from it."""
    if not isinstance(descriptor, dict):
        raise ValueError(f"image-ref descriptor must be a dict, got {type(descriptor).__name__}")
    version = descriptor.get("version")
    if version != SFT_IMAGE_REF_DESCRIPTOR_VERSION:
        raise ValueError(
            f"unsupported image-ref descriptor version {version!r} (expected {SFT_IMAGE_REF_DESCRIPTOR_VERSION})"
        )
    kind = descriptor.get("kind")
    if kind not in SUPPORTED_IMAGE_REF_DESCRIPTOR_KINDS:
        raise ValueError(
            f"unsupported image-ref descriptor kind {kind!r}; supported kinds: "
            f"{sorted(SUPPORTED_IMAGE_REF_DESCRIPTOR_KINDS)}"
        )
    refs = descriptor.get("image_refs")
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
        raise ValueError("image-ref descriptor must carry a non-empty ordered list of string image_refs")
    return descriptor


def split_pixel_values_from_mm_inputs(
    mm_inputs: dict[str, Any],
    image_refs: list[str],
    *,
    idx: int,
    source_name: str,
    kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace ``pixel_values`` in a processed sample's mm inputs with a
    descriptor.

    Returns ``(mm_inputs_without_pixels, descriptor)``. The producer keeps the
    grid: metrics and non-vision PP stages still read it, and the consumer
    validates its rebuilt grid against it before re-attaching pixels.
    """
    pixel_values = mm_inputs.get("pixel_values")
    grid = mm_inputs.get("image_grid_thw")
    if pixel_values is None or grid is None:
        raise ValueError(
            f"sample idx={idx} in {source_name!r}: cannot replace pixel_values with an image reference; "
            f"mm inputs carry keys {sorted(mm_inputs)} and need both pixel_values and image_grid_thw."
        )
    grid = torch.as_tensor(grid)
    if grid.dim() != 2 or grid.shape[0] != len(image_refs):
        raise ValueError(
            f"sample idx={idx} in {source_name!r}: image grid rows {tuple(grid.shape)} do not match "
            f"{len(image_refs)} ordered image references."
        )
    retained = {key: value for key, value in mm_inputs.items() if key != "pixel_values"}
    return retained, build_image_ref_descriptor(image_refs, pixel_values, grid, kind)


def extract_image_ref_descriptors(raw_batch: Any) -> list[Any] | None:
    """Read per-sample image-ref descriptors from a raw (pre-conversion) batch.

    Used by the background SFT prefetch thread under ``--sft-image-preprocess-
    on-rank``, before ``get_data_from_transfer_queue`` converts the TensorDict
    payload: the returned descriptors feed the rank-side pixel rebuild while
    the training thread keeps training. Values may be ``None`` for text-only
    samples. Returns ``None`` when the batch carries no descriptor field.
    """
    if raw_batch is None:
        return None
    try:
        values = raw_batch[SFT_IMAGE_REFS_FIELD]
    except (KeyError, IndexError):
        return None
    from tensordict.tensorclass import NonTensorData

    descriptors: list[Any] = []
    for item in list(values):
        raw = item.data if isinstance(item, NonTensorData) else item
        if raw is None:
            descriptors.append(None)
        elif isinstance(raw, dict):
            descriptors.append(raw)
        elif hasattr(raw, "items"):
            descriptors.append(dict(raw.items()))
        else:
            raise TypeError(f"Unsupported image-ref descriptor payload type: {type(raw)}")
    return descriptors
