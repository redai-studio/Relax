# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import base64
import math
import os
import time
import urllib.parse
from io import BytesIO
from typing import Any, ByteString, Dict, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image

from relax.utils.env import Envs

from .config import MultimodalConfig, get_image_max_token_num, get_image_min_token_num, get_image_resize_scale_factor


SPATIAL_MERGE_SIZE = 2
QWEN_VL_MAX_ASPECT_RATIO = 200
QWEN_VL_SAFE_ASPECT_RATIO = 199


ImageInput = Union[
    Image.Image,
    np.ndarray,
    ByteString,
    str,
]


def resize_qwen_vl_extreme_aspect_ratio(image: Image.Image) -> Image.Image:
    """Resize images that exceed Qwen-VL's hard aspect-ratio limit.

    Qwen-VL processors reject images whose long-to-short-side ratio is greater
    than 200 before their normal pixel and patch-alignment resize runs. Expand
    only the short side to keep the long-side pixels intact; Qwen's processor
    remains responsible for its checkpoint-specific pixel and patch limits.
    """
    height, width = image.height, image.width
    short_side = max(min(height, width), 1)
    aspect_ratio = max(height, width) / short_side
    if aspect_ratio <= QWEN_VL_MAX_ASPECT_RATIO:
        return image

    target_short_side = math.ceil(max(height, width) / QWEN_VL_SAFE_ASPECT_RATIO)
    if height > width:
        target_height = height
        target_width = target_short_side
    else:
        target_width = width
        target_height = target_short_side
    return image.resize((target_width, target_height), Image.LANCZOS)


def get_resize_height_width(
    max_ratio: Optional[float],
    height: int,
    width: int,
    scale_factor: Optional[int],
    max_pixels: int,
    min_pixels: int,
) -> Tuple[int, int]:
    """Compute a resized height and width that respects pixel and aspect
    constraints.

    Parameters
    - max_ratio: Optional maximum allowed aspect ratio (larger dimension / smaller dimension).
    - height: Original image height in pixels.
    - width: Original image width in pixels.
    - scale_factor: Optional integer multiple that the returned dimensions should align to.
    - max_pixels: Maximum allowed total pixels (height * width) after resize.
    - min_pixels: Minimum required total pixels after resize.

    Returns
    - Tuple `(new_height, new_width)` of integers that satisfy the constraints.
    """
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"Absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)

    return h_bar, w_bar


def image_smart_resize(
    image: Image.Image,
    height: int,
    width: int,
    scale_factor: Optional[int] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    max_ratio: Optional[float] = None,
    config: MultimodalConfig = None,
    **kwargs: Any,
) -> Image.Image:
    """Resize a PIL image while respecting token/pixel constraints.

    Parameters
    - image: Input `PIL.Image` to resize.
    - height, width: Target (original) dimensions used as baseline for resizing.
    - scale_factor: Optional integer alignment for the resulting dimensions.
    - image_min_pixels, image_max_pixels: Optional pixel bounds; if omitted,
    - max_ratio: Optional maximum allowed aspect ratio.

    Returns
    - A resized `PIL.Image`.
    """
    image_max_pixels = (
        image_max_pixels if image_max_pixels is not None else (get_image_max_token_num(config) * scale_factor**2)
    )
    image_min_pixels = (
        image_min_pixels if image_min_pixels is not None else (get_image_min_token_num(config) * scale_factor**2)
    )
    assert image_max_pixels >= image_min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    h_bar, w_bar = get_resize_height_width(max_ratio, height, width, scale_factor, image_max_pixels, image_min_pixels)
    image = image.resize((w_bar, h_bar))
    return image


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert an image to RGB, compositing alpha over white when needed."""
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def load_image_from_path(image: str, **kwargs: Any) -> Image.Image:
    """Load an image from a local path or HTTP(S) URL.

    Parameters
    - image: Local filesystem path, `file://` URI, or HTTP(S) URL.
    - **kwargs: Ignored, present for forward compatibility.

    Returns
    - A `PIL.Image` instance opened from the provided path/URL.
    """
    if image.startswith("data:image/"):
        header, _, encoded = image.partition(",")
        if ";base64" not in header or not encoded:
            raise ValueError("data:image payload must use 'data:image/...;base64,...' format")
        with BytesIO(base64.b64decode(encoded)) as bio:
            image_obj = Image.open(bio)
            image_obj.load()
        return image_obj

    if image.startswith(("http://", "https://")):
        # Read per call rather than at import: a Ray worker's runtime_env is
        # applied to the process, and reading here keeps the knob tunable
        # regardless of when this module first got imported.
        attempts = max(1, Envs.ROLLOUT_IMAGE_FETCH_ATTEMPTS)
        backoff = Envs.ROLLOUT_IMAGE_FETCH_BACKOFF_S
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                with requests.get(image, stream=True, timeout=Envs.ROLLOUT_IMAGE_FETCH_TIMEOUT_S) as response:
                    response.raise_for_status()
                    with BytesIO(response.content) as bio:
                        image_obj = Image.open(bio)
                        image_obj.load()  # force pixel read before bio/response close (Image.open is lazy)
                break
            except Exception as exc:  # transient network / HTTP — retry with backoff
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(backoff * (attempt + 1))
        else:
            raise last_exc
    else:
        if image.startswith("file://"):
            image = image[7:]
        assert os.path.exists(image), f"Image path {image} does not exist."
        with open(image, "rb") as f:
            image_obj = Image.open(f)
            image_obj.load()

    return image_obj


def load_image_from_bytes(image: bytes, **kwargs: Any) -> Image.Image:
    """Load a `PIL.Image` from raw image bytes.

    Parameters
    - image: Raw image bytes (e.g. PNG/JPEG file contents).
    - **kwargs: Ignored, present for forward compatibility.

    Returns
    - A `PIL.Image` instance.
    """
    # Fully decode before the backing buffer goes out of scope. `Image.open`
    # is lazy and keeps a reference to the underlying file object; if the
    # `BytesIO` is garbage-collected (or the decode is deferred onto another
    # thread, as the Qwen-VL processor does), a later `.load()` fails with
    # truncated-stream errors ("assert self.png is not None" / OSError:
    # unrecognized data stream contents).
    with BytesIO(image) as bio:
        image_obj = Image.open(bio)
        image_obj.load()
    return image_obj


def decode_data_uri(uri: str) -> bytes:
    """Decode an RFC 2397 `data:` URI to raw bytes.

    Handles both `;base64` and URL-encoded payloads. Caller is responsible for
    confirming the input starts with `data:`.
    """
    head, _, body = uri.partition(",")
    if not body:
        raise ValueError(f"Malformed data URI: {uri[:32]!r}")
    if "base64" in head.lower():
        return base64.b64decode(body)
    return urllib.parse.unquote_to_bytes(body)


def load_image(image: ImageInput, **kwargs: Any) -> Image.Image:
    """Generic loader for different image input types.

    Parameters
    - image: One of:
        - `PIL.Image.Image`: returned as-is.
        - `str`: a path, URL, or `data:` URI.
        - `bytes`: raw image bytes.
        - `dict`: with one of `bytes`, `base64`, or `path` fields
          (HF `datasets` / OpenAI-style payloads).

    Returns
    - A `PIL.Image` instance.
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        if image.startswith("data:"):
            return load_image_from_bytes(decode_data_uri(image), **kwargs)
        return load_image_from_path(image, **kwargs)
    if isinstance(image, (bytes, bytearray)):
        return load_image_from_bytes(bytes(image), **kwargs)
    if isinstance(image, dict):
        raw = image.get("bytes")
        if isinstance(raw, (bytes, bytearray)):
            return load_image_from_bytes(bytes(raw), **kwargs)
        b64 = image.get("base64")
        if isinstance(b64, str):
            return load_image_from_bytes(base64.b64decode(b64), **kwargs)
        path = image.get("path")
        if isinstance(path, str):
            if path.startswith("data:"):
                return load_image_from_bytes(decode_data_uri(path), **kwargs)
            return load_image_from_path(path, **kwargs)
    raise NotImplementedError(f"Unsupported image input type: {type(image)}")


def fetch_image(
    info: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14, config: MultimodalConfig = None
) -> Image.Image:
    """Extract, load and resize an image according to `info` metadata.

    Parameters
    - info: Mapping containing either `"image"` (a `PIL.Image` or other supported
      image input) or `"image_url"`/path. May also contain resizing keys:
      `resized_height`, `resized_width`, `min_pixels`, `max_pixels`.
    - image_patch_size: Base patch size used to compute alignment and pixel limits.

    Returns
    - A resized `PIL.Image` in `RGB` mode.
    """
    if "image" in info:
        image = info["image"]
    else:
        image = info["image_url"]

    # load image
    image_obj = None
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    if isinstance(image, Image.Image):
        image_obj = image
    else:
        image_obj = load_image(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    # resize
    config_scale_factor = get_image_resize_scale_factor(config)
    if config_scale_factor is None:
        resize_scale_factor = patch_factor
    elif config_scale_factor == 0:
        resize_scale_factor = None
    else:
        resize_scale_factor = config_scale_factor

    if "resized_height" in info and "resized_width" in info:
        image = image_smart_resize(
            image,
            info["resized_height"],
            info["resized_width"],
            scale_factor=resize_scale_factor,
            image_min_pixels=get_image_min_token_num(config) * patch_factor**2,
            image_max_pixels=get_image_max_token_num(config) * patch_factor**2,
            config=config,
        )
    else:
        width, height = image.size
        min_pixels = info.get("min_pixels", get_image_min_token_num(config) * patch_factor**2)
        max_pixels = info.get("max_pixels", get_image_max_token_num(config) * patch_factor**2)
        image = image_smart_resize(
            image,
            height,
            width,
            scale_factor=resize_scale_factor,
            image_min_pixels=min_pixels,
            image_max_pixels=max_pixels,
            config=config,
        )

    return image
