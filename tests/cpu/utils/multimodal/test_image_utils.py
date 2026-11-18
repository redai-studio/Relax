# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import base64
import errno
from io import BytesIO
from typing import BinaryIO
from unittest.mock import Mock

import pytest
from PIL import Image

from relax.utils.multimodal import image_utils
from relax.utils.multimodal.image_utils import (
    decode_data_uri,
    get_resize_height_width,
    image_smart_resize,
    load_image,
    resize_qwen_vl_extreme_aspect_ratio,
    to_rgb,
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_get_resize_height_width_downscales_and_aligns_to_scale_factor():
    height, width = get_resize_height_width(None, 1000, 2000, 14, 200_000, 196)

    assert (height, width) == (308, 630)
    assert height % 14 == 0 and width % 14 == 0
    assert 196 <= height * width <= 200_000


def test_get_resize_height_width_upscales_to_minimum_pixels():
    height, width = get_resize_height_width(None, 10, 20, 4, 2_000, 1_000)

    assert (height, width) == (24, 48)
    assert height % 4 == 0 and width % 4 == 0
    assert 1_000 <= height * width <= 2_000


def test_get_resize_height_width_rejects_excessive_aspect_ratio():
    with pytest.raises(ValueError, match="Absolute aspect ratio"):
        get_resize_height_width(3.0, 100, 400, 2, 100_000, 1)


def test_get_resize_height_width_keeps_size_when_already_in_range():
    height, width = get_resize_height_width(None, 112, 112, 28, 28 * 28 * 1024, 28 * 28)

    assert (height, width) == (112, 112)


def test_get_resize_height_width_rounds_to_scale_factor_28():
    height, width = get_resize_height_width(None, 100, 100, 28, 28 * 28 * 1024, 28 * 28)

    assert (height, width) == (112, 112)


def test_image_smart_resize_preserves_mode_and_patch_alignment():
    image = Image.new("RGB", (200, 100), (12, 34, 56))

    resized = image_smart_resize(
        image,
        height=100,
        width=200,
        scale_factor=28,
        image_min_pixels=28 * 28,
        image_max_pixels=4 * 28 * 28,
    )

    assert resized.mode == "RGB"
    assert resized.size == (56, 28)
    assert resized.size[0] % 28 == 0 and resized.size[1] % 28 == 0
    assert 28 * 28 <= resized.width * resized.height <= 4 * 28 * 28


@pytest.mark.parametrize("size", [(750, 1), (1, 750), (201, 1), (1, 201)])
def test_resize_qwen_vl_extreme_aspect_ratio_makes_image_safe(size):
    image = Image.new("RGB", size, (12, 34, 56))

    resized = resize_qwen_vl_extreme_aspect_ratio(image)

    assert max(resized.size) / min(resized.size) < 200
    assert resized.mode == image.mode
    assert resized.width >= resized.height if image.width >= image.height else resized.height >= resized.width
    assert resize_qwen_vl_extreme_aspect_ratio(resized) is resized


@pytest.mark.parametrize("size", [(200, 1), (1, 200), (100, 100)])
def test_resize_qwen_vl_extreme_aspect_ratio_leaves_valid_image_unchanged(size):
    image = Image.new("RGB", size, (12, 34, 56))

    assert resize_qwen_vl_extreme_aspect_ratio(image) is image


def test_to_rgb_composites_rgba_over_white():
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 128))

    converted = to_rgb(image)

    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (255, 127, 127)


def test_to_rgb_converts_non_rgba_image():
    image = Image.new("L", (1, 1), 42)

    converted = to_rgb(image)

    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (42, 42, 42)


def test_to_rgb_composites_fully_transparent_over_white():
    image = Image.new("RGBA", (1, 1), (10, 20, 30, 0))

    converted = to_rgb(image)

    assert converted.mode == "RGB"
    assert converted.getpixel((0, 0)) == (255, 255, 255)


def test_decode_data_uri_supports_base64_and_url_encoded_payloads():
    payload = b"image bytes / 1, 2"
    base64_uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    text_uri = "data:text/plain," + "image%20bytes%20%2F%201%2C%202"

    assert decode_data_uri(base64_uri) == payload
    assert decode_data_uri(text_uri) == payload


def test_decode_data_uri_rejects_missing_payload():
    with pytest.raises(ValueError, match="Malformed data URI"):
        decode_data_uri("data:image/png;base64,")


@pytest.mark.parametrize(
    "input_kind", ["pil", "bytes", "bytearray", "dict_bytes", "dict_bytes_base64", "dict_base64", "data_uri"]
)
def test_load_image_supports_memory_input_types(input_kind):
    source = Image.new("RGB", (3, 2), (1, 2, 3))
    encoded = _png_bytes(source)
    inputs = {
        "pil": source,
        "bytes": encoded,
        "bytearray": bytearray(encoded),
        "dict_bytes": {"bytes": encoded},
        "dict_bytes_base64": {"bytes": base64.b64encode(encoded).decode(), "path": None},
        "dict_base64": {"base64": base64.b64encode(encoded).decode()},
        "data_uri": "data:image/png;base64," + base64.b64encode(encoded).decode(),
    }

    loaded = load_image(inputs[input_kind])

    assert loaded.size == (3, 2)
    assert loaded.convert("RGB").getpixel((0, 0)) == (1, 2, 3)


def test_load_image_supports_local_path_and_file_uri(tmp_path):
    source = Image.new("RGB", (2, 2), (9, 8, 7))
    path = tmp_path / "sample.png"
    source.save(path)

    from_path = load_image(str(path))
    from_uri = load_image(path.as_uri())

    assert from_path.size == (2, 2)
    assert from_uri.getpixel((1, 1)) == (9, 8, 7)


@pytest.mark.parametrize("during_decode", [False, True])
def test_load_image_retries_transient_local_io(monkeypatch: pytest.MonkeyPatch, during_decode: bool) -> None:
    payload = _png_bytes(Image.new("RGB", (2, 2), (9, 8, 7)))
    first = BytesIO(payload)
    second = BytesIO(payload)
    error = OSError(errno.EIO, "temporary filesystem error")
    open_file = Mock(side_effect=[first if during_decode else error, second])
    sleep = Mock()
    monkeypatch.setattr(image_utils, "open", open_file, raising=False)
    monkeypatch.setattr(image_utils.time, "sleep", sleep)
    if during_decode:
        open_image = Image.open

        def fail_first_decode(stream: BinaryIO) -> Image.Image:
            decoded = open_image(stream)
            if stream is first:
                monkeypatch.setattr(decoded, "load", Mock(side_effect=error))
            return decoded

        monkeypatch.setattr(Image, "open", fail_first_decode)

    loaded = load_image("file:///sample.png")

    assert loaded.getpixel((1, 1)) == (9, 8, 7)
    assert open_file.call_count == 2
    assert second.closed and (not during_decode or first.closed)
    sleep.assert_called_once_with(0.2)


def test_load_image_surfaces_io_error_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    open_file = Mock(side_effect=OSError(errno.EIO, "persistent filesystem error"))
    sleep = Mock()
    monkeypatch.setattr(image_utils, "open", open_file, raising=False)
    monkeypatch.setattr(image_utils.time, "sleep", sleep)

    with pytest.raises(OSError, match="persistent filesystem error") as error:
        load_image("/sample.png")

    assert error.value.errno == errno.EIO
    assert open_file.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.2, 0.4]


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (FileNotFoundError(errno.ENOENT, "missing"), AssertionError),
        (NotADirectoryError(errno.ENOTDIR, "invalid parent directory"), AssertionError),
        (PermissionError(errno.EACCES, "permission denied"), PermissionError),
        (None, Image.UnidentifiedImageError),
    ],
)
def test_load_image_does_not_retry_permanent_errors(
    monkeypatch: pytest.MonkeyPatch, failure: OSError | None, expected: type[Exception]
) -> None:
    open_file = Mock(side_effect=failure, return_value=BytesIO(b"not an image"))
    sleep = Mock()
    monkeypatch.setattr(image_utils, "open", open_file, raising=False)
    monkeypatch.setattr(image_utils.time, "sleep", sleep)

    with pytest.raises(expected):
        load_image("/sample.png")

    open_file.assert_called_once_with("/sample.png", "rb")
    sleep.assert_not_called()


def test_load_image_supports_truncated_jpeg():
    source = Image.new("RGB", (8, 8), (1, 2, 3))
    buffer = BytesIO()
    source.save(buffer, format="JPEG")

    loaded = load_image(buffer.getvalue()[:-4])

    assert loaded.size == (8, 8)


def test_load_image_rejects_unsupported_type():
    with pytest.raises(NotImplementedError, match="Unsupported image input type"):
        load_image(123)
