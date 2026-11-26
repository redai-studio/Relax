# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.version import Version
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = pytest.mark.skipif(Version(version("sglang")) < Version("0.5.21"), reason="SGLang 0.5.21 patch")


def jpeg(mode: str, size: tuple[int, int]) -> bytes:
    stream = BytesIO()
    Image.new(mode, size, 127 if mode == "L" else (32, 96, 180)).save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture
def processor_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], Any]]:
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor, SGLangBaseProcessor

    # The service constructor allocates runtime resources. Use only a local I/O worker.
    def init_io(self: Any, config: Any, server_args: Any, processor: Any) -> None:
        self.hf_config = config
        self.io_executor = executor

    monkeypatch.setattr(SGLangBaseProcessor, "__init__", init_io)
    tokens = SimpleNamespace(tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: ["<video>"]))

    def make_processor(model_type: str) -> Any:
        config = SimpleNamespace(
            model_type=model_type,
            vision_config=SimpleNamespace(spatial_merge_size=2),
            vision_start_token_id=1,
            vision_end_token_id=2,
            image_token_id=3,
            video_token_id=4,
        )
        return QwenVLImageProcessor(config, None, tokens)

    with ThreadPoolExecutor(max_workers=1) as executor:
        yield make_processor


@pytest.fixture
def cpu_jpeg_backend(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    import torchvision.io as image_io
    from sglang.srt.utils import common

    # Exercise the GPU JPEG dispatch but execute the requested color conversion on CPU.
    native_decode = image_io.decode_jpeg
    calls: list[Any] = []

    def decode_on_cpu(encoded: Any, mode: Any = image_io.ImageReadMode.UNCHANGED, *, device: str) -> Any:
        assert device == "cuda"
        calls.append(mode)
        return native_decode(encoded, mode=mode, device="cpu")

    monkeypatch.setattr(common, "is_cuda", lambda: True)
    monkeypatch.setattr(image_io, "decode_jpeg", decode_on_cpu)
    return calls


def load_images(processor: Any, payloads: list[bytes], loader: str) -> list[Any]:
    from sglang.srt.multimodal.processors.base_processor import Modality

    if loader == "fast":
        jobs = processor._submit_mm_data_loading_tasks_simple(payloads, Modality.IMAGE, None, True)
        return [future.result() for _, _, future in jobs]
    futures, _ = processor.submit_data_loading_tasks(
        ["<image>"] * len(payloads), processor.mm_tokens, {Modality.IMAGE: iter(payloads)}
    )
    return [future.result() for future in futures]


@pytest.mark.parametrize("loader", ["fast", "legacy"])
def test_qwen_mixed_jpegs_reach_image_preprocessor(
    processor_factory: Callable[[str], Any], cpu_jpeg_backend: list[Any], loader: str
) -> None:
    from torchvision.io import ImageReadMode
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    payloads = [jpeg("L", (1088, 1344)), jpeg("RGB", (1089, 1343))]
    decoded = load_images(processor_factory("qwen4_exp"), payloads, loader)
    hf = Qwen2VLImageProcessor(patch_size=16, merge_size=2, size={"shortest_edge": 1024, "longest_edge": 4000000})
    features = hf(
        images=decoded,
        do_convert_rgb=True,
        input_data_format="channels_first",
        disable_grouping=False,
        return_tensors="pt",
    )

    assert features.image_grid_thw.tolist() == [[1, 84, 68], [1, 84, 68]]
    assert features.pixel_values.shape == (11424, 1536)
    assert cpu_jpeg_backend == [ImageReadMode.RGB, ImageReadMode.RGB]
    assert [tuple(image.shape) for image in decoded] == [(3, 1344, 1088), (3, 1343, 1089)]
    assert torch.equal(decoded[0][0], decoded[0][1]) and torch.equal(decoded[0][0], decoded[0][2])
    with Image.open(BytesIO(payloads[1])) as color:
        assert tuple(decoded[1][:, 0, 0].tolist()) == color.getpixel((0, 0))


def test_other_model_sharing_qwen_processor_keeps_native_jpeg_mode(
    processor_factory: Callable[[str], Any], cpu_jpeg_backend: list[Any]
) -> None:
    from torchvision.io import ImageReadMode

    (gray,) = load_images(processor_factory("intern_s2_preview"), [jpeg("L", (32, 32))], "fast")
    assert gray.shape == (1, 32, 32)
    assert cpu_jpeg_backend == [ImageReadMode.UNCHANGED]


def test_qwen_pil_fallback_converts_gray_to_rgb(
    processor_factory: Callable[[str], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sglang.srt.utils import common

    monkeypatch.setattr(common, "is_cuda", lambda: False)
    (gray,) = load_images(processor_factory("qwen4_exp"), [jpeg("L", (32, 32))], "fast")
    assert isinstance(gray, Image.Image)
    assert gray.mode == "RGB"
    assert gray.getpixel((0, 0)) == (127, 127, 127)
