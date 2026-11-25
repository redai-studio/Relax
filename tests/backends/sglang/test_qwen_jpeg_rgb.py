# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image


torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA nvJPEG")


def jpeg(mode, size):
    stream = BytesIO()
    Image.new(mode, size, 127 if mode == "L" else (32, 96, 180)).save(stream, format="JPEG")
    return stream.getvalue()


@pytest.fixture
def processor_factory(monkeypatch):
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor, SGLangBaseProcessor

    # The service constructor allocates runtime resources. Use only a local I/O worker.
    def init_io(self, config, server_args, processor):
        self.hf_config = config
        self.io_executor = executor

    monkeypatch.setattr(SGLangBaseProcessor, "__init__", init_io)
    tokens = SimpleNamespace(tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: ["<video>"]))

    def make_processor(model_type):
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


def load_images(processor, payloads, loader):
    from sglang.srt.multimodal.processors.base_processor import Modality

    if loader == "artifact":
        return [
            processor.decode_media_snapshot(processor.snapshot_media_source(data, Modality.IMAGE), Modality.IMAGE)
            for data in payloads
        ]
    if loader == "fast":
        jobs = processor._submit_mm_data_loading_tasks_simple(payloads, Modality.IMAGE, None, True)
        return [future.result() for _, _, future in jobs]
    futures, _ = processor.submit_data_loading_tasks(
        ["<image>"] * len(payloads), processor.mm_tokens, {Modality.IMAGE: iter(payloads)}
    )
    return [future.result() for future in futures]


@pytest.mark.parametrize("loader", ["fast", "legacy", "artifact"])
def test_mixed_qwen_jpegs(processor_factory, loader):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    payloads = [jpeg("L", (64, 64)), jpeg("RGB", (64, 64))]
    decoded = load_images(processor_factory("qwen4_exp"), payloads, loader)
    pixels = torch.stack(decoded)
    assert pixels.is_cuda
    assert pixels.shape == (2, 3, 64, 64)
    expected = torch.tensor([[127, 127, 127], [32, 96, 180]], device=pixels.device)
    torch.testing.assert_close(pixels[:, :, 0, 0].long(), expected, rtol=0, atol=2)
    hf = Qwen2VLImageProcessor(patch_size=16, merge_size=2)
    result = hf(
        images=decoded,
        do_resize=False,
        input_data_format="channels_first",
        disable_grouping=False,
        return_tensors="pt",
    )
    assert result.image_grid_thw.tolist() == [[1, 4, 4], [1, 4, 4]]
    assert result.pixel_values.shape == (32, 1536)


def test_other_model_sharing_qwen_processor_keeps_native_jpeg_mode(processor_factory):
    (gray,) = load_images(processor_factory("intern_s2_preview"), [jpeg("L", (32, 32))], "fast")
    assert gray.is_cuda
    assert gray.shape == (1, 32, 32)
