# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-GPU Qwen regressions using local RELAX_TEST_MODEL_PATH processor
files."""

import asyncio
import gc
from typing import Any

import pytest
from PIL import Image

from .multimodal_test_utils import image_bytes


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.usefixtures("sglang_cuda")


def test_gpu_native_mixed_jpeg_preprocessing(qwen_processor: Any) -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    req = GenerateReqInput(rid="mixed-jpegs", video_data=None, audio_data=None)
    result = asyncio.run(
        qwen_processor.process_mm_data_async(
            [image_bytes("L", (1088, 1344)), image_bytes("RGB", (1089, 1343))],
            "<image><image>",
            req,
        )
    )
    assert len(result.mm_items) == 2
    shapes = [tuple(item.feature.shape) for item in result.mm_items]
    assert shapes[0] == shapes[1] and shapes[0][-1] == 1536, shapes
    assert result.input_ids


def test_real_gpu_oom_retries_image_processor_on_cpu(qwen_processor: Any) -> None:
    # Limit this process's allocator rather than fill the device or fake an OOM.
    # IPC transport uses the ordinary CUDA allocator, as production tokenizer workers do.
    qwen_processor.mm_feature_transport = "cuda_ipc"
    image = Image.new("RGB", (2048, 2048), (32, 96, 180))
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    hf = qwen_processor._processor
    expected = hf(text=[prompt], images=[image], padding=True, return_tensors="pt", device="cpu")
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(64 * 1024**2 / total)
    try:
        actual = qwen_processor.process_mm_data(prompt, images=[image])
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    assert actual["pixel_values"].device.type == "cpu"
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"], rtol=0, atol=0)
    assert torch.equal(actual["input_ids"], expected["input_ids"])


def test_public_qwen_preprocessing_releases_idle_cuda_cache(qwen_processor: Any) -> None:
    from sglang.srt.managers.io_struct import GenerateReqInput

    torch.cuda.empty_cache()
    temporary = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
    del temporary
    gc.collect()
    before = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert before >= 2 * 1024**3
    req = GenerateReqInput(rid="cache-release", video_data=None, audio_data=None)
    result = asyncio.run(qwen_processor.process_mm_data_async([image_bytes()], "<image>", req))
    torch.cuda.synchronize()
    after = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert len(result.mm_items) == 1
    assert after < 1024**3, "idle preprocessing blocks remain reserved after the request"
