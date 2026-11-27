# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-GPU regression for Qwen preprocessing cache device ownership."""

import gc
from collections.abc import Iterator
from typing import Any

import pytest
from PIL import Image

from tests.backends.sglang.multimodal_test_utils import (
    assert_image_reference,
    assert_same_image_output,
    cpu_image_reference,
    process_image,
    qwen_processor_context,
    require_cuda_devices,
)


torch = pytest.importorskip("torch")


@pytest.fixture
def gpu1_cached_qwen_processor() -> Iterator[Any]:
    require_cuda_devices(2)
    with qwen_processor_context(
        base_gpu_id=1,
        cache_size_mb=16,
        mm_process_config={"image": {"size": {"shortest_edge": 65536, "longest_edge": 262144}}},
    ) as processor:
        from sglang.srt.multimodal.cache import build_processor_fingerprint

        assert processor.mm_preprocess_cache.enabled
        assert processor.processor_fingerprint == build_processor_fingerprint(processor, processor.hf_config)
        yield processor


@pytest.mark.parametrize("warm_cache", [False, True], ids=["first-miss", "cache-hit"])
def test_cached_public_release_uses_processor_device_and_preserves_current_device(
    gpu1_cached_qwen_processor: Any, warm_cache: bool
) -> None:
    processor = gpu1_cached_qwen_processor
    assert processor._fast_image_processor_device(processor._processor) == "cuda:1"
    image = Image.new("RGB", (256, 256), (32, 96, 180))
    expected = cpu_image_reference(processor, image)
    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        warm = process_image(processor, image, "artifact-gpu1-warm") if warm_cache else None
        before_stats = processor.mm_preprocess_cache.stats()
        assert before_stats["entries"] == int(warm_cache), before_stats
        for device in (0, 1):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                reserve_bytes = 128 * 1024**2 if device == 0 else 2 * 1024**3
                temporary = torch.empty(reserve_bytes, dtype=torch.uint8, device=f"cuda:{device}")
                del temporary
        gc.collect()
        idle = lambda device: torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        before = {device: idle(device) for device in (0, 1)}
        assert 0 <= before[0] < 1024**3 and before[1] >= 2 * 1024**3, before
        assert torch.cuda.current_device() == 0
        result = process_image(processor, image, "artifact-gpu1-release")
        for device in (0, 1):
            torch.cuda.synchronize(device)
        after = {device: idle(device) for device in (0, 1)}
        after_stats = processor.mm_preprocess_cache.stats()
        assert_image_reference(result, expected)
        if warm_cache:
            assert after_stats["hits"] == before_stats["hits"] + 1, after_stats
            assert after_stats["misses"] == before_stats["misses"], after_stats
            assert_same_image_output(warm, result)
        else:
            assert after_stats["misses"] == before_stats["misses"] + 1, after_stats
            assert after_stats["entries"] == 1, after_stats
        assert after[1] < 1024**3, "processor GPU 1 retained more than 1 GiB of idle CUDA cache"
        assert torch.cuda.current_device() == 0, "request changed the caller's current CUDA device"
    finally:
        for device in (0, 1):
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        torch.cuda.set_device(original_device)
