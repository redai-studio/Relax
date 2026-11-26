# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-GPU public artifact regressions: real allocator OOM, cache miss and
hit."""

import gc
from typing import Any

import pytest
from PIL import Image

from tests.backends.sglang.multimodal_test_utils import (
    assert_image_reference,
    assert_same_image_output,
    cpu_image_reference,
    process_image,
)


torch = pytest.importorskip("torch")

pytestmark = pytest.mark.usefixtures("sglang_cuda")


def test_cached_public_miss_recovers_real_gpu_oom_then_reuses_cpu_artifact(cached_qwen_processor: Any) -> None:
    processor = cached_qwen_processor
    # Use the ordinary CUDA allocator, as CUDA IPC tokenizer preprocessing does.
    # The cache still mandates CPU features, so no transport proxy is needed.
    processor.mm_feature_transport = "cuda_ipc"
    image = Image.new("RGB", (4096, 4096), (32, 96, 180))
    expected = cpu_image_reference(processor, image)
    assert expected["pixel_values"].numel() * expected["pixel_values"].element_size() < 16 * 1024**2
    initial = processor.mm_preprocess_cache.stats()
    assert initial["entries"] == initial["misses"] == initial["hits"] == 0
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    before_ooms = torch.cuda.memory_stats()["num_ooms"]
    torch.cuda.set_per_process_memory_fraction(64 * 1024**2 / total)
    try:
        first = process_image(processor, image, "artifact-real-oom-miss")
        after_miss_ooms = torch.cuda.memory_stats()["num_ooms"]
        assert after_miss_ooms > before_ooms, "the fixture did not trigger a real CUDA allocator OOM"
        assert_image_reference(first, expected, exact=True)
        cached = processor.mm_preprocess_cache.stats()
        assert cached["entries"] == cached["misses"] == 1 and cached["inflight"] == 0, cached
        second = process_image(processor, image, "artifact-real-oom-hit")
        hit = processor.mm_preprocess_cache.stats()
        assert hit["hits"] == cached["hits"] + 1 and hit["misses"] == cached["misses"], hit
        assert torch.cuda.memory_stats()["num_ooms"] == after_miss_ooms, "a cache hit retried GPU preprocessing"
        assert_same_image_output(first, second)
        assert_image_reference(second, expected, exact=True)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)


@pytest.mark.parametrize("warm_cache", [False, True], ids=["first-miss", "cache-hit"])
def test_cached_public_request_releases_idle_cache_without_changing_output(
    cached_qwen_processor: Any, warm_cache: bool
) -> None:
    processor = cached_qwen_processor
    image = Image.new("RGB", (256, 256), (32, 96, 180))
    expected = cpu_image_reference(processor, image)
    warm = process_image(processor, image, "artifact-warm") if warm_cache else None
    before_stats = processor.mm_preprocess_cache.stats()
    assert before_stats["entries"] == int(warm_cache), before_stats

    torch.cuda.empty_cache()
    temporary = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
    del temporary
    gc.collect()
    before = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    assert before >= 2 * 1024**3, "fixture did not reserve the intended real idle CUDA cache"
    result = process_image(processor, image, "artifact-cache-release")
    torch.cuda.synchronize()
    after = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    after_stats = processor.mm_preprocess_cache.stats()
    if warm_cache:
        assert after_stats["hits"] == before_stats["hits"] + 1, after_stats
        assert after_stats["misses"] == before_stats["misses"], after_stats
        assert_same_image_output(warm, result)
    else:
        assert after_stats["misses"] == before_stats["misses"] + 1, after_stats
        assert after_stats["entries"] == 1, after_stats
    assert_image_reference(result, expected)
    assert after < 1024**3, "cached public preprocessing retained more than 1 GiB of idle GPU cache"
