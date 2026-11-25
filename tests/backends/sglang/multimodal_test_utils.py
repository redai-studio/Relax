# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small inputs for real SGLang multimodal lifecycle regressions."""

import array
import asyncio
import gc
import importlib.util
import os
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from typing import Any

import pytest
from PIL import Image


torch = pytest.importorskip("torch")


def image_bytes(mode: str = "RGB", size: tuple[int, int] = (128, 128)) -> bytes:
    image = Image.new(mode, size, 127 if mode == "L" else (32, 96, 180))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def make_request(manager: Any, pool: Any, rid: str = "ipc", proxy: Any = None) -> tuple[Any, Any, Any]:
    from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput
    from sglang.srt.managers.tokenizer_manager import ReqState
    from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
    from sglang.srt.sampling.sampling_params import SamplingParams

    if proxy is None:
        proxy = pool.wrap_tensor(torch.arange(64, device="cuda", dtype=torch.float32), use_pool_handle_cache=True)
    assert proxy is not None
    mm = MultimodalProcessorOutput(
        input_ids=[1], mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)]
    )
    stats = APIServerReqTimeStats()
    state = ReqState([], False, asyncio.Event(), GenerateReqInput(rid=rid), stats)
    manager.rid_to_state[rid] = state
    req = TokenizedGenerateReqInput(
        rid=rid,
        input_text="x",
        input_ids=array.array("q", [1]),
        input_embeds=None,
        mm_inputs=mm,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        time_stats=stats,
    )
    return req, state, proxy


def recycle(pool: Any) -> None:
    torch.cuda.synchronize()
    with pool._pool._lock:
        pool._pool._recycle_ready_leases_locked()


def require_cuda_devices(count: int = 1) -> None:
    """Skip before importing SGLang when the needed CUDA devices are absent."""
    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} real CUDA GPU(s)")
    if importlib.util.find_spec("sglang") is None:
        pytest.skip("requires the patched SGLang dependency")


@contextmanager
def qwen_processor_context(
    *,
    base_gpu_id: int = 0,
    cache_size_mb: int | None = None,
    mm_process_config: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Construct the real processor once per test, with a shared cleanup
    path."""
    model_path = os.environ.get("RELAX_TEST_MODEL_PATH")
    if not model_path:
        pytest.skip("set RELAX_TEST_MODEL_PATH to a local Qwen3.5 model with processor files")

    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    settings = {}
    if mm_process_config is not None:
        settings["mm_process_config"] = mm_process_config
    args = ServerArgs(
        model_path=model_path,
        device="cuda",
        base_gpu_id=base_gpu_id,
        mm_processor_worker_num=1,
        mm_io_worker_num=1,
        mm_feature_transport="cpu",
        mm_preprocess_cache_size_mb=(
            int(os.environ.get("RELAX_TEST_MM_CACHE_MB", "0")) if cache_size_mb is None else cache_size_mb
        ),
        **settings,
    )
    processor = None
    publish(args, role="tokenizer")
    try:
        processor = QwenVLImageProcessor(config, args, hf, "default", skip_mm_pool=True)
        yield processor
    finally:
        if processor is not None:
            processor.io_executor.shutdown(wait=True)
            for name in ("cpu_executor", "hash_executor", "mm_processor_executor"):
                executor = getattr(processor, name, None)
                if executor is not None and hasattr(executor, "shutdown"):
                    executor.shutdown()
        reset_context()
        gc.collect()
        with torch.cuda.device(base_gpu_id):
            torch.cuda.empty_cache()


NATIVE_IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>"


def image_request(rid: str) -> Any:
    from sglang.srt.managers.io_struct import GenerateReqInput

    return GenerateReqInput(rid=rid, video_data=None, audio_data=None)


def process_image(processor: Any, image: Any, rid: str) -> Any:
    return asyncio.run(processor.process_mm_data_async([image], NATIVE_IMAGE_PROMPT, image_request(rid)))


def cpu_image_reference(processor: Any, image: Any) -> Any:
    return processor._processor(
        text=[NATIVE_IMAGE_PROMPT],
        images=[image],
        padding=True,
        return_tensors="pt",
        device="cpu",
        images_kwargs=processor.image_config,
    )


def assert_image_reference(result: Any, expected: Any, *, exact: bool = False) -> None:
    assert len(result.mm_items) == 1
    item = result.mm_items[0]
    assert item.feature.device.type == "cpu", "cached artifacts must retain CPU features"
    torch.testing.assert_close(
        item.feature,
        expected["pixel_values"],
        rtol=0 if exact else 1e-6,
        atol=0 if exact else 1e-6,
    )
    assert result.input_ids == expected["input_ids"].flatten().tolist()
    torch.testing.assert_close(item.image_grid_thw, expected["image_grid_thw"], rtol=0, atol=0)


def assert_same_image_output(first: Any, second: Any) -> None:
    assert first.input_ids == second.input_ids
    assert first.padded_input_ids == second.padded_input_ids
    torch.testing.assert_close(first.mrope_positions, second.mrope_positions, rtol=0, atol=0)
    torch.testing.assert_close(first.mrope_position_delta, second.mrope_position_delta, rtol=0, atol=0)
    assert first.mm_items[0].hash == second.mm_items[0].hash
    assert first.mm_items[0].offsets == second.mm_items[0].offsets
    torch.testing.assert_close(first.mm_items[0].feature, second.mm_items[0].feature, rtol=0, atol=0)
