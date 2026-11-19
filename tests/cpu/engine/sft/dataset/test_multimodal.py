# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for multimodal preprocessing wrapper."""

import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.engine.sft.dataset.multimodal import (
    SFTMultimodalMediaLoadError,
    has_multimodal_content,
    preprocess_multimodal,
    preprocess_multimodal_async,
    sample_media_paths,
)
from relax.engine.sft.dataset.sample import (
    CanonicalMessage,
    CanonicalSample,
)
from relax.utils.data import processor_pool as worker_processor_pool
from relax.utils.data.processor_pool import MediaLoadError, _load_media_for_worker


@pytest.fixture(autouse=True)
def stub_processor_pool_module(monkeypatch):
    processor_pool = ModuleType("relax.utils.data.processor_pool")
    processor_pool.MediaLoadError = MediaLoadError
    processor_pool.process_sample_from_paths_in_worker = MagicMock()
    monkeypatch.setitem(sys.modules, "relax.utils.data.processor_pool", processor_pool)

    import relax.utils.data as data_package

    monkeypatch.setattr(data_package, "processor_pool", processor_pool, raising=False)


def _text_sample():
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="hi", learn=False),
            CanonicalMessage(role="assistant", content="ok", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
    )


def _image_sample():
    return CanonicalSample(
        messages=[
            CanonicalMessage(role="user", content="<|vision_start|><|image_pad|><|vision_end|>describe", learn=False),
            CanonicalMessage(role="assistant", content="cat", learn=True),
        ],
        metadata={"source_dataset": "x", "row_index": 0},
        images=["/tmp/cat.png"],
    )


def test_has_multimodal_content_text_only():
    assert has_multimodal_content(_text_sample()) is False


def test_has_multimodal_content_image():
    assert has_multimodal_content(_image_sample()) is True


def test_preprocess_text_only_returns_none():
    prompt_ids, mm_inputs = preprocess_multimodal(_text_sample(), processor_pool=None)
    assert prompt_ids is None
    assert mm_inputs is None


def test_preprocess_image_calls_processor_pool():
    """When sample has images, we must dispatch to processor_pool.executor and
    return the (expanded prompt_ids, mm_train_inputs) pair the worker produces.

    — `prompt_ids` is what the model actually consumes after image-pad
    expansion.
    """
    fake_pool = MagicMock()
    fake_executor = MagicMock()
    fake_pool.executor = fake_executor
    fake_future = MagicMock()
    fake_future.result.return_value = (
        [1, 2, 3],
        {"pixel_values": torch.zeros(1, 3, 224, 224), "image_grid_thw": torch.tensor([[1, 16, 16]])},
    )
    fake_executor.submit.return_value = fake_future
    prompt_ids, mm_inputs = preprocess_multimodal(_image_sample(), processor_pool=fake_pool)
    assert prompt_ids == [1, 2, 3]
    assert "pixel_values" in mm_inputs
    assert "image_grid_thw" in mm_inputs
    fake_executor.submit.assert_called_once()


def test_preprocess_without_pool_when_multimodal_raises():
    with pytest.raises(ValueError, match="processor_pool"):
        preprocess_multimodal(_image_sample(), processor_pool=None)


def _patch_loaders(monkeypatch, load_image):
    for kind in ("image", "video", "audio"):
        module_name = f"relax.utils.multimodal.{kind}_utils"
        loader_module = ModuleType(module_name)
        setattr(loader_module, f"load_{kind}", load_image if kind == "image" else MagicMock())
        monkeypatch.setitem(sys.modules, module_name, loader_module)


def test_worker_media_loader_loads_images_without_video_or_audio_dependencies(monkeypatch):
    image_utils = ModuleType("relax.utils.multimodal.image_utils")
    image_utils.load_image = MagicMock(return_value="decoded image")
    monkeypatch.setitem(sys.modules, image_utils.__name__, image_utils)
    monkeypatch.setitem(sys.modules, "relax.utils.multimodal.video_utils", None)
    monkeypatch.setitem(sys.modules, "relax.utils.multimodal.audio_utils", None)

    assert _load_media_for_worker({"image": ["cat.png"], "video": [], "audio": []}) == {"images": ["decoded image"]}
    image_utils.load_image.assert_called_once_with("cat.png")


def test_worker_media_loader_reports_position(monkeypatch):
    load_image = MagicMock(side_effect=AssertionError("missing"))
    _patch_loaders(monkeypatch, load_image)

    with pytest.raises(MediaLoadError, match="image position=0") as exc_info:
        _load_media_for_worker({"image": ["/tmp/cat.png"], "video": [], "audio": []})
    load_image.assert_called_once_with("/tmp/cat.png")
    assert isinstance(exc_info.value.__cause__, AssertionError)


def test_worker_media_loader_unpacks_video_and_uses_singular_audio_key(monkeypatch):
    video_utils = ModuleType("relax.utils.multimodal.video_utils")
    frames = torch.zeros(2, 3, 56, 56)
    metadata = {"fps": 2, "total_num_frames": 2, "frames_indices": torch.tensor([0, 1]), "sample_fps": 2}
    video_utils.fetch_video = MagicMock(return_value=(frames, metadata, None, None))
    audio_utils = ModuleType("relax.utils.multimodal.audio_utils")
    waveform = torch.zeros(6400)
    audio_utils.load_audio = MagicMock(return_value=waveform)
    monkeypatch.setitem(sys.modules, video_utils.__name__, video_utils)
    monkeypatch.setitem(sys.modules, audio_utils.__name__, audio_utils)

    config = worker_processor_pool.MultimodalConfig(video_fps_max_frames=4, video_max_token_num=16)
    monkeypatch.setattr(worker_processor_pool, "_worker_multimodal_config", config)
    monkeypatch.setattr(
        worker_processor_pool, "_worker_processor", SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))
    )
    loaded = _load_media_for_worker({"video": ["clip.mp4"], "audio": ["tone.wav"]})

    assert set(loaded) == {"videos", "video_metadata", "audio"}
    assert loaded["video_metadata"] == [{"fps": 2, "total_num_frames": 2, "frames_indices": [0, 1]}]
    assert loaded["videos"][0] is frames
    assert loaded["audio"][0] is waveform
    video_utils.fetch_video.assert_called_once_with(
        {"video": "clip.mp4"}, image_patch_size=16, use_audio_in_video=False, config=config
    )
    audio_utils.load_audio.assert_called_once_with("tone.wav")


def test_preprocess_wraps_worker_media_load_error():
    pool = MagicMock()
    pool.executor.submit.return_value.result.side_effect = MediaLoadError(
        "failed to load image position=0: '/data/missing.jpg'"
    )
    with pytest.raises(SFTMultimodalMediaLoadError, match=r"image position=0.*row_index=0") as exc_info:
        preprocess_multimodal(_image_sample(), processor_pool=pool)
    assert isinstance(exc_info.value.__cause__, MediaLoadError)


def test_sample_media_paths_keeps_kind_and_order():
    paths = sample_media_paths(_image_sample())
    assert paths["image"] == ["/tmp/cat.png"]
    assert paths["video"] == [] and paths["audio"] == []


@pytest.mark.asyncio
async def test_preprocess_multimodal_async_dispatches_to_executor():
    """``preprocess_multimodal_async`` must hand the work to
    ``processor_pool.executor`` via ``run_in_executor`` and return the
    ``(prompt_ids, mm_train_inputs)`` tuple the worker produces."""
    expected_inputs = {"pixel_values": torch.zeros(1, 3, 4, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    expected_prompt_ids = [10, 20, 30]

    class _FakePool:
        def __init__(self) -> None:
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.calls = 0

    pool = _FakePool()

    def _fake_worker(text, mm_inputs_ipc, kwargs):  # noqa: ARG001
        pool.calls += 1
        return expected_prompt_ids, expected_inputs

    with patch("relax.utils.data.processor_pool.process_sample_from_paths_in_worker", side_effect=_fake_worker):
        prompt_ids, mm_inputs = await preprocess_multimodal_async(_image_sample(), processor_pool=pool)

    pool.executor.shutdown(wait=True)
    assert prompt_ids == expected_prompt_ids
    assert mm_inputs is expected_inputs
    assert pool.calls == 1


@pytest.mark.asyncio
async def test_preprocess_multimodal_async_text_only_returns_none():
    prompt_ids, mm_inputs = await preprocess_multimodal_async(_text_sample(), processor_pool=None)
    assert prompt_ids is None
    assert mm_inputs is None


@pytest.mark.asyncio
async def test_preprocess_multimodal_async_without_pool_raises():
    with pytest.raises(ValueError, match="processor_pool"):
        await preprocess_multimodal_async(_image_sample(), processor_pool=None)


def test_worker_rejects_video_audio_before_loading_media(monkeypatch):
    processor_pool = worker_processor_pool

    load_video = MagicMock()
    module = ModuleType("relax.utils.multimodal.video_utils")
    module.load_video = load_video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    process = MagicMock()
    monkeypatch.setattr(processor_pool, "process_sample_in_worker", process)

    with pytest.raises(ValueError, match="does not support extracting audio from video"):
        processor_pool.process_sample_from_paths_in_worker(
            "video prompt", {"video": ["clip.mp4"]}, {"use_audio_in_video": True}
        )
    load_video.assert_not_called()
    process.assert_not_called()


def test_worker_forwards_disabled_video_audio_option(monkeypatch):
    processor_pool = worker_processor_pool

    frames = torch.zeros(2, 3, 28, 28)
    load = MagicMock(return_value={"videos": [frames]})
    process = MagicMock(return_value=([1, 2], None))
    monkeypatch.setattr(processor_pool, "_load_media_for_worker", load)
    monkeypatch.setattr(processor_pool, "process_sample_in_worker", process)
    options = {"use_audio_in_video": False}
    assert processor_pool.process_sample_from_paths_in_worker("video prompt", {"video": ["clip.mp4"]}, options) == (
        [1, 2],
        None,
    )
    load.assert_called_once_with({"video": ["clip.mp4"]}, use_audio_in_video=False)
    process.assert_called_once_with(
        "video prompt",
        {"videos": [frames]},
        {**options, "videos_kwargs": {"do_sample_frames": False, "do_resize": False}},
    )
    assert process.call_args.args[1]["videos"][0] is frames
    assert not frames.is_shared()


def test_worker_video_applies_frame_and_pixel_limits_before_processing(monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from relax.utils.multimodal import video_utils

    frames = torch.arange(12, dtype=torch.float32).view(12, 1, 1, 1).expand(12, 3, 224, 224).clone()
    monkeypatch.setattr(
        video_utils, "load_video", lambda *a, **kw: (frames, {"fps": 2, "total_num_frames": 12}, None, None)
    )
    monkeypatch.setattr(
        worker_processor_pool,
        "_worker_multimodal_config",
        worker_processor_pool.MultimodalConfig(
            video_fps_max_frames=4,
            video_fps_min_frames=2,
            video_min_token_num=4,
            video_max_token_num=16,
        ),
    )
    vocab = {"<unk>": 0, **{token: i + 1 for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"],
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(do_rescale=False, do_normalize=False),
    )
    monkeypatch.setattr(worker_processor_pool, "_worker_processor", processor)

    media = worker_processor_pool._load_media_for_worker({"video": ["clip.mp4"]})
    sampled = media["videos"][0]
    assert sampled.shape[0] == 4
    assert sampled.shape[-1] * sampled.shape[-2] <= 16 * 32**2
    assert not sampled.is_shared() and not frames.is_shared()

    prompt_ids, train_inputs = worker_processor_pool.process_sample_from_paths_in_worker(
        "Describe <|video_pad|>", {"video": ["clip.mp4"]}, {"return_tensors": "pt"}
    )
    assert train_inputs["video_grid_thw"].tolist() == [[2, 8, 8]]
    assert torch.unique(train_inputs["pixel_values_videos"].float()).tolist() == [0, 4, 7, 11]
    rendered = tokenizer.decode(prompt_ids, clean_up_tokenization_spaces=False)
    assert "<1.0 seconds>" in rendered
    assert "<4.5 seconds>" in rendered
