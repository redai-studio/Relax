# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for processor worker multimodal preprocessing."""

import numpy as np
from PIL import Image

from relax.utils.data import processor_pool


class Qwen2VLImageProcessor:
    pass


Qwen2VLImageProcessor.__module__ = "transformers.models.qwen2_vl.image_processing_qwen2_vl"


class CustomQwen2VLImageProcessor(Qwen2VLImageProcessor):
    pass


class ProcessorWithQwenImageProcessor:
    def __init__(self) -> None:
        self.image_processor = CustomQwen2VLImageProcessor()
        self.received_sizes: list[tuple[int, int]] = []

    def __call__(self, **kwargs):
        self.received_sizes = [image.size for image in kwargs["images"]]
        return {"input_ids": [[1]]}


class OtherProcessor:
    def __init__(self) -> None:
        self.received_image: Image.Image | None = None

    def __call__(self, **kwargs):
        self.received_image = kwargs["images"][0]
        return {"input_ids": [[1]]}


class SameNamedNonTransformersImageProcessor:
    pass


SameNamedNonTransformersImageProcessor.__name__ = "Qwen2VLImageProcessor"


class ProcessorWithSameNamedNonTransformersImageProcessor:
    image_processor = SameNamedNonTransformersImageProcessor()


def test_process_sample_resizes_only_extreme_images_for_qwen_vl(monkeypatch):
    processor = ProcessorWithQwenImageProcessor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)

    prompt_ids, train_inputs = processor_pool.process_sample_in_worker(
        "prompt",
        {
            "images": [
                np.asarray(Image.new("RGB", (750, 1))),
                np.asarray(Image.new("RGB", (100, 50))),
                np.asarray(Image.new("RGB", (1, 750))),
            ]
        },
        {},
    )

    assert prompt_ids == [1]
    assert train_inputs is None
    assert processor.received_sizes == [(750, 4), (100, 50), (4, 750)]


def test_process_sample_does_not_resize_extreme_image_for_other_processors(monkeypatch):
    processor = OtherProcessor()
    monkeypatch.setattr(processor_pool, "_worker_processor", processor)

    prompt_ids, train_inputs = processor_pool.process_sample_in_worker(
        "prompt", {"images": [np.asarray(Image.new("RGB", (750, 1)))]}, {}
    )

    assert prompt_ids == [1]
    assert train_inputs is None
    assert processor.received_image is not None
    assert processor.received_image.size == (750, 1)


def test_qwen_vl_detection_rejects_same_class_name_from_other_module():
    processor = ProcessorWithSameNamedNonTransformersImageProcessor()

    assert not processor_pool._is_qwen_vl_processor(processor)
