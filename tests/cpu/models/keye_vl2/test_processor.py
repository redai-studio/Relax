# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks against the real Keye HF processor selected by
KEYE_MODEL_PATH."""

import logging
import os
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="module")
def hf_processor():
    model_path = os.environ.get("KEYE_MODEL_PATH")
    if not model_path:
        pytest.skip("Set KEYE_MODEL_PATH to a local Keye checkpoint with its HF processor files")
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return model_path, config, processor


@pytest.fixture
def processor(hf_processor, monkeypatch):
    from sglang.srt import runtime_context
    from sglang.srt.server_args import ServerArgs

    from relax.models.keye_vl2.sglang.processor import KeyeProcessor

    model_path, config, hf = hf_processor
    monkeypatch.setenv("SGLANG_MM_AVOID_RETOKENIZE", "1")
    monkeypatch.setattr(runtime_context, "_CONTEXT", runtime_context.RuntimeContext(runtime_context.ParallelContext()))
    args = ServerArgs(
        model_path=model_path,
        trust_remote_code=True,
        device="cpu",
        context_length=4096,
        enable_multimodal=True,
        disable_fast_image_processor=True,
        mm_feature_transport="cpu",
        mm_processor_worker_num=1,
    )
    runtime_context.publish(args, role="tokenizer")
    instance = KeyeProcessor(config, args, deepcopy(hf), "default", skip_mm_pool=True)
    try:
        yield instance
    finally:
        instance.shutdown()


@pytest.mark.parametrize("image_type", ["pil", "numpy", "torch"])
@pytest.mark.parametrize(
    ("height", "width", "settings"),
    [
        (73, 101, {}),
        (14, 14, {"min_pixels": 56 * 56}),
        (448, 336, {"max_pixels": 56 * 112}),
        (73, 101, {"size": {"height": 56, "width": 84}}),
        (56, 84, {"do_resize": False}),
    ],
    ids=["rounded", "upscaled", "downscaled", "fixed-size", "no-resize"],
)
def test_keye_image_token_counts_match_real_hf_grid(processor, monkeypatch, image_type, height, width, settings):
    image_processor = processor._processor.image_processor
    for name, value in settings.items():
        monkeypatch.setattr(image_processor, name, value)
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    image = {
        "pil": lambda: Image.fromarray(pixels),
        "numpy": lambda: pixels,
        "torch": lambda: torch.from_numpy(pixels).permute(2, 0, 1),
    }[image_type]()

    reference = image_processor(images=[image], return_tensors="pt")
    grid = reference["image_grid_thw"]
    expected = (grid.prod(dim=-1) // image_processor.merge_size**2).tolist()

    assert processor.resolve_image_token_counts([image]) == expected
    assert reference["pixel_values"].device.type == "cpu"
    assert reference["pixel_values"].shape[0] == int(grid.prod())


def test_keye_processor_preserves_original_tokens_and_image_offsets(processor, caplog):
    from sglang.srt.multimodal.processors.base_processor import BaseMultiModalProcessorOutput

    tokenizer = processor._processor.tokenizer
    # These adjacent tokens decode to "ab", whose canonical encoding merges them.
    noncanonical = tokenizer.encode("a", add_special_tokens=False) + tokenizer.encode("b", add_special_tokens=False)
    assert tokenizer.encode(tokenizer.decode(noncanonical), add_special_tokens=False) != noncanonical
    marker = tokenizer.encode("<|vision_start|><|image_pad|><|vision_end|>", add_special_tokens=False)
    original_ids = noncanonical + marker + noncanonical + marker + noncanonical
    text = tokenizer.decode(original_ids)
    images = [Image.new("RGB", (84, 56), "gray"), Image.new("RGB", (112, 84), "gray")]
    reference = processor._processor(text=[text], images=images, return_tensors="pt", padding=True)
    counts = (reference["image_grid_thw"].prod(dim=-1) // processor._processor.image_processor.merge_size**2).tolist()
    placeholder = processor.hf_config.image_token_id
    expected = []
    expected_offsets = []
    image_index = 0
    for token in original_ids:
        if token == placeholder:
            count = counts[image_index]
            expected_offsets.append((len(expected), len(expected) + count - 1))
            expected.extend([placeholder] * count)
            image_index += 1
        else:
            expected.append(token)
    assert reference["input_ids"].flatten().tolist() != expected
    base = BaseMultiModalProcessorOutput(input_text=text, input_ids=original_ids, images=images)

    with caplog.at_level(logging.WARNING):
        items, actual_ids, result = processor.process_and_combine_mm_data(base, processor.mm_tokens)

    assert actual_ids.tolist() == expected
    assert [token for token in actual_ids.tolist() if token != placeholder] == [
        token for token in original_ids if token != placeholder
    ]
    assert len(items) == len(images)
    assert [offset for item in items for offset in item.offsets] == expected_offsets
    assert all(item.feature.device.type == "cpu" for item in items)
    torch.testing.assert_close(result["image_grid_thw"], reference["image_grid_thw"])
    assert "falling back to decode+retokenize" not in caplog.text
