# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from relax.utils.data.data_utils import check_sample_length
from relax.utils.types import Sample


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    model_path = os.environ.get("RELAX_TEST_QWEN3_4B")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Qwen3-4B is unavailable; set RELAX_TEST_QWEN3_4B to its local directory")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        pytest.skip(f"transformers AutoTokenizer unavailable: {exc}")
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


@pytest.fixture(scope="module")
def qwen3_vl_processor():
    model_path = os.environ.get("RELAX_TEST_QWEN3_VL_4B")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("Qwen3-VL-4B-Instruct is unavailable; set RELAX_TEST_QWEN3_VL_4B to its local directory")
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        pytest.skip(f"transformers AutoProcessor unavailable: {exc}")
    return AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)


def _render_image_prompt(processor: Any, image: Image.Image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert isinstance(prompt, str)
    return prompt


def _multimodal_token_count(processor: Any, prompt: str, image: Image.Image) -> int:
    return len(processor(text=prompt, images=[image])["input_ids"][0])


def test_multimodal_processor_expands_image_tokens(qwen3_vl_processor):
    image = Image.new("RGB", (224, 224), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, image)

    text_token_count = len(qwen3_vl_processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    multimodal_token_count = _multimodal_token_count(qwen3_vl_processor, prompt, image)

    assert multimodal_token_count > text_token_count


def test_multimodal_length_boundary_is_inclusive(qwen3_vl_processor):
    image = Image.new("RGB", (224, 224), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, image)
    sample = Sample(prompt=prompt, multimodal_inputs={"images": [image]})
    actual_length = _multimodal_token_count(qwen3_vl_processor, prompt, image)

    assert check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=actual_length,
    )
    assert not check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=actual_length - 1,
    )


def test_larger_image_resolution_produces_more_tokens(qwen3_vl_processor):
    small_image = Image.new("RGB", (224, 224), (32, 96, 160))
    large_image = Image.new("RGB", (448, 448), (32, 96, 160))
    prompt = _render_image_prompt(qwen3_vl_processor, small_image)

    small_length = _multimodal_token_count(qwen3_vl_processor, prompt, small_image)
    large_length = _multimodal_token_count(qwen3_vl_processor, prompt, large_image)

    assert large_length > small_length

    sample = Sample(prompt=prompt, multimodal_inputs={"images": [large_image]})
    assert not check_sample_length(
        sample,
        qwen3_vl_processor.tokenizer,
        qwen3_vl_processor,
        max_length=small_length,
    )


def test_text_prompt_length_boundary_without_processor(qwen3_tokenizer):
    prompt = "Compare two integers and explain the answer. " * 8
    sample = Sample(prompt=prompt)
    actual_length = len(qwen3_tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert check_sample_length(sample, qwen3_tokenizer, processor=None, max_length=actual_length)
    assert not check_sample_length(sample, qwen3_tokenizer, processor=None, max_length=actual_length - 1)
