# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise real SGLang media loading with a tiny local CPU tokenizer."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import pytest


pytest.importorskip("sglang")
from PIL import Image
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


NATIVE = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture
def media_loader():
    # Tokenization is an external boundary; use a real local tokenizer, without
    # downloading model weights/configs or replacing the loading implementation.
    special = ["[UNK]", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    alphabet = list(dict.fromkeys("<image>assistanttool text"))
    tokenizer = Tokenizer(models.WordLevel({s: i for i, s in enumerate(special + alphabet)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, additional_special_tokens=special)
    processor = object.__new__(QwenVLImageProcessor)
    processor._tokenizer = tokenizer
    processor.IM_START_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    processor.skip_tokenizer_init = False
    tokens = MultimodalSpecialTokens(
        image_token=NATIVE,
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        image_token_regex=re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>|<image>"),
    ).build(None)
    with ThreadPoolExecutor(max_workers=2) as executor:
        processor.io_executor = executor
        yield processor, tokens


@pytest.mark.parametrize(
    ("text", "count", "tokenized"),
    [
        (NATIVE + "assistant <image>", 1, True),
        (NATIVE + "assistant <image>" + NATIVE + "tool <image>", 2, True),
        ("<image>", 1, True),
        ("<image>", 1, False),
        (NATIVE + "<image>", 2, True),
    ],
)
def test_qwen_media_loading_preserves_text_and_legacy_inputs(media_loader, text, count, tokenized):
    processor, tokens = media_loader
    prompt = processor._tokenizer.encode(text, add_special_tokens=False) if tokenized else text
    original = list(prompt) if tokenized else prompt
    original_regex = tokens.get_combined_regex()
    images = [Image.new("RGB", (28, 28)) for _ in range(count)]
    output = asyncio.run(processor.load_mm_data(prompt=prompt, multimodal_tokens=tokens, image_data=images))
    assert len(output.images) == count
    assert prompt == original
    assert tokens.get_combined_regex() is original_regex
    assert tokens.image_token_regex.fullmatch("<image>") is not None
    if tokenized and text.startswith(NATIVE) and count == text.count(NATIVE):
        assert output.input_text.count("<image>") == text.count("<image>")


def test_qwen_media_loading_still_rejects_missing_image(media_loader):
    processor, tokens = media_loader
    prompt = processor._tokenizer.encode(NATIVE + NATIVE, add_special_tokens=False)
    with pytest.raises(RuntimeError, match="loading multimodal data"):
        asyncio.run(
            processor.load_mm_data(prompt=prompt, multimodal_tokens=tokens, image_data=[Image.new("RGB", (28, 28))])
        )


def test_qwen_concurrent_native_and_legacy_requests_keep_shared_rules(media_loader):
    processor, tokens = media_loader
    native = processor._tokenizer.encode(NATIVE + "assistant <image>", add_special_tokens=False)

    async def load_both():
        return await asyncio.gather(
            processor.load_mm_data(prompt=native, multimodal_tokens=tokens, image_data=[Image.new("RGB", (28, 28))]),
            processor.load_mm_data(
                prompt="<image>", multimodal_tokens=tokens, image_data=[Image.new("RGB", (28, 28))]
            ),
        )

    native_output, legacy_output = asyncio.run(load_both())
    assert len(native_output.images) == len(legacy_output.images) == 1
    assert "assistant <image>" in native_output.input_text
    assert tokens.image_token_regex.fullmatch("<image>") is not None
