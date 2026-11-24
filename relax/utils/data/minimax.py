# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MiniMax's processor expands each image placeholder into framed patch
tokens."""

import re
from collections import Counter
from typing import Any

import torch


IMAGE_TOKEN = "]<]image[>["
_VISION_START = "]<]start of image[>["
_VISION_END = "]<]end of image[>["
_IMAGE_PLACEHOLDER = re.compile(re.escape(IMAGE_TOKEN))
_IMAGE_RUN = re.compile(f"{re.escape(_VISION_START)}(?:{re.escape(IMAGE_TOKEN)})+{re.escape(_VISION_END)}")


def is_minimax_processor(processor: Any) -> bool:
    return IMAGE_TOKEN in (getattr(processor, "IMAGE_TOKEN", None), getattr(processor, "image_token", None))


def minimax_image_token_id(tokenizer: Any) -> int | None:
    get_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_vocab):
        return None
    vocab = get_vocab()
    if all(name in vocab for name in (IMAGE_TOKEN, _VISION_START, _VISION_END)):
        return vocab[IMAGE_TOKEN]
    return None


def align_minimax_loss_mask(
    *,
    tokenizer: Any,
    text: str,
    short_ids: torch.Tensor,
    short_mask: torch.Tensor,
    expanded_ids: torch.Tensor,
) -> torch.Tensor:
    """Map supervised tokens through the processor's image replacements."""
    output_ids = expanded_ids.tolist()
    expanded = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # The official tokenizer encodes in NFC. Use its normalization on both
    # strings before asking for offsets, which are relative to the input text.
    normalizer = getattr(getattr(tokenizer, "backend_tokenizer", None), "normalizer", None)
    if normalizer is not None:
        text = normalizer.normalize_str(text)
        expanded = normalizer.normalize_str(expanded)
    original_encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    expanded_encoding = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if original_encoding["input_ids"] != short_ids.tolist() or expanded_encoding["input_ids"] != output_ids:
        raise ValueError("MiniMax SFT requires lossless tokenization of the original and expanded prompts.")

    char_offsets: list[int | None] = [None] * len(text)
    before = cursor = 0
    for placeholder in _IMAGE_PLACEHOLDER.finditer(text):
        prefix = text[before : placeholder.start()]
        if not expanded.startswith(prefix, cursor):
            raise ValueError("MiniMax processor changed text before an image placeholder.")
        char_offsets[before : placeholder.start()] = range(cursor, cursor + len(prefix))
        cursor += len(prefix)
        span = _IMAGE_RUN.match(expanded, cursor)
        if span is None:
            raise ValueError("MiniMax processor returned an invalid image expansion.")
        cursor = span.end()
        before = placeholder.end()
    if text[before:] != expanded[cursor:]:
        raise ValueError("MiniMax processor changed text after the last image placeholder.")
    char_offsets[before:] = range(cursor, len(expanded))

    # Only supervised tokens must retain their exact IDs and spans.
    targets: Counter[tuple[int, int, int]] = Counter()
    for token, learn, (start, end) in zip(
        original_encoding["input_ids"], short_mask.tolist(), original_encoding["offset_mapping"], strict=True
    ):
        if not learn or end <= start:
            continue
        mapped = char_offsets[start]
        if mapped is None:
            continue
        targets[mapped, mapped + end - start, token] += 1
    mask = []
    for token, (start, end) in zip(output_ids, expanded_encoding["offset_mapping"], strict=True):
        key = (start, end, token)
        learn = int(targets[key] > 0)
        mask.append(learn)
        if not learn:
            continue
        targets[key] -= 1
    if any(targets.values()):
        raise ValueError("MiniMax processor changed a supervised token during media expansion.")
    return torch.tensor(mask, dtype=torch.long)
