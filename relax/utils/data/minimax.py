# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MiniMax's processor expands each media placeholder into framed patch
tokens."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch


IMAGE_TOKEN = "]<]image[>["
VIDEO_TOKEN = "]<]video[>["
_VISION_START = "]<]start of image[>["
_VISION_END = "]<]end of image[>["
_MEDIA = re.compile(f"{re.escape(IMAGE_TOKEN)}|{re.escape(VIDEO_TOKEN)}")
_TIMESTAMP = r"\]<\]\d+\.\d+ seconds\[>\["
_IMAGE_RUN = re.compile(f"{re.escape(_VISION_START)}(?:{re.escape(IMAGE_TOKEN)})+{re.escape(_VISION_END)}")
_VIDEO_FRAME = re.compile(
    f"(?:{_TIMESTAMP})?{re.escape(_VISION_START)}(?:{re.escape(VIDEO_TOKEN)})+{re.escape(_VISION_END)}"
)


def is_minimax_processor(processor: Any) -> bool:
    return IMAGE_TOKEN in (getattr(processor, "IMAGE_TOKEN", None), getattr(processor, "image_token", None))


@dataclass(frozen=True)
class MiniMaxMediaTokens:
    image: int
    video: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "MiniMaxMediaTokens | None":
        get_vocab = getattr(tokenizer, "get_added_vocab", None)
        if not callable(get_vocab):
            return None
        vocab = get_vocab()
        token_names = (IMAGE_TOKEN, VIDEO_TOKEN, _VISION_START, _VISION_END)
        if all(name in vocab for name in token_names):
            return cls(vocab[IMAGE_TOKEN], vocab[VIDEO_TOKEN])
        return None


def align_minimax_loss_mask(
    *,
    tokenizer: Any,
    text: str,
    short_ids: torch.Tensor,
    short_mask: torch.Tensor,
    expanded_ids: torch.Tensor,
    video_grids: torch.Tensor | None,
) -> torch.Tensor:
    """Map supervised tokens through the processor's image and video
    replacements."""
    output_ids = expanded_ids.tolist()
    expanded = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    original_encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    expanded_encoding = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    if original_encoding["input_ids"] != short_ids.tolist() or expanded_encoding["input_ids"] != output_ids:
        raise ValueError("MiniMax SFT requires lossless tokenization of the original and expanded prompts.")

    frames = iter(()) if video_grids is None else (int(grid[0]) for grid in video_grids)
    char_offsets: list[int | None] = [None] * len(text)
    before = cursor = 0
    for placeholder in _MEDIA.finditer(text):
        prefix = text[before : placeholder.start()]
        if not expanded.startswith(prefix, cursor):
            raise ValueError("MiniMax processor changed text before a media placeholder.")
        char_offsets[before : placeholder.start()] = range(cursor, cursor + len(prefix))
        cursor += len(prefix)
        if placeholder.group() == VIDEO_TOKEN:
            count = next(frames, 0)
            pattern = _VIDEO_FRAME
        else:
            count = 1
            pattern = _IMAGE_RUN
        if count <= 0:
            raise ValueError("MiniMax video placeholder has no matching temporal grid.")
        for _ in range(count):
            span = pattern.match(expanded, cursor)
            if span is None:
                raise ValueError("MiniMax processor returned an invalid media expansion.")
            cursor = span.end()
        before = placeholder.end()
    if text[before:] != expanded[cursor:] or next(frames, None) is not None:
        raise ValueError("MiniMax processor changed trailing text or returned unmatched video grids.")
    char_offsets[before:] = range(cursor, len(expanded))

    # Timestamp insertion can merge with the preceding user whitespace during
    # tokenization. Only supervised tokens must retain their exact IDs and spans.
    targets: Counter[tuple[int, int, int]] = Counter()
    for token, learn, (start, end) in zip(
        original_encoding["input_ids"], short_mask.tolist(), original_encoding["offset_mapping"], strict=True
    ):
        if learn and end > start and (mapped := char_offsets[start]) is not None:
            targets[mapped, mapped + end - start, token] += 1
    mask = []
    for token, (start, end) in zip(output_ids, expanded_encoding["offset_mapping"], strict=True):
        key = (start, end, token)
        learn = int(targets[key] > 0)
        mask.append(learn)
        if learn:
            targets[key] -= 1
    if any(targets.values()):
        raise ValueError("MiniMax processor changed a supervised token during media expansion.")
    return torch.tensor(mask, dtype=torch.long)
