# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""LamRA query/candidate messages shared by preparation and rerank
evaluation."""

import csv
import math
from pathlib import Path
from typing import Any


# Retain the reference prompt, including its whitespace and spelling.
POINTWISE_PROMPT = (
    "I will provide you with a query and a candidate. Please evaluate whether the candidate"
    "                        meets the requirements of the query. If it does, respond with 'Yes'; "
    "if it doesn't, responed with 'No'."
)
LISTWISE_PROMPT = (
    "I will provide you with a query followed by multiple candidates in the format: (1) cand1 (2) cand2, etc. "
    "Each candidate is independent of the others. "
    "                        Evaluate each candidate against the query, and respond with the number "
    "corresponding to the candidate that best meets the requirements of the query."
)


def format_text(text: str | None) -> str:
    """Apply LamRA's text normalization before optional token truncation."""
    text = (text or "").replace("\r", "").strip().strip('"')
    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
    return text


def load_instructions(path: Path) -> dict[tuple[str, str, str], list[str]]:
    result = {}
    with path.open() as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader)
        for fields in reader:
            result[(fields[3], fields[0], fields[1])] = [text for text in fields[4:] if text]
    return result


def make_entity(text: str, image: str | None, image_root: Path) -> dict[str, str]:
    result = {}
    if image:
        path = image_root / image
        if not path.is_relative_to(image_root):
            raise ValueError(f"Image path escapes image root: {image!r}")
        result["image"] = str(path)
    if text or not image:
        result["txt"] = text
    return result


def _append_entity(content: list[dict[str, str]], entity: dict[str, str]) -> None:
    if "image" in entity:
        content.append({"type": "image", "image": entity["image"]})
    if "txt" in entity:
        content.append({"type": "text", "text": entity["txt"]})


def pointwise_messages(
    query: dict[str, str], candidate: dict[str, str], answer: str | None = None
) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": POINTWISE_PROMPT}, {"type": "text", "text": "Query:"}]
    _append_entity(content, query)
    content.append({"type": "text", "text": "Candidate:"})
    _append_entity(content, candidate)
    messages = [{"role": "user", "content": content}]
    if answer is not None:
        if answer not in {"Yes", "No"}:
            raise ValueError("Pointwise labels must be Yes or No")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def listwise_messages(query: dict[str, str], candidates: list[dict[str, str]], answer: int) -> list[dict[str, Any]]:
    if not 2 <= len(candidates) <= 5 or not 1 <= answer <= len(candidates):
        raise ValueError("Listwise supervision requires 2..5 candidates and a valid one-based answer")
    content = [{"type": "text", "text": LISTWISE_PROMPT}, {"type": "text", "text": "Query:"}]
    _append_entity(content, query)
    content.append({"type": "text", "text": "Candidates:"})
    for index, candidate in enumerate(candidates, 1):
        content.append({"type": "text", "text": f"({index}) "})
        _append_entity(content, candidate)
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": f"Ans: ({answer})"}]},
    ]


def load_rank_image(path: str | Path) -> Any:
    """Match the reference RGB conversion and factor-28, 4..300-token
    resize."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if max(width, height) / min(width, height) > 200:
        raise ValueError(f"Image aspect ratio exceeds the reference's 200 limit: {path}")
    resized_h = max(28, round(height / 28) * 28)
    resized_w = max(28, round(width / 28) * 28)
    if resized_h * resized_w > 300 * 28**2:
        scale = math.sqrt(height * width / (300 * 28**2))
        resized_h = math.floor(height / scale / 28) * 28
        resized_w = math.floor(width / scale / 28) * 28
    elif resized_h * resized_w < 4 * 28**2:
        scale = math.sqrt(4 * 28**2 / (height * width))
        resized_h = math.ceil(height * scale / 28) * 28
        resized_w = math.ceil(width * scale / 28) * 28
    if resized_h / resized_w > 200:
        resized_h = resized_w * 200
    elif resized_w / resized_h > 200:
        resized_w = resized_h * 200
    return image.resize((resized_w, resized_h))
