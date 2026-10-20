# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Text-only continued pretraining on the SFT data contract."""

import re
from typing import Any

import torch

from relax.engine.sft.dataset.sample import CanonicalMessage, CanonicalSample


# Match ms-swift's all+ignore_empty_think configuration, including literal
# non-Qwen markers that may occur in a text document. No chat scaffold is added.
_IGNORE_EMPTY_THINK = (
    r"^<think>\s*</think>\s*",
    r"^<seed:think><seed:cot_budget_reflect>The current thinking budget is 0, so I will directly start answering "
    r"the question.</seed:cot_budget_reflect>\n</seed:think>\s*",
    r"^</think>\s*",
    r"^<\|channel>thought\n<channel\|>",
    r"^</mm:think>\s*",
)
_EMPTY_THINK_SPLIT = re.compile("|".join(f"({pattern})" for pattern in _IGNORE_EMPTY_THINK), re.DOTALL)


def _normalize_qwen3_5_text(text: str) -> str:
    """Follow Qwen3_5Template._swift_prepare_inputs, including its trim
    rules."""
    text = text.strip()
    if "</think>" in text and "<think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.rstrip("\n").rsplit("<think>", 1)[-1].lstrip("\n").strip()
        rest = after.lstrip("\n")
        text = f"<think>\n{reasoning}\n</think>\n\n{rest}"
    return text


def _qwen3_5_loss_segments(text: str) -> list[tuple[str, int]]:
    """Split and merge BEFORE tokenization to match Swift's BPE boundaries."""
    segments: list[tuple[str, int]] = []
    for part in _EMPTY_THINK_SPLIT.split(text):
        if not part:
            continue
        # Swift reclassifies each split part, including the trailing remainder.
        # Thus consecutive empty blocks can mask the remainder as well. Keep
        # that behavior for baseline parity instead of fixing it in this port.
        weight = int(not any(re.match(pattern, part, re.DOTALL) for pattern in _IGNORE_EMPTY_THINK))
        if segments and segments[-1][1] == weight:
            segments[-1] = (segments[-1][0] + part, weight)
        else:
            segments.append((part, weight))
    return segments


def _render_qwen3_5_cpt(
    text: str, *, tokenizer: Any, loss_scale: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    text = _normalize_qwen3_5_text(text)
    ids: list[int] = []
    mask: list[int] = []
    # A supplied unit message weight bypasses Swift's modifier strategy.
    segments = [(text, 1)] if loss_scale is not None else _qwen3_5_loss_segments(text)
    for part, weight in segments:
        part_ids = tokenizer.encode(part, add_special_tokens=False)
        ids.extend(part_ids)
        mask.extend([weight] * len(part_ids))
    # QwenTemplateMeta supplies endoftext; TemplateMeta.init also adds EOS.
    stop_words = ("<|endoftext|>", tokenizer.eos_token)
    if not any(word and text.endswith(word) for word in stop_words):
        ids.append(tokenizer.eos_token_id)
        mask.append(1)
    if len(ids) < 2:
        raise ValueError("CPT requires at least two tokens for next-token prediction")
    mask[0] = 0  # Swift labels[0] = -100; no preceding token can predict it.
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def truncate_qwen3_5_cpt(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tokenizer: Any,
    capacity: int,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Swift's protected placeholder tokens and first-label reset."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    protected_ids = set()
    if convert is not None:
        for name in ("<|image_pad|>", "<|video_pad|>"):
            token_id = convert(name)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                protected_ids.add(token_id)
    # These are CPU dataset tensors, before any GPU transfer or packing.
    ids = tokens.tolist()
    protected = {index for index, token in enumerate(ids) if token in protected_ids}
    if len(protected) > capacity:
        raise ValueError(f"CPT protected placeholder count {len(protected)} exceeds token capacity {capacity}")
    remaining = [index for index in range(len(ids)) if index not in protected]
    budget = max(0, capacity - len(protected))
    if budget:
        protected.update(remaining[-budget:] if strategy == "truncate_left" else remaining[:budget])
    indices = sorted(protected)
    result_ids, result_mask = tokens[indices].contiguous(), loss_mask[indices].clone()
    if result_mask.numel():
        result_mask[0] = 0
    return result_ids, result_mask


def build_cpt_sample(row: dict[str, Any], *, prompt_key: str, source_name: str, row_index: int) -> CanonicalSample:
    """Accept a text column or ms-swift's single-assistant pretraining
    format."""
    context = f"CPT row idx={row_index} in {source_name!r}"
    for key in ("images", "image", "videos", "video", "audios", "audio", "tools"):
        value = row.get(key)
        if value is not None and not (isinstance(value, (list, tuple, str)) and len(value) == 0):
            raise ValueError(f"{context}: text-only CPT does not support {key!r}")

    # Auto mode accepts either supported CPT schema per row while preserving
    # the strict behavior of explicitly configured column names.
    key = prompt_key
    if key == "auto":
        candidates = [candidate for candidate in ("text", "messages") if candidate in row]
        if len(candidates) != 1:
            raise ValueError(f"{context}: CPT auto input requires exactly one of 'text' or 'messages'")
        key = candidates[0]
    elif key == "input" and key not in row:
        key = "text" if "text" in row else "messages"
    if key not in row:
        raise ValueError(f"{context}: missing text column {key!r}")
    text = row[key]
    loss_scale = None
    if isinstance(text, list):
        if len(text) != 1 or not isinstance(text[0], dict) or text[0].get("role") != "assistant":
            raise ValueError(f"{context}: CPT messages must contain exactly one assistant text message")
        message = text[0]
        if message.get("tool_calls") or message.get("loss") is False or message.get("learn") is False:
            raise ValueError(f"{context}: CPT requires plain text with all-token supervision")
        if message.get("loss_scale") not in (None, 1, 1.0):
            raise ValueError(f"{context}: CPT does not support per-message loss weights")
        loss_scale = message.get("loss_scale")
        text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{context}: CPT requires non-empty text")
    return CanonicalSample(
        messages=[CanonicalMessage(role="assistant", content=text, learn=True)],
        metadata={"source_dataset": source_name, "row_index": row_index, "cpt_loss_scale": loss_scale},
    )


def render_cpt(sample: CanonicalSample, *, tokenizer: Any, template: str = "raw") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode CPT text using raw or ms-swift-compatible Qwen3.5 semantics."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("CPT requires tokenizer.eos_token_id to delimit documents")
    if template == "qwen3_5":
        return _render_qwen3_5_cpt(
            sample.messages[0].content, tokenizer=tokenizer, loss_scale=sample.metadata.get("cpt_loss_scale")
        )
    if template != "raw":
        raise ValueError(f"Unknown CPT template: {template!r}")
    # Let the tokenizer supply its normal special tokens (including BOS if
    # applicable). Qwen3.5 adds none here. Do not hardcode chat/EOD token IDs.
    ids = list(tokenizer.encode(sample.messages[0].content, add_special_tokens=True))
    if not ids:
        raise ValueError("CPT text produced no tokens")
    if ids[-1] != eos_id:
        ids.append(eos_id)
    if len(ids) < 2:
        raise ValueError("CPT requires at least two tokens for next-token prediction")
    tokens = torch.tensor(ids, dtype=torch.long)
    # Megatron aligns this mask once, per document, and masks its last label.
    return tokens, torch.ones_like(tokens)
