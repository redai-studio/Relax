# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native DeepSeek-V4.1 multimodal SFT inputs from a trusted local checkpoint."""

import io
import runpy
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any


_reference_lock = Lock()


@lru_cache(maxsize=8)
def _reference(model_dir: str, source: str) -> SimpleNamespace:
    path = Path(model_dir) / source
    if not path.is_file():
        raise FileNotFoundError(f"DeepSeek-V4.1 multimodal checkpoint is missing {path}")
    # run_path temporarily registers a module; simultaneous cold starts must not replace it.
    with _reference_lock:
        return SimpleNamespace(**runpy.run_path(str(path), run_name=f"_relax_deepseek_v41_{path.stem}"))


def render_conversation(
    model_dir: str,
    messages: list[dict],
    tools: list[dict] | None,
    options: dict,
    *,
    ignore_empty_think: bool = False,
) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Use the reference's message boundaries and return supervised spans and image order."""
    from relax.engine.sft.dataset.deepseek_chat_template_patch import split_embedded_thinking

    enc = _reference(model_dir, "encoding/encoding.py")
    mode = options.get("thinking_mode", "thinking" if options.get("enable_thinking", True) else "chat")
    drop = options.get("drop_thinking", not options.get("preserve_thinking", True)) and not tools
    normalized = []
    image_count = 0
    for message in messages:
        msg = dict(message)
        if msg["role"] == "function_call":
            msg["role"] = "assistant"
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] in ("image", "image_url"):
                    if msg["role"] not in ("user", "tool"):
                        raise ValueError(f"DeepSeek-V4.1 cannot insert images in a {msg['role']} message")
                    # Let the official encoder reorder parallel tool results, without copying media bytes.
                    parts.append({"type": "image_url", "image_url": {"url": str(image_count)}})
                    image_count += 1
                elif part["type"] == "text":
                    parts.append(part)
                else:
                    raise ValueError(f"Unsupported DeepSeek-V4.1 content type: {part['type']}")
            if msg["role"] == "user":
                # merge_tool_messages reads structured user turns from content_blocks.
                msg["content_blocks"] = parts
                content = ""
            elif msg["role"] == "tool":
                content = parts
            else:
                content = "".join(part["text"] for part in parts)
        if msg["role"] == "assistant" and msg.get("reasoning_content") is None:
            reasoning, content = split_embedded_thinking(content)
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
        if mode == "chat" and msg.get("reasoning_content"):
            raise ValueError("DeepSeek-V4.1 SFT with assistant reasoning requires thinking_mode='thinking'")
        msg["content"] = content
        enc._validate_no_image_sp_tokens(msg)
        normalized.append(msg)
    if tools:
        if not normalized or normalized[0]["role"] != "system":
            normalized.insert(0, {"role": "system", "content": ""})
        normalized[0]["tools"] = tools
    merged = enc.sort_tool_results_by_call_order(enc.merge_tool_messages(normalized))
    processed, media = enc.process_image_messages(merged)
    drop = drop and not any(msg.get("tools") for msg in processed)
    if mode == "thinking" and drop:
        processed = enc._drop_thinking_messages(processed)
    text_parts = [enc.bos_token]
    offset = len(enc.bos_token)
    spans = []
    for i, msg in enumerate(processed):
        part = enc.render_message(
            i, processed, thinking_mode=mode, drop_thinking=drop, reasoning_effort=options.get("reasoning_effort")
        )
        if msg["role"] == "assistant" and msg.get("learn"):
            start = offset
            if ignore_empty_think and not msg.get("reasoning_content") and part.startswith(enc.thinking_end_token):
                start += len(enc.thinking_end_token)
            spans.append((start, offset + len(part)))
        offset += len(part)
        text_parts.append(part)
    order = [int(item["url"]) for item in media]
    if sorted(order) != list(range(image_count)):
        raise ValueError("DeepSeek-V4.1 renderer lost or duplicated an image")
    return "".join(text_parts), spans, order


def compact_ids_and_mask(tokenizer: Any, text: str, spans: list[tuple[int, int]]) -> tuple[Any, Any]:
    """Tokenize the complete prompt once; BPE segmentation can depend on adjacent text."""
    import torch

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [left for left, _ in spans]
    mask = []
    for left, right in encoded["offset_mapping"]:
        idx = bisect_right(starts, left) - 1
        mask.append(int(idx >= 0 and left < right <= spans[idx][1]))
    return torch.tensor(encoded["input_ids"], dtype=torch.long), torch.tensor(mask, dtype=torch.long)


class DeepSeekV41Processor:
    def __init__(self, tokenizer: Any, config: dict) -> None:
        self.tokenizer = tokenizer
        self.image_token_id = int(config["image_token_id"])
        vision = config["vision_config"]
        self.image_processor = SimpleNamespace(
            patch_size=vision["patch_size"],
            vision_patch_size=vision["patch_size"],
            vision_downsample_ratio=vision["downsample_ratio"],
            vision_min_pixels=vision["min_pixels"],
            vision_max_n_token=vision["max_image_tokens"],
            vision_max_wh_ratio=vision.get("max_wh_ratio"),
        )

    def prompt_token_count(self, text: str, images: list) -> int:
        ref = _reference(self.tokenizer.deepseek_v41_model_dir, "inference/image_processor.py")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if tokens.count(self.image_token_id) != len(images):
            raise ValueError("DeepSeek-V4.1 rendered image slots and media count disagree")
        length = len(tokens) - len(images)
        for image in images:
            h, w, _, _ = ref.plan_image_grid(*image.size, self.image_processor)
            length += ref.num_image_tokens(h, w)
        return length

    def __call__(self, text: str, images: list | None = None, **kwargs: Any) -> dict[str, Any]:
        import torch

        if kwargs.get("videos") or kwargs.get("audio") or kwargs.get("audios"):
            raise ValueError("DeepSeek-V4.1 preprocessing accepts images and text only")
        ref = _reference(self.tokenizer.deepseek_v41_model_dir, "inference/image_processor.py")
        image_list = [] if images is None else images
        compact = self.tokenizer.encode(text, add_special_tokens=False)
        slots = [i for i, token in enumerate(compact) if token == self.image_token_id]
        if len(slots) != len(image_list):
            raise ValueError(f"DeepSeek-V4.1 prompt contains {len(slots)} image slots for {len(image_list)} images")
        pixels, grids, spans = [], [], []
        expanded = []
        cursor = 0
        for pos, image in zip(slots, image_list, strict=True):
            # The shared worker supplies decoded PIL images. PNG preserves their RGB pixels.
            with io.BytesIO() as buffer:
                rgb = image if image.mode == "RGB" else image.convert("RGB")
                rgb.save(buffer, format="PNG", compress_level=0)
                patches, h, w, llm_h, llm_w = ref.load_image({"data": buffer.getvalue()}, self.image_processor)
            count = ref.num_image_tokens(llm_h, llm_w)
            expanded.extend(compact[cursor:pos])
            spans.append((len(expanded), count))
            expanded.extend([self.image_token_id] * count)
            pixels.append(patches)
            grids.append((h, w))
            cursor = pos + 1
        expanded.extend(compact[cursor:])
        result = {"input_ids": torch.tensor([expanded], dtype=torch.long)}
        if pixels:
            result.update(
                pixel_values=torch.cat(pixels),
                image_grid_hws=torch.tensor(grids, dtype=torch.long),
                image_spans=torch.tensor(spans, dtype=torch.long),
            )
        return result


def expand_sft_mask(compact: Any, mask: Any, expanded: Any, spans: Any) -> Any:
    """Check sample-local image replacements and keep all image positions out of SFT loss."""
    import torch

    collapsed = expanded.tolist()
    counts = [1] * compact.numel()
    for start, length in reversed(spans.tolist()):
        if length < 1 or start < 0 or start + length > len(collapsed):
            raise ValueError(f"Invalid DeepSeek-V4.1 image span {(start, length)}")
        if any(token != collapsed[start] for token in collapsed[start : start + length]):
            raise ValueError("DeepSeek-V4.1 image span contains a text token")
        collapsed[start : start + length] = [collapsed[start]]
    if collapsed != compact.tolist():
        raise ValueError("DeepSeek-V4.1 compact and expanded token sequences disagree")
    extra = 0
    for start, length in spans.tolist():
        pos = start - extra
        if mask[pos]:
            raise ValueError("DeepSeek-V4.1 image occurs inside assistant supervision")
        counts[pos] = length
        extra += length - 1
    return torch.repeat_interleave(mask, torch.tensor(counts), output_size=expanded.numel())
