# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""M-BEIR pairs with native sampling in the shared contrastive SFT loop."""

import random
from pathlib import Path
from typing import Any

import torch

from examples.lamra_rerank.data import format_text, load_rank_image, make_entity
from relax.engine.sft.dataset.retrieval import NLIPairDataset, _encode_pair


def retrieval_messages(entity: dict[str, str]) -> list[dict[str, Any]]:
    """Match native LamRA's text, image and mixed one-word prompts."""
    if not entity or set(entity) - {"txt", "image"}:
        raise ValueError("A retrieval entity requires text and/or one image")
    content = []
    if "image" in entity:
        content.append({"type": "image", "image": entity["image"]})
    if "txt" in entity:
        kind = "image and sentence" if "image" in entity else "sentence"
        text = entity["txt"] + f"\nSummarize above {kind} in one word: "
    else:
        text = "\nSummarize above image in one word: "
    content.append({"type": "text", "text": text})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


def select_visit(
    query: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    instructions: dict[tuple[str, str, str], list[str]],
    rng: random.Random,
) -> dict[str, Any]:
    """Choose anew on every occurrence, including the padded epoch prefix."""
    did = rng.choice(query["pos_cand_list"])
    positive = candidates[did]
    key = (query["qid"].split(":")[0], query["query_modality"], positive["modality"])
    instruction = format_text(rng.choice(instructions[key]))
    if not instruction:
        raise ValueError(f"Empty retrieval instruction for {key}")
    return {
        "qid": query["qid"],
        "positive_id": did,
        "instruction": instruction,
        "query_text": format_text(f"{instruction} {query.get('query_txt') or ''}"),
        "positive_text": format_text(positive.get("txt")),
        "query_image": query.get("query_img_path"),
        "positive_image": positive.get("img_path"),
    }


def visit_row(visit: dict[str, Any], query_text: str, positive_text: str, image_root: Path) -> dict[str, Any]:
    """The two text arguments have already undergone native 480-token
    truncation."""
    return {
        "query": {"messages": retrieval_messages(make_entity(query_text, visit["query_image"], image_root))},
        "positive": {"messages": retrieval_messages(make_entity(positive_text, visit["positive_image"], image_root))},
        "metadata": {key: value for key, value in visit.items() if key not in {"query_text", "positive_text"}},
    }


def _encode_mbeir_pair(
    row: dict[str, Any], emb_token_id: int, max_length: int, processor: Any = None
) -> dict[str, Any]:
    if processor is None:
        from relax.utils.data.processor_pool import get_worker_processor

        processor = get_worker_processor()
    source = row["native_mbeir"]
    candidates = {candidate["did"]: candidate for candidate in source["positives"]}
    instructions = {tuple(key): values for key, values in source["instructions"]}
    visit = select_visit(source["query"], candidates, instructions, random)
    encoded_text = processor.tokenizer(
        [visit["query_text"], visit["positive_text"]],
        truncation=True,
        max_length=480,
        padding=False,
        add_special_tokens=False,
    )
    query_text, positive_text = processor.tokenizer.batch_decode(encoded_text["input_ids"])
    image_root = Path(source["image_root"])
    row = visit_row(visit, query_text, positive_text, image_root)
    result = {}
    for side in ("query", "positive"):
        messages = row[side]["messages"]
        if len(messages) != 2 or [item["role"] for item in messages] != ["user", "assistant"]:
            raise ValueError("M-BEIR branches require one user/assistant conversation")
        images = []
        for message in messages:
            for part in message["content"]:
                if part["type"] == "image":
                    images.append(load_rank_image(part["image"]))
                elif part["type"] != "text":
                    raise ValueError("M-BEIR retrieval supports text/images only")
        if len(images) > 1:
            raise ValueError("Native M-BEIR has at most one image per branch")
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = processor(
            text=[rendered], images=images or None, padding=False, truncation=False, return_tensors="pt"
        )
        ids = encoded["input_ids"][0].contiguous()
        positions = torch.nonzero(ids == emb_token_id, as_tuple=False).flatten()
        if ids.device.type != "cpu" or ids.ndim != 1 or not 2 <= ids.numel() <= max_length:
            raise ValueError("M-BEIR branch is invalid or overlength; no visits are dropped/truncated")
        if positions.numel() == 0 or int(positions[0]) == 0:
            raise ValueError("M-BEIR branch requires <emb> with a preceding token")
        branch = {"input_ids": ids}
        # Only model tensors cross TQ. Processor control flags and the padded
        # attention mask do not belong in THD multimodal_train_inputs.
        for key in ("pixel_values", "image_grid_thw"):
            if key in encoded:
                if not isinstance(encoded[key], torch.Tensor) or encoded[key].device.type != "cpu":
                    raise TypeError(f"M-BEIR processor returned a non-CPU tensor for {key}")
                branch[key] = encoded[key].contiguous()
        if images and ("pixel_values" not in branch or len(branch.get("image_grid_thw", [])) != len(images)):
            raise ValueError("M-BEIR image tensors do not match the branch image")
        result[side] = branch
    return result


class _MbeirExecutor:
    def __init__(self, executor: Any) -> None:
        self.executor = executor

    def submit(self, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        if function is _encode_pair:
            function = _encode_mbeir_pair
        return self.executor.submit(function, *args, **kwargs)


class _MbeirPool:
    def __init__(self, pool: Any) -> None:
        self.executor = _MbeirExecutor(pool.executor)


class MbeirPairDataset(NLIPairDataset):
    """Use Relax's default IndexManager and draw a positive/prompt on access.

    Preparation retains raw query order and every positive/prompt choice. The
    existing processor pool performs those draws and multimedia encoding; no
    shared permutation or per-occurrence choice is replayed.
    """

    _preload_jsonl = False

    def __init__(self, *args: Any, processor_pool: Any = None, **kwargs: Any) -> None:
        if processor_pool is None:
            raise ValueError("M-BEIR retrieval requires the existing SFT processor pool")
        super().__init__(*args, processor_pool=_MbeirPool(processor_pool), **kwargs)
