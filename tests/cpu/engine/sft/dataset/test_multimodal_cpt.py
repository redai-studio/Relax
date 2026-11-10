# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPT must retain media while supervising text from every conversation
role."""

import json
from unittest.mock import patch

import pytest
import torch

from relax.engine.sft.dataset.streaming import SFTStreamingDataset


class Tokenizer:
    unk_token_id = -1
    eos_token_id = 9
    pad_token_id = 9
    chat_template = "test"

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 3, "<|vision_start|>": 2, "<|vision_end|>": 4}.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        assert text == "rendered"
        assert not add_special_tokens
        return [1, 2, 3, 4, 5, 9]


def dataset(tmp_path, **kwargs):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "<image>question", "loss": False},
                    {"role": "assistant", "content": "answer", "loss": False},
                ],
                "images": ["image.png"],
            }
        )
        + "\n"
    )
    return SFTStreamingDataset(
        str(path),
        tokenizer=Tokenizer(),
        prompt_key="messages",
        training_mode="cpt",
        cpt_template="multimodal",
        multimodal_keys={"image": "images"},
        pad_token_ids=[3],
        prefetch_max_cached=0,
        **kwargs,
    )


def test_all_text_is_supervised_and_expanded_media_is_masked(tmp_path):
    ds = dataset(tmp_path)
    with patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered") as render:
        sample = ds._render_one(0)
    render.assert_called_once()
    assert sample.sample.images == ["image.png"]
    assert sample.rendered_text == "rendered"
    assert sample.short_mask.tolist() == [1, 0, 0, 0, 1, 1]
    media = {"pixel_values": torch.ones(1), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    processed = ds._build_processed(sample, [1, 2, 3, 3, 3, 4, 5, 9], media)
    assert processed.loss_mask.tolist() == [1, 0, 0, 0, 0, 0, 1, 1]
    assert processed.multimodal_train_inputs is media


def test_cpt_rejects_generation_only_prefix(tmp_path):
    ds = dataset(tmp_path, apply_chat_template_kwargs={"add_generation_prompt": True})
    with pytest.raises(ValueError, match="generation prompts"):
        ds._render_one(0)


def test_expanded_multimodal_cpt_cannot_be_truncated(tmp_path):
    ds = dataset(tmp_path, capacity=7, oversize_strategy="truncate_right")
    with patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered"):
        sample = ds._render_one(0)
    with pytest.raises(ValueError, match="misalign"):
        ds._build_processed(sample, [1, 2, 3, 3, 3, 4, 5, 9], {"pixel_values": torch.ones(1)})


def test_cpt_video_inserted_timestamps_skip_sft_alignment(tmp_path):
    ds = dataset(tmp_path)
    with patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered"):
        sample = ds._render_one(0)
    # A processor may add a timestamp before each expanded video frame.
    expanded = [1, 77, 2, 3, 3, 4, 78, 2, 3, 4, 5, 9]
    with patch("relax.engine.sft.dataset.streaming._expand_loss_mask_via_alignment") as align:
        processed = ds._build_processed(sample, expanded, {"pixel_values": torch.ones(1)})
    align.assert_not_called()
    assert processed.tokens.tolist() == expanded
    assert processed.loss_mask.tolist() == [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]
