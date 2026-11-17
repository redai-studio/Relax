# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate benchmark token/mask semantics through the native SFT renderer."""

import json
from types import SimpleNamespace

import pytest
import torch

from examples.long_context_repeat import DoubledSFTDataset, FilledSFTDataset, RepeatedSFTDataset


class Tokenizer:
    chat_template = "{% generation %}assistant{% endgeneration %}"

    def apply_chat_template(self, messages, *, tokenize=True, return_assistant_tokens_mask=False, **kwargs):
        ids, mask, text = [], [], ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            text += content
            ids.extend(ord(character) for character in content)
            mask.extend([int(message["role"] == "assistant")] * len(content))
        if not tokenize:
            return text
        tokens = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tokens, "assistant_masks": [mask]} if return_assistant_tokens_mask else tokens


def dataset(tmp_path, cls, limit, *, rows=None, **overrides):
    path = tmp_path / "data.jsonl"
    rows = rows or [{"messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "Answer"}]}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(
        seq_length=limit,
        prompt_data=str(path),
        input_key="messages",
        seed=9714,
        sft_prefetch_buffer_size=0,
        sft_prefetch_num_workers=1,
        **overrides,
    )
    return cls.from_args(args, tokenizer=Tokenizer(), processor_pool=None, pad_token_ids=set())


@pytest.mark.parametrize(
    "cls,limit,target",
    [
        (RepeatedSFTDataset, 524288, 8),
        (DoubledSFTDataset, 1048576, 16),
        (FilledSFTDataset, 524288, 524288),
        (FilledSFTDataset, 1048576, 1048576),
    ],
)
def test_rendered_tokens_and_supervision_repeat_together(tmp_path, cls, limit, target):
    ds = dataset(tmp_path, cls, limit)
    try:
        ds.shuffle(0)
        samples, _ = ds.get_batch(1)
        sample = samples[0]
        original_tokens = torch.tensor([ord(c) for c in "Q?Answer"])
        original_mask = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
        indices = torch.arange(target) % len(original_tokens)
        assert sample.total_length == target
        assert torch.equal(sample.tokens, original_tokens[indices])
        assert torch.equal(sample.loss_mask, original_mask[indices])
        assert sample.source_idx == 0 and sample.pre_shifted_labels is None
    finally:
        ds.stop()


@pytest.mark.parametrize(
    "cls,limit", [(RepeatedSFTDataset, 524288), (DoubledSFTDataset, 1048576), (FilledSFTDataset, 1048576)]
)
@pytest.mark.parametrize("source_length", [524288, 524289])
def test_all_scenarios_share_the_same_source_length_filter(tmp_path, cls, limit, source_length):
    rows = [
        {"messages": [{"role": "user", "content": "Q" * (source_length - 1)}, {"role": "assistant", "content": "A"}]}
    ]
    ds = dataset(tmp_path, cls, limit, rows=rows)
    try:
        rendered = ds._render_one(0)
        sample = ds._build_processed(rendered, None, None)
        if source_length > 524288:
            assert sample is None
        else:
            assert sample.total_length == limit
    finally:
        ds.stop()


def test_repeat_keeps_partial_final_copy_and_rejects_supervised_start(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "AB"}]}]
    ds = dataset(tmp_path, FilledSFTDataset, 524288, rows=rows)
    try:
        rendered = ds._render_one(0)
        sample = ds._build_processed(rendered, None, None)
        assert sample.tokens[-2:].tolist() == [ord("Q"), ord("A")]
        assert sample.loss_mask[-2:].tolist() == [0, 1]
        rendered.short_mask[0] = 1
        with pytest.raises(NotImplementedError, match="supervised first token"):
            ds._build_processed(rendered, None, None)
    finally:
        ds.stop()


def test_multimodal_repeat_rejected(tmp_path):
    ds = dataset(tmp_path, FilledSFTDataset, 524288)
    try:
        with pytest.raises(NotImplementedError, match="rendered text"):
            ds._build_processed(ds._render_one(0), None, {"pixel_values": torch.ones(1)})
    finally:
        ds.stop()


@pytest.mark.parametrize(
    "cls,limit,overrides,exception",
    [
        (RepeatedSFTDataset, 1048576, {}, ValueError),
        (DoubledSFTDataset, 524288, {}, ValueError),
        (FilledSFTDataset, 2048, {}, ValueError),
        (FilledSFTDataset, 524288, {"sft_training_mode": "cpt"}, NotImplementedError),
        (FilledSFTDataset, 524288, {"task_type": "seq_cls"}, NotImplementedError),
    ],
)
def test_invalid_dataset_contract_rejected(tmp_path, cls, limit, overrides, exception):
    with pytest.raises(exception):
        dataset(tmp_path, cls, limit, **overrides)


def test_dataset_cursor_resume_preserves_repeated_sample(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": str(i)}]} for i in range(5)
    ]
    ds = dataset(tmp_path, DoubledSFTDataset, 1048576, rows=rows)
    restored = dataset(tmp_path, DoubledSFTDataset, 1048576, rows=rows)
    try:
        ds.shuffle(0)
        ds.get_batch(2)
        expected, _ = ds.get_batch(1)
        restored.shuffle(0, position=2)
        actual, _ = restored.get_batch(1)
        assert actual[0].source_idx == expected[0].source_idx
        assert torch.equal(actual[0].tokens, expected[0].tokens)
        assert torch.equal(actual[0].loss_mask, expected[0].loss_mask)
    finally:
        ds.stop()
        restored.stop()
