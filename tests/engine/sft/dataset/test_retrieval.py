# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise pair atomicity and padded-epoch resume with the real SFT reader."""

import asyncio
import json

import pytest
import torch

from relax.engine.sft.dataset.retrieval import NLIPairDataset, _encode_pair
from relax.utils.training.retrieval_utils import expand_retrieval_rollout_data


def _dataset(tmp_path, *, count=5, batch_size=4):
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"query": {"input_ids": [index + 10, 99]}, "positive": {"input_ids": [index + 20, 99]}}
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return NLIPairDataset(str(path), emb_token_id=99, batch_size=batch_size, max_length=8, processor_pool=None)


def _batch(dataset):
    samples, crossed = asyncio.run(dataset.get_batch_async(dataset.batch_size))
    return [sample.source_idx for sample in samples], samples, crossed


def test_pair_rows_remain_atomic_until_dp_assignment(tmp_path):
    dataset = _dataset(tmp_path)
    indices, samples, crossed = _batch(dataset)
    assert len(set(indices)) == 4
    assert crossed is False
    # This list represents an arbitrary post-sampler DP slice, including reordering.
    selected = [samples[3].multimodal_train_inputs, samples[0].multimodal_train_inputs]
    expanded = expand_retrieval_rollout_data({"multimodal_train_inputs": selected})
    assert expanded["retrieval_pair_count"] == 2
    for offset, sample_index in enumerate([3, 0]):
        row_id = samples[sample_index].source_idx
        assert expanded["tokens"][2 * offset].tolist() == [row_id + 10, 99]
        assert expanded["tokens"][2 * offset + 1].tolist() == [row_id + 20, 99]
    assert expanded["total_lengths"] == [2] * 4
    assert "multimodal_train_inputs" not in expanded
    # Re-entering post-processing must not alias a dict subsequently clear()ed.
    assert expand_retrieval_rollout_data(expanded) is not expanded


def test_tail_repeats_same_epoch_head_then_advances(tmp_path):
    dataset = _dataset(tmp_path)
    first, _, _ = _batch(dataset)
    tail, _, crossed = _batch(dataset)
    assert set(first + tail) == set(range(5))
    assert tail[1:] == first[:3]
    assert crossed is False
    _indices, _, crossed = _batch(dataset)
    assert crossed is True
    assert dataset.index_manager.current_epoch == 1


@pytest.mark.parametrize("resume_step", [1, 2, 3, 4, 7])
def test_resume_matches_uninterrupted_order_across_padded_epochs(tmp_path, resume_step):
    original = _dataset(tmp_path)
    for _ in range(resume_step):
        _batch(original)
    expected = [_batch(original)[0] for _ in range(3)]
    resumed = _dataset(tmp_path)
    # The existing SFT component encodes consumed rows using the *raw* length.
    consumed_pairs = resume_step * resumed.batch_size
    epoch, position = divmod(consumed_pairs, len(resumed))
    resumed.shuffle(epoch, position=position)
    assert [_batch(resumed)[0] for _ in range(3)] == expected


def test_small_dataset_can_pad_a_complete_pair_batch(tmp_path):
    dataset = _dataset(tmp_path, count=2, batch_size=8)
    indices, _, _ = _batch(dataset)
    assert indices == indices[:2] * 4


@pytest.mark.parametrize("ids", [[99, 1], [1, 2], [1] * 9 + [99]])
def test_invalid_pairs_fail_without_filtering_or_truncation(ids):
    with pytest.raises(ValueError):
        _encode_pair({"query": {"input_ids": ids}, "positive": {"input_ids": [2, 99]}}, 99, 8)


def test_nli_does_not_silently_discard_media():
    with pytest.raises(ValueError, match="media"):
        _encode_pair({"query": {"input_ids": [1, 99], "pixel_values": [0]}, "positive": {"input_ids": [2, 99]}}, 99, 8)


def test_raw_native_conversations_use_existing_processor_without_truncation(monkeypatch):
    from relax.utils.data import processor_pool

    class Processor:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is False
            return messages[0]["content"]

        def __call__(self, *, text, padding, truncation, return_tensors):
            assert padding is False and truncation is False and return_tensors == "pt"
            return {"input_ids": torch.tensor([[10 if text == ["query"] else 20, 99]])}

    monkeypatch.setattr(processor_pool, "get_worker_processor", lambda: Processor())
    messages = [
        {"role": role, "content": content}
        for role, content in (("user", "query"), ("assistant", "<emb>"), ("user", "positive"), ("assistant", "<emb>"))
    ]
    pair = _encode_pair({"messages": messages}, 99, 8)
    assert pair["query"]["input_ids"].tolist() == [10, 99]
    assert pair["positive"]["input_ids"].tolist() == [20, 99]
