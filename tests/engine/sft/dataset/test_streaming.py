# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the prompt-data based SFT streaming dataset."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from relax.engine.sft.dataset import streaming as streaming_module
from relax.engine.sft.dataset.streaming import (
    ProcessedSample,
    SFTMultimodalContractError,
    SFTStreamingDataset,
    _canonicalize_messages,
    _expand_loss_mask_via_alignment,
    pack_samples_for_tq,
)
from relax.utils.utils import dict_to_tensordict


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_restrict_training_size_excludes_held_out_tail_from_every_epoch(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"messages": []} for _ in range(10)])
    dataset = SFTStreamingDataset(path=str(path), prefetch_max_cached=0, seed=7)

    dataset.restrict_training_size(8)
    dataset.shuffle(0)
    indices, crossed_epoch = dataset.index_manager.get_next_indices(24)

    assert crossed_epoch is True
    assert set(indices) == set(range(8))
    assert all(index < 8 for index in indices)
    assert len(dataset) == 10


def test_restrict_training_indices_shuffles_only_physical_row_ids_and_resumes(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"messages": []} for _ in range(10)])
    train_indices = (0, 2, 5, 7, 9)

    uninterrupted = SFTStreamingDataset(path=str(path), prefetch_max_cached=0, seed=7)
    uninterrupted.restrict_training_indices(train_indices)
    uninterrupted.shuffle(0)
    uninterrupted.index_manager.get_next_indices(7)
    expected, _ = uninterrupted.index_manager.get_next_indices(8)

    resumed = SFTStreamingDataset(path=str(path), prefetch_max_cached=0, seed=7)
    resumed.restrict_training_indices(train_indices)
    resumed.shuffle(1, position=2)
    actual, crossed_epoch = resumed.index_manager.get_next_indices(8)

    assert actual == expected
    assert crossed_epoch is True
    assert set(actual).issubset(train_indices)


def test_restrict_training_indices_prefetches_physical_row_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"messages": []} for _ in range(10)])
    train_indices = (0, 2, 5, 7, 9)
    dataset = SFTStreamingDataset(path=str(path), prefetch_max_cached=0, seed=7)
    dataset._prefetch = MagicMock()

    dataset.restrict_training_indices(train_indices)
    dataset.shuffle(0)

    prefetched_indices = dataset._prefetch.set_index_order.call_args.args[0]
    assert set(prefetched_indices) == set(train_indices)
    assert len(prefetched_indices) == len(train_indices)


@pytest.mark.parametrize("indices", [(), (1, 1), (0, 10)])
def test_restrict_training_indices_rejects_invalid_row_ids(tmp_path, indices):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"messages": []} for _ in range(10)])
    dataset = SFTStreamingDataset(path=str(path), prefetch_max_cached=0)

    with pytest.raises(ValueError):
        dataset.restrict_training_indices(indices)


def test_get_batch_by_indices_preserves_requested_physical_order(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": f"Q{index}"},
                    {"role": "assistant", "content": f"A{index}"},
                ]
            }
            for index in range(6)
        ],
    )
    dataset = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        prompt_key="messages",
        prefetch_max_cached=0,
    )

    samples = dataset.get_batch_by_indices((4, 1, 5))

    assert [sample.source_idx for sample in samples] == [4, 1, 5]


class _FakeTokenizer:
    """Minimal tokenizer with generation-mask chat-template support."""

    chat_template = "{% generation %}assistant{% endgeneration %}"

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,  # noqa: ARG002
        tokenize=True,
        return_tensors=None,  # noqa: ARG002
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,  # noqa: ARG002
    ):
        ids: list[int] = []
        masks: list[int] = []
        rendered_parts: list[str] = []
        next_id = 1
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if part.get("type") == "text")
            else:
                text = content
            rendered_parts.append(text)
            n = max(1, len(text))
            ids.extend(range(next_id, next_id + n))
            masks.extend([1 if message["role"] in ("assistant", "function_call") else 0] * n)
            next_id += n
        if not tokenize:
            return "".join(rendered_parts)
        input_ids = torch.tensor([ids], dtype=torch.long)
        if return_assistant_tokens_mask:
            return {"input_ids": input_ids, "assistant_masks": [masks]}
        return input_ids


def test_streaming_dataset_reads_prompt_data_messages(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, crossed = ds.get_batch(1)

    assert crossed is False
    assert len(samples) == 1
    assert samples[0].loss_mask.sum().item() == len("Answer")
    assert samples[0].source_idx == 0
    ds.stop()


def test_pack_samples_for_tq_marks_samples_as_sft(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, _ = ds.get_batch(1)
    batch = pack_samples_for_tq(samples)

    assert batch is not None
    assert batch["response_lengths"] == batch["total_lengths"]
    assert batch["response_lengths"][0] == batch["tokens"][0].numel()
    assert int(batch["loss_masks"][0].sum().item()) == len("A")
    ds.stop()


def _make_classification_dataset(
    path: Path,
    *,
    problem_type: str = "single_label_classification",
    num_labels: int = 3,
    capacity: int | None = None,
) -> SFTStreamingDataset:
    return SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=capacity,
        prompt_key="messages",
        label_key="label",
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
        oversize_strategy="truncate_right",
        task_type="seq_cls",
        num_labels=num_labels,
        problem_type=problem_type,
        classification_sentinel_token_id=99,
        require_response=False,
    )


def test_streaming_dataset_builds_single_label_classification_sample(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "Question"}], "label": 2}])
    ds = _make_classification_dataset(path)

    sample = ds.get_batch_in_order(0, 1)[0]
    batch = pack_samples_for_tq([sample])

    assert sample.tokens[-1].item() == 99
    assert sample.loss_mask.tolist() == [1]
    assert sample.classification_label.item() == 2
    assert batch["response_lengths"] == [1]
    assert batch["classification_labels"] == [2]
    ds.stop()


def test_streaming_dataset_builds_multimodal_classification_sample(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Classify this image."},
                            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                        ],
                    }
                ],
                "images": [],
                "label": 2,
            }
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=MagicMock(),
        capacity=None,
        prompt_key="messages",
        label_key="label",
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
        task_type="seq_cls",
        num_labels=3,
        problem_type="single_label_classification",
        classification_sentinel_token_id=99,
        require_response=False,
    )
    mm_inputs = {
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    with (
        patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered multimodal prompt"),
        patch(
            "relax.engine.sft.dataset.streaming.preprocess_multimodal",
            return_value=([10, 11, 12], mm_inputs),
        ),
    ):
        sample = ds.get_batch_in_order(0, 1)[0]
    batch = pack_samples_for_tq([sample], force_multimodal_field=True)

    assert sample.tokens.tolist() == [10, 11, 12, 99]
    assert sample.loss_mask.tolist() == [1]
    assert sample.classification_label.item() == 2
    assert sample.multimodal_train_inputs is mm_inputs
    assert batch["classification_labels"] == [2]
    assert batch["multimodal_train_inputs"] == [mm_inputs]
    ds.stop()


def test_streaming_dataset_builds_dense_multi_label_targets(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "Question"}], "label": [0, 2]}])
    ds = _make_classification_dataset(
        path,
        problem_type="multi_label_classification",
        num_labels=4,
    )

    sample = ds.get_batch_in_order(0, 1)[0]
    batch = pack_samples_for_tq([sample], sample_weights=[1.0])

    assert sample.classification_label.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert batch["classification_labels"] == [[1.0, 0.0, 1.0, 0.0]]
    assert batch["sample_weights"] == [1.0]
    ds.stop()


def test_streaming_dataset_truncates_prompt_but_preserves_classification_sentinel(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "long prompt"}], "label": 1}])
    ds = _make_classification_dataset(path, capacity=5)

    sample = ds.get_batch_in_order(0, 1)[0]

    assert sample.total_length == 5
    assert sample.tokens[-1].item() == 99
    assert sample.loss_mask.tolist() == [1]
    ds.stop()


@pytest.mark.parametrize(
    ("label", "problem_type", "match"),
    [
        (True, "single_label_classification", "requires an integer label"),
        (3, "single_label_classification", "expected 0 <= label < 3"),
        ([0, 0], "multi_label_classification", "duplicate indices"),
        ([3], "multi_label_classification", "expected 0 <= index < 3"),
    ],
)
def test_streaming_dataset_rejects_invalid_classification_labels(
    tmp_path: Path,
    label,
    problem_type: str,
    match: str,
):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "Question"}], "label": label}])
    ds = _make_classification_dataset(path, problem_type=problem_type)

    with pytest.raises((TypeError, ValueError), match=match):
        ds.get_batch_in_order(0, 1)
    ds.stop()


class _CountingReader:
    """Reader wrapper that counts ``__getitem__`` calls without changing
    behaviour."""

    def __init__(self, inner):
        self._inner = inner
        self.count = 0

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        self.count += 1
        return self._inner[idx]


def test_render_one_reads_reader_once_for_seq_cls(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "Question"}], "label": 1}])
    ds = _make_classification_dataset(path)
    counting = _CountingReader(ds.reader)
    ds.reader = counting

    rendered = ds._render_one(0)

    assert rendered is not None
    assert rendered.classification_label.item() == 1
    assert counting.count == 1
    ds.stop()


def test_render_one_reads_reader_once_for_causal_lm(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )
    counting = _CountingReader(ds.reader)
    ds.reader = counting

    rendered = ds._render_one(0)

    assert rendered is not None
    assert rendered.classification_label is None
    assert counting.count == 1
    ds.stop()


def _make_text_only_sample() -> ProcessedSample:
    """A ProcessedSample with no multimodal inputs (text-only)."""
    return ProcessedSample(
        tokens=torch.tensor([1, 2, 3], dtype=torch.long),
        loss_mask=torch.tensor([0, 1, 1], dtype=torch.long),
        total_length=3,
        multimodal_train_inputs=None,
        source_idx=0,
    )


def test_pack_samples_for_tq_omits_multimodal_field_for_text_only_batch():
    # Default behaviour: an all-text batch carries no multimodal key.
    sample = _make_text_only_sample()
    batch = pack_samples_for_tq([sample])

    assert batch is not None
    assert batch["tokens"][0] is sample.tokens
    assert batch["loss_masks"][0] is sample.loss_mask
    assert "multimodal_train_inputs" not in batch


def test_dict_to_tensordict_preserves_tensor_list_as_nested_tensor():
    batch = {
        "tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "loss_masks": [torch.tensor([0, 1, 1]), torch.tensor([1, 1])],
        "total_lengths": [3, 2],
        "response_lengths": [3, 2],
    }

    td = dict_to_tensordict(batch, batch_size=2)

    assert td["tokens"].is_nested
    assert td["loss_masks"].is_nested
    assert [tensor.tolist() for tensor in td["tokens"]] == [[1, 2, 3], [4, 5]]
    assert [tensor.tolist() for tensor in td["loss_masks"]] == [[0, 1, 1], [1, 1]]
    assert td["total_lengths"].tolist() == [3, 2]


def test_pack_samples_for_tq_forces_multimodal_field_for_text_only_batch():
    # A VL run (multimodal_keys configured) must always emit the field so the
    # consumer's fixed TQ field list stays satisfied even for text-only batches.
    batch = pack_samples_for_tq([_make_text_only_sample()], force_multimodal_field=True)

    assert batch is not None
    assert "multimodal_train_inputs" in batch
    assert batch["multimodal_train_inputs"] == [None]


def test_streaming_dataset_builds_messages_from_prompt_and_label_keys(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"prompt": "What is 2+2?", "answer": "4"}])

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="prompt",
        label_key="answer",
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    ds.shuffle(0)
    samples, _ = ds.get_batch(1)

    assert len(samples) == 1
    assert samples[0].loss_mask.sum().item() == len("4")
    ds.stop()


def test_streaming_dataset_rejects_messages_when_label_key_is_set(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
                "answer": "extra",
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key="answer",
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(TypeError, match="--label-key is set"):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_rejects_prompt_string_without_label_key(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"prompt": "No label"}])

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="prompt",
        label_key=None,
        multimodal_keys=None,
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(TypeError, match="--label-key is not set"):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_collects_inline_structured_image_url(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "_sample_id": "inline-image-url",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == ["https://example.test/image.png"]
    assert sample.messages[0].content[1] == {
        "type": "image",
        "image": "https://example.test/image.png",
    }
    ds.stop()


@pytest.mark.parametrize(
    ("messages", "expected_images"),
    [
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/inline.png"}},
                    ],
                },
                {"role": "user", "content": "<image>\nDescribe both images."},
                {"role": "assistant", "content": "Answer"},
            ],
            ["https://example.test/inline.png", "/data/top-level.png"],
        ),
        (
            [
                {"role": "user", "content": "<image>\nFirst image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/inline.png"}},
                    ],
                },
                {"role": "assistant", "content": "Answer"},
            ],
            ["/data/top-level.png", "https://example.test/inline.png"],
        ),
    ],
)
def test_streaming_dataset_preserves_mixed_inline_and_top_level_image_order(
    tmp_path: Path,
    messages: list[dict],
    expected_images: list[str],
):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"messages": messages, "images": ["/data/top-level.png"]}])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == expected_images
    ds.stop()


def test_streaming_dataset_preserves_repeated_inline_image_url(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    image_url = "https://example.test/repeated.png"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [],
            }
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == [image_url, image_url]
    ds.stop()


@pytest.mark.parametrize("image_url", [{"url": 123}, {"url": ""}, ""])
def test_streaming_dataset_rejects_malformed_inline_image_url(tmp_path: Path, image_url):
    path = tmp_path / "train.jsonl"
    rows = _invalid_image_url_rows()
    rows[0]["messages"][0]["content"][1]["image_url"] = image_url
    _write_jsonl(path, rows[:1])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(SFTMultimodalContractError, match="top_level_required_count"):
        ds.get_canonical_sample(0)
    ds.stop()


def _invalid_image_url_rows() -> list[dict]:
    return [
        {
            "_sample_id": "invalid-image-url",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {}},
                    ],
                },
                {"role": "assistant", "content": "Bad sample"},
            ],
            "images": [],
        },
        {
            "_sample_id": "text-only",
            "messages": [
                {"role": "user", "content": "Text question"},
                {"role": "assistant", "content": "Good sample"},
            ],
            "images": [],
        },
    ]


def _media_fetch_rows() -> list[dict]:
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image."},
                        {"type": "image_url", "image_url": {"url": f"https://example.test/{idx}.png"}},
                    ],
                },
                {"role": "assistant", "content": f"Answer {idx}"},
            ],
            "images": [],
        }
        for idx in range(2)
    ]


def test_streaming_dataset_invalid_multimodal_defaults_to_error(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows()[:1])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)

    try:
        with pytest.raises(SFTMultimodalContractError, match="sample idx=0, sample_id='invalid-image-url'"):
            ds.get_batch(1)
    finally:
        ds.stop()


def test_streaming_dataset_media_fetch_error_defaults_to_error(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _media_fetch_rows()[:1])
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=MagicMock(),
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)

    try:
        with (
            patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered prompt"),
            patch(
                "relax.engine.sft.dataset.streaming.preprocess_multimodal",
                side_effect=RuntimeError("429 Too Many Requests"),
            ),
            pytest.raises(RuntimeError, match="429 Too Many Requests"),
        ):
            ds.get_batch(1)
    finally:
        ds.stop()


@pytest.mark.parametrize("batch_mode", ["inline", "async", "prefetch"])
def test_streaming_dataset_media_fetch_error_skip_refills_batch(tmp_path: Path, caplog, batch_mode: str):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _media_fetch_rows())
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=MagicMock(),
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=4 if batch_mode == "prefetch" else 0,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
        invalid_multimodal_strategy="skip",
    )

    def _preprocess(sample, **_kwargs):
        if sample.metadata["row_index"] == 0:
            raise RuntimeError("429 Too Many Requests")
        return None, None

    async def _preprocess_async(sample, **_kwargs):
        return _preprocess(sample)

    with (
        patch("relax.engine.sft.dataset.streaming.render_to_text", return_value="rendered prompt"),
        patch("relax.engine.sft.dataset.streaming.preprocess_multimodal", side_effect=_preprocess),
        patch("relax.engine.sft.dataset.streaming.preprocess_multimodal_async", side_effect=_preprocess_async),
    ):
        ds.shuffle(0)
        try:
            if batch_mode == "async":
                samples, _ = asyncio.run(ds.get_batch_async(1))
            else:
                samples, _ = ds.get_batch(1)
        finally:
            ds.stop()

    assert [sample.source_idx for sample in samples] == [1]
    assert "SFTStreamingDataset[invalid-multimodal=skip]" in caplog.text
    assert "sample idx=0" in caplog.text
    assert "429 Too Many Requests" in caplog.text


@pytest.mark.parametrize("batch_mode", ["inline", "async", "prefetch"])
def test_streaming_dataset_invalid_multimodal_skip_refills_batch(tmp_path: Path, caplog, batch_mode: str):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows())
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=4 if batch_mode == "prefetch" else 0,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
        invalid_multimodal_strategy="skip",
    )
    ds.shuffle(0)

    try:
        if batch_mode == "async":
            samples, _ = asyncio.run(ds.get_batch_async(1))
        else:
            samples, _ = ds.get_batch(1)
        assert [sample.source_idx for sample in samples] == [1]
        assert "SFTStreamingDataset[invalid-multimodal=skip]" in caplog.text
        assert "sample idx=0, sample_id='invalid-image-url'" in caplog.text
    finally:
        ds.stop()


@pytest.mark.parametrize("batch_mode", ["inline", "async", "prefetch"])
def test_streaming_dataset_missing_media_skip_refills_batch(tmp_path: Path, monkeypatch, batch_mode: str):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>\nDescribe the image."},
                    {"role": "assistant", "content": "Bad sample"},
                ],
                "images": [str(tmp_path / "missing.png")],
            },
            {
                "messages": [
                    {"role": "user", "content": "Text question"},
                    {"role": "assistant", "content": "Good sample"},
                ],
                "images": [],
            },
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=object(),
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=0,
        prefetch_max_cached=4 if batch_mode == "prefetch" else 0,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
        invalid_multimodal_strategy="skip",
    )
    ds.shuffle(0)
    warning_messages: list[str] = []
    original_warning = streaming_module.logger.warning

    def _capture_warning(message, *args, **kwargs):
        warning_messages.append(message % args)
        return original_warning(message, *args, **kwargs)

    monkeypatch.setattr(streaming_module.logger, "warning", _capture_warning)

    try:
        if batch_mode == "async":
            samples, _ = asyncio.run(ds.get_batch_async(1))
        else:
            samples, _ = ds.get_batch(1)
        assert [sample.source_idx for sample in samples] == [1]
        captured_warnings = "\n".join(warning_messages)
        assert "SFTStreamingDataset[invalid-multimodal=skip]" in captured_warnings
        assert "image position=0" in captured_warnings
    finally:
        ds.stop()


def test_streaming_dataset_async_prefetch_waits_without_foreground_fallback(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "assistant", "content": "A"}]},
            {"messages": [{"role": "assistant", "content": "B"}]},
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)
    ds.index_manager.position = 0
    first_idx = ds.index_manager.indices[0]

    class _FakePrefetch:
        cache_size = 1
        is_alive = True

        def __init__(self) -> None:
            self.get_called = False
            self.get_cached_calls = 0

        def get(self, idx: int):  # noqa: ARG002
            self.get_called = True
            raise AssertionError("async prefetch path must not use synchronous fallback")

        def get_cached(self, idx: int, *, record_miss: bool = True):  # noqa: ARG002
            self.get_cached_calls += 1
            if self.get_cached_calls == 1:
                return False, None
            return True, ProcessedSample(
                tokens=torch.tensor([1], dtype=torch.long),
                loss_mask=torch.tensor([1], dtype=torch.long),
                total_length=1,
                multimodal_train_inputs=None,
                source_idx=idx,
            )

        def set_index_order(self, indices: list[int]) -> None:  # noqa: ARG002
            raise AssertionError("test should not re-prime prefetch")

        def stop(self) -> None:
            pass

    prefetch = _FakePrefetch()
    ds._prefetch = prefetch

    try:
        samples, crossed = asyncio.run(ds.get_batch_async(1))
        assert crossed is False
        assert [item.source_idx for item in samples] == [first_idx]
        assert prefetch.get_called is False
        assert prefetch.get_cached_calls == 2
    finally:
        ds.stop()


def test_streaming_dataset_async_prefetch_rechecks_cache_after_worker_exit(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "assistant", "content": "A"}]},
            {"messages": [{"role": "assistant", "content": "B"}]},
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)
    ds.index_manager.position = 0
    first_idx = ds.index_manager.indices[0]

    class _ExitedPrefetch:
        cache_size = 1
        is_alive = False

        def __init__(self) -> None:
            self.get_cached_calls = 0

        def get_cached(self, idx: int, *, record_miss: bool = True):  # noqa: ARG002
            self.get_cached_calls += 1
            if self.get_cached_calls == 1:
                return False, None
            return True, ProcessedSample(
                tokens=torch.tensor([1], dtype=torch.long),
                loss_mask=torch.tensor([1], dtype=torch.long),
                total_length=1,
                multimodal_train_inputs=None,
                source_idx=idx,
            )

        def stop(self) -> None:
            pass

    prefetch = _ExitedPrefetch()
    ds._prefetch = prefetch

    try:
        samples, crossed = asyncio.run(ds.get_batch_async(1))
        assert crossed is False
        assert [item.source_idx for item in samples] == [first_idx]
        assert prefetch.get_cached_calls == 2
    finally:
        ds.stop()


def test_streaming_dataset_async_prefetch_reprimes_current_epoch_boundary_index(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "assistant", "content": "A"}]},
            {"messages": [{"role": "assistant", "content": "B"}]},
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)
    ds.index_manager.position = len(ds.index_manager.indices)

    class _FakePrefetch:
        cache_size = 0
        is_alive = True

        def __init__(self) -> None:
            self.index_orders: list[list[int]] = []

        def set_index_order(self, indices: list[int]) -> None:
            self.index_orders.append(list(indices))

        def get_cached(self, idx: int, *, record_miss: bool = True):  # noqa: ARG002
            return True, ProcessedSample(
                tokens=torch.tensor([1], dtype=torch.long),
                loss_mask=torch.tensor([1], dtype=torch.long),
                total_length=1,
                multimodal_train_inputs=None,
                source_idx=idx,
            )

        def stop(self) -> None:
            pass

    prefetch = _FakePrefetch()
    ds._prefetch = prefetch

    try:
        samples, crossed = asyncio.run(ds.get_batch_async(3))
        assert crossed is True
        assert len(prefetch.index_orders) == 2
        assert [order[0] for order in prefetch.index_orders] == [samples[0].source_idx, samples[2].source_idx]
    finally:
        ds.stop()


def test_streaming_dataset_async_prefetch_raises_when_worker_exits(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "assistant", "content": "A"}]},
            {"messages": [{"role": "assistant", "content": "B"}]},
        ],
    )
    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        seed=0,
        prefetch_max_cached=0,
    )
    ds.shuffle(0)
    ds.index_manager.position = 0

    class _DeadPrefetch:
        cache_size = 0
        is_alive = False

        def get_cached(self, idx: int, *, record_miss: bool = True):  # noqa: ARG002
            return False, None

        def wait_for(self, idx: int, timeout: float | None = None):  # noqa: ARG002
            return False

        def stop(self) -> None:
            pass

    ds._prefetch = _DeadPrefetch()

    try:
        with pytest.raises(RuntimeError, match="prefetch worker exited"):
            asyncio.run(ds.get_batch_async(1))
    finally:
        ds.stop()


def test_streaming_dataset_rejects_unknown_invalid_multimodal_strategy(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, _invalid_image_url_rows())

    with pytest.raises(ValueError, match="invalid_multimodal_strategy must be one of"):
        SFTStreamingDataset(
            path=str(path),
            tokenizer=_FakeTokenizer(),
            processor_pool=None,
            capacity=None,
            prompt_key="messages",
            label_key=None,
            multimodal_keys={"image": "images"},
            prefetch_max_cached=0,
            invalid_multimodal_strategy="ignore",
        )


def test_streaming_dataset_rejects_duplicate_inline_and_top_level_image_sources(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                        ],
                    },
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": ["/data/image.png"],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    with pytest.raises(
        SFTMultimodalContractError,
        match=r"'inline_count': 1, 'top_level_required_count': 0.*'top_level_count': 1",
    ):
        ds.get_canonical_sample(0)
    ds.stop()


def test_streaming_dataset_accepts_literal_image_placeholder_with_matching_top_level_image(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "messages": [
                    {"role": "user", "content": "<image>\nDescribe the image."},
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": ["/data/image.png"],
            }
        ],
    )

    ds = SFTStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        processor_pool=None,
        capacity=None,
        prompt_key="messages",
        label_key=None,
        multimodal_keys={"image": "images"},
        seed=42,
        prefetch_max_cached=0,
    )

    sample = ds.get_canonical_sample(0)

    assert sample.images == ["/data/image.png"]
    assert sample.messages[0].content[0] == {"type": "image", "image": "/data/image.png"}
    ds.stop()


def test_streaming_dataset_prefetch_and_inline_yield_same_samples(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(8)
        ],
    )

    def _collect(prefetch: int) -> list[int]:
        ds = SFTStreamingDataset(
            path=str(path),
            tokenizer=_FakeTokenizer(),
            processor_pool=None,
            capacity=None,
            prompt_key="messages",
            label_key=None,
            multimodal_keys=None,
            seed=42,
            prefetch_max_cached=prefetch,
        )
        ds.shuffle(0)
        seen: list[int] = []
        for _ in range(2):
            batch, _ = ds.get_batch(4)
            seen.extend(s.source_idx for s in batch)
        ds.stop()
        return seen

    inline_order = _collect(0)
    prefetch_order = _collect(64)
    assert inline_order == prefetch_order
    assert sorted(inline_order) == list(range(8))


# ----------------------------------------------------------------------
# _expand_loss_mask_via_alignment
# ----------------------------------------------------------------------


def test_expand_loss_mask_alignment_text_only_is_identity():
    short = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    mask = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=mask, expanded_ids=short, pad_token_ids=frozenset({99})
    )
    assert out.tolist() == [0, 1, 1, 0]


def test_expand_loss_mask_alignment_single_image_pad_run():
    pad = 99
    short = torch.tensor([1, pad, 5], dtype=torch.long)
    short_mask = torch.tensor([0, 0, 1], dtype=torch.long)
    expanded = torch.tensor([1, pad, pad, pad, pad, pad, 5], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=short_mask, expanded_ids=expanded, pad_token_ids=frozenset({pad})
    )
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 1]


def test_expand_loss_mask_alignment_multiple_pad_kinds():
    img, aud = 99, 88
    short = torch.tensor([1, img, 2, aud, 3], dtype=torch.long)
    short_mask = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
    expanded = torch.tensor([1, img, img, img, 2, aud, aud, 3], dtype=torch.long)
    out = _expand_loss_mask_via_alignment(
        short_ids=short, short_mask=short_mask, expanded_ids=expanded, pad_token_ids=frozenset({img, aud})
    )
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 0, 1]


def test_expand_loss_mask_alignment_disagreement_raises():
    short = torch.tensor([1, 2, 3], dtype=torch.long)
    expanded = torch.tensor([1, 999, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="alignment failed"):
        _expand_loss_mask_via_alignment(
            short_ids=short,
            short_mask=torch.zeros_like(short),
            expanded_ids=expanded,
            pad_token_ids=frozenset(),
        )


def test_expand_loss_mask_alignment_trailing_mismatch_raises():
    short = torch.tensor([1, 2], dtype=torch.long)
    expanded = torch.tensor([1, 2, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="alignment ended early"):
        _expand_loss_mask_via_alignment(
            short_ids=short,
            short_mask=torch.zeros_like(short),
            expanded_ids=expanded,
            pad_token_ids=frozenset(),
        )


def test_canonicalize_messages_extracts_tool_calls():
    """Raw OpenAI-style assistant messages may carry a ``tool_calls`` field;
    ``_canonicalize_messages`` must propagate it onto ``CanonicalMessage`` so
    the chat template can render the tool calls.

    Messages without the field stay None.
    """
    tool_call = {"type": "function", "function": {"name": "f", "arguments": {"x": 1}}}
    raw = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "content": "ok"},
    ]
    msgs = _canonicalize_messages(raw, require_response=True)
    assert msgs[0].tool_calls is None
    assert msgs[1].tool_calls == [tool_call]
    assert msgs[2].tool_calls is None
