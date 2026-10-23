# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for shared data processing utilities."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from relax.utils.data import data_utils
from relax.utils.data.data import Dataset
from relax.utils.data.data_utils import build_messages, collect_message_multimodal_data
from relax.utils.types import Sample


@pytest.mark.parametrize(
    "image_url",
    [
        "https://example.test/image.png",
        {"url": "https://example.test/image.png"},
    ],
)
def test_build_messages_normalizes_inline_image_url(image_url):
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ]
    }

    messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys=None,
    )

    assert messages[0]["content"][1] == {
        "type": "image",
        "image": "https://example.test/image.png",
    }
    assert collect_message_multimodal_data(messages)["image"] == ["https://example.test/image.png"]


def test_build_messages_uses_top_level_media_without_mutating_input():
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {}},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ],
        "images": ["/data/image.png"],
    }

    first_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    second_messages = build_messages(
        row,
        prompt_key="messages",
        system_prompt=None,
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )

    assert row["images"] == ["/data/image.png"]
    assert first_messages == second_messages
    assert first_messages[0]["content"][0] == {"type": "image", "image": "/data/image.png"}
    assert collect_message_multimodal_data(first_messages)["image"] == ["/data/image.png"]


@pytest.mark.parametrize("multimodal_keys", [None, {"image": "images"}])
def test_build_messages_accepts_null_tool_body_without_mutating_metadata(multimodal_keys):
    tool_calls = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
        "reasoning_content": "Need a lookup",
        "loss": False,
        "learn": False,
    }
    row = {"messages": [message], "images": []}
    result = build_messages(row, "messages", None, True, multimodal_keys)
    assert result[0]["content"] == ([] if multimodal_keys else "")
    assert {k: v for k, v in result[0].items() if k != "content"} == {
        k: v for k, v in message.items() if k != "content"
    }
    assert message["content"] is None


@pytest.mark.parametrize("content", [None, 123])
def test_build_messages_does_not_hide_invalid_non_tool_content(content):
    row = {"messages": [{"role": "assistant", "content": content}]}
    with pytest.raises(ValueError, match="Unsupported content type"):
        build_messages(row, "messages", None, True, {"image": "images"})


@pytest.fixture
def complete_prompt_first(monkeypatch):
    """Let only the chosen worker finish before result iteration begins."""
    worker = data_utils._check_sample_length_worker

    def configure(first_prompt):
        release = threading.Event()

        def check_length(sample, max_length):
            if sample.prompt != first_prompt:
                assert release.wait(timeout=5)
            return worker(sample, max_length)

        def progress(futures, **kwargs):
            for future in futures:
                release.set()
                yield future

        monkeypatch.setattr(data_utils, "ProcessPoolExecutor", ThreadPoolExecutor)
        monkeypatch.setattr(data_utils, "_check_sample_length_worker", check_length)
        monkeypatch.setattr(data_utils.multiprocessing, "cpu_count", lambda: 3)
        monkeypatch.setattr(data_utils.tqdm, "tqdm", progress)

    return configure


def tokenize_characters(text, **kwargs):
    return {"input_ids": list(range(len(text)))}


@pytest.mark.parametrize("max_length", [1, 20])
def test_filter_long_prompts_preserves_input_order(complete_prompt_first, max_length):
    samples = [Sample(prompt=prompt) for prompt in ["a", "too long", "c"]]
    complete_prompt_first("c")

    filtered = data_utils.filter_long_prompts(samples, tokenize_characters, object(), max_length, num_workers=3)

    assert [sample.prompt for sample in filtered] == [
        sample.prompt for sample in samples if len(sample.prompt) <= max_length
    ]


def test_dataset_same_seed_preserves_batches_across_completion_orders(tmp_path, monkeypatch, complete_prompt_first):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"text": prompt}) + "\n" for prompt in ["a", "b", "c"]))
    monkeypatch.setattr(Dataset, "_process_data", lambda self, row: Sample(prompt=row["text"]))
    datasets = []
    for first_prompt in ["a", "c"]:
        complete_prompt_first(first_prompt)
        datasets.append(Dataset(str(path), tokenize_characters, object(), max_length=1, seed=42))

    for epoch in [0, 1]:
        for dataset in datasets:
            dataset.shuffle(epoch)
        batches = [
            [[sample.prompt for sample in dataset.samples[start : start + 2]] for start in range(0, len(dataset), 2)]
            for dataset in datasets
        ]
        assert batches[0] == batches[1]
