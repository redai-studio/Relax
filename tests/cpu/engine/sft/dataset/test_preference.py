# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preference-pair schema, rendering, truncation, and queue tests."""

import json
from pathlib import Path

import pytest
import torch

from relax.engine.sft.dataset.preference import (
    PreferenceDataError,
    PreferenceStreamingDataset,
    pack_preference_pairs_for_tq,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


class _FakeTokenizer:
    chat_template = "{% generation %}assistant{% endgeneration %}"

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,  # noqa: ARG002
        tokenize=True,  # noqa: ARG002
        return_tensors=None,  # noqa: ARG002
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,  # noqa: ARG002
    ):
        ids: list[int] = []
        masks: list[int] = []
        for message in messages:
            prefix = {"system": 10, "user": 20, "assistant": 30}[message["role"]]
            content = message["content"]
            encoded = [prefix + (ord(char) % 10) for char in content]
            ids.extend(encoded)
            masks.extend([int(message["role"] == "assistant")] * len(encoded))
        input_ids = torch.tensor([ids], dtype=torch.long)
        if return_assistant_tokens_mask:
            return {"input_ids": input_ids, "assistant_masks": [masks]}
        return input_ids


def _dataset(path: Path | list[Path], **kwargs) -> PreferenceStreamingDataset:
    return PreferenceStreamingDataset(
        path=[str(item) for item in path] if isinstance(path, list) else str(path),
        tokenizer=_FakeTokenizer(),
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        prefetch_max_cached=0,
        **kwargs,
    )


def test_explicit_pair_builds_identical_prompt_and_completion_only_masks(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": {"role": "assistant", "content": "good"},
                "rejected": {"role": "assistant", "content": "bad"},
            }
        ],
    )

    dataset = _dataset(path, max_length=32, max_completion_length=8, pair_capacity=64)
    dataset.shuffle(0)
    pairs, crossed = dataset.get_batch(1)

    assert crossed is False
    assert len(pairs) == 1
    pair = pairs[0]
    chosen_prompt = pair.chosen_tokens[: pair.chosen_prompt_length]
    rejected_prompt = pair.rejected_tokens[: pair.rejected_prompt_length]
    assert torch.equal(chosen_prompt, rejected_prompt)
    assert pair.chosen_loss_mask[: pair.chosen_prompt_length].sum().item() == 0
    assert pair.rejected_loss_mask[: pair.rejected_prompt_length].sum().item() == 0
    assert pair.chosen_loss_mask[pair.chosen_prompt_length :].all()
    assert pair.rejected_loss_mask[pair.rejected_prompt_length :].all()


@pytest.mark.parametrize("implicit_prompt", [False, True])
@pytest.mark.parametrize("history_turns", [1, 2])
def test_preference_pair_masks_historical_assistant_turns(tmp_path: Path, implicit_prompt: bool, history_turns: int):
    prompt = [
        message
        for _ in range(history_turns)
        for message in (
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        )
    ]
    prompt.append({"role": "user", "content": "question"})
    chosen = {"role": "assistant", "content": "good"}
    rejected = {"role": "assistant", "content": "bad"}
    row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if implicit_prompt:
        row = {"chosen": [*prompt, chosen], "rejected": [*prompt, rejected]}
    path = tmp_path / "history.jsonl"
    _write_jsonl(path, [row])

    pair = _dataset(path).get_processed_pair(0)

    assert pair.pair_id == 0
    expected_prompt = _FakeTokenizer().apply_chat_template(prompt).squeeze(0)
    for branch, expected_completion in (("chosen", [33, 31, 31, 30]), ("rejected", [38, 37, 30])):
        tokens = getattr(pair, f"{branch}_tokens")
        mask = getattr(pair, f"{branch}_loss_mask")
        prompt_length = getattr(pair, f"{branch}_prompt_length")
        assert prompt_length == expected_prompt.numel()
        assert torch.equal(tokens[:prompt_length], expected_prompt)
        assert tokens[prompt_length:].tolist() == expected_completion
        assert not mask[:prompt_length].any()
        assert mask[prompt_length:].all()


@pytest.mark.parametrize("prefetch_max_cached", [0, 2])
def test_preference_dataset_preserves_error_context(tmp_path: Path, prefetch_max_cached: int):
    path = tmp_path / "identical.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": {"role": "assistant", "content": "same"},
                "rejected": {"role": "assistant", "content": "same"},
            }
        ],
    )
    dataset = PreferenceStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        prefetch_max_cached=prefetch_max_cached,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
    )
    try:
        dataset.shuffle(0)
        if prefetch_max_cached:
            assert dataset._prefetch.wait_for(0, timeout=5)
        with pytest.raises(PreferenceDataError) as exc_info:
            dataset.get_batch(1)
        assert exc_info.value.reason_code == "identical"
        assert exc_info.value.source_idx == 0
        assert exc_info.value.pair_id == 0
        assert exc_info.value.__cause__ is not None
        assert "identical" in str(exc_info.value.__cause__)
    finally:
        dataset.stop()


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"chosen": None}, "message object"),
        ({"rejected": {"role": "user", "content": "bad"}}, "assistant"),
    ],
)
def test_pair_schema_rejects_invalid_rows(tmp_path: Path, update: dict, match: str):
    row = {
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
    }
    row.update(update)
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match=match):
        _dataset(path, max_length=32, max_completion_length=8, pair_capacity=64).get_processed_pair(0)


@pytest.mark.parametrize("extra", [{}, {"prompt_id": "reused"}, {"prompt_id": 7}, {"prompt_id": None}])
def test_preference_rows_do_not_require_external_pair_ids(tmp_path: Path, extra: dict):
    row = {
        **extra,
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
    }
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [row, row])
    pairs = _dataset(path).get_batch_by_indices([1, 0])
    batch, _ = pack_preference_pairs_for_tq(pairs)

    assert batch["pair_ids"] == [1, 0]
    assert all(pair.chosen_tokens[-4:].tolist() == [33, 31, 31, 30] for pair in pairs)
    assert all(pair.rejected_tokens[-3:].tolist() == [38, 37, 30] for pair in pairs)


def test_preference_initialization_does_not_read_rows_and_errors_do_not_reread(tmp_path: Path, monkeypatch):
    from relax.utils.data.streaming_dataset import StreamingReader

    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [{}, {}, {}])
    reads = []
    original_getitem = StreamingReader.__getitem__

    def getitem(reader, index):
        reads.append(index)
        return original_getitem(reader, index)

    monkeypatch.setattr(StreamingReader, "__getitem__", getitem)
    dataset = _dataset(path)
    assert reads == []
    with pytest.raises(PreferenceDataError) as error:
        dataset.get_processed_pair(2)
    assert reads == [2]
    assert error.value.source_idx == error.value.pair_id == 2


def test_shared_prompt_and_completion_truncation_preserves_pair_difference(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "0123456789"}],
                "chosen": {"role": "assistant", "content": "chosen"},
                "rejected": {"role": "assistant", "content": "reject"},
            }
        ],
    )

    pair = _dataset(path, max_length=8, max_completion_length=3, pair_capacity=8).get_processed_pair(0)

    assert pair.chosen_prompt_length == pair.rejected_prompt_length == 5
    assert pair.chosen_completion_length == pair.rejected_completion_length == 3
    assert pair.chosen_total_length + pair.rejected_total_length == 16
    assert not torch.equal(
        pair.chosen_tokens[pair.chosen_prompt_length :], pair.rejected_tokens[pair.rejected_prompt_length :]
    )


@pytest.fixture
def oversize_pair_path(tmp_path: Path) -> Path:
    path = tmp_path / "oversize.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "abcd"}],
                "chosen": {"role": "assistant", "content": "chosen"},
                "rejected": {"role": "assistant", "content": "badly"},
            }
        ],
    )
    return path


@pytest.mark.parametrize("capacity", [1, 12, 64])
def test_preference_keep_does_not_change_tokens_with_batch_budget(oversize_pair_path: Path, capacity: int):
    expected = _dataset(oversize_pair_path).get_processed_pair(0)
    actual = _dataset(oversize_pair_path, pair_capacity=capacity).get_processed_pair(0)
    assert pack_preference_pairs_for_tq([actual]) == pack_preference_pairs_for_tq([expected])


@pytest.mark.parametrize(
    ("strategy", "chosen_slice", "rejected_slice", "prompt_length"),
    [("truncate_left", slice(-6, None), slice(4, None), 0), ("truncate_right", slice(None, 6), slice(None, 6), 4)],
)
def test_preference_explicit_truncation_preserves_shared_prompt_and_masks(
    oversize_pair_path: Path, strategy: str, chosen_slice: slice, rejected_slice: slice, prompt_length: int
):
    original = _dataset(oversize_pair_path).get_processed_pair(0)
    pair = _dataset(oversize_pair_path, pair_capacity=12, oversize_strategy=strategy).get_processed_pair(0)
    torch.testing.assert_close(pair.chosen_tokens, original.chosen_tokens[chosen_slice])
    torch.testing.assert_close(pair.rejected_tokens, original.rejected_tokens[rejected_slice])
    assert pair.chosen_prompt_length == pair.rejected_prompt_length == prompt_length
    torch.testing.assert_close(pair.chosen_tokens[:prompt_length], pair.rejected_tokens[:prompt_length])
    for mask in (pair.chosen_loss_mask, pair.rejected_loss_mask):
        assert not mask[:prompt_length].any()
        assert mask[prompt_length:].all()
    assert pair.chosen_total_length + pair.rejected_total_length <= 12


@pytest.mark.parametrize("prefetch_max_cached", [0, 2])
def test_preference_skip_refills_training_but_does_not_repeat_eval(oversize_pair_path: Path, prefetch_max_cached: int):
    rows = [json.loads(oversize_pair_path.read_text())]
    rows.append({**rows[0], "prompt": [{"role": "user", "content": ""}]})
    _write_jsonl(oversize_pair_path, rows)
    dataset = PreferenceStreamingDataset(
        path=str(oversize_pair_path),
        tokenizer=_FakeTokenizer(),
        pair_capacity=12,
        oversize_strategy="skip",
        prefetch_max_cached=prefetch_max_cached,
        prefetch_chunk_size=1,
        prefetch_num_workers=1,
    )
    try:
        dataset.shuffle(0)
        if prefetch_max_cached:
            assert dataset._prefetch.wait_for(0, timeout=5)
        pairs, crossed = dataset.get_batch(3)
        assert crossed
        assert [pair.source_idx for pair in pairs] == [1, 1, 1]
        cursor = dataset.index_manager.position
        assert [pair.source_idx for pair in dataset.get_batch_in_order(0, 2)] == [1]
        assert [pair.source_idx for pair in dataset.get_batch_by_indices([1, 0])] == [1]
        assert dataset.index_manager.position == cursor
    finally:
        dataset.stop()


def test_preference_skip_fails_when_no_pair_fits(oversize_pair_path: Path):
    dataset = _dataset(oversize_pair_path, pair_capacity=1, oversize_strategy="skip")
    dataset.shuffle(0)
    with pytest.raises(RuntimeError, match="partial batch: expected 1, got 0"):
        dataset.get_batch(1)
    assert dataset.get_batch_in_order(0, 1) == []


@pytest.mark.parametrize(("rejected", "capacities"), [("badly", [6, 6]), ("n", [7])])
@pytest.mark.parametrize("skip", [False, True])
def test_preference_custom_uses_branch_budget_and_skips_whole_pair(
    oversize_pair_path: Path, rejected: str, capacities: list[int], skip: bool
):
    row = json.loads(oversize_pair_path.read_text())
    row["rejected"]["content"] = rejected
    _write_jsonl(oversize_pair_path, [row])
    calls = []

    def truncate(*, tokens, loss_mask, capacity, idx):
        calls.append((capacity, idx))
        return None if skip else (tokens[:capacity], loss_mask[:capacity])

    pair = _dataset(
        oversize_pair_path, pair_capacity=12, oversize_strategy="custom", oversize_custom_fn=truncate
    ).get_processed_pair(0)
    assert calls == [(capacity, 0) for capacity in (capacities[:1] if skip else capacities)]
    if skip:
        assert pair is None
    else:
        assert pair.chosen_prompt_length == pair.rejected_prompt_length == 4
        assert pair.chosen_total_length + pair.rejected_total_length == 12
        assert pair.chosen_completion_length == capacities[0] - 4


def test_preference_truncation_reports_when_budget_removes_completion(oversize_pair_path: Path):
    with pytest.raises(PreferenceDataError, match="completion has no supervised tokens") as error:
        _dataset(oversize_pair_path, pair_capacity=8, oversize_strategy="truncate_right").get_processed_pair(0)
    assert error.value.pair_id == 0
    assert error.value.source_idx == 0


def test_pack_pair_rows_and_custom_meta_are_aligned(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    rows = []
    for idx in range(2):
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"q{idx}"}],
                "chosen": {"role": "assistant", "content": f"yes{idx}"},
                "rejected": {"role": "assistant", "content": f"no{idx}"},
            }
        )
    _write_jsonl(path, rows)
    dataset = _dataset(path, max_length=16, max_completion_length=8, pair_capacity=32)

    batch, custom_meta = pack_preference_pairs_for_tq([dataset.get_processed_pair(0), dataset.get_processed_pair(1)])

    assert batch["pair_ids"] == [0, 1]
    assert len(custom_meta) == 2
    for idx, metadata in enumerate(custom_meta):
        assert metadata["total_lengths"] == (batch["chosen_total_lengths"][idx] + batch["rejected_total_lengths"][idx])


@pytest.mark.parametrize("multiple_files", [False, True])
def test_cross_epoch_batch_preserves_repeated_pairs_and_resume_order(tmp_path: Path, multiple_files: bool):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": f"q{index}"}],
                "chosen": {"role": "assistant", "content": f"yes{index}"},
                "rejected": {"role": "assistant", "content": f"no{index}"},
            }
            for index in range(10)
        ],
    )
    paths = path
    if multiple_files:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        second_path = tmp_path / "more-pairs.jsonl"
        _write_jsonl(path, rows[:5])
        _write_jsonl(second_path, rows[5:])
        paths = [path, second_path]
    dataset = _dataset(paths, seed=42)
    dataset.shuffle(0)
    dataset.get_batch(8)
    pairs, crossed = dataset.get_batch(4)
    assert crossed
    assert [pair.source_idx for pair in pairs] == [2, 3, 7, 2]
    batch, metadata = pack_preference_pairs_for_tq(pairs)
    assert batch["pair_ids"] == [2, 3, 7, 2]
    assert len(metadata) == 4
    for index, pair in enumerate(pairs):
        assert batch["chosen_tokens"][index] == pair.chosen_tokens.tolist()
        assert batch["rejected_tokens"][index] == pair.rejected_tokens.tolist()

    restored = _dataset(paths, seed=42)
    restored.shuffle(0, position=8)
    resumed, _ = restored.get_batch(4)
    assert [pair.source_idx for pair in resumed] == [pair.source_idx for pair in pairs]
    assert pack_preference_pairs_for_tq(resumed) == (batch, metadata)


def test_preference_split_eval_and_resume_preserve_pair_order(tmp_path: Path):
    from relax.engine.sft.runtime import resolve_sft_split_indices

    path = tmp_path / "split.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": {"role": "assistant", "content": "good"},
                "rejected": {"role": "assistant", "content": "bad"},
            }
            for _ in range(10)
        ],
    )
    train_indices, eval_indices = resolve_sft_split_indices(10, 0.3, seed=42)
    dataset = _dataset(path)
    dataset.restrict_training_indices(train_indices, dataset_seed_offset=1)
    dataset.shuffle(0)
    dataset.get_batch(5)
    cursor = dataset.index_manager.position
    assert [pair.source_idx for pair in dataset.get_batch_by_indices(eval_indices)] == list(eval_indices)
    assert dataset.index_manager.position == cursor
    expected, _ = dataset.get_batch(12)
    assert {pair.source_idx for pair in expected} <= set(train_indices)
    restored = _dataset(path)
    restored.restrict_training_indices(train_indices, dataset_seed_offset=1)
    restored.shuffle(0, position=5)
    actual, _ = restored.get_batch(12)
    assert [pair.pair_id for pair in actual] == [pair.pair_id for pair in expected]
    with pytest.raises(RuntimeError, match="before"):
        restored.restrict_training_indices(train_indices)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [2], [0.5]])
def test_preference_split_rejects_invalid_training_indices(tmp_path: Path, indices):
    path = tmp_path / "split.jsonl"
    _write_jsonl(path, [{}, {}])
    with pytest.raises(ValueError):
        _dataset(path).restrict_training_indices(indices)
