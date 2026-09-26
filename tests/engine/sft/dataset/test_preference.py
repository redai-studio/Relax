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


def _dataset(path: Path, **kwargs) -> PreferenceStreamingDataset:
    return PreferenceStreamingDataset(
        path=str(path),
        tokenizer=_FakeTokenizer(),
        prompt_key="prompt",
        chosen_key="chosen",
        rejected_key="rejected",
        pair_id_key="prompt_id",
        prefetch_max_cached=0,
        **kwargs,
    )


def test_explicit_pair_builds_identical_prompt_and_completion_only_masks(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": "pair-1",
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


def test_implicit_ultrafeedback_pair_extracts_strict_common_prefix(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": "pair-1",
                "chosen": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "good"},
                ],
                "rejected": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "bad"},
                ],
            }
        ],
    )

    dataset = _dataset(path, max_length=32, max_completion_length=8, pair_capacity=64)
    pair = dataset.get_processed_pair(0)

    assert pair.pair_id == "pair-1"
    assert pair.chosen_completion_length == 4
    assert pair.rejected_completion_length == 3


def test_rejection_reason_is_classified_counted_and_still_fail_fast(tmp_path: Path):
    path = tmp_path / "identical.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": "pair-identical",
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": {"role": "assistant", "content": "same"},
                "rejected": {"role": "assistant", "content": "same"},
            }
        ],
    )
    dataset = _dataset(path)
    with pytest.raises(PreferenceDataError) as exc_info:
        dataset.get_processed_pair(0)
    assert exc_info.value.reason_code == "identical"
    assert dataset.rejection_counts == {"identical": 1}
    assert dataset.rejection_records[0]["pair_id"] == "pair-identical"


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"prompt_id": None}, "prompt_id"),
        ({"rejected": {"role": "user", "content": "bad"}}, "assistant"),
        ({"rejected": {"role": "assistant", "content": "good"}}, "identical"),
    ],
)
def test_pair_schema_rejects_invalid_rows(tmp_path: Path, update: dict, match: str):
    row = {
        "prompt_id": "pair-1",
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
    }
    row.update(update)
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match=match):
        _dataset(path, max_length=32, max_completion_length=8, pair_capacity=64).get_processed_pair(0)


def test_preference_dataset_rejects_duplicate_pair_ids(tmp_path: Path):
    row = {
        "prompt_id": "duplicate",
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
    }
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [row, row])

    with pytest.raises(ValueError, match="duplicate preference pair ID"):
        _dataset(path, max_length=32, max_completion_length=8, pair_capacity=64)


def test_shared_prompt_and_completion_truncation_preserves_pair_difference(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": "pair-1",
                "prompt": [{"role": "user", "content": "0123456789"}],
                "chosen": {"role": "assistant", "content": "chosen"},
                "rejected": {"role": "assistant", "content": "reject"},
            }
        ],
    )

    pair = _dataset(path, max_length=8, max_completion_length=3, pair_capacity=16).get_processed_pair(0)

    assert pair.chosen_prompt_length == pair.rejected_prompt_length == 5
    assert pair.chosen_completion_length == pair.rejected_completion_length == 3
    assert pair.chosen_total_length + pair.rejected_total_length == 16
    assert not torch.equal(
        pair.chosen_tokens[pair.chosen_prompt_length :], pair.rejected_tokens[pair.rejected_prompt_length :]
    )


def test_pack_pair_rows_and_custom_meta_are_aligned(tmp_path: Path):
    path = tmp_path / "pairs.jsonl"
    rows = []
    for idx in range(2):
        rows.append(
            {
                "prompt_id": f"pair-{idx}",
                "prompt": [{"role": "user", "content": f"q{idx}"}],
                "chosen": {"role": "assistant", "content": f"yes{idx}"},
                "rejected": {"role": "assistant", "content": f"no{idx}"},
            }
        )
    _write_jsonl(path, rows)
    dataset = _dataset(path, max_length=16, max_completion_length=8, pair_capacity=32)

    batch, custom_meta = pack_preference_pairs_for_tq([dataset.get_processed_pair(0), dataset.get_processed_pair(1)])

    assert len(batch["pair_ids"]) == len(custom_meta) == 2
    for idx, metadata in enumerate(custom_meta):
        assert metadata["total_lengths"] == (batch["chosen_total_lengths"][idx] + batch["rejected_total_lengths"][idx])


def test_preference_split_eval_and_resume_preserve_pair_order(tmp_path: Path):
    from relax.engine.sft.runtime import resolve_sft_split_indices

    path = tmp_path / "split.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt_id": f"pair-{i}",
                "prompt": [{"role": "user", "content": "question"}],
                "chosen": {"role": "assistant", "content": "good"},
                "rejected": {"role": "assistant", "content": "bad"},
            }
            for i in range(10)
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
    _write_jsonl(path, [{"prompt_id": "pair-0"}, {"prompt_id": "pair-1"}])
    with pytest.raises(ValueError):
        _dataset(path).restrict_training_indices(indices)
