# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from relax.engine.sft.data_state import restore_data_state, save_data_state
from relax.engine.sft.dataset.lightning_opd import LightningOPDDataset
from relax.engine.sft.dataset.streaming import pack_samples_for_tq
from relax.utils.training.data_fields import build_data_fields


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]

    def __len__(self):
        return 128


def row(prompt="ab"):
    return {
        "prompt": prompt,
        "metadata": {"response_tokens": [5, 6, 7], "loss_mask": [1, 0, 1], "teacher_log_probs": [-1.0, -0.5, -2.0]},
    }


def test_parquet_to_actor_preserves_response_positions_and_batch_order(tmp_path):
    path = tmp_path / "rollouts.parquet"
    pq.write_table(pa.Table.from_pylist([row("a"), row("ab"), row("abc")]), path, row_group_size=1)
    dataset = LightningOPDDataset(str(path), tokenizer=Tokenizer())
    batch = pack_samples_for_tq(dataset.get_batch_by_indices([2, 0]))
    args = Namespace(loss_type="lightning_opd")
    assert set(batch) == set(build_data_fields(args))
    assert [r.tolist() for r in batch["tokens"]] == [[97, 98, 99, 5, 6, 7], [97, 5, 6, 7]]
    assert batch["total_lengths"] == [6, 4]
    assert batch["response_lengths"] == [3, 3]
    assert [r.tolist() for r in batch["loss_masks"]] == [[1, 0, 1]] * 2
    assert [r.tolist() for r in batch["teacher_log_probs"]] == [[-1, -0.5, -2]] * 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"teacher_log_probs": [-1.0]},
        {"teacher_log_probs": [0.0, float("nan"), -1.0]},
        {"prompt_ids": [97]},
        {"loss_mask": [0, 0, 0]},
        {"response_tokens": [128, 6, 7]},
    ],
)
def test_invalid_cached_trajectory_reports_file_and_row(tmp_path, overrides):
    path = tmp_path / "bad.parquet"
    sample = row()
    sample["metadata"].update(overrides)
    pq.write_table(pa.Table.from_pylist([sample]), path)
    with pytest.raises(ValueError, match=r"bad\.parquet row 0"):
        LightningOPDDataset(str(path), tokenizer=Tokenizer()).get_batch(1)


def test_checkpoint_restores_next_offline_batch_across_epoch(tmp_path):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([row("a"), row("ab"), row("abc")]), path)
    args = Namespace(loss_type="lightning_opd", prompt_data=str(path), load=str(tmp_path), save=str(tmp_path))
    dataset = LightningOPDDataset(str(path), tokenizer=Tokenizer())
    first, _ = dataset.get_batch(2)
    save_data_state(args, dataset, 1, len(dataset), first)
    expected, _ = dataset.get_batch(2)
    resumed = LightningOPDDataset(str(path), tokenizer=Tokenizer())
    restore_data_state(args, resumed, 1, len(resumed))
    actual, crossed = resumed.get_batch(2)
    assert crossed
    assert [s.source_idx for s in actual] == [s.source_idx for s in expected]
