# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections import Counter

import pytest
import torch

from examples.osworld_agentic.fixed_task_data_source import FixedTaskDataSource
from relax.engine.rollout import data_source as source_module
from relax.utils.types import Sample


class _Dataset:
    def __init__(self, task_ids):
        self.samples = [
            Sample(prompt=f"Prompt {i}", tokens=[100, 200], metadata={"task_id": task_id})
            for i, task_id in enumerate(task_ids)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def shuffle(self, _epoch):
        raise AssertionError("Fixed tasks must never shuffle.")

    def load_state_dict(self, _state):
        raise AssertionError("An eager fixed dataset must not restore a streaming cursor.")


@pytest.fixture
def make_source(monkeypatch):
    # Real parent init/get_samples/load; only model loading and dataset construction are stubbed.
    monkeypatch.setattr(source_module, "prepare_model_maybe_update_args", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(source_module, "load_tokenizer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(source_module, "load_processor", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(source_module.MultimodalConfig, "from_args", lambda _args: None)

    def build(task_ids=None, **overrides):
        dataset = _Dataset(task_ids if task_ids is not None else [f"task-{i:02d}" for i in range(74)])
        monkeypatch.setattr(source_module, "_create_dataset", lambda *_args, **_kwargs: dataset)
        values = dict(
            rollout_global_dataset=True,
            use_streaming_dataset=False,
            rollout_shuffle=False,
            rollout_batch_size=74,
            n_samples_per_prompt=8,
            buffer_filter_path=None,
            hf_checkpoint="unused-model",
            dump_details=None,
            load=None,
            save=None,
        )
        values.update(overrides)
        return FixedTaskDataSource(Namespace(**values))

    return build


@pytest.mark.parametrize("chunks", [(74,), (1, 73), (11, 17, 3, 43), (8, 8, 8, 8, 8, 8, 8, 8, 8, 2)])
def test_multiple_steps_cover_all_74_tasks_with_real_parent_clones(make_source, chunks):
    source = make_source()
    expected_tasks = [f"task-{i:02d}" for i in range(74)]
    all_indices = set()
    for step in range(5):
        assert source.begin_rollout_step(step) == tuple(expected_tasks)
        groups = [group for chunk in chunks for group in source.get_samples(chunk)]
        assert [group[0].metadata["task_id"] for group in groups] == expected_tasks
        assert [group[0].group_index for group in groups] == list(range(step * 74, (step + 1) * 74))
        samples = [sample for group in groups for sample in group]
        assert len(samples) == 592
        assert all(len(group) == 8 for group in groups)
        assert Counter(sample.metadata["task_id"] for sample in samples) == dict.fromkeys(expected_tasks, 8)
        indices = {sample.index for sample in samples}
        assert indices == set(range(step * 592, (step + 1) * 592))
        assert not all_indices.intersection(indices)
        all_indices.update(indices)
        assert all(len({sample.group_index for sample in group}) == 1 for group in groups)
        assert len({id(sample) for sample in samples}) == 592
        groups[0][0].tokens.append(999)
        groups[0][0].metadata["test_mutation"] = True
        assert groups[0][1].tokens == [100, 200]
        assert "test_mutation" not in groups[0][1].metadata
        assert source.dataset.samples[0].tokens == [100, 200]
        assert "test_mutation" not in source.dataset.samples[0].metadata
        with pytest.raises(RuntimeError, match="refusing to substitute"):
            source.get_samples(1)
        source.finish_rollout_step(step)
    assert source.sample_group_index == 370
    assert source.sample_index == 2960


def test_load_old_streaming_cursor_then_restart_task_order_preserves_counters(make_source, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    torch.save(
        dict(
            sample_offset=45,
            epoch_id=9,
            sample_group_index=789,
            sample_index=6312,
            metadata={"checkpoint_marker": "preserved"},
            streaming_state={"sample_offset": 45},
        ),
        dataset_dir / "global_dataset_state_dict_4.pt",
    )
    source = make_source(load=str(tmp_path))
    source.load(4)
    assert source.sample_offset == 45
    assert source.metadata == {"checkpoint_marker": "preserved"}
    source.begin_rollout_step(5)
    assert source.sample_offset == 0
    assert (source.sample_group_index, source.sample_index) == (789, 6312)
    groups = source.get_samples(74)
    assert [group[0].metadata["task_id"] for group in groups] == list(source.task_ids)
    assert [group[0].group_index for group in groups] == list(range(789, 863))
    assert [sample.index for group in groups for sample in group] == list(range(6312, 6904))
    source.finish_rollout_step(5)


def test_requires_begin_and_complete_matching_finish(make_source):
    source = make_source()
    with pytest.raises(RuntimeError, match="begin_rollout_step"):
        source.get_samples(1)
    source.begin_rollout_step(0)
    with pytest.raises(RuntimeError, match="has not finished"):
        source.begin_rollout_step(1)
    with pytest.raises(ValueError, match="nonnegative"):
        source.get_samples(-1)
    assert source.get_samples(0) == []
    source.get_samples(10)
    with pytest.raises(RuntimeError, match="Incomplete"):
        source.finish_rollout_step(0)
    source.get_samples(64)
    with pytest.raises(RuntimeError, match="Incomplete"):
        source.finish_rollout_step(1)
    source.finish_rollout_step(0)
    source.begin_rollout_step(1)
    assert source.get_samples(1)[0][0].metadata["task_id"] == "task-00"


def test_oversized_request_is_atomic_and_remaining_can_still_be_served(make_source):
    source = make_source()
    source.begin_rollout_step(7)
    source.get_samples(70)
    counters = (source.sample_offset, source.sample_group_index, source.sample_index)
    with pytest.raises(RuntimeError, match="only 4 unissued"):
        source.get_samples(5)
    assert (source.sample_offset, source.sample_group_index, source.sample_index) == counters
    assert [group[0].metadata["task_id"] for group in source.get_samples(4)] == [
        "task-70",
        "task-71",
        "task-72",
        "task-73",
    ]
    source.finish_rollout_step(7)


def test_rejects_buffered_replacement_groups(make_source):
    source = make_source()
    source.add_samples([])
    with pytest.raises(RuntimeError, match="buffered or partial"):
        source.add_samples([[source.dataset.samples[0]]])
    source.buffer.append([source.dataset.samples[0]])
    with pytest.raises(RuntimeError, match="buffered groups"):
        source.begin_rollout_step(0)


@pytest.mark.parametrize("size", [0, 73, 75])
def test_rejects_dataset_size_different_from_fixed_batch(make_source, size):
    with pytest.raises(ValueError, match="dataset size"):
        make_source([f"task-{i}" for i in range(size)])


@pytest.mark.parametrize("invalid_id", [None, "", 0, 123, False])
def test_rejects_missing_or_invalid_task_ids(make_source, invalid_id):
    task_ids = [f"task-{i}" for i in range(74)]
    task_ids[20] = invalid_id
    with pytest.raises(ValueError, match="nonempty metadata.task_id"):
        make_source(task_ids)


def test_rejects_duplicate_task_ids(make_source):
    task_ids = [f"task-{i}" for i in range(74)]
    task_ids[73] = task_ids[0]
    with pytest.raises(ValueError, match="duplicate task_id"):
        make_source(task_ids)


@pytest.mark.parametrize(
    ("name", "value"),
    [("rollout_global_dataset", False), ("use_streaming_dataset", True), ("rollout_shuffle", True)],
)
def test_requires_eager_unshuffled_global_dataset(make_source, name, value):
    with pytest.raises(ValueError, match="eager, unshuffled global"):
        make_source(**{name: value})
