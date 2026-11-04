# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import threading
from types import SimpleNamespace

import pytest

from relax.engine.sft.data_state import restore_data_state, save_data_state
from relax.engine.sft.dataset.streaming import SFTStreamingDataset
from relax.utils.data.streaming_dataset import IndexManager


def dataset():
    # Exercise the production index manager and accepted-sample refill loop.
    result = object.__new__(SFTStreamingDataset)
    result.index_manager = IndexManager(13, seed=42, index_pool=range(10, 23))
    result._prefetch = None
    result._first_error = None
    result._error_lock = threading.Lock()
    result._process_one = lambda idx: None if idx % 3 == 0 else SimpleNamespace(source_idx=idx)
    result.shuffle(0)
    return result


def config(tmp_path):
    return SimpleNamespace(load=str(tmp_path), save=str(tmp_path), seed=42, global_batch_size=4, prompt_data="dataset")


def ids(batch):
    return [sample.source_idx for sample in batch[0]]


@pytest.mark.parametrize("step", [1, 3, 7])
def test_filtered_resume_uses_checkpoint_boundary_not_prefetch_ahead(tmp_path, step):
    args, original = config(tmp_path), dataset()
    for batch in range(step):
        samples, _ = original.get_batch(4)
        save_data_state(args, original, batch + 1, 13, samples)
    expected = ids(original.get_batch(4))
    for future in range(step + 1, step + 4):
        samples, _ = original.get_batch(4)
        save_data_state(args, original, future, 13, samples)
    restored = dataset()
    restore_data_state(args, restored, step, 13)
    assert ids(restored.get_batch(4)) == expected


@pytest.mark.parametrize("step", [0, 1, 3, 7])
def test_legacy_checkpoint_replays_accepted_samples_across_epochs(tmp_path, step):
    args, original = config(tmp_path), dataset()
    for _ in range(step):
        original.get_batch(4)
    expected = ids(original.get_batch(4))
    restored = dataset()
    restored.get_batch(4)  # Restore also resets a reused epoch-zero index manager.
    restore_data_state(args, restored, step, 13)
    assert ids(restored.get_batch(4)) == expected


def test_data_state_rejects_configuration_change_but_allows_longer_run(tmp_path):
    args, original = config(tmp_path), dataset()
    samples, _ = original.get_batch(4)
    save_data_state(args, original, 1, 13, samples)
    args.num_rollout = 200
    restore_data_state(args, dataset(), 1, 13)
    args.seed = 43
    with pytest.raises(ValueError, match="configuration"):
        restore_data_state(args, dataset(), 1, 13)


@pytest.mark.parametrize("key,value", [("position", 14), ("epoch", -1), ("next_step", 9), ("version", 0)])
def test_corrupt_cursor_is_not_silently_replayed(tmp_path, key, value):
    args, original = config(tmp_path), dataset()
    samples, _ = original.get_batch(4)
    save_data_state(args, original, 1, 13, samples)
    path = tmp_path / "sft_data_state/0000001.json"
    state = json.loads(path.read_text())
    state[key] = value
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        restore_data_state(args, dataset(), 1, 13)


@pytest.mark.parametrize("step", [1, 3, 7])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_with_real_prefetch_and_filtered_epoch_boundaries(tmp_path, step, legacy):
    from relax.utils.data.streaming_dataset import PrefetchBuffer

    args, original, restored = config(tmp_path), dataset(), dataset()
    for value in (original, restored):
        value._prefetch = PrefetchBuffer(value._process_one, chunk_size=2, max_cached=4, num_workers=2)
        value.shuffle(0)
    try:
        for batch in range(step):
            samples, _ = original.get_batch(4)
            if not legacy:
                save_data_state(args, original, batch + 1, 13, samples)
        expected = ids(original.get_batch(4))
        restored.get_batch(4)
        restore_data_state(args, restored, step, 13)
        assert ids(restored.get_batch(4)) == expected
    finally:
        original._prefetch.stop()
        restored._prefetch.stop()
