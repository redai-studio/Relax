# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checkpoint recovery must not skip a batch fetched only into process
memory."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from relax.engine.rollout.data_source import RolloutDataSource
from relax.utils.types import Sample


try:
    from relax.engine.rollout import sglang_rollout
except ModuleNotFoundError as exc:
    if exc.name.split(".")[0] not in {"ray", "sglang", "sglang_router", "pybase64"}:
        raise
    pytest.skip(f"Missing inference dependency: {exc.name}", allow_module_level=True)


@pytest.mark.parametrize("interval,rotate,total", [(3, False, 5), (10, False, 3), (10, True, 5)])
@pytest.mark.parametrize("dataset_size", [10, 100])
def test_resume_reads_next_unconsumed_group(monkeypatch, tmp_path, interval, rotate, total, dataset_size):
    args = Namespace(
        rollout_global_dataset=False,
        save=str(tmp_path),
        load=str(tmp_path),
        save_interval=interval,
        rotate_ckpt=rotate,
        num_rollout=total,
        fully_async=False,
        over_sampling_batch_size=4,
        n_samples_per_prompt=4,
        rollout_shuffle=True,
    )

    def source():
        args.rollout_global_dataset = False
        result = RolloutDataSource(args)
        args.rollout_global_dataset = True
        dataset = MagicMock()
        dataset.__len__.return_value = dataset_size
        original = [Sample(prompt=str(i)) for i in range(dataset_size)]

        def shuffle(epoch):
            offset = epoch % dataset_size
            dataset.samples = original[offset:] + original[:offset]

        dataset.shuffle.side_effect = shuffle
        shuffle(0)
        result.dataset = dataset
        return result

    data = source()
    state = SimpleNamespace(prefetched_samples_ref=None)
    proxy = SimpleNamespace(get_samples=SimpleNamespace(remote=data.get_samples))

    async def generate(args, rollout_id, buffer, client):
        batch = state.prefetched_samples_ref
        state.prefetched_samples_ref = None
        return (batch if batch is not None else buffer.get_samples.remote(4)), []

    monkeypatch.setattr(sglang_rollout, "generate_rollout_async", generate)
    monkeypatch.setattr(sglang_rollout, "run", asyncio.run)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    for step in range(3):
        batch = sglang_rollout.generate_rollout(args, step, proxy, None)
        assert batch[0][0].index == step * 16
    data.save(2)
    expected = data.get_samples(4)
    restored = source()
    restored.load(2)
    actual = restored.get_samples(4)
    assert actual[0][0].index == 48
    assert actual[0][0].group_index == 12
    assert [[(s.index, s.group_index, s.prompt) for s in group] for group in actual] == [
        [(s.index, s.group_index, s.prompt) for s in group] for group in expected
    ]
