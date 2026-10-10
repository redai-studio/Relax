# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


try:
    from relax.engine.rollout import sglang_rollout

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.mark.parametrize(
    "step,save,interval,rotate,fully_async,expected_fetches",
    [
        (0, "checkpoint", 2, False, False, 1),
        (1, "checkpoint", 2, False, False, 0),
        (2, "checkpoint", 2, False, False, 0),
        (0, "checkpoint", 2, True, False, 0),
        (1, None, 2, False, False, 1),
        (1, "checkpoint", None, False, False, 1),
        (0, "checkpoint", 2, False, True, 0),
    ],
)
def test_rollout_checkpoint_does_not_advance_unpersisted_cursor(
    monkeypatch, step, save, interval, rotate, fully_async, expected_fetches
):
    args = SimpleNamespace(
        rollout_global_dataset=True,
        fully_async=fully_async,
        save=save,
        save_interval=interval,
        rotate_ckpt=rotate,
        num_rollout=3,
        over_sampling_batch_size=8,
    )
    output = object()
    monkeypatch.setattr(sglang_rollout, "generate_rollout_async", lambda *a: (output, []))
    monkeypatch.setattr(sglang_rollout, "run", lambda result: result)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda a: SimpleNamespace())
    source = MagicMock()

    assert sglang_rollout.generate_rollout(args, step, source, None) is output
    assert source.get_samples.remote.call_count == expected_fetches
