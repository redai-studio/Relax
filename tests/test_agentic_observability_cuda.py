# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA snapshot ownership regression for Agentic training observations."""

import pytest
import torch

from relax.utils.agentic_obs import resolve_agentic_obs_config
from relax.utils.agentic_obs.writers import close_observation_writers
from tests.cpu.test_agentic_observability import _read_train_samples, _rollout_data, _train_args
from tests.cpu.test_agentic_observability import fake_mpu as fake_mpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA snapshot validation requires a GPU")
def test_cuda_training_snapshot_owns_input_before_later_mutation(tmp_path, fake_mpu):
    from relax.utils.agentic_obs import train_hooks

    fake_mpu()
    args = _train_args(tmp_path)
    batch = _rollout_data([7])
    for key in ("loss_masks", "advantages", "returns"):
        batch[key] = [value.cuda() for value in batch[key]]
    with train_hooks.observe_train_step(args, 1, 0):
        train_hooks.capture_training_input(batch)
        batch["loss_masks"][0].zero_()
        batch["advantages"][0].fill_(99)
        train_hooks.record_optimizer_result(True)
    train_hooks.close_train_appenders()
    close_observation_writers()
    (row,) = _read_train_samples(resolve_agentic_obs_config(args).train_scalar_path(1, 0))
    assert row["loss_mask"] == [1, 1, 1, 1]
    assert row["advantage_mean"] == pytest.approx(0.5)
