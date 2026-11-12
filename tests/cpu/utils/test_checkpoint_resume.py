# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from relax.utils.utils import checkpoint_resume_step, recovery_load_path


@pytest.mark.parametrize("tracker,override,iteration", [(0, None, 0), (199, None, 199), (199, 0, 199), (199, 99, 99)])
def test_recovery_configures_committed_iteration_and_load_flags(tmp_path, tracker, override, iteration):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(tracker))
    (tmp_path / f"iter_{iteration:07d}").mkdir()
    (tmp_path / "iter_0009999").mkdir()  # An uncommitted future directory is not a recovery point.
    args = Namespace(save=str(tmp_path), no_save_optim=False, no_save_rng=False, ckpt_step=override)

    assert checkpoint_resume_step(str(tmp_path), override) == iteration + 1
    args.start_rollout_id = iteration + 1
    recovery_load_path(args)

    assert args.start_rollout_id is None
    assert args.load == str(tmp_path)
    assert not args.no_load_optim and not args.no_load_rng and not args.finetune


def test_recovery_uses_checkpoint_committed_after_restart_configuration(tmp_path):
    from relax.components.actor import _resolve_start_rollout_id

    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("19")
    (tmp_path / "iter_0000019").mkdir()
    args = Namespace(save=str(tmp_path), no_save_optim=False, no_save_rng=False, start_rollout_id=20)
    recovery_load_path(args)

    (tmp_path / "iter_0000023").mkdir()
    tracker.write_text("23")
    loaded_step = checkpoint_resume_step(args.load)
    assert _resolve_start_rollout_id(args.start_rollout_id, loaded_step, explicitly_set=True) == 24


@pytest.mark.parametrize("tracker", ["release", "-1", "incomplete", ""])
def test_resume_rejects_non_resumable_tracker(tmp_path, tracker):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker)
    with pytest.raises(ValueError, match="numeric checkpoint tracker"):
        checkpoint_resume_step(str(tmp_path), 99)


def test_resume_rejects_missing_committed_directory(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("199")
    (tmp_path / "iter_0000399").mkdir()
    with pytest.raises(ValueError, match="Committed checkpoint directory is missing"):
        checkpoint_resume_step(str(tmp_path))


def test_fsdp_recovery_keeps_backend_specific_checkpoint_layout(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("199")
    (tmp_path / "image_sft" / "iter_0000199").mkdir(parents=True)
    args = Namespace(save=str(tmp_path), no_save_optim=False, no_save_rng=False, train_backend="fsdp")

    recovery_load_path(args)

    assert args.start_rollout_id is None
    assert args.load == str(tmp_path)
