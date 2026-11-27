# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import errno
import io
import os
from argparse import Namespace
from pathlib import Path

import pytest
import ray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.summary.writer.record_writer import RecordWriter

from relax.utils import tracking_utils
from relax.utils.metrics.adapters.tensorboard import _TensorboardAdapter
from relax.utils.metrics.adapters.tensorboard_sync import _EventFileCopy
from relax.utils.misc import SingletonMeta


@pytest.fixture
def tensorboard_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    monkeypatch.setattr(SingletonMeta, "_instances", {})
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)
    monkeypatch.setenv("TENSORBOARD_SYNC_INTERVAL_SECONDS", "3600")
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    return Namespace(
        save=tmp_path,
        use_tensorboard=True,
        use_metrics_service=False,
        use_wandb=False,
        use_clearml=False,
        tb_project_name=None,
        tb_experiment_name=None,
        notify_urls=None,
    )


def test_tensorboard_sync_direct_logging_persists_without_final_sync(tensorboard_args: Namespace) -> None:
    tracking_utils.log(tensorboard_args, {"step": 1, "loss": 0.5}, "step")
    adapter = _TensorboardAdapter(tensorboard_args)
    try:
        adapter._writer.flush()
        (event,) = (tensorboard_args.save / "tensorboard_log").glob("events.out.tfevents.*")
        (point,) = EventAccumulator(str(event)).Reload().Scalars("loss")
        assert (point.step, point.value) == (1, 0.5)
    finally:
        adapter.finish()


def test_tensorboard_sync_finishes_if_exit_actor_cannot_start(
    tensorboard_args: Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensorboard_args.use_metrics_service = True
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    def reject_actor_start(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Ray could not start the recovery actor")

    monkeypatch.setattr(ray, "remote", reject_actor_start)
    adapter = _TensorboardAdapter(tensorboard_args)
    try:
        adapter.log({"loss": 0.5}, 1)
        assert not any((tensorboard_args.save / "tensorboard_log").glob("events.out.tfevents.*"))
    finally:
        adapter.finish()
    (event,) = (tensorboard_args.save / "tensorboard_log").glob("events.out.tfevents.*")
    (point,) = EventAccumulator(str(event)).Reload().Scalars("loss")
    assert (point.step, point.value) == (1, 0.5)


def test_tensorboard_sync_resumes_partial_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    destination = tmp_path / "cloud"
    source = staging / "events.out.tfevents.test"
    target = destination / source.name

    buffer = io.BytesIO()
    writer = RecordWriter(buffer)
    writer.write(b"first")
    first_record = buffer.getvalue()
    writer.write(b"second")
    both_records = buffer.getvalue()
    source.write_bytes(both_records[:-1])

    # Simulate storage persisting only a fragment of the attempted append.
    partial_size = len(first_record) // 2

    def fail_sync(fd: int) -> None:
        os.ftruncate(fd, partial_size)
        raise OSError(errno.ENOSPC, "simulated destination full")

    event_copy = _EventFileCopy(staging, destination)
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_sync)
        with pytest.raises(OSError, match="simulated destination full"):
            event_copy.sync_once()
    assert target.read_bytes() == first_record[:partial_size]
    inode = target.stat().st_ino

    assert not event_copy.sync_once()
    assert target.read_bytes() == first_record
    with source.open("ab") as file:
        file.write(both_records[-1:])
    assert event_copy.sync_once()
    assert target.read_bytes() == both_records
    assert target.stat().st_ino == inode
