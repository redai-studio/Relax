# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import errno
import io
import os
from pathlib import Path

import pytest
from tensorboard.summary.writer.record_writer import RecordWriter

from relax.utils.metrics.adapters.tensorboard_sync import _EventFileCopy


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
