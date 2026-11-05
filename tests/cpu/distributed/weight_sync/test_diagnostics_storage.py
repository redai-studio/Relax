# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Diagnostic I/O keeps a bounded wait and preserves functional errors."""

import threading

import pytest

from relax.distributed.weight_sync import diagnostics, store, traffic


def test_optional_workers_remain_bounded_when_storage_hangs():
    release = threading.Event()
    try:
        for _ in range(2):
            result = diagnostics.observe(lambda: release.wait(5), "stalled read", timeout=0.01)
            assert "pending" in result.error
        result = diagnostics.observe(lambda: 7, "another read", timeout=0.01)
        assert "busy" in result.error
    finally:
        release.set()
        for worker in threading.enumerate():
            if worker.name == "weight-sync-observation":
                worker.join(2)
    assert diagnostics.observe(lambda: 7, "recovered").value == 7


def test_malformed_io_observation_preserves_function_and_error(monkeypatch):
    monkeypatch.setattr(traffic, "process_io", lambda: {"read_bytes": 10})

    def malformed(path):
        raise IndexError("invalid mount record")

    monkeypatch.setattr(traffic, "filesystem_info", malformed)
    with pytest.raises(ValueError, match="real failure"):
        with traffic.IOObservation(path=".") as observation:
            raise ValueError("real failure")
    assert "invalid mount record" in str(observation.report["errors"])


def test_delayed_diagnostic_cannot_overwrite_newer_snapshot(tmp_path, monkeypatch):
    release, entered = threading.Event(), threading.Event()
    path = tmp_path / "metrics.json"
    real_write = store.atomic_write

    def delayed(target, payload):
        if payload == b"old":
            entered.set()
            release.wait(5)
        real_write(target, payload)

    monkeypatch.setattr(store, "atomic_write", delayed)
    try:
        assert diagnostics.save_diagnostic(path, b"old").error
        assert entered.is_set()
        assert diagnostics.save_diagnostic(path, b"new").error
    finally:
        release.set()
        for worker in threading.enumerate():
            if worker.name == "weight-sync-observation":
                worker.join(2)
    assert path.read_bytes() == b"new"
