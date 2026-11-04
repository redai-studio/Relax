# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run outcomes use actual driver completion, independent of sample stages."""

import errno
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.utils.agentic_obs.run_hooks import RunRecorder, annotate_failure_stage


def _args(tmp_path, **kwargs):
    return Namespace(dump_agentic_obs=str(tmp_path), agentic_obs_run_id="run_test", **kwargs)


def _records(tmp_path):
    path = tmp_path / "run_test/stages/run.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_recorder_is_disabled_without_dump_directory(tmp_path):
    recorder = RunRecorder(Namespace())
    recorder.start()
    recorder.set_phase("training")
    recorder.record_outcome("completed")
    recorder.flush()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_run_recorder_pairs_execution_boundaries_and_flushes_once(tmp_path, status):
    recorder = RunRecorder(_args(tmp_path))
    recorder.start()
    recorder.set_phase("training")
    error = RuntimeError("backend failed") if status == "failed" else None
    recorder.record_outcome(status, error=error)
    assert len(_records(tmp_path)) == 1  # The captured result is flushed before teardown.
    recorder.flush()
    recorder.flush()
    recorder.start()
    start, end = _records(tmp_path)
    assert start["scope"] == end["scope"] == "run"
    assert start["stage"] == end["stage"] == "execution"
    assert start["stage_id"] == end["stage_id"]
    assert [start["event_seq"], end["event_seq"]] == [1, 2]
    assert not start["is_terminal"] and end["is_terminal"]
    assert start["occurred_at"] <= end["occurred_at"]
    assert start["record_id"] != end["record_id"]
    assert end["status"] == status and end["operation"] == "training"
    assert "sample_export_id" not in end
    if error:
        assert end["reason"] == "RuntimeError: backend failed"
        assert end["reason_code"] == "run_failed"


@pytest.mark.parametrize("first", ["completed", "failed", "cancelled"])
def test_run_recorder_preserves_first_outcome_during_cleanup(tmp_path, first):
    recorder = RunRecorder(_args(tmp_path))
    recorder.start()
    recorder.set_phase("training")
    recorder.record_outcome(first)
    recorder.set_phase("shutdown")
    recorder.record_outcome("cancelled", reason="Received SIGTERM")
    recorder.flush()
    assert _records(tmp_path)[1]["status"] == first
    assert _records(tmp_path)[1]["operation"] == "training"


def test_run_failure_stage_survives_ray_transport_and_service_wrapping(tmp_path):
    import cloudpickle
    from ray.exceptions import RayTaskError

    error = OSError(errno.EDQUOT, "Disk quota exceeded")

    @annotate_failure_stage("checkpoint")
    def save_model(owner):
        raise error

    with pytest.raises(OSError) as caught:
        save_model(SimpleNamespace(args=_args(tmp_path)))
    assert caught.value is error
    remote = cloudpickle.loads(cloudpickle.dumps(RayTaskError("save_model", "remote traceback", error)))
    try:
        raise remote
    except Exception:
        try:
            raise RuntimeError("Service task failed")
        except RuntimeError as wrapped:
            recorder = RunRecorder(_args(tmp_path))
            recorder.start()
            recorder.set_phase("training")
            recorder.record_outcome("failed", error=wrapped)
            recorder.flush()
    end = _records(tmp_path)[1]
    assert end["operation"] == "checkpoint"
    assert end["reason_code"] == "run_failed"
    assert "Disk quota exceeded" in end["reason"] and "OSError" in end["reason"]


def test_run_failure_annotation_does_not_change_disabled_exception(tmp_path):
    error = ValueError("save failed")

    @annotate_failure_stage("checkpoint")
    def save_model(owner):
        raise error

    with pytest.raises(ValueError) as caught:
        save_model(SimpleNamespace(args=Namespace()))
    assert caught.value is error and error.__dict__ == {}


def test_run_recorder_handles_cyclic_exception_links(tmp_path):
    error = RuntimeError("failed")
    error.__cause__ = error
    recorder = RunRecorder(_args(tmp_path))
    recorder.record_outcome("failed", error=error)
    recorder.flush()
    assert _records(tmp_path)[1]["reason"] == "RuntimeError: failed"


def test_run_recorder_does_not_replace_outcome_during_error_formatting(tmp_path):
    recorder = RunRecorder(_args(tmp_path))
    recorder.start()

    class InterruptedError(Exception):
        def __str__(self):
            recorder.record_outcome("cancelled", reason="Received SIGTERM")
            return "training interrupted"

    recorder.record_outcome("failed", error=InterruptedError())
    recorder.flush()
    assert _records(tmp_path)[1]["status"] == "cancelled"


def test_run_recorder_keeps_write_failures_best_effort(tmp_path):
    path = tmp_path / "run_test/stages/run.jsonl"
    path.mkdir(parents=True)  # Simulate an unwritable JSONL destination.
    recorder = RunRecorder(_args(tmp_path))
    recorder.start()
    recorder.record_outcome("failed", error=OSError(errno.EDQUOT, "Disk quota exceeded"))
    recorder.flush()
    assert path.is_dir()
