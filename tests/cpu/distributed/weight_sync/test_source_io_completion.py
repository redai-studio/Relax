"""Completion and actual deadline failures remain distinct."""

import logging
import time

import pytest

from relax.distributed.weight_sync import source_io


@pytest.mark.parametrize("committing", [False, True])
def test_successful_slow_file_call_returns_result_after_draining(committing, caplog, monkeypatch):
    # Keep lazy application logger setup from removing pytest's capture handler.
    monkeypatch.setattr(source_io, "logger", logging.getLogger(__name__))

    def slow():
        time.sleep(0.06)
        return "completed-data"

    result = source_io.coordinated_io(
        slow,
        lambda state: [state],
        stage="slow read",
        timeout=0.02,
        committing=committing,
    )
    assert result == "completed-data"
    assert "result retained" in caplog.text


def test_cooperative_file_deadline_still_propagates_failure():
    from relax.distributed.weight_sync.deadline import checkpoint

    def slow():
        time.sleep(0.06)
        checkpoint("block read")

    with pytest.raises(RuntimeError, match="deadline"):
        source_io.coordinated_io(slow, lambda state: [state], stage="read", timeout=0.02)
