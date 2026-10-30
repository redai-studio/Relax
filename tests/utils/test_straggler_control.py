# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for switching the profiler's level through the metrics service."""

import pytest

from relax.utils.straggler.analysis.control import CONTROL_PAGE, ControlState, LevelPoller


def test_requests_are_validated_and_reach_the_next_poll_with_the_primary_status():
    times = iter(range(100, 200))
    state = ControlState(clock=lambda: float(next(times)))
    assert state.poll({"level": 1, "rollout": 9, "health": "active"}) == {"requested": None, "version": 0}
    with pytest.raises(ValueError, match="0..3"):
        state.request(4)
    state.request(0)
    assert state.poll({"level": 1, "rollout": 19, "health": "active"}) == {"requested": 0, "version": 1}
    snapshot = state.state()
    assert snapshot["requested"] == 0 and snapshot["version"] == 1
    assert snapshot["report"]["rollout"] == 19 and snapshot["now"] > snapshot["report"]["time"]


class _Reply:
    def __init__(self, body=None, error=None):
        self._body, self._error = body, error

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def json(self):
        return self._body


def test_the_poller_reports_status_and_keeps_the_last_answer_when_a_poll_fails():
    calls = []
    replies = [_Reply({"requested": 2, "version": 1}), _Reply(error=RuntimeError("503")), ConnectionError("down")]

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    status = {"level": 1, "rollout": 4, "health": "active"}
    poller = LevelPoller("http://head:8000/metrics/", lambda: status, interval=3600, timeout=2.0, post=post)
    try:
        for _ in range(3):
            poller.poll_once()
            assert poller.requested == 2
    finally:
        poller.stop()
    assert calls[0] == ("http://head:8000/metrics/straggler/poll", status, 2.0)


def test_the_metrics_service_serves_the_page_and_the_level_endpoints(monkeypatch):
    from fastapi.testclient import TestClient

    from relax.utils.metrics.service import app
    from relax.utils.straggler import control_routes

    monkeypatch.setattr(control_routes, "state", ControlState())
    client = TestClient(app)
    assert client.get("/straggler").text == CONTROL_PAGE
    assert client.post("/straggler/control", json={"level": 2}).json()["requested"] == 2
    assert client.post("/straggler/control", json={"level": 5}).status_code == 400
    poll = {"level": 1, "rollout": 9, "health": "active"}
    assert client.post("/straggler/poll", json=poll).json() == {"requested": 2, "version": 1}
    assert client.get("/straggler/control").json()["report"]["level"] == 1
