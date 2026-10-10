# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for :class:`relax.utils.health_monitor.RolloutHealthMonitor`.

Regression target (self-heal prune/rejoin): ``resume()`` used to call
``self._consecutive_failures.clear()`` on every train step. Because a soft
failure (sglang subprocess dead, actor alive -> ``RayTaskError``) is usually
probed only once per resume window, the consecutive-failure counter never
reached the threshold and a dead engine was never killed/rebuilt.

Fixes under test:
- resume() no longer clears the counter (soft failures accumulate across
  resume windows).
- Terminal ``RayActorError`` bypasses the counter and fences resource reuse.
- only a *successful* health check resets the counter (overload mitigation).
- ``_kill_engine`` pops the counter so a rebuilt engine reusing the slot
  starts clean.

Tests build a bare shell via ``object.__new__`` + stubbed attributes so no
Ray runtime / GPU is needed.
"""

import threading
from unittest import mock

import requests
from ray.exceptions import ActorUnavailableError, RayActorError, RayTaskError

import relax.utils.health_monitor as hm
from relax.utils.health_monitor import RolloutHealthMonitor


def _monitor(max_failures=2):
    """Build a bare RolloutHealthMonitor shell with the attributes the health
    check / kill paths touch."""
    m = object.__new__(RolloutHealthMonitor)
    m._intentionally_removed = set()
    m._consecutive_failures = {}
    m._max_consecutive_failures = max_failures
    m._check_timeout = 30
    m._engine_group = _Group([])
    return m


class _Engine:
    def __init__(self):
        self.health_generate = mock.MagicMock()
        self.shutdown = mock.MagicMock()
        self.unregister_dcs = mock.MagicMock()


class _Group:
    def __init__(self, engines):
        self.all_engines = list(engines)
        self.nodes_per_engine = 1


def _install_ray_get(monkeypatch, seq):
    """Patch ``health_monitor.ray.get`` to consume a scripted sequence.

    Each item is either raised (if it is an exception) or returned.
    """
    it = iter(seq)

    def _fake_get(ref, *args, **kwargs):
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(hm.ray, "get", _fake_get)


def _soft_error():
    return RayTaskError(
        "health_generate",
        "traceback",
        requests.exceptions.ConnectionError("connection refused"),
    )


# ---------------------------------------------------------------------------
# Test 1: soft path accumulates (subprocess dead -> RayTaskError)
# ---------------------------------------------------------------------------
def test_health_monitor_soft_failure_accumulates_then_kills(monkeypatch):
    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    engine = _Engine()
    _install_ray_get(monkeypatch, [_soft_error(), _soft_error()])

    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 1
    m._kill_engine.assert_not_called()

    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 2
    m._kill_engine.assert_called_once_with(rollout_engine_id=0)


# ---------------------------------------------------------------------------
# Test 2: hard path kills immediately (whole actor dead -> RayActorError)
# ---------------------------------------------------------------------------
def test_health_monitor_actor_error_kills_immediately(monkeypatch):
    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    engine = _Engine()
    _install_ray_get(monkeypatch, [RayActorError()])

    m._check_engine_health(0, engine)

    # Killed on the very first probe, without going through the soft counter.
    m._kill_engine.assert_called_once_with(rollout_engine_id=0)
    assert m._consecutive_failures.get(0, 0) == 0


# ---------------------------------------------------------------------------
# Test 3: resume() no longer clears the counter (main-fix regression)
# ---------------------------------------------------------------------------
def test_health_monitor_resume_does_not_reset_counter(monkeypatch):
    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    # resume() touches these attributes.
    m._pause_event = threading.Event()
    m._need_first_wait = False
    m._is_checking_enabled = True
    engine = _Engine()

    # Soft failure -> pause/resume window boundary -> soft failure again.
    _install_ray_get(monkeypatch, [_soft_error()])
    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 1
    m._kill_engine.assert_not_called()

    m.resume()  # must NOT wipe the counter
    assert m._consecutive_failures[0] == 1

    _install_ray_get(monkeypatch, [_soft_error()])
    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 2
    m._kill_engine.assert_called_once_with(rollout_engine_id=0)


# ---------------------------------------------------------------------------
# Test 4: an intermittent success rescues a slow engine (overload mitigation)
# ---------------------------------------------------------------------------
def test_health_monitor_success_resets_counter_no_false_kill(monkeypatch):
    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    engine = _Engine()
    # fail, success, fail -> should never reach 2 in a row.
    _install_ray_get(monkeypatch, [_soft_error(), None, _soft_error()])

    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 1

    m._check_engine_health(0, engine)  # success resets
    assert m._consecutive_failures[0] == 0

    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 1

    m._kill_engine.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: _kill_engine pops the counter (clean slate after rebuild)
# ---------------------------------------------------------------------------
def test_health_monitor_kill_engine_clears_counter(monkeypatch):
    m = _monitor(max_failures=2)
    engine = _Engine()
    m._engine_group = _Group([engine])
    m._consecutive_failures[0] = 5  # pretend a stale count is present

    # Confirmed cleanup is delegated to the group; the monitor only resets its counter.
    m._engine_group.shutdown_engines = mock.Mock()

    m._kill_engine(rollout_engine_id=0)

    assert 0 not in m._consecutive_failures
    m._engine_group.shutdown_engines.assert_called_once_with({0})
    assert m._engine_group.all_engines[0] is engine


def test_health_monitor_unconfirmed_cleanup_keeps_worker_and_counter():
    m = _monitor()
    engine = _Engine()
    m._engine_group = _Group([engine])
    m._engine_group.shutdown_engines = mock.Mock(side_effect=hm.InferenceCleanupError("unconfirmed"))
    m._consecutive_failures[0] = 2
    m._kill_engine(0)
    assert m._engine_group.all_engines == [engine]
    assert m._consecutive_failures[0] == 2


def test_health_monitor_terminal_actor_is_not_probed_again(monkeypatch):
    m = _monitor()
    engine = _Engine()
    m._engine_group = _Group([engine])
    m._engine_group.shutdown_engines = mock.Mock(side_effect=hm.InferenceRecoveryRequired("unconfirmed"))
    get = mock.Mock(side_effect=RayActorError())
    monkeypatch.setattr(hm.ray, "get", get)
    m._check_engine_health(0, engine)
    m._check_engine_health(0, engine)
    assert get.call_count == 1
    assert m._engine_group.all_engines == [engine]


def test_health_monitor_temporary_actor_unavailability_does_not_trigger_cleanup(monkeypatch):
    m = _monitor()
    m._kill_engine = mock.Mock()
    _install_ray_get(monkeypatch, [ActorUnavailableError("restarting", None)])
    m._check_engine_health(0, _Engine())
    m._kill_engine.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: intentionally removed engines are never health-checked / killed
# ---------------------------------------------------------------------------
def test_health_monitor_intentionally_removed_is_skipped(monkeypatch):
    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    m.mark_intentionally_removed(0)
    engine = _Engine()
    # Even a hard/soft error must not matter: the check returns before ray.get.
    _install_ray_get(monkeypatch, [RayActorError(), _soft_error()])

    m._check_engine_health(0, engine)
    m._check_engine_health(0, engine)

    m._kill_engine.assert_not_called()
    assert 0 not in m._consecutive_failures


# ---------------------------------------------------------------------------
# Test 7 (optional): RayTaskError.cause is accessible; Timeout and
# ConnectionError both flow through the single soft counter (we deliberately
# did NOT split by cause -- documents current behavior).
# ---------------------------------------------------------------------------
def test_health_monitor_soft_causes_share_single_counter(monkeypatch):
    conn = RayTaskError("health_generate", "tb", requests.exceptions.ConnectionError("refused"))
    timeout = RayTaskError("health_generate", "tb", requests.exceptions.Timeout("slow"))
    assert isinstance(conn.cause, requests.exceptions.ConnectionError)
    assert isinstance(timeout.cause, requests.exceptions.Timeout)

    m = _monitor(max_failures=2)
    m._kill_engine = mock.MagicMock()
    engine = _Engine()
    _install_ray_get(monkeypatch, [timeout, conn])

    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 1
    m._check_engine_health(0, engine)
    assert m._consecutive_failures[0] == 2
    m._kill_engine.assert_called_once_with(rollout_engine_id=0)
