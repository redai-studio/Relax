# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for AutoscalerService."""

import asyncio
import json

from relax.utils.autoscaler.autoscaler_service import AutoscalerService, AutoscalerState
from relax.utils.autoscaler.config import AutoscalerConfig
from relax.utils.autoscaler.metrics_collector import AggregatedMetrics
from relax.utils.autoscaler.scaling_decision import ScalingAction, ScalingDecision


# Underlying class behind the @serve.deployment decorator.
_ServiceCls = getattr(AutoscalerService, "func_or_class", AutoscalerService)


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class _FakeSession:
    """Minimal aiohttp-like session: GET returns a preconfigured response."""

    def __init__(self, get_payload=None, get_status=200):
        self._get_payload = get_payload
        self._get_status = get_status
        self.get_urls = []

    def get(self, url):
        self.get_urls.append(url)
        return _FakeResp(self._get_status, self._get_payload)


def _service(session=None) -> "_ServiceCls":
    svc = object.__new__(_ServiceCls)
    svc.config = AutoscalerConfig(rollout_service_url="http://rollout")
    svc._http_session = session
    svc._state = AutoscalerState()
    return svc


def _engines_payload(engines):
    return {"models": {"default": {"engine_groups": [{"engines": engines}]}}}


def test_fetch_engines_filters_dead_slots():
    payload = _engines_payload(
        [
            {"rank": 0, "url": "http://e0", "status": "active"},
            {"rank": 1, "url": None, "status": "dead"},
            {"rank": 2, "url": "http://e2", "status": "active"},
        ]
    )
    svc = _service(_FakeSession(get_payload=payload))
    engines = asyncio.run(svc._fetch_engines())
    ids = {e["id"] for e in engines}
    assert ids == {"engine_0", "engine_2"}
    assert all(e["status"] == "active" for e in engines)


def test_fetch_engines_all_dead_returns_empty():
    payload = _engines_payload(
        [
            {"rank": 0, "url": None, "status": "dead"},
            {"rank": 1, "url": None, "status": "dead"},
        ]
    )
    svc = _service(_FakeSession(get_payload=payload))
    engines = asyncio.run(svc._fetch_engines())
    assert engines == []


def test_fetch_engines_skips_active_without_url():
    """An 'active' slot without a URL cannot be scraped -> excluded."""
    payload = _engines_payload(
        [
            {"rank": 0, "url": "", "status": "active"},
            {"rank": 1, "url": "http://e1", "status": "active"},
        ]
    )
    svc = _service(_FakeSession(get_payload=payload))
    engines = asyncio.run(svc._fetch_engines())
    assert [e["id"] for e in engines] == ["engine_1"]


def test_fetch_engines_multinode_counts_logical_engines_only():
    """Count logical engines rather than physical node actors."""
    # 2 logical engines x 2 nodes each = 4 physical node-actor rows.
    # node-0 rows (rank 0, 2) carry a url; worker rows (rank 1, 3) do not.
    payload = _engines_payload(
        [
            {"rank": 0, "url": "http://e0", "status": "active"},  # engine A node-0
            {"rank": 1, "url": None, "status": "active"},  # engine A worker (no url)
            {"rank": 2, "url": "http://e2", "status": "active"},  # engine B node-0
            {"rank": 3, "url": None, "status": "active"},  # engine B worker (no url)
        ]
    )
    svc = _service(_FakeSession(get_payload=payload))
    engines = asyncio.run(svc._fetch_engines())
    assert [e["id"] for e in engines] == ["engine_0", "engine_2"]
    assert len(engines) == 2  # logical engines, NOT 4 physical node actors


def test_evaluate_reconciles_pending_when_no_engines():
    svc = _service()
    called = {"pending": False}

    async def _no_engines():
        return []

    async def _spy_pending():
        called["pending"] = True

    class _Dec:
        def reset_condition_trackers(self):
            pass

    svc._fetch_engines = _no_engines
    svc._update_pending_requests = _spy_pending
    svc.decision_engine = _Dec()

    asyncio.run(svc._evaluate_and_scale())
    assert called["pending"] is True


def test_evaluate_resets_debounce_when_no_engines():
    """when engine discovery yields nothing (empty pool or transient.

    /engines failure), the debounce trackers must be reset before returning, so
    an accumulated scale-in streak does not fire immediately on recovery.
    """
    svc = _service()
    called = {"reset": False}

    async def _no_engines():
        return []

    async def _noop_pending():
        return None

    class _Dec:
        def reset_condition_trackers(self):
            called["reset"] = True

    svc._fetch_engines = _no_engines
    svc._update_pending_requests = _noop_pending
    svc.decision_engine = _Dec()

    asyncio.run(svc._evaluate_and_scale())
    assert called["reset"] is True


def test_update_pending_requests_treats_partial_as_terminal():
    svc = _service(session=_FakeSession())  # terminal-at-entry: GET is never issued
    svc._state.pending_requests = [{"request_id": "r1", "action": "scale_out", "status": "PARTIAL", "delta": 2}]
    asyncio.run(svc._update_pending_requests())
    assert svc._state.pending_requests == []
    assert len(svc._state.scale_history) == 1
    assert svc._state.scale_history[0]["request_id"] == "r1"
    assert svc._state.total_scale_operations == 1


def test_update_pending_requests_keeps_non_terminal():
    """A non-terminal scale-out (CREATING) is not moved when the server still
    reports it as non-terminal."""
    session = _FakeSession(get_payload={"status": "CREATING"}, get_status=200)
    svc = _service(session=session)
    svc._state.pending_requests = [{"request_id": "r1", "action": "scale_out", "status": "CREATING", "delta": 2}]
    asyncio.run(svc._update_pending_requests())
    assert len(svc._state.pending_requests) == 1
    assert svc._state.pending_requests[0]["status"] == "CREATING"
    assert len(svc._state.scale_history) == 0


def test_update_pending_requests_scale_in_completed_terminal():
    """scale_in terminal state is COMPLETED (not ACTIVE) -> moved to
    history."""
    svc = _service(session=_FakeSession())  # terminal-at-entry: GET is never issued
    svc._state.pending_requests = [{"request_id": "s1", "action": "scale_in", "status": "COMPLETED", "delta": 1}]
    asyncio.run(svc._update_pending_requests())
    assert svc._state.pending_requests == []
    assert len(svc._state.scale_history) == 1


def test_update_pending_requests_propagates_failure_categories():
    """A failed scale-out's source-classified failure_categories must be copied
    from the rollout status back into the tracked request and survive
    serialization to /scale_history (ScaleHistoryItem), so the monitor can
    bucket it without re-parsing error_message."""
    from relax.utils.autoscaler.autoscaler_service import ScaleHistoryItem

    session = _FakeSession(
        get_payload={
            "status": "FAILED",
            "error_message": "scale-out failed: NCCL transport mismatch: wrong_type",
            "failure_categories": ["NCCL_PRECHECK_TRANSPORT_MISMATCH"],
        },
        get_status=200,
    )
    svc = _service(session=session)
    svc._state.pending_requests = [
        {"request_id": "r1", "action": "scale_out", "status": "CREATING", "delta": 2, "triggered_at": 1.0}
    ]
    asyncio.run(svc._update_pending_requests())

    assert len(svc._state.scale_history) == 1
    record = svc._state.scale_history[0]
    assert record["failure_categories"] == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]
    # The /scale_history pydantic model must keep the field (not drop it).
    assert ScaleHistoryItem(**record).failure_categories == ["NCCL_PRECHECK_TRANSPORT_MISMATCH"]


class _FakePostSession:
    """aiohttp-like session that records POSTs and returns a preconfigured
    body.

    ``post_status``/``post_payload`` control the response; ``post_calls``
    captures ``(url, json_payload)`` tuples for assertions. A GET response can
    also be configured for orphan-404 checks.
    """

    def __init__(self, post_status=200, post_payload=None, get_status=200, get_payload=None):
        self._post_status = post_status
        self._post_payload = post_payload or {}
        self._get_status = get_status
        self._get_payload = get_payload or {}
        self.post_calls = []
        self.get_urls = []

    def post(self, url, json=None):
        self.post_calls.append((url, json))
        return _FakeResp(self._post_status, self._post_payload)

    def get(self, url):
        self.get_urls.append(url)
        return _FakeResp(self._get_status, self._get_payload)


def _decision(action, delta=1):
    return ScalingDecision(
        action=action,
        delta=delta,
        reason="test",
        triggered_conditions=["cond"],
        metrics_snapshot={"m": 1},
    )


def test_scale_out_noop_not_pending_records_history():
    session = _FakePostSession(post_status=200, post_payload={"request_id": "noop-xyz", "status": "NOOP"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=2), current_engines=3))

    assert svc._state.pending_requests == []
    assert len(svc._state.scale_history) == 1
    assert svc._state.scale_history[0]["status"] == "NOOP"
    assert svc._state.scale_history[0]["action"] == "scale_out"
    assert svc._state.total_scale_operations == 1
    # NOOP scaled nothing -> cooldown clock untouched.
    assert svc._state.last_scale_time is None


def test_scale_in_noop_not_pending_records_history():
    session = _FakePostSession(post_status=200, post_payload={"request_id": "noop-abc", "status": "NOOP"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_in(_decision(ScalingAction.SCALE_IN, delta=1), current_engines=3))

    assert svc._state.pending_requests == []
    assert len(svc._state.scale_history) == 1
    assert svc._state.scale_history[0]["status"] == "NOOP"
    assert svc._state.scale_history[0]["action"] == "scale_in"
    assert svc._state.last_scale_time is None


def test_scale_out_pending_status_enters_pending():
    """A genuinely non-terminal status (PENDING) is tracked as pending."""
    session = _FakePostSession(post_status=200, post_payload={"request_id": "r1", "status": "PENDING"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=2), current_engines=3))

    assert len(svc._state.pending_requests) == 1
    assert svc._state.pending_requests[0]["request_id"] == "r1"
    assert svc._state.pending_requests[0]["status"] == "PENDING"
    assert svc._state.pending_requests[0]["delta"] == 2
    assert svc._state.last_scale_action == ScalingAction.SCALE_OUT
    assert svc._state.last_scale_time is not None


def test_scale_out_conflict_body_not_pending():
    """CONFLICT reported with a 2xx body must not enter pending."""
    session = _FakePostSession(post_status=200, post_payload={"request_id": "c1", "status": "CONFLICT"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=1), current_engines=2))

    assert svc._state.pending_requests == []
    assert len(svc._state.scale_history) == 0


def test_scale_out_conflict_409_not_pending():
    """CONFLICT surfaced as HTTP 409 (non-2xx) must not enter pending."""
    session = _FakePostSession(post_status=409, post_payload={"status": "CONFLICT", "detail": "scale in progress"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=1), current_engines=2))

    assert svc._state.pending_requests == []


def test_noop_does_not_block_next_cycle():
    session = _FakePostSession(post_status=200, post_payload={"request_id": "noop-1", "status": "NOOP"})
    svc = _service(session=session)
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=1), current_engines=2))
    assert svc._state.pending_requests == []

    # Next cycle: nothing pending -> _update_pending_requests issues no GET and
    # the active_pending gate would be open.
    asyncio.run(svc._update_pending_requests())
    assert svc._state.pending_requests == []
    assert session.get_urls == []


def test_scale_out_payload_omits_timeout_by_default():
    """default: config leaves scale_out_request_timeout_secs=None, so the
    payload must NOT carry timeout_secs (server inherits its 300s default)."""
    session = _FakePostSession(post_status=200, post_payload={"request_id": "r1", "status": "PENDING"})
    svc = _service(session=session)
    assert svc.config.scale_out_request_timeout_secs is None
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=1), current_engines=2))

    _, payload = session.post_calls[0]
    assert "timeout_secs" not in payload


def test_scale_out_payload_includes_configured_timeout():
    """when ops sets a short timeout, it is forwarded as timeout_secs."""
    session = _FakePostSession(post_status=200, post_payload={"request_id": "r1", "status": "PENDING"})
    svc = _service(session=session)
    svc.config.scale_out_request_timeout_secs = 120.0
    asyncio.run(svc._execute_scale_out(_decision(ScalingAction.SCALE_OUT, delta=1), current_engines=2))

    _, payload = session.post_calls[0]
    assert payload["timeout_secs"] == 120.0


def test_config_scale_out_timeout_defaults_none():
    cfg = AutoscalerConfig()
    assert cfg.scale_out_request_timeout_secs is None
    assert cfg.to_dict()["scale_out_request_timeout_secs"] is None


def test_config_scale_out_timeout_roundtrip(tmp_path):
    import yaml

    path = tmp_path / "autoscaler.yaml"
    path.write_text(yaml.safe_dump({"scale_out_request_timeout_secs": 90.0}))
    cfg = AutoscalerConfig.from_yaml(str(path))
    assert cfg.scale_out_request_timeout_secs == 90.0
    assert cfg.to_dict()["scale_out_request_timeout_secs"] == 90.0


def test_config_scale_out_timeout_rejects_non_positive():
    import pytest

    with pytest.raises(ValueError):
        AutoscalerConfig(scale_out_request_timeout_secs=0)


class _RecordingCollector:
    """Fake MetricsCollector that records the ``num_candidates`` passed to
    ``add_snapshot`` so tests can assert the coverage denominator is bound to
    the cycle's own engine count (not shared ``_last_num_candidates`` state
    that an interleaving /status collection could overwrite)."""

    def __init__(self):
        self.add_snapshot_num_candidates = []

    async def collect_all(self, engines):
        return {}

    def add_snapshot(self, snapshot, num_candidates=None):
        self.add_snapshot_num_candidates.append(num_candidates)

    def get_aggregated_metrics(self):
        # is_empty freezes the decision engine -> no scale execution needed.
        return AggregatedMetrics(is_empty=True, coverage=0.0)

    def get_history(self):
        return []


class _FreezeDecisionEngine:
    def evaluate(self, **kwargs):
        return ScalingDecision(
            action=ScalingAction.NONE,
            delta=0,
            reason="frozen",
            triggered_conditions=[],
            metrics_snapshot={},
        )


def test_evaluate_binds_coverage_denominator_to_current_engines():
    """``_evaluate_and_scale`` must pass an explicit ``num_candidates`` equal
    to this cycle's engine count.

    Relying on the collector's shared ``_last_num_candidates`` lets a
    concurrent /status collection overwrite the denominator, turning partial
    coverage into a false 1.0.
    """
    svc = _service()
    engines = [
        {"id": "engine_0", "url": "http://e0"},
        {"id": "engine_1", "url": "http://e1"},
    ]
    collector = _RecordingCollector()

    async def _engines():
        return engines

    async def _noop_pending():
        return None

    svc._fetch_engines = _engines
    svc._update_pending_requests = _noop_pending
    svc.metrics_collector = collector
    svc.decision_engine = _FreezeDecisionEngine()

    asyncio.run(svc._evaluate_and_scale())

    assert collector.add_snapshot_num_candidates == [len(engines)]


def test_status_binds_coverage_denominator_to_current_engines():
    """the /status real-time collection path must likewise bind the denominator
    to its own engine count."""
    svc = _service()
    engines = [
        {"id": "engine_0", "url": "http://e0"},
        {"id": "engine_1", "url": "http://e1"},
        {"id": "engine_2", "url": "http://e2"},
    ]
    collector = _RecordingCollector()

    async def _engines():
        return engines

    svc._fetch_engines = _engines
    svc.metrics_collector = collector

    asyncio.run(svc.get_autoscaler_status())

    assert collector.add_snapshot_num_candidates == [len(engines)]


def test_update_config_patches_coverage_and_timeout_fields():
    """PATCH must actually apply the three fields that already round-trip
    through YAML/to_dict (min_coverage_scale_out/in,
    scale_out_request_timeout_secs).

    Previously these were silently ignored because ConfigUpdateRequest lacked
    them.
    """
    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    svc = _service()
    req = ConfigUpdateRequest(
        min_coverage_scale_out=0.25,
        min_coverage_scale_in=0.75,
        scale_out_request_timeout_secs=45.0,
    )
    resp = asyncio.run(svc.update_config(req))

    assert svc.config.min_coverage_scale_out == 0.25
    assert svc.config.min_coverage_scale_in == 0.75
    assert svc.config.scale_out_request_timeout_secs == 45.0
    cfg = resp.config
    assert cfg["min_coverage_scale_out"] == 0.25
    assert cfg["min_coverage_scale_in"] == 0.75
    assert cfg["scale_out_request_timeout_secs"] == 45.0


def test_update_config_rejects_out_of_range_coverage():
    """coverage fields must be constrained to [0, 1]."""
    import pytest
    from pydantic import ValidationError

    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    with pytest.raises(ValidationError):
        ConfigUpdateRequest(min_coverage_scale_out=1.5)
    with pytest.raises(ValidationError):
        ConfigUpdateRequest(min_coverage_scale_in=-0.1)


def test_update_config_rejects_non_positive_timeout():
    """scale_out_request_timeout_secs must be > 0 when provided."""
    import pytest
    from pydantic import ValidationError

    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    with pytest.raises(ValidationError):
        ConfigUpdateRequest(scale_out_request_timeout_secs=0)


def test_config_request_rejects_zero_coverage():
    import pytest
    from pydantic import ValidationError

    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    with pytest.raises(ValidationError):
        ConfigUpdateRequest(min_coverage_scale_out=0)
    with pytest.raises(ValidationError):
        ConfigUpdateRequest(min_coverage_scale_in=0)


def test_update_config_coverage_takes_effect_in_decision_engine():
    """a legal coverage PATCH must update ``svc.config`` AND be visible on the
    freshly-rebuilt ``decision_engine.config`` -- proving the update actually
    takes effect on the object the decision loop reads (not just the request
    model)."""
    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    svc = _service()
    req = ConfigUpdateRequest(min_coverage_scale_out=0.7, min_coverage_scale_in=0.7)
    asyncio.run(svc.update_config(req))

    assert svc.config.min_coverage_scale_out == 0.7
    assert svc.config.min_coverage_scale_in == 0.7
    # The rebuilt decision engine must read the new values (it holds a reference
    # to the same config instance).
    assert svc.decision_engine.config.min_coverage_scale_out == 0.7
    assert svc.decision_engine.config.min_coverage_scale_in == 0.7


def test_update_config_validate_backstop_rejects_invariant_violation():
    """defense-in-depth: a PATCH can pass every per-field pydantic bound
    yet still violate a cross-field dataclass invariant.

    ``metrics_interval_secs`` (gt=0) alone is a valid field, but pushing it
    above the current ``evaluation_interval_secs`` (default 30) breaks
    ``evaluation_interval_secs >= metrics_interval_secs``. The end-of-update
    ``self.config.validate()`` must catch it and raise, instead of committing a
    self-inconsistent config and rebuilding the decision engine around it.
    """
    import pytest

    from relax.utils.autoscaler.autoscaler_service import ConfigUpdateRequest

    svc = _service()
    assert svc.config.evaluation_interval_secs == 30.0
    req = ConfigUpdateRequest(metrics_interval_secs=100.0)  # > evaluation_interval_secs
    with pytest.raises(ValueError):
        asyncio.run(svc.update_config(req))
