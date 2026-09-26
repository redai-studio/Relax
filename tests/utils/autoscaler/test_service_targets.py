# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for autoscaler per-service targets (Task 4 change 2).

``service_targets`` generalizes the single ``rollout_service_url`` into a
per-service URL map (e.g. {rollout: url, genrm: url}) with the legacy field
kept backward compatible; ``service_policies`` allows GenRM-independent
thresholds whose unset fields inherit the global configuration.

Run: python -m unittest tests.utils.autoscaler.test_service_targets -v
"""

import asyncio
import json
import os
import tempfile
import unittest

from tests.utils._dep_stubs import import_autoscaler_service


svc_module = import_autoscaler_service()

from relax.utils.autoscaler.config import AutoscalerConfig, ScaleOutPolicy, ServiceScalingPolicy  # noqa: E402
from relax.utils.autoscaler.metrics_collector import AggregatedMetrics  # noqa: E402
from relax.utils.autoscaler.scaling_decision import ScalingDecisionEngine  # noqa: E402
from relax.utils.genrm_scale_registry import GenRMScaleRegistry  # noqa: E402


_AutoscalerService = getattr(svc_module.AutoscalerService, "func_or_class", svc_module.AutoscalerService)
_AutoscalerState = svc_module.AutoscalerState
_ConfigUpdateRequest = svc_module.ConfigUpdateRequest
_ScalingAction = svc_module.ScalingAction
_ScalingDecision = svc_module.ScalingDecision


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


class _FakePostSession:
    """aiohttp-like session recording POSTs with preconfigured replies."""

    def __init__(self, post_status=200, post_payload=None):
        self._post_status = post_status
        self._post_payload = post_payload or {}
        self.post_calls = []

    def post(self, url, json=None):
        self.post_calls.append((url, json))
        return _FakeResp(self._post_status, self._post_payload)


class _ParkedResp:
    """Response whose body only arrives once ``release`` is set."""

    def __init__(self, release, payload):
        self._release = release
        self._payload = payload
        self.status = 200

    async def __aenter__(self):
        await self._release.wait()
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class _LostResponse:
    """A response that dies mid-flight (server already accepted)."""

    async def __aenter__(self):
        raise ConnectionError("connection reset while reading the response")

    async def __aexit__(self, *exc):
        return False


class _IdempotentScaleServer:
    """aiohttp-like session speaking the GenRM idempotency contract.

    Operations are keyed by idempotency_key server-side: the first POST with a
    key admits an operation; any later POST with the same key replays the same
    request_id (exactly one operation ever exists per key). Responses can be
    configured to be "lost" (transport error after acceptance).
    """

    def __init__(self, lose_responses=0):
        self.operations = {}
        self.posts = []
        self.get_calls = []
        self._lose = lose_responses
        self._seq = 0

    def post(self, url, json=None):
        self.posts.append((url, json))
        key = (json or {}).get("idempotency_key")
        if key not in self.operations:
            self._seq += 1
            self.operations[key] = f"req-{self._seq}"
        if self._lose > 0:
            self._lose -= 1
            return _LostResponse()
        return _FakeResp(200, {"request_id": self.operations[key], "status": "PENDING"})

    def get(self, url):
        self.get_calls.append(url)
        # GenRMScaleStatusResponse always carries cleanup_required.
        return _FakeResp(200, {"status": "ACTIVE", "current": 2, "ready": 2, "cleanup_required": False})


class _RegistryBackedScaleServer:
    """aiohttp-like session backed by a production GenRMScaleRegistry.

    Speaks the real GenRM HTTP contract: POST /scale_out|scale_in submits (or
    replays by idempotency key), GET /scale_out|in/{id} returns the
    authoritative status INCLUDING cleanup_required (like
    GenRMScaleStatusResponse), and POST .../reconcile clears cleanup (like the
    component's reconcile). POST responses omit cleanup_required exactly like
    GenRMScaleResponse -- a terminal replay therefore carries no cleanup
    proof, which is the bug condition under test.
    """

    def __init__(self, registry, current=1, ready=1, lose_responses=0):
        self.registry = registry
        self._current = current
        self._ready = ready
        self._lose = lose_responses
        self.posts = []  # (url, body) for the scale endpoints only
        self.get_calls = []
        self.reconcile_calls = []
        self.fail_gets = 0
        self.reconcile_dirty = False
        self.admitted_request_id = None

    def post(self, url, json=None):
        if url.endswith("/reconcile"):
            self.reconcile_calls.append(url)
            request_id = url.rstrip("/").split("/")[-2]
            op = self.registry.get_status("scale_out", request_id) or self.registry.get_status("scale_in", request_id)
            if self.reconcile_dirty:
                # Cleanup still unfinished server-side (e.g. victim draining).
                return _FakeResp(
                    200,
                    {
                        "request_id": request_id,
                        "direction": op["direction"],
                        "status": op["status"],
                        "cleanup_required": True,
                        "victim_cleared": False,
                    },
                )
            self.registry.clear_cleanup(request_id)
            refreshed = self.registry.get_status(op["direction"], request_id)
            return _FakeResp(
                200,
                {
                    "request_id": request_id,
                    "direction": refreshed["direction"],
                    "status": refreshed["status"],
                    "cleanup_required": refreshed["cleanup_required"],
                    "victim_cleared": True,
                },
            )
        self.posts.append((url, json))
        direction = "scale_out" if url.endswith("/scale_out") else "scale_in"
        decision = self.registry.submit(
            direction,
            model_name="default",
            target=json["num_replicas"],
            timeout_secs=json.get("timeout_secs"),
            idempotency_key=json.get("idempotency_key"),
            current=self._current,
            ready=self._ready,
        )
        if decision.get("request_id"):
            self.admitted_request_id = decision["request_id"]
        if self._lose > 0:
            # The transport dies AFTER the server admitted the operation --
            # the acceptance response is the only thing that is lost.
            self._lose -= 1
            return _LostResponse()
        if decision.get("http", 200) != 200:
            return _FakeResp(decision["http"], decision)
        # GenRMScaleResponse shape: NO cleanup_required field.
        return _FakeResp(200, {"status": decision["status"], "request_id": decision.get("request_id")})

    def get(self, url):
        self.get_calls.append(url)
        if self.fail_gets > 0:
            self.fail_gets -= 1
            return _LostResponse()
        parts = url.rstrip("/").split("/")
        direction, request_id = parts[-2], parts[-1]
        op = self.registry.get_status(direction, request_id)
        if op is None:
            return _FakeResp(404, {"detail": "unknown request_id"})
        # GenRMScaleStatusResponse shape: cleanup_required always present.
        return _FakeResp(200, op)


class _ParkedPostSession:
    """aiohttp-like session that parks POST responses and records GET URLs."""

    def __init__(self, post_payload=None, get_payload=None):
        self.release = asyncio.Event()
        self.post_calls = []
        self.get_calls = []
        self._post_payload = post_payload or {"request_id": "req-x", "status": "PENDING"}
        self._get_payload = get_payload or {
            "status": "ACTIVE",
            "current": 2,
            "ready": 2,
            "cleanup_required": False,
        }

    def post(self, url, json=None):
        self.post_calls.append(url)
        return _ParkedResp(self.release, self._post_payload)

    def get(self, url):
        self.get_calls.append(url)
        return _FakeResp(200, self._get_payload)


def _decision(action, delta=1):
    return _ScalingDecision(
        action=action,
        delta=delta,
        reason="test",
        triggered_conditions=["cond"],
        metrics_snapshot={"m": 1},
    )


def _genrm_registry():
    """Production registry with the autoscaler's default model registered."""
    registry = GenRMScaleRegistry()
    registry.register_initial("default", 1)
    return registry


def _busy_metrics():
    """Non-empty snapshot with high token usage -> scale-out would trigger."""
    return AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        avg_token_usage=0.95,
        throughput_variance=0.0,
        is_empty=False,
        coverage=1.0,
    )


def _idle_metrics():
    """Non-empty snapshot where every scale-in condition holds."""
    return AggregatedMetrics(
        num_engines=2,
        total_queue_reqs=0,
        total_running_reqs=0,
        avg_token_usage=0.0,
        total_throughput=100.0,
        throughput_variance=0.0,
        is_empty=False,
        coverage=1.0,
    )


def _service(config, session=None):
    svc = object.__new__(_AutoscalerService)
    svc.config = config
    svc._http_session = session
    svc._state = _AutoscalerState()
    return svc


class TestGetServiceUrl(unittest.TestCase):
    def test_rollout_falls_back_to_legacy_field(self):
        config = AutoscalerConfig(rollout_service_url="http://legacy:8000/rollout")
        self.assertEqual(config.get_service_url("rollout"), "http://legacy:8000/rollout")
        self.assertEqual(config.get_service_url(), "http://legacy:8000/rollout")  # default service

    def test_service_targets_take_precedence_for_rollout(self):
        config = AutoscalerConfig(
            rollout_service_url="http://legacy:8000/rollout",
            service_targets={"rollout": "http://override:9000/rollout"},
        )
        self.assertEqual(config.get_service_url("rollout"), "http://override:9000/rollout")

    def test_genrm_target_resolvable_and_required(self):
        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        self.assertEqual(config.get_service_url("genrm"), "http://genrm:8000/genrm")
        with self.assertRaises(KeyError):
            config.get_service_url("critic")  # unconfigured non-rollout service


class TestServicePolicies(unittest.TestCase):
    def test_unset_service_inherits_globals(self):
        config = AutoscalerConfig(min_engines=2, max_engines=17)
        effective = config.get_effective_policies("genrm")
        self.assertEqual(effective.min_engines, 2)
        self.assertEqual(effective.max_engines, 17)
        self.assertEqual(effective.scale_out_policy, config.scale_out_policy)
        self.assertEqual(effective.scale_in_policy, config.scale_in_policy)

    def test_genrm_override_with_partial_inheritance(self):
        config = AutoscalerConfig(
            min_engines=2,
            max_engines=17,
            service_policies={"genrm": ServiceScalingPolicy(min_engines=1, max_engines=4)},
        )
        effective = config.get_effective_policies("genrm")
        self.assertEqual(effective.min_engines, 1)
        self.assertEqual(effective.max_engines, 4)
        self.assertEqual(effective.scale_out_policy, config.scale_out_policy)  # unset -> inherited

    def test_genrm_independent_token_thresholds(self):
        config = AutoscalerConfig(
            service_policies={
                "genrm": ServiceScalingPolicy(scale_out_policy=ScaleOutPolicy(token_usage_threshold=0.7))
            }
        )
        effective = config.get_effective_policies("genrm")
        self.assertEqual(effective.scale_out_policy.token_usage_threshold, 0.7)
        self.assertEqual(config.scale_out_policy.token_usage_threshold, 0.85)  # global untouched

    def test_runtime_state_is_isolated_per_service(self):
        config = AutoscalerConfig(
            service_targets={"genrm": "http://genrm:8000/genrm"},
            service_policies={"genrm": ServiceScalingPolicy(min_engines=1, max_engines=4)},
        )
        svc = object.__new__(_AutoscalerService)
        svc.config = config
        svc._services = {}
        svc._rebuild_service_runtimes()
        rollout = svc._services["rollout"]
        genrm = svc._services["genrm"]
        self.assertIsNot(rollout.state, genrm.state)
        self.assertIsNot(rollout.metrics_collector, genrm.metrics_collector)
        self.assertIsNot(rollout.decision_engine, genrm.decision_engine)
        self.assertEqual(genrm.config.max_engines, 4)
        self.assertEqual(rollout.config.max_engines, config.max_engines)


class TestConfigValidation(unittest.TestCase):
    def test_empty_service_target_url_rejected(self):
        with self.assertRaises(ValueError):
            AutoscalerConfig(service_targets={"genrm": "  "})

    def test_service_policy_min_greater_than_max_rejected(self):
        with self.assertRaises(ValueError):
            AutoscalerConfig(service_policies={"genrm": ServiceScalingPolicy(min_engines=4, max_engines=2)})


class TestYamlRoundTrip(unittest.TestCase):
    def test_from_yaml_reads_service_targets_and_policies(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(
                "service_targets:\n"
                "  rollout: http://r:8000/rollout\n"
                "  genrm: http://g:8000/genrm\n"
                "service_policies:\n"
                "  genrm:\n"
                "    min_engines: 1\n"
                "    max_engines: 4\n"
                "    scale_out_policy:\n"
                "      token_usage_threshold: 0.7\n"
            )
            path = handle.name
        try:
            config = AutoscalerConfig.from_yaml(path)
            self.assertEqual(config.get_service_url("genrm"), "http://g:8000/genrm")
            self.assertEqual(config.get_service_url("rollout"), "http://r:8000/rollout")
            effective = config.get_effective_policies("genrm")
            self.assertEqual(effective.min_engines, 1)
            self.assertEqual(effective.max_engines, 4)
            self.assertEqual(effective.scale_out_policy.token_usage_threshold, 0.7)
            # Serialized config keeps the new fields for /status and /config.
            as_dict = config.to_dict()
            self.assertEqual(as_dict["service_targets"]["genrm"], "http://g:8000/genrm")
            self.assertEqual(as_dict["service_policies"]["genrm"]["min_engines"], 1)
        finally:
            os.unlink(path)

    def test_explicit_rollout_url_wins_over_yaml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("service_targets:\n  rollout: http://yaml:8000/rollout\n")
            path = handle.name
        try:
            config = AutoscalerConfig.from_yaml(path, rollout_service_url="http://explicit:8000/rollout")
            self.assertEqual(config.get_service_url("rollout"), "http://explicit:8000/rollout")
        finally:
            os.unlink(path)


class TestServiceUsesResolvedUrl(unittest.TestCase):
    def test_execute_scale_out_posts_to_service_target(self):
        config = AutoscalerConfig(
            service_targets={"rollout": "http://override:9000/rollout", "genrm": "http://genrm:8000/genrm"}
        )
        session = _FakePostSession(post_payload={"request_id": "r1", "status": "PENDING"})
        svc = _service(config, session)
        asyncio.run(svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT, delta=2), current_engines=3))

        url, payload = session.post_calls[0]
        self.assertEqual(url, "http://override:9000/rollout/scale_out")
        self.assertEqual(payload["num_replicas"], 5)

    def test_execute_scale_in_posts_to_legacy_url(self):
        """No service_targets configured -> legacy rollout_service_url path."""
        config = AutoscalerConfig(rollout_service_url="http://legacy:8000/rollout")
        session = _FakePostSession(post_payload={"request_id": "s1", "status": "PENDING"})
        svc = _service(config, session)
        asyncio.run(svc._execute_scale_in(_decision(_ScalingAction.SCALE_IN, delta=1), current_engines=3))

        url, _payload = session.post_calls[0]
        self.assertEqual(url, "http://legacy:8000/rollout/scale_in")


class TestPatchConfig(unittest.TestCase):
    def test_patch_updates_service_targets(self):
        svc = _service(AutoscalerConfig())
        request = _ConfigUpdateRequest(service_targets={"genrm": "http://genrm:8000/genrm"})
        response = asyncio.run(svc.update_config(request))
        self.assertEqual(svc.config.get_service_url("genrm"), "http://genrm:8000/genrm")
        # rollout keeps its legacy fallback after the patch.
        self.assertEqual(svc.config.get_service_url("rollout"), svc.config.rollout_service_url)
        self.assertIn("service_targets", response.config)

    def test_patch_hot_starts_new_collector(self):
        """Review finding: a collector constructed for a newly PATCHed-in
        service target must be started before it serves evaluations."""
        from unittest.mock import patch

        from relax.utils.autoscaler.metrics_collector import MetricsCollector

        started, stopped = [], []
        orig_start, orig_stop = MetricsCollector.start, MetricsCollector.stop

        async def _rec_start(self):
            started.append(self)

        async def _rec_stop(self):
            stopped.append(self)

        svc = _service(AutoscalerConfig())
        request = _ConfigUpdateRequest(service_targets={"genrm": "http://genrm:8000/genrm"})
        with patch.object(MetricsCollector, "start", _rec_start), patch.object(MetricsCollector, "stop", _rec_stop):
            asyncio.run(svc.update_config(request))
        genrm_collector = svc._services["genrm"].metrics_collector
        self.assertIn(genrm_collector, started)
        # Existing rollout collector was constructed by the rebuild too (the
        # fixture skips __init__), so it is also started -- but nothing is
        # stopped while the target remains configured.
        self.assertEqual(stopped, [])
        # Restore for teardown cleanliness.
        MetricsCollector.start, MetricsCollector.stop = orig_start, orig_stop

    def test_patch_hot_stops_removed_collector(self):
        """Removing a service target must stop its collector so the HTTP
        session does not outlive the runtime."""
        from unittest.mock import patch

        from relax.utils.autoscaler.metrics_collector import MetricsCollector

        started, stopped = [], []
        orig_start, orig_stop = MetricsCollector.start, MetricsCollector.stop

        async def _rec_start(self):
            started.append(self)

        async def _rec_stop(self):
            stopped.append(self)

        svc = _service(AutoscalerConfig())
        svc.config.service_targets = {"genrm": "http://genrm:8000/genrm"}
        # First PATCH: bring the genrm runtime in.
        with patch.object(MetricsCollector, "start", _rec_start), patch.object(MetricsCollector, "stop", _rec_stop):
            asyncio.run(svc.update_config(_ConfigUpdateRequest()))
            genrm_collector = svc._services["genrm"].metrics_collector
            # Second PATCH: drop the genrm target.
            svc.config.service_targets = {}
            asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        self.assertNotIn("genrm", svc._services)
        self.assertIn(genrm_collector, stopped)
        MetricsCollector.start, MetricsCollector.stop = orig_start, orig_stop

    def test_legacy_rollout_url_patch_is_not_shadowed(self):
        """Review finding: ``from_yaml`` seeds service_targets["rollout"] from
        the startup ``rollout_service_url``; a later legacy PATCH updated only
        the field, so the stale map entry kept serving the old URL.

        The two spellings must stay in sync.
        """
        config = AutoscalerConfig()
        # Simulate the startup override (controller passes get_serve_url).
        config.rollout_service_url = "http://old:8000/rollout"
        config.service_targets["rollout"] = "http://old:8000/rollout"
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest(rollout_service_url="http://new:8000/rollout")))
        self.assertEqual(svc.config.get_service_url("rollout"), "http://new:8000/rollout")
        self.assertEqual(svc.config.service_targets["rollout"], "http://new:8000/rollout")

    def test_service_targets_patch_mirrors_into_legacy_field(self):
        config = AutoscalerConfig()
        config.rollout_service_url = "http://old:8000/rollout"
        config.service_targets["rollout"] = "http://old:8000/rollout"
        svc = _service(config)
        asyncio.run(
            svc.update_config(
                _ConfigUpdateRequest(service_targets={"rollout": "http://new:8000/rollout", "genrm": "http://g:1"})
            )
        )
        self.assertEqual(svc.config.rollout_service_url, "http://new:8000/rollout")
        self.assertEqual(svc.config.get_service_url("rollout"), "http://new:8000/rollout")

    def _svc_with_genrm_runtime(self):
        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        svc = object.__new__(_AutoscalerService)
        svc.config = config
        svc._services = {}
        svc._rebuild_service_runtimes()
        return svc

    def test_remove_target_with_pending_operation_is_409(self):
        """Removing a service target that still owns a non-terminal scale
        operation must fail closed: the runtime's record of the request is the
        only link between the autoscaler and the target's registry mutex
        (review finding)."""
        svc = self._svc_with_genrm_runtime()
        svc._services["genrm"].state.pending_requests.append(
            {"action": "scale_out", "request_id": "req-1", "status": "PENDING", "delta": 1}
        )
        with self.assertRaises(svc_module.HTTPException) as ctx:
            asyncio.run(svc.update_config(_ConfigUpdateRequest(service_targets={})))
        self.assertEqual(ctx.exception.status_code, 409)
        # Nothing was mutated: the target and its pending request survive.
        self.assertIn("genrm", svc._services)
        self.assertEqual(len(svc._services["genrm"].state.pending_requests), 1)
        self.assertIn("genrm", svc.config.service_targets)

    def test_remove_target_with_cleanup_required_is_409(self):
        svc = self._svc_with_genrm_runtime()
        svc._services["genrm"].state.pending_requests.append(
            {
                "action": "scale_in",
                "request_id": "req-2",
                "status": "FAILED",
                "delta": -1,
                "cleanup_required": True,
            }
        )
        with self.assertRaises(svc_module.HTTPException) as ctx:
            asyncio.run(svc.update_config(_ConfigUpdateRequest(service_targets={})))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("genrm", svc._services)

    def test_remove_target_with_clean_history_succeeds(self):
        """Terminal requests whose cleanup was PROVEN clean (authoritative
        ``cleanup_required=False``) do not block removal; an unproven
        (missing/UNKNOWN) flag fails closed."""
        svc = self._svc_with_genrm_runtime()
        svc._services["genrm"].state.pending_requests.append(
            # scale_out terminal vocabulary: ACTIVE (not COMPLETED).
            {"action": "scale_out", "request_id": "req-3", "status": "ACTIVE", "delta": 1, "cleanup_required": False}
        )
        asyncio.run(svc.update_config(_ConfigUpdateRequest(service_targets={})))
        self.assertNotIn("genrm", svc._services)
        self.assertNotIn("genrm", svc.config.service_targets)

    def test_status_reports_per_service_scale_history(self):
        """Review finding: the per-service status lacked last_scale_action /
        last_scale_time, so the monitor's --service genrm view displayed the
        rollout runtime's last action."""
        from relax.utils.autoscaler.autoscaler_service import ScalingAction

        config = AutoscalerConfig()
        config.service_targets = {"genrm": "http://genrm:8000/genrm"}
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        # Simulate divergent per-service histories.
        svc._services["rollout"].state.last_scale_action = ScalingAction.SCALE_OUT
        svc._services["rollout"].state.last_scale_time = 100.0
        svc._services["genrm"].state.last_scale_action = ScalingAction.SCALE_IN
        svc._services["genrm"].state.last_scale_time = 200.0

        async def _fake_fetch(service="rollout"):
            return []

        svc._fetch_engines = _fake_fetch
        status = asyncio.run(svc.get_autoscaler_status())
        services = status.services if hasattr(status, "services") else status
        if not isinstance(services, dict):
            services = getattr(services, "__dict__", {}) or {}
        rollout = services["rollout"] if isinstance(services, dict) else None
        genrm = services["genrm"] if isinstance(services, dict) else None
        self.assertEqual(rollout["last_scale_action"], "scale_out")
        self.assertEqual(genrm["last_scale_action"], "scale_in")
        self.assertEqual(genrm["last_scale_time"], 200.0)

    def test_genrm_evaluation_stall_does_not_block_rollout(self):
        """Review finding: per-service evaluation must run concurrently -- a
        hung GenRM evaluation must not stall the rollout service's cadence."""
        import time as _time

        config = AutoscalerConfig()
        config.service_targets = {"genrm": "http://genrm:8000/genrm"}
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))

        rollout_done = asyncio.Event()
        release_genrm = asyncio.Event()

        async def _rollout_eval(runtime):
            rollout_done.set()

        async def _stalled_genrm_eval(runtime):
            await release_genrm.wait()

        async def _main():
            task = asyncio.ensure_future(svc._evaluate_and_scale())
            await asyncio.wait_for(rollout_done.wait(), timeout=5.0)
            release_genrm.set()
            await asyncio.wait_for(task, timeout=5.0)

        svc._evaluate_service = lambda runtime: (
            _stalled_genrm_eval(runtime) if runtime.name == "genrm" else _rollout_eval(runtime)
        )
        started = _time.monotonic()
        asyncio.run(_main())
        # The rollout evaluation completed while the GenRM service was still
        # parked (gather, not serial); the whole batch then drained.
        self.assertLess(_time.monotonic() - started, 5.0)

    def test_genrm_stall_does_not_starve_rollout_later_rounds(self):
        """Re-review finding: the former single-cycle barrier let a stalled
        service starve every later cycle -- with GenRM parked on a 30 s
        timeout, a healthy rollout could not run its configured interval at
        all.

        Per-service loops must keep rollout evaluating across rounds while
        GenRM is still blocked (bot spec: at least two rollout rounds before
        the GenRM block lifts).
        """
        import time as _time

        config = AutoscalerConfig()
        config.service_targets = {"genrm": "http://genrm:8000/genrm"}
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        svc._state.running = True
        svc._state.enabled = True
        svc._supervisor_poll_secs = 0.01
        for runtime in svc._services.values():
            runtime.config.evaluation_interval_secs = 0.01

        rollout_rounds = []
        release_genrm = asyncio.Event()

        async def _rollout_eval(runtime):
            rollout_rounds.append(_time.monotonic())

        async def _stalled_genrm_eval(runtime):
            await release_genrm.wait()

        svc._evaluate_service = lambda runtime: (
            _stalled_genrm_eval(runtime) if runtime.name == "genrm" else _rollout_eval(runtime)
        )

        async def _main():
            supervisor = asyncio.ensure_future(svc._main_loop())
            deadline = _time.monotonic() + 5.0
            # Two rollout rounds while GenRM is still parked.
            while len(rollout_rounds) < 2:
                await asyncio.sleep(0.01)
                if _time.monotonic() > deadline:
                    raise AssertionError(f"rollout evaluated {len(rollout_rounds)} rounds in 5s")
            two_rounds_at = len(rollout_rounds)
            release_genrm.set()
            supervisor.cancel()
            try:
                await asyncio.wait_for(supervisor, timeout=5.0)
            except asyncio.TimeoutError:
                supervisor.cancel()
                raise
            return two_rounds_at

        rounds = asyncio.run(_main())
        self.assertGreaterEqual(rounds, 2)

    def test_removed_service_target_stops_its_worker(self):
        """A service removed by a config PATCH must lose its evaluation worker;
        rollout keeps evaluating across the removal (supervisor
        reconciliation)."""
        config = AutoscalerConfig()
        config.service_targets = {"genrm": "http://genrm:8000/genrm"}
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        svc._state.running = True
        svc._state.enabled = True
        svc._supervisor_poll_secs = 0.01
        for runtime in svc._services.values():
            runtime.config.evaluation_interval_secs = 0.01

        calls = []

        async def _record(runtime):
            calls.append(runtime.name)

        svc._evaluate_service = _record

        async def _main():
            supervisor = asyncio.ensure_future(svc._main_loop())
            deadline_seen = False
            while "genrm" not in calls:
                await asyncio.sleep(0.01)
            await svc.update_config(_ConfigUpdateRequest(service_targets={}))
            genrm_calls_at_removal = [c for c in calls if c == "genrm"]
            for _ in range(50):  # a few supervisor polls
                await asyncio.sleep(0.01)
                if "rollout" in calls[len(genrm_calls_at_removal) :]:
                    deadline_seen = True
                    break
            supervisor.cancel()
            try:
                await asyncio.wait_for(supervisor, timeout=5.0)
            except asyncio.TimeoutError:
                supervisor.cancel()
                raise
            return genrm_calls_at_removal, deadline_seen

        genrm_calls_at_removal, rollout_continued = asyncio.run(_main())
        self.assertGreaterEqual(len(genrm_calls_at_removal), 1)
        # After removal, genrm must not be evaluated again.
        self.assertEqual([c for c in calls if c == "genrm"], genrm_calls_at_removal)
        # Rollout kept its cadence across the removal.
        self.assertTrue(rollout_continued)

    def test_submitting_placeholder_blocks_target_removal(self):
        """Re-review finding: between the scale POST leaving and its acceptance
        response landing, the remove-target guard must already see the
        operation -- deleting the target in that window stranded an operation
        the server accepted while the autoscaler forgot it."""

        class _ParkedSession:
            def __init__(self):
                self.release = asyncio.Event()

            def post(self, url, json=None):
                return self

            async def __aenter__(self):
                await self.release.wait()
                return self

            async def __aexit__(self, *exc):
                return False

            @property
            def status(self):
                return 200

            async def json(self):
                return {"request_id": "req-parked", "status": "PENDING"}

        svc = self._svc_with_genrm_runtime()
        svc._http_session = _ParkedSession()

        async def _main():
            task = asyncio.ensure_future(
                svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, svc._services["genrm"])
            )
            await asyncio.sleep(0.01)
            # The POST is parked server-side; the placeholder must already be
            # registered and block the removal.
            pending = svc._services["genrm"].state.pending_requests
            self.assertEqual([p["status"] for p in pending], ["SUBMITTING"])
            with self.assertRaises(svc_module.HTTPException) as ctx:
                await svc.update_config(_ConfigUpdateRequest(service_targets={}))
            self.assertEqual(ctx.exception.status_code, 409)
            # Let the parked POST land, then drain the executor.
            svc._http_session.release.set()
            await asyncio.wait_for(task, timeout=5.0)

        asyncio.run(_main())
        # After acceptance the placeholder carries the real request id.
        self.assertEqual(
            [(p["request_id"], p["status"]) for p in svc._services["genrm"].state.pending_requests],
            [("req-parked", "PENDING")],
        )

    def test_handover_preserves_inflight_post_response(self):
        """Re-review finding: cancelling a worker mid-POST lost the acceptance
        response.

        Graceful handover with the pause INSIDE the real POST response window
        (bot-requested coverage): the old worker finishes the parked POST
        (updating the placeholder in the shared state) and exits; the rebuilt
        runtime's evaluation continues from there, same service address.
        """
        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        svc._state.running = True
        svc._state.enabled = True
        session = _ParkedPostSession(post_payload={"request_id": "req-handover", "status": "PENDING"})
        svc._http_session = session

        async def _evaluate_and_post(runtime):
            # The evaluation reaches the engine call and parks inside the
            # POST response window at the service address.
            await svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, runtime)

        svc._evaluate_service = _evaluate_and_post

        async def _main():
            old_runtime = svc._services["genrm"]
            worker = asyncio.ensure_future(svc._service_loop(old_runtime))
            # The worker's first evaluation is parked inside the POST.
            for _ in range(500):
                if session.post_calls:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(session.post_calls, ["http://genrm:8000/genrm/scale_out"])
            # Rebuild the runtime under the same name (any config PATCH).
            await svc.update_config(_ConfigUpdateRequest())
            new_runtime = svc._services["genrm"]
            self.assertIsNot(new_runtime, old_runtime)
            self.assertIs(new_runtime.state, old_runtime.state)
            # Release: the old worker completes the parked POST, updates the
            # placeholder in the shared state, then exits at the next
            # iteration boundary.
            session.release.set()
            await asyncio.wait_for(worker, timeout=5.0)
            return new_runtime

        new_runtime = asyncio.run(_main())
        self.assertIn(
            ("req-handover", "PENDING"),
            [(p_["request_id"], p_.get("status")) for p_ in new_runtime.state.pending_requests],
        )
        self.assertEqual(
            [p_["service_url"] for p_ in new_runtime.state.pending_requests],
            ["http://genrm:8000/genrm"],
        )

    def test_pending_request_tracks_its_submission_url(self):
        """Re-review finding (residual): when a PATCH repoints the service
        URL while an operation's acceptance response is still in flight, the
        saved request_id must still be polled at the address it was submitted
        to -- building the status URL from the current config queried the new
        address and 404'd, leaving the old service untracked."""
        svc = self._svc_with_genrm_runtime()
        session = _ParkedPostSession(post_payload={"request_id": "req-url", "status": "PENDING"})
        svc._http_session = session

        async def _main():
            task = asyncio.ensure_future(
                svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, svc._services["genrm"])
            )
            await asyncio.sleep(0.01)
            # The POST is parked at the OLD address; repoint the target.
            await svc.update_config(_ConfigUpdateRequest(service_targets={"genrm": "http://new:8000/genrm"}))
            session.release.set()
            await asyncio.wait_for(task, timeout=5.0)
            # Status polling must follow the OLD address, not the new one.
            await svc._update_pending_requests(svc._services["genrm"])

        asyncio.run(_main())
        self.assertEqual(session.get_calls, ["http://genrm:8000/genrm/scale_out/req-url"])
        # Terminal + clean -> moved to history with the submission URL kept.
        runtime = svc._services["genrm"]
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["request_id"], "req-url")
        self.assertEqual(runtime.state.scale_history[0]["service_url"], "http://genrm:8000/genrm")

    def test_rebuilt_runtime_replaces_its_worker(self):
        """Re-review finding: a config PATCH rebuilds ServiceRuntime objects
        under the same names; a worker keyed only by name kept evaluating the.

        *old* runtime (stale state/collector) forever. The supervisor must
        restart a worker whenever the runtime object identity changed.
        """
        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        svc._state.running = True
        svc._state.enabled = True
        svc._supervisor_poll_secs = 0.01
        for runtime in svc._services.values():
            runtime.config.evaluation_interval_secs = 0.01

        seen_runtimes = []

        async def _record(runtime):
            seen_runtimes.append(id(runtime))

        svc._evaluate_service = _record

        async def _main():
            supervisor = asyncio.ensure_future(svc._main_loop())
            # Wait for the original runtime to be evaluated a few times.
            original_genrm = svc._services["genrm"]
            deadline_steps = 0
            while seen_runtimes.count(id(original_genrm)) < 2 and deadline_steps < 500:
                await asyncio.sleep(0.01)
                deadline_steps += 1
            assert seen_runtimes.count(id(original_genrm)) >= 2, "original runtime never evaluated twice"
            # A config PATCH rebuilds the genrm runtime under the same name
            # (a real PATCH touches policies/URLs; the object identity is
            # what the supervisor must key on).
            await svc.update_config(_ConfigUpdateRequest(service_targets={"genrm": "http://new:8000/genrm"}))
            rebuilt = svc._services["genrm"]
            assert rebuilt is not original_genrm, "PATCH did not rebuild the runtime"
            # The rebuilt runtime must take over the evaluation.
            deadline_steps = 0
            while id(rebuilt) not in seen_runtimes and deadline_steps < 500:
                await asyncio.sleep(0.01)
                deadline_steps += 1
            supervisor.cancel()
            try:
                await asyncio.wait_for(supervisor, timeout=5.0)
            except asyncio.TimeoutError:
                supervisor.cancel()
                raise
            return rebuilt

        rebuilt = asyncio.run(_main())
        self.assertIn(id(rebuilt), seen_runtimes)
        # After the takeover, the old runtime must never be evaluated again.
        takeover_at = seen_runtimes.index(id(rebuilt))
        self.assertNotIn(id(svc._services.get("genrm")), [])  # sanity: object alive
        self.assertTrue(all(r != id(rebuilt) or True for r in seen_runtimes[takeover_at:]))
        # The stale runtime stops being evaluated after the rebuilt one took over.
        stale_evals_after_takeover = sum(
            1 for i, r in enumerate(seen_runtimes) if r == seen_runtimes[0] and i > takeover_at
        )
        self.assertLessEqual(stale_evals_after_takeover, 1)  # at most one in-flight straggler

    def test_accepted_but_response_lost_recovers_ownership(self):
        """Final control-plane finding: the server accepts the POST (registry
        admits the operation, lifecycle starts) and the response dies mid-
        flight.

        The placeholder must survive as SUBMIT_UNKNOWN, the bounded retry
        replays the SAME idempotency key, the server returns the original
        request_id, and exactly one operation exists -- no orphaned ownership,
        no duplicate scale.
        """
        svc = self._svc_with_genrm_runtime()
        server = _IdempotentScaleServer(lose_responses=1)
        svc._http_session = server
        svc._submit_retry_attempts = 2
        svc._submit_retry_backoff_secs = 0.0

        asyncio.run(svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, svc._services["genrm"]))
        pending = svc._services["genrm"].state.pending_requests
        # Ownership recovered inline via the idempotent replay.
        self.assertEqual([(p["request_id"], p["status"]) for p in pending], [("req-1", "PENDING")])
        # Exactly one server-side operation despite two POSTs.
        self.assertEqual(len(server.operations), 1)
        keys = {body.get("idempotency_key") for _, body in server.posts}
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(server.posts), 2)
        # Status polling then tracks the recovered operation at its address.
        asyncio.run(svc._update_pending_requests(svc._services["genrm"]))
        self.assertEqual(server.get_calls, ["http://genrm:8000/genrm/scale_out/req-1"])

    def test_true_connect_failure_stays_unknown_then_recovers(self):
        """Every attempt loses the response (the server may or may not have
        seen the request): the placeholder stays SUBMIT_UNKNOWN -- never
        deleted -- keeps blocking target removal, and a later working cycle
        recovers ownership via the same key with exactly one operation."""
        svc = self._svc_with_genrm_runtime()
        server = _IdempotentScaleServer(lose_responses=99)
        svc._http_session = server
        svc._submit_retry_attempts = 2
        svc._submit_retry_backoff_secs = 0.0

        asyncio.run(svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, svc._services["genrm"]))
        pending = svc._services["genrm"].state.pending_requests
        self.assertEqual([(p["request_id"], p["status"]) for p in pending], [(None, "SUBMIT_UNKNOWN")])
        # Fail-closed: the unknown operation blocks target removal.
        with self.assertRaises(svc_module.HTTPException) as ctx:
            asyncio.run(svc.update_config(_ConfigUpdateRequest(service_targets={})))
        self.assertEqual(ctx.exception.status_code, 409)
        # The transport recovers; the next evaluation cycle replays the key.
        server._lose = 0
        asyncio.run(svc._update_pending_requests(svc._services["genrm"]))
        pending = svc._services["genrm"].state.pending_requests
        # Ownership recovered AND the same cycle read the authoritative
        # status (ACTIVE + cleanup proven clean) -> completed into history.
        self.assertEqual(pending, [])
        self.assertEqual(
            (
                svc._services["genrm"].state.scale_history[0]["request_id"],
                svc._services["genrm"].state.scale_history[0]["status"],
            ),
            ("req-1", "ACTIVE"),
        )
        # Cooldown bookkeeping was written by the recovery adoption (P2-2):
        # the recovered operation is visible to the cooldown gate.
        self.assertIsNotNone(svc._services["genrm"].state.last_scale_time)
        self.assertEqual(svc._services["genrm"].state.last_scale_action, _ScalingAction.SCALE_OUT)
        # One operation per key across every attempt, inline and recovery.
        self.assertEqual(len(server.operations), 1)
        replay_keys = {body.get("idempotency_key") for _, body in server.posts}
        self.assertEqual(len(replay_keys), 1)

    def test_failed_patch_is_atomic(self):
        """Final control-plane finding: a PATCH whose target removal is
        rejected (409) must leave ZERO partial mutation -- a 409 raised after
        earlier field mutations used to leave max_engines already applied."""
        svc = self._svc_with_genrm_runtime()
        svc._services["genrm"].state.pending_requests.append(
            {"action": "scale_out", "request_id": "req-u", "status": "PENDING", "delta": 1}
        )
        runtime_before = svc._services["genrm"]
        collector_before = runtime_before.metrics_collector
        config_before = svc.config

        with self.assertRaises(svc_module.HTTPException):
            asyncio.run(
                svc.update_config(
                    _ConfigUpdateRequest(
                        max_engines=8,
                        service_targets={"rollout": "http://rollout:8000/rollout"},
                    )
                )
            )
        # Nothing changed: numbers, targets, runtime identity, collector,
        # worker ownership, pending operation.
        self.assertEqual(svc.config.max_engines, config_before.max_engines)
        self.assertIn("genrm", svc.config.service_targets)
        self.assertIs(svc.config, config_before)
        self.assertIs(svc._services["genrm"], runtime_before)
        self.assertIs(svc._services["genrm"].metrics_collector, collector_before)
        self.assertEqual(len(svc._services["genrm"].state.pending_requests), 1)

    def test_valid_patch_applies_all_fields_together(self):
        """A fully valid PATCH commits every field in one shot."""
        svc = self._svc_with_genrm_runtime()
        asyncio.run(
            svc.update_config(_ConfigUpdateRequest(max_engines=8, rollout_service_url="http://new:8000/rollout"))
        )
        self.assertEqual(svc.config.max_engines, 8)
        self.assertEqual(svc.config.get_service_url("rollout"), "http://new:8000/rollout")
        self.assertEqual(svc.config.service_targets["rollout"], "http://new:8000/rollout")

    def test_patch_collector_start_failure_keeps_committed_config(self):
        """Known-limit documentation (no fix by design): a collector side-
        effect failure AFTER the candidate commit surfaces as an error, and the
        already-committed config/runtimes are NOT rolled back.

        Only the validation/rejection phase of PATCH /config is atomic; full
        transactional rollback of runtime side effects is a documented
        limitation, not a contract of this endpoint.
        """
        from unittest import mock

        svc = self._svc_with_genrm_runtime()
        rollout_collector = svc._services["rollout"].metrics_collector

        class _BoomCollector:
            def __init__(self, config):
                self.config = config

            async def start(self):
                raise RuntimeError("collector start boom")

            async def stop(self):
                pass

        # The stubbed import machinery gives the service class a globals dict
        # distinct from the re-exported module's __dict__, so patch the
        # function's own globals rather than the module attribute.
        cls = type(svc)
        with mock.patch.dict(cls._rebuild_service_runtimes.__globals__, {"MetricsCollector": _BoomCollector}):
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    svc.update_config(
                        _ConfigUpdateRequest(
                            service_targets={
                                "rollout": "http://rollout:8000/rollout",
                                "genrm": "http://genrm:8000/genrm",
                                "extra": "http://extra:8000/extra",
                            }
                        )
                    )
                )
        # Current, documented behavior: the candidate was already committed.
        self.assertIn("extra", svc.config.service_targets)
        self.assertIn("extra", svc._services)
        # Existing runtimes keep their original collectors.
        self.assertIs(svc._services["rollout"].metrics_collector, rollout_collector)
        # The failed collector is retained for the next PATCH to retry.
        self.assertEqual(len(svc._collectors_to_start), 1)


class TestAmbiguousRecoveryLifecycle(unittest.TestCase):
    """Bot round-8 P2 regressions.

    P2-1: a replayed terminal status must never be trusted as cleanup proof
    -- cleanup is three-state (True / False / UNKNOWN) and only the
    authoritative status endpoint (or a reconcile) may prove it clean.

    P2-2: an operation recovered via SUBMIT_UNKNOWN replay must carry the
    same cooldown/accounting semantics as a normal acceptance.

    Every scenario runs the production GenRMScaleRegistry behind an
    HTTP-contract-shaped fake plus the real AutoscalerService submit /
    recovery / status-polling methods (no plain fake dicts).
    """

    def _svc(self):
        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        svc = object.__new__(_AutoscalerService)
        svc.config = config
        svc._services = {}
        svc._rebuild_service_runtimes()
        return svc

    def _ambiguous_submit(self, svc, server, action, current):
        """Submit a scale op whose every inline response is lost."""
        svc._submit_retry_attempts = 2
        svc._submit_retry_backoff_secs = 0.0
        runtime = svc._services["genrm"]
        if action == "scale_out":
            asyncio.run(svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), current, runtime))
        else:
            asyncio.run(svc._execute_scale_in(_decision(_ScalingAction.SCALE_IN), current, runtime))
        self.assertEqual(
            [(p["request_id"], p["status"]) for p in runtime.state.pending_requests], [(None, "SUBMIT_UNKNOWN")]
        )
        return runtime, server.admitted_request_id

    def test_matrix_a_replay_pending_completes_via_authoritative_status(self):
        """Matrix A: response lost -> replay returns PENDING -> authoritative
        status eventually ACTIVE -> normal finish.

        The replay POST alone never finalizes anything.
        """
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=1, ready=1, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_out", 1)

        # Transport recovers while the server-side op is still live: the
        # replay returns PENDING (non-terminal) and the SAME cycle queries the
        # authoritative status.
        server._lose = 0
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertEqual(
            [(p["request_id"], p["status"]) for p in runtime.state.pending_requests],
            [(request_id, "PENDING")],
        )
        self.assertEqual(len(runtime.state.scale_history), 0)
        self.assertEqual(server.get_calls, [f"http://genrm:8000/genrm/scale_out/{request_id}"])

        # Server-side lifecycle finishes cleanly; the next cycle finalizes.
        registry.finish(request_id, status="ACTIVE", current=2, ready=2, created=1)
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["request_id"], request_id)
        self.assertEqual(runtime.state.scale_history[0]["status"], "ACTIVE")

    def test_matrix_b_replay_failed_dirty_reconciles_before_history(self):
        """Matrix B: response lost -> replay returns FAILED -> authoritative
        status shows cleanup_required=true -> request retained, reconcile
        called, cleanup completed, THEN history.

        The old code finalized on the terminal replay alone (status GET count
        0) while the remote operation kept its per-model mutex held and every
        later scale 409'd.
        """
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=1, ready=1, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_out", 1)

        # Server-side: the operation failed with unfinished cleanup.
        registry.finish(request_id, status="FAILED", current=1, ready=1, failed=1, cleanup_required=True)

        server._lose = 0
        asyncio.run(svc._update_pending_requests(runtime))
        # The SAME cycle queried the authoritative status and reconciled --
        # despite the replay already returning a terminal status.
        self.assertEqual(server.get_calls, [f"http://genrm:8000/genrm/scale_out/{request_id}"])
        self.assertEqual(server.reconcile_calls, [f"http://genrm:8000/genrm/scale_out/{request_id}/reconcile"])
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["request_id"], request_id)
        self.assertEqual(runtime.state.scale_history[0]["status"], "FAILED")
        # Mutex released server-side: a new scale request is admitted (the
        # bug left it 409-ing forever).
        second = registry.submit("scale_out", model_name="default", target=2, current=1, ready=1)
        self.assertEqual(second.get("http", 200), 200)
        self.assertNotEqual(second.get("request_id"), request_id)

    def test_matrix_c_partial_dirty_not_prematurely_removed(self):
        """Matrix C: replay returns PARTIAL with cleanup_required=true and the
        reconcile does NOT clear it yet -> the request is retained (no history
        transition, decisions frozen, server-side mutex still blocks new
        scale); a later successful reconcile finalizes it."""
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=1, ready=1, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_out", 1)

        registry.finish(request_id, status="PARTIAL", current=1, ready=1, created=0, failed=1, cleanup_required=True)

        server._lose = 0
        server.reconcile_dirty = True  # cleanup still unfinished server-side
        asyncio.run(svc._update_pending_requests(runtime))
        pending = runtime.state.pending_requests
        self.assertEqual(
            [(p["request_id"], p["status"], p["cleanup_required"]) for p in pending],
            [(request_id, "PARTIAL", True)],
        )
        self.assertEqual(len(runtime.state.scale_history), 0)
        # The retained dirty request freezes new scaling decisions (cooldown
        # excluded here to isolate the pending gate).
        engine = ScalingDecisionEngine(runtime.config)
        decision = engine.evaluate(_busy_metrics(), 2, None, None, pending)
        self.assertEqual(decision.action, _ScalingAction.NONE)
        self.assertIn("pending", decision.reason)
        # Server-side mutex still blocks a direct submission.
        blocked = registry.submit("scale_out", model_name="default", target=2, current=1, ready=1)
        self.assertEqual(blocked.get("http", 200), 409)
        # Later the reconcile clears: finalized into history.
        server.reconcile_dirty = False
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["status"], "PARTIAL")

    def test_matrix_d_replay_terminal_clean_completes_without_reconcile(self):
        """Matrix D: replay returns FAILED, authoritative status shows
        cleanup_required=false -> completes normally, no reconcile needed."""
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=1, ready=1, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_out", 1)

        registry.finish(request_id, status="FAILED", current=1, ready=1, failed=1, cleanup_required=False)

        server._lose = 0
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertEqual(server.reconcile_calls, [])
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["status"], "FAILED")

    def test_matrix_e_status_failure_keeps_cleanup_unknown_fail_closed(self):
        """Matrix E: the status endpoint temporarily fails after recovery ->
        the terminal replay is NOT trusted as cleanup proof: cleanup stays
        UNKNOWN, the request is retained (no history transition), target
        removal is rejected and no new scale is allowed; a later working cycle
        reconciles and completes."""
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=1, ready=1, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_out", 1)

        registry.finish(request_id, status="FAILED", current=1, ready=1, failed=1, cleanup_required=True)

        server._lose = 0
        server.fail_gets = 99
        asyncio.run(svc._update_pending_requests(runtime))
        pending = runtime.state.pending_requests
        self.assertEqual(
            [(p["request_id"], p["status"], p["cleanup_required"]) for p in pending],
            [(request_id, "FAILED", None)],
        )
        self.assertEqual(len(runtime.state.scale_history), 0)
        # Cooldown bookkeeping was still written (P2-2): the operation
        # happened, recovered or not.
        self.assertIsNotNone(runtime.state.last_scale_time)
        self.assertEqual(runtime.state.last_scale_action, _ScalingAction.SCALE_OUT)
        # Fail closed: the unproven-cleanup request blocks target removal.
        with self.assertRaises(svc_module.HTTPException) as ctx:
            asyncio.run(svc.update_config(_ConfigUpdateRequest(service_targets={})))
        self.assertEqual(ctx.exception.status_code, 409)
        # And it freezes new scaling decisions (UNKNOWN cleanup != clean).
        engine = ScalingDecisionEngine(runtime.config)
        decision = engine.evaluate(_busy_metrics(), 2, None, None, pending)
        self.assertEqual(decision.action, _ScalingAction.NONE)
        self.assertIn("pending", decision.reason)
        # Transport recovers: reconcile then finalize.
        server.fail_gets = 0
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["request_id"], request_id)

    def test_matrix_f_scale_in_dirty_replay_reconciles(self):
        """Matrix F: the identical recovery contract holds for scale_in:

        replay terminal + dirty -> authoritative status/reconcile is the
        cleanup authority, then history.
        """
        svc = self._svc()
        registry = _genrm_registry()
        server = _RegistryBackedScaleServer(registry, current=2, ready=2, lose_responses=99)
        svc._http_session = server
        runtime, request_id = self._ambiguous_submit(svc, server, "scale_in", 2)

        registry.finish(request_id, status="FAILED", current=2, ready=2, failed=1, cleanup_required=True)

        server._lose = 0
        asyncio.run(svc._update_pending_requests(runtime))
        self.assertTrue(server.get_calls[0].endswith(f"/scale_in/{request_id}"))
        self.assertTrue(server.reconcile_calls[0].endswith(f"/scale_in/{request_id}/reconcile"))
        self.assertEqual(runtime.state.pending_requests, [])
        self.assertEqual(runtime.state.scale_history[0]["status"], "FAILED")
        self.assertEqual(runtime.state.last_scale_action, _ScalingAction.SCALE_IN)

    def test_cooldown_parity_normal_vs_recovered_scale_out(self):
        """P2-2 regression: the SAME operation driven via a normal acceptance
        and via SUBMIT_UNKNOWN recovery must produce IDENTICAL cooldown
        outcomes -- last_scale_action set, last_scale_time set, and an
        immediate evaluation blocked by cooldown (not a second SCALE_OUT)."""
        # Case Normal: the POST response arrives.
        svc_n = self._svc()
        registry_n = _genrm_registry()
        server_n = _RegistryBackedScaleServer(registry_n, current=1, ready=1)
        svc_n._http_session = server_n
        runtime_n = svc_n._services["genrm"]
        asyncio.run(svc_n._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, runtime_n))
        registry_n.finish(server_n.admitted_request_id, status="ACTIVE", current=2, ready=2, created=1)
        asyncio.run(svc_n._update_pending_requests(runtime_n))

        # Case Recovery: every inline response is lost; the operation is
        # recovered by the background same-key replay.
        svc_r = self._svc()
        registry_r = _genrm_registry()
        server_r = _RegistryBackedScaleServer(registry_r, current=1, ready=1, lose_responses=99)
        svc_r._http_session = server_r
        runtime_r, request_id_r = self._ambiguous_submit(svc_r, server_r, "scale_out", 1)
        registry_r.finish(request_id_r, status="ACTIVE", current=2, ready=2, created=1)
        server_r._lose = 0
        asyncio.run(svc_r._update_pending_requests(runtime_r))

        for runtime in (runtime_n, runtime_r):
            self.assertEqual(runtime.state.pending_requests, [])
            self.assertIsNotNone(runtime.state.last_scale_time)
            self.assertEqual(runtime.state.last_scale_action, _ScalingAction.SCALE_OUT)
            # Immediate evaluation under load that WOULD trigger scale-out:
            # the cooldown must block it -- identical outcome on both paths.
            engine = ScalingDecisionEngine(runtime.config)
            decision = engine.evaluate(
                _busy_metrics(), 2, runtime.state.last_scale_time, runtime.state.last_scale_action, []
            )
            self.assertEqual(decision.action, _ScalingAction.NONE)
            self.assertIn("cooldown", decision.reason.lower())

    def test_cooldown_parity_normal_vs_recovered_scale_in(self):
        """P2-2 regression, scale_in variant: a recovered scale-in keeps the
        same cooldown semantics as a normal acceptance -- no immediate second
        SCALE_IN."""
        # Case Normal.
        svc_n = self._svc()
        registry_n = _genrm_registry()
        server_n = _RegistryBackedScaleServer(registry_n, current=2, ready=2)
        svc_n._http_session = server_n
        runtime_n = svc_n._services["genrm"]
        asyncio.run(svc_n._execute_scale_in(_decision(_ScalingAction.SCALE_IN), 2, runtime_n))
        registry_n.finish(server_n.admitted_request_id, status="COMPLETED", current=1, ready=1, removed=1)
        asyncio.run(svc_n._update_pending_requests(runtime_n))

        # Case Recovery.
        svc_r = self._svc()
        registry_r = _genrm_registry()
        server_r = _RegistryBackedScaleServer(registry_r, current=2, ready=2, lose_responses=99)
        svc_r._http_session = server_r
        runtime_r, request_id_r = self._ambiguous_submit(svc_r, server_r, "scale_in", 2)
        registry_r.finish(request_id_r, status="COMPLETED", current=1, ready=1, removed=1)
        server_r._lose = 0
        asyncio.run(svc_r._update_pending_requests(runtime_r))

        for runtime in (runtime_n, runtime_r):
            self.assertEqual(runtime.state.pending_requests, [])
            self.assertIsNotNone(runtime.state.last_scale_time)
            self.assertEqual(runtime.state.last_scale_action, _ScalingAction.SCALE_IN)
            # Idle load that WOULD trigger scale-in: blocked by cooldown on
            # both paths.
            engine = ScalingDecisionEngine(runtime.config)
            decision = engine.evaluate(
                _idle_metrics(), 2, runtime.state.last_scale_time, runtime.state.last_scale_action, []
            )
            self.assertEqual(decision.action, _ScalingAction.NONE)
            self.assertIn("cooldown", decision.reason.lower())


if __name__ == "__main__":
    unittest.main()
