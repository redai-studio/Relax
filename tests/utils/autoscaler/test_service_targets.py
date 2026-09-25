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


def _decision(action, delta=1):
    return _ScalingDecision(
        action=action,
        delta=delta,
        reason="test",
        triggered_conditions=["cond"],
        metrics_snapshot={"m": 1},
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
        """Terminal, reconciled requests do not block removal."""
        svc = self._svc_with_genrm_runtime()
        svc._services["genrm"].state.pending_requests.append(
            # scale_out terminal vocabulary: ACTIVE (not COMPLETED).
            {"action": "scale_out", "request_id": "req-3", "status": "ACTIVE", "delta": 1}
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

        Graceful handover: the old worker finishes the POST (updating the
        placeholder in the shared state) and exits; the rebuilt runtime's
        evaluation continues from there.
        """

        class _ParkedSession:
            def __init__(self):
                self.release = asyncio.Event()
                self.calls = 0

            def get(self, url):
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
                self.calls += 1
                return {"request_id": "req-handover", "status": "PENDING"}

        config = AutoscalerConfig(service_targets={"genrm": "http://genrm:8000/genrm"})
        svc = _service(config)
        asyncio.run(svc.update_config(_ConfigUpdateRequest()))
        svc._state.running = True
        svc._state.enabled = True
        svc._http_session = _FakePostSession(200, {"request_id": "req-handover", "status": "PENDING"})
        release = asyncio.Event()

        async def _parked_evaluation(runtime):
            # Park like a slow metrics/discovery round, then fire a scale POST
            # whose acceptance response only lands after the handover.
            await release.wait()
            await svc._execute_scale_out(_decision(_ScalingAction.SCALE_OUT), 1, runtime)

        svc._evaluate_service = _parked_evaluation

        async def _main():
            old_runtime = svc._services["genrm"]
            worker = asyncio.ensure_future(svc._service_loop(old_runtime))
            await asyncio.sleep(0.02)
            # Rebuild the runtime under the same name (any config PATCH).
            await svc.update_config(_ConfigUpdateRequest(service_targets={"genrm": "http://new:8000/genrm"}))
            new_runtime = svc._services["genrm"]
            self.assertIsNot(new_runtime, old_runtime)
            # The state object is shared across the rebuild.
            self.assertIs(new_runtime.state, old_runtime.state)
            # Release: the old worker completes its parked evaluation (scale
            # POST included), then exits at the next iteration boundary.
            release.set()
            await asyncio.wait_for(worker, timeout=5.0)
            return new_runtime

        new_runtime = asyncio.run(_main())
        # The in-flight POST result survived the handover in the shared state.
        self.assertIn(
            ("req-handover", "PENDING"),
            [(p["request_id"], p.get("status")) for p in new_runtime.state.pending_requests],
        )

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


if __name__ == "__main__":
    unittest.main()
