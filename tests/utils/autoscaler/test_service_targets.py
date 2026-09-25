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


if __name__ == "__main__":
    unittest.main()
