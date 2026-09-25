# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the GenRM Serve scale endpoints (Task 4).

The endpoints are thin wrappers over ``GenRMScaleRegistry``; these tests
drive the real (deployment-unwrapped) ``GenRM`` class methods without Ray:
managers are ``SimpleNamespace`` fakes, engine discovery is injected per
instance, and the module is imported through ``_dep_stubs`` which prefers
real dependencies and only stubs the missing ones (fastapi/pydantic/ray and
the placement-group/tokenizer bridge modules).

Run: python -m unittest tests.components.test_genrm_scale_endpoints -v
"""

import asyncio
import unittest
from types import SimpleNamespace

import ray

from tests.utils._dep_stubs import import_genrm_component


genrm_module = import_genrm_component()

# The /metrics capacity query goes through ray.get with plain dicts from the
# SimpleNamespace manager fakes; a real ray.get would spin up a local Ray
# instance to reject them. Pass ray.get through for this module (same
# direct-attribute pattern as test_genrm_scale_watcher.py).
_orig_ray_get = ray.get


def _passthrough_ray_get(ref, timeout=None, **kwargs):
    return ref


def setUpModule():
    ray.get = _passthrough_ray_get


def tearDownModule():
    ray.get = _orig_ray_get


# Deployment-unwrapped class (the @serve.deployment wrapper stores it).
_GenRM = genrm_module.GenRM.func_or_class
_GenRMScaleRequest = genrm_module.GenRMScaleRequest
_GenRMScaleRegistry = genrm_module.GenRMScaleRegistry
_HTTPException = genrm_module.HTTPException

_ENGINES = [("192.0.2.1", 16001)]


def _fake_manager(hooks: dict = None):
    """A manager fake: only ``get_engine_hosts_ports`` and optional scale
    hooks."""
    manager = SimpleNamespace(
        get_engine_hosts_ports=SimpleNamespace(remote=lambda: _ENGINES),
    )
    for name, remote in (hooks or {}).items():
        setattr(manager, name, SimpleNamespace(remote=remote))
    return manager


def _replica(manager=None, engines=_ENGINES):
    replica = object.__new__(_GenRM)
    replica.genrm_managers = {"__default__": manager or _fake_manager()}
    replica._scale_registry = _GenRMScaleRegistry()
    replica._scale_registry.register_initial("__default__", 1)
    replica._logger_instance = None  # Base.__init__ normally sets this
    replica._engine_inflight = {}
    replica._engine_served = {}
    replica._engine_caches = {"__default__": genrm_module._EngineCacheState()}
    if engines is not None:
        replica._genrm_engine_list = lambda key: list(engines)
    return replica


def _run(coro):
    return asyncio.run(coro)


class TestScaleOutEndpoint(unittest.TestCase):
    def test_pending_with_manager_hook_detail_when_manager_lacks_hooks(self):
        """Admitted operation stays PENDING and the response carries
        detail='manager_scale_not_implemented' (GenRMManager has no scale
        hooks)."""
        replica = _replica()
        response = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2)))
        self.assertEqual(response.status, "PENDING")
        self.assertIsNotNone(response.request_id)
        self.assertEqual(response.current, 1)
        self.assertEqual(response.detail, "manager_scale_not_implemented")
        status = replica._scale_registry.get_status("scale_out", response.request_id)
        self.assertEqual(status["status"], "PENDING")
        self.assertEqual(status["detail"], "manager_scale_not_implemented")

    def test_pending_without_detail_when_manager_hook_exists(self):
        calls = []
        replica = _replica(manager=_fake_manager({"execute_genrm_scale_out": lambda rid: calls.append(rid)}))
        response = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2)))
        self.assertEqual(response.status, "PENDING")
        self.assertIsNone(response.detail)
        self.assertEqual(calls, [response.request_id])  # fire-and-forget hook invoked

    def test_noop_when_target_already_satisfied(self):
        replica = _replica(engines=[("192.0.2.1", 16001), ("192.0.2.2", 16002)])
        response = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2)))
        self.assertEqual(response.status, "NOOP")
        self.assertIsNone(response.request_id)  # NOOP carries no request_id (contract)
        self.assertEqual(response.current, 2)

    def test_idempotent_replay_returns_original_operation(self):
        replica = _replica()
        first = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2, idempotency_key="k")))
        replay = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2, idempotency_key="k")))
        self.assertEqual(replay.request_id, first.request_id)
        self.assertEqual(replay.status, "PENDING")

    def test_conflicting_fingerprint_is_409(self):
        replica = _replica()
        _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2, idempotency_key="k")))
        with self.assertRaises(_HTTPException) as ctx:
            _run(replica.scale_out(_GenRMScaleRequest(num_replicas=3, idempotency_key="k")))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_second_in_flight_request_is_409(self):
        replica = _replica()
        _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2)))
        with self.assertRaises(_HTTPException) as ctx:
            _run(replica.scale_out(_GenRMScaleRequest(num_replicas=3)))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_target_below_initial_is_400(self):
        replica = _replica()
        replica._scale_registry.register_initial("__default__", 2)
        with self.assertRaises(_HTTPException) as ctx:
            _run(replica.scale_in(_GenRMScaleRequest(num_replicas=1)))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_unknown_model_is_400(self):
        replica = _replica()
        with self.assertRaises(_HTTPException) as ctx:
            _run(replica.scale_out(_GenRMScaleRequest(model_name="nope", num_replicas=2)))
        self.assertEqual(ctx.exception.status_code, 400)


class TestScaleInEndpoint(unittest.TestCase):
    def test_pending_then_state_machine_completes(self):
        replica = _replica(engines=[("192.0.2.1", 16001), ("192.0.2.2", 16002)])
        response = _run(replica.scale_in(_GenRMScaleRequest(num_replicas=1)))
        self.assertEqual(response.status, "PENDING")
        self.assertEqual(response.detail, "manager_scale_not_implemented")
        request_id = response.request_id
        for status in ("DRAINING", "REMOVING"):
            replica._scale_registry.advance(request_id, status)
        result = replica._scale_registry.finish(request_id, status="COMPLETED", current=1, ready=1, removed=1)
        self.assertEqual(result["status"], "COMPLETED")

    def test_noop_when_target_at_current(self):
        replica = _replica(engines=[("192.0.2.1", 16001)])
        response = _run(replica.scale_in(_GenRMScaleRequest(num_replicas=1)))
        self.assertEqual(response.status, "NOOP")


class TestStatusEndpoints(unittest.TestCase):
    def test_unknown_request_id_is_404(self):
        replica = _replica()
        with self.assertRaises(_HTTPException) as ctx:
            _run(replica.get_scale_out_status("no-such-id"))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_status_reports_operation_fields(self):
        replica = _replica()
        created = _run(replica.scale_out(_GenRMScaleRequest(num_replicas=2, timeout_secs=90.0)))
        result = _run(replica.get_scale_out_status(created.request_id))
        self.assertEqual(result.request_id, created.request_id)
        self.assertEqual(result.status, "PENDING")
        self.assertEqual(result.target, 2)
        self.assertEqual(result.timeout_secs, 90.0)
        self.assertEqual(result.model_name, "__default__")


class TestEnginesEndpoint(unittest.TestCase):
    def test_single_instance_flat_shape(self):
        replica = _replica(engines=[("192.0.2.1", 16001), ("192.0.2.2", 16002)])
        result = _run(replica.get_engines())
        self.assertEqual(result["service"], "genrm")
        self.assertEqual(result["current"], 2)
        self.assertEqual(result["ready"], 2)
        self.assertEqual(
            result["engines"],
            [
                {"host": "192.0.2.1", "port": 16001, "inflight": 0, "served": 0},
                {"host": "192.0.2.2", "port": 16002, "inflight": 0, "served": 0},
            ],
        )

    def test_multi_instance_nested_shape(self):
        replica = object.__new__(_GenRM)
        replica.genrm_managers = {
            "quality": _fake_manager(),
            "safety": _fake_manager(),
        }
        replica._logger_instance = None
        replica._engine_inflight = {}
        replica._engine_served = {}
        replica._genrm_engine_list = lambda key: [("192.0.2.1", 16001)] if key == "quality" else []
        result = _run(replica.get_engines())
        self.assertEqual(result["service"], "genrm")
        self.assertEqual(result["instances"]["quality"]["current"], 1)
        self.assertEqual(result["instances"]["safety"]["current"], 0)

    def test_scale_routes_by_model_name(self):
        """Multi-instance: model_name selects the instance and its registry slot."""
        replica = object.__new__(_GenRM)
        replica.genrm_managers = {"quality": _fake_manager(), "safety": _fake_manager()}
        replica._scale_registry = _GenRMScaleRegistry()
        replica._scale_registry.register_initial("quality", 1)
        replica._scale_registry.register_initial("safety", 1)
        replica._logger_instance = None
        replica._genrm_engine_list = lambda key: [("192.0.2.1", 16001)]

        response = _run(replica.scale_out(_GenRMScaleRequest(model_name="quality", num_replicas=2)))
        self.assertEqual(response.status, "PENDING")
        status = replica._scale_registry.get_status("scale_out", response.request_id)
        self.assertEqual(status["model_name"], "quality")


class TestMetricsDynamicCapacity(unittest.TestCase):
    """/metrics must report the manager's live capacity, not the startup
    spec: after an elastic scale-out the engine count changes at runtime."""

    _SPEC = {"model_path": "/model", "num_gpus": 1, "num_gpus_per_engine": 1}

    def _replica_with_spec(self, manager):
        replica = _replica(manager=manager)
        replica.instance_specs = {"__default__": dict(self._SPEC)}
        return replica

    def test_metrics_reflects_manager_capacity(self):
        capacity = {"current": 2, "ready": 2, "occupied": 2, "pending_cleanup": 1}
        manager = _fake_manager({"get_engine_capacity": lambda: capacity})
        out = _run(self._replica_with_spec(manager).metrics())
        self.assertEqual(out["num_engines"], 2)  # live current, not startup spec (1)
        self.assertEqual(out["ready_engines"], 2)
        self.assertEqual(out["occupied"], 2)
        self.assertEqual(out["pending_cleanup"], 1)
        self.assertNotIn("capacity_error", out)

    def test_metrics_falls_back_to_startup_count_on_query_failure(self):
        def boom():
            raise RuntimeError("manager unreachable")

        manager = _fake_manager({"get_engine_capacity": boom})
        out = _run(self._replica_with_spec(manager).metrics())
        self.assertEqual(out["num_engines"], 1)  # startup spec fallback
        self.assertIn("capacity_error", out)
        # No fabricated live numbers alongside the fallback.
        self.assertNotIn("ready_engines", out)
        self.assertNotIn("pending_cleanup", out)

    def test_metrics_without_capacity_hook_keeps_startup_count(self):
        replica = self._replica_with_spec(_fake_manager())  # no get_engine_capacity hook
        out = _run(replica.metrics())
        self.assertEqual(out["num_engines"], 1)
        self.assertNotIn("capacity_error", out)


class TestGenerateEngineAttribution(unittest.TestCase):
    """/generate must attribute each reply to the engine that served it and
    surface finish metadata — the basis for per-engine reward-consistency
    evidence and truncation checks."""

    def test_call_engine_stashes_final_engine(self):
        replica = _replica()

        async def fake_tracked(route_key, key, host, port, inflight_holder, messages, sampling_params=None):
            return {"text": " Judgement: 1", "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 4}}

        replica._call_engine_tracked = fake_tracked
        out = _run(replica._call_engine(None, [{"role": "user", "content": "q"}]))
        self.assertEqual(out["engine_host"], _ENGINES[0][0])
        self.assertEqual(out["engine_port"], _ENGINES[0][1])
        self.assertEqual(out["text"], " Judgement: 1")

    def test_generate_response_exposes_engine_and_finish_metadata(self):
        replica = _replica()

        async def fake_call(route_key, messages, sampling_params=None):
            return {
                "text": " Judgement: 1 ",
                "engine_host": "192.0.2.9",
                "engine_port": 16007,
                "meta_info": {"finish_reason": {"type": "length"}, "completion_tokens": 8},
            }

        replica._call_engine = fake_call
        request = genrm_module.GenerateRequest(messages=[{"role": "user", "content": "q"}])
        response = _run(replica.generate(request))
        self.assertEqual(response.response, "Judgement: 1")
        self.assertEqual(response.engine_host, "192.0.2.9")
        self.assertEqual(response.engine_port, 16007)
        self.assertEqual(response.finish_reason, "length")  # truncation is observable
        self.assertEqual(response.completion_tokens, 8)

    def test_generate_response_fields_default_to_none(self):
        response = genrm_module.GenerateResponse(response="Judgement: 0")
        self.assertIsNone(response.engine_host)
        self.assertIsNone(response.engine_port)
        self.assertIsNone(response.finish_reason)
        self.assertIsNone(response.completion_tokens)


if __name__ == "__main__":
    unittest.main()
