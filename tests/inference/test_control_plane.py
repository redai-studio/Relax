# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from relax.components.inference_gateway import InferenceGateway
from relax.inference.lifecycle import LifecycleCoordinator
from relax.inference.manager import InferenceManager
from relax.inference.placement import PlacementPlanner, validate_bundle_span
from relax.inference.routing import ModelUnavailable, RoutingError, engine_record, select_endpoint, select_model
from relax.utils.inference_client import InferenceClient


def snapshot():
    models = {
        name: {
            "state": "ready",
            "router_url": None,
            "engines": [engine_record(f"{name}/0", f"http://{name}-head", "ready")],
        }
        for name in ("a", "b")
    }
    return {
        "role": "teacher",
        "phase": "inference",
        "topology_revision": 1,
        "models": models,
        "default_model": "a",
        "routes": {"math": "b"},
    }


@pytest.mark.parametrize("model,route,expected", [("a", "math", "a"), (None, "math", "b"), (None, None, "a")])
def test_routing_precedence(model, route, expected):
    assert select_model(snapshot(), model, route) == expected


def test_explicit_unknown_model_never_falls_back():
    with pytest.raises(RoutingError):
        select_model(snapshot(), "unknown", "math")


def test_pd_uses_router_and_sleeping_never_routes():
    data = snapshot()
    data["models"]["a"]["engines"][0]["direct_eligible"] = False
    with pytest.raises(ModelUnavailable):
        select_endpoint(data, "a")


async def test_gateway_lists_and_routes_legacy_served_model_alias():
    data = snapshot()
    data["routes"] = {}
    data["models"]["a"]["model_aliases"] = ["/models/checkpoint"]
    manager = SimpleNamespace(
        get_inference_snapshot=SimpleNamespace(remote=lambda: asyncio.sleep(0, result=data)),
    )
    gateway = InferenceGateway("rollout", {"default": manager})

    discovered = await gateway.discovery()
    assert select_model(discovered, "/models/checkpoint") == "a"
    model_ids = {item["id"] for item in (await gateway.models())["data"]}
    assert model_ids == {"a", "b", "/models/checkpoint"}

    await gateway.aclose()
    data["models"]["a"]["router_url"] = "http://router"
    assert select_endpoint(data, "a") == "http://router"
    data["models"]["a"]["state"] = "sleeping"
    with pytest.raises(ModelUnavailable):
        select_endpoint(data, "a")


async def test_gateway_health_reports_failed_models():
    data = snapshot()
    gateway = InferenceGateway("teacher", {})
    gateway.discovery = lambda: asyncio.sleep(0, result=data)

    assert (await gateway.health())["status"] == "healthy"
    data["models"]["a"]["state"] = "failed"
    health = await gateway.health()
    assert health == {"status": "unhealthy", "role": "teacher", "models": {"a": "failed", "b": "ready"}}

    await gateway.aclose()


async def test_gateway_admits_generation_after_initial_weight_sync():
    data = snapshot()
    data["models"] = {"a": {**data["models"]["a"], "router_url": "http://backend", "state": "unavailable"}}
    manager = SimpleNamespace(
        get_inference_snapshot=SimpleNamespace(remote=lambda: asyncio.sleep(0, result=data)),
    )
    gateway = InferenceGateway("rollout", {"default": manager})
    backend = httpx.MockTransport(lambda request: httpx.Response(200, json={"text": "initial policy"}))
    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=backend)
    app = FastAPI()

    @app.get("/engines")
    async def engines():
        return await gateway.discovery()

    @app.post("/generate")
    async def generate(request: Request):
        return await gateway.forward(request, "generate")

    payload = {"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 1}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as client:
        blocked = await client.post("/generate", json=payload)
        assert blocked.status_code == 503

        data["models"]["a"]["state"] = "ready"
        admitted = await client.post("/generate", json=payload)
        assert admitted.status_code == 200
        assert admitted.json() == {"text": "initial policy"}

    await gateway.aclose()


def test_manager_idempotency_partial_onload_and_immutable_snapshots():
    owner = InferenceManager("teacher")
    models = snapshot()["models"]
    first = owner.snapshot(models)
    assert owner.snapshot(models)["topology_revision"] == first["topology_revision"]
    calls = []
    owner.transition("deactivate", lambda: calls.append("off"))
    owner.transition("deactivate", lambda: calls.append("off"))
    sleeping = owner.snapshot(models)
    assert sleeping["models"]["a"]["engines"][0]["direct_eligible"] is False
    assert first["models"]["a"]["engines"][0]["direct_eligible"] is True
    owner.transition("activate", lambda: calls.append("weights"), ["weights"])
    owner.transition("activate", lambda: calls.append("weights"), ["weights"])
    assert owner.state == "onloading"
    owner.transition("activate", lambda: calls.append("kv"), ["kv_cache", "cuda_graph"])
    assert owner.state == "ready"
    assert calls == ["off", "weights", "kv"]
    changed = copy.deepcopy(models)
    changed["a"]["engines"][0]["base_url"] = "http://replacement"
    assert owner.snapshot(changed)["topology_revision"] > sleeping["topology_revision"]


def test_manager_failure_closes_admission():
    owner = InferenceManager("rollout")

    def failed():
        raise TimeoutError("drain")

    with pytest.raises(TimeoutError):
        owner.transition("deactivate", failed)
    with pytest.raises(ModelUnavailable):
        select_endpoint(owner.snapshot(snapshot()["models"]), "a")


def config(**overrides):
    values = dict(
        resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 2], "teacher": [1, 2]},
        colocate=True,
        hybrid=False,
        fully_async=False,
        num_gpus_per_node=4,
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        teacher_num_gpus_per_engine=2,
        _genrm_instances_resolved={"judge": {"num_gpus": 2, "num_gpus_per_engine": 2}},
        offload_rollout=True,
        offload_train=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_placement_combined_split_has_disjoint_regions():
    args = config()
    plan = PlacementPlanner.apply(args)
    assert [(p.role, p.offset) for p in plan] == [("rollout", 0), ("genrm", 4), ("teacher", 6)]
    assert args._teacher_bundle_start == 6
    assert all(p.pg_owner == "controller" for p in plan)


def test_deferred_ppo_reserves_ray_capacity_for_all_phase_actors():
    args = config(
        rollout_num_gpus=8, opd_teacher_defer=True, defer_reward_to_post_process=True, advantage_estimator="ppo"
    )
    args.resource["critic"] = args.resource["actor"]
    PlacementPlanner.apply(args)
    assert 0 < args._inference_ray_gpu_fraction
    assert 0.8 + 3 * args._inference_ray_gpu_fraction <= 1.0


def test_placement_decoupled_starts_each_pool_at_zero():
    plan = PlacementPlanner.resolve(config(colocate=False))
    assert all(p.offset == 0 for p in plan)
    assert [p.pg_owner for p in plan] == ["service", "service", "manager"]


@pytest.mark.parametrize("flags", [{"colocate": False}, {"colocate": True, "hybrid": True}])
def test_independent_rollout_rejects_request_above_resource_budget(flags):
    with pytest.raises(ValueError, match="independent resource budget"):
        PlacementPlanner.resolve(config(rollout_num_gpus=8, **flags))


@pytest.mark.parametrize("flags", [{"colocate": False}, {"colocate": True, "hybrid": True}])
@pytest.mark.parametrize("count", [2, 4])
def test_independent_rollout_allows_request_within_resource_budget(flags, count):
    plan = PlacementPlanner.resolve(config(rollout_num_gpus=count, **flags))
    assert plan[0].num_gpus == count


def test_placement_defer_allows_reuse_only_in_distinct_phases():
    args = config(rollout_num_gpus=8, opd_teacher_defer=True, defer_reward_to_post_process=True)
    plan = PlacementPlanner.resolve(args)
    assert [p.offset for p in plan] == [0, 0, 0]
    assert [p.phase for p in plan] == ["inference", "genrm", "teacher"]
    with pytest.raises(ValueError, match="co-residency"):
        PlacementPlanner.resolve(config(rollout_num_gpus=8))


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_num_gpus_per_engine": 3},
        {"teacher_num_gpus_per_engine": 0},
        {"opd_teacher_defer": True, "fully_async": True},
        {"opd_teacher_defer": True, "offload_rollout": False},
        {"offload_rollout": False},
        {"offload_train": False},
        {"defer_reward_to_post_process": True, "_genrm_instances_resolved": {}},
    ],
)
def test_placement_rejects_invalid_before_allocation(overrides):
    with pytest.raises(ValueError):
        PlacementPlanner.resolve(config(**overrides))


def test_topology_checks_physical_node_and_gpu_mapping():
    validate_bundle_span([3, 1, 0, 2], [0, 1, 0, 1], {3: "n1", 1: "n1", 0: "n2", 2: "n2"}, 0, 2)
    with pytest.raises(ValueError, match="node boundaries"):
        validate_bundle_span([3, 1, 0, 2], [0, 1, 0, 1], {3: "n1", 1: "n1", 0: "n2", 2: "n2"}, 1, 2)
    with pytest.raises(ValueError, match="contiguous"):
        validate_bundle_span([0, 1], [0, 2], {0: "n1", 1: "n1"}, 0, 2)


async def test_lifecycle_serializes_phases_and_rolls_back_failed_activation():
    owner = LifecycleCoordinator()
    events = []

    async def on():
        events.append("on")
        raise RuntimeError("partial activation")

    async def off():
        events.append("off")

    async def work():
        events.append("work")

    with pytest.raises(RuntimeError, match="partial activation"):
        await owner.run_phase("teacher", on, off, work)
    assert events == ["on", "off"]
    assert owner.phase is None

    async def ok():
        await asyncio.sleep(0)
        events.append(owner.phase)

    await asyncio.gather(*(owner.run_phase(name, ok, ok, ok) for name in ("teacher", "genrm")))
    assert events[2:] == ["teacher"] * 3 + ["genrm"] * 3


async def test_lifecycle_failed_drain_prevents_next_activation():
    owner = LifecycleCoordinator()

    async def ok():
        return None

    async def failed():
        raise TimeoutError("drain")

    with pytest.raises(TimeoutError):
        await owner.run_phase("teacher", ok, failed, ok)
    with pytest.raises(RuntimeError, match="retained"):
        await owner.run_phase("genrm", ok, ok, ok)


async def test_gateway_and_direct_client_choose_same_model_and_preserve_payload():
    data = snapshot()
    gateway = InferenceGateway("teacher", {})
    gateway.discovery = lambda: asyncio.sleep(0, result=data)
    seen = []

    def backend(request):
        seen.append((request.url.host, json.loads(request.content)))
        return httpx.Response(200, json={"meta_info": {"input_token_logprobs": [[-1, 3, None]]}})

    await gateway._client.aclose()
    gateway._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    app = FastAPI()

    @app.get("/engines")
    async def engines():
        return data

    @app.post("/generate")
    async def generate(request: Request):
        return await gateway.forward(request, "generate")

    payload = {"input_ids": [1, 2, 3], "return_logprob": True, "logprob_start_len": 1}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://gateway") as transport:
        response = await transport.post("/generate", json={**payload, "route_key": "math"})
        assert response.status_code == 200
        client = InferenceClient("http://gateway", direct=True)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        client.discovery = lambda: asyncio.sleep(0, result=data)
        await client.generate(payload, route_key="math")
        await client.aclose()
        assert seen == [("b-head", payload), ("b-head", payload)]
        data["models"]["b"]["state"] = "sleeping"
        response = await transport.post("/generate", json={"model": "b", **payload})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        assert len(seen) == 2
    await gateway.aclose()


@pytest.mark.parametrize("direct", [True, False])
async def test_streaming_client_preserves_sse_and_closes_response(direct):
    client = InferenceClient("http://gateway", direct=direct)
    await client._client.aclose()
    responses = []

    def backend(request):
        assert json.loads(request.content)["stream"]
        response = httpx.Response(
            200, content=b'data: {"text":"hello"}\n\ndata: [DONE]\n\n', headers={"content-type": "text/event-stream"}
        )
        responses.append(response)
        return response

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    client.discovery = lambda: asyncio.sleep(0, result=snapshot())
    parts = [chunk async for chunk in client.stream({"messages": []}, path="v1/chat/completions")]
    assert b"".join(parts).endswith(b"data: [DONE]\n\n")
    assert responses[0].is_closed
    await client.aclose()


def test_planner_rejects_replica_crossing_node_before_ray_allocation():
    with pytest.raises(ValueError, match="node boundary"):
        PlacementPlanner.resolve(
            config(
                resource={"actor": [1, 12], "rollout": [1, 9], "teacher": [1, 3]},
                rollout_num_gpus=9,
                rollout_num_gpus_per_engine=3,
                num_gpus_per_node=8,
                teacher_num_gpus_per_engine=3,
                _genrm_instances_resolved={},
            )
        )
