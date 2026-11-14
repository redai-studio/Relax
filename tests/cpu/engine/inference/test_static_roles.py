# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Auxiliary entry points use real snapshots/executor with HTTP isolated."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from relax.distributed.ray.inference_manager import InferenceManager
from relax.distributed.ray.static_inference_pool import StaticInferencePool
from relax.engine.inference.config_adapters import static_role_spec
from relax.engine.inference.types import LifecycleState, Role, RouteMode, WeightSource


def _owner(role):
    owner = InferenceManager.__ray_metadata__.modified_class()
    pool = StaticInferencePool.__new__(StaticInferencePool)
    pool.model_id = "judge" if role is Role.GENRM else "default"
    pool.backend = SimpleNamespace(
        all_engines=[object(), object()],
        nodes_per_engine=2,
        _engine_addr_and_ports={0: {"host": "engine", "port": 16000}},
    )
    owner.pool = pool
    owner.role = role
    owner.memory_ready = True
    owner.state = LifecycleState.READY
    owner._last_health = float("inf")
    remote = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(owner, name))
            for name in ("snapshot", "admit_request", "complete_request", "wait_ready")
        }
    )
    return owner, remote


@pytest.mark.asyncio
async def test_genrm_managed_request_resolves_current_head_and_preserves_sampling():
    from relax.components.genrm import GenRM

    owner, remote = _owner(Role.GENRM)
    replica = object.__new__(GenRM.func_or_class)
    replica._managed = True
    replica.genrm_managers = {"judge": remote}
    replica.instance_specs = {"judge": {"sampling_config": {"temperature": 0.4}}}
    replica.tokenizers = {"judge": SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])}
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"text": "score"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        replica._http_client = client
        assert await replica._call_engine("judge", [], {"temperature": 0.7}) == {"text": "score"}
        owner.pool.backend._engine_addr_and_ports[0]["port"] = 16001
        await replica._call_engine("judge", [])
    assert [request.url.port for request in sent] == [16000, 16001]
    assert json.loads(sent[0].content)["sampling_params"]["temperature"] == 0.7
    assert json.loads(sent[1].content)["sampling_params"]["temperature"] == 0.4
    assert len((await owner.snapshot(Role.GENRM)).models[0].replicas) == 1
    assert not owner.ledger.permits


@pytest.mark.asyncio
async def test_managed_teacher_total_timeout_aborts_and_returns_permit(monkeypatch):
    from relax.engine.rollout.on_policy_distillation import OpdManager
    from relax.utils import http_utils
    from relax.utils.types import Sample

    owner, remote = _owner(Role.TEACHER)
    worker = OpdManager.__new__(OpdManager)
    worker.args = SimpleNamespace(_teacher_inference_manager_handle=remote, opd_teacher_timeout_s=0.02)
    requests = []

    async def respond(request):
        requests.append(request)
        if request.url.path == "/generate":
            await asyncio.Event().wait()
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(http_utils, "_http_client", client)
        result = await asyncio.wait_for(
            worker._post_logprob(
                None, "http://stale/generate", {"input_ids": [1]}, Sample(index=1), "teacher prefill"
            ),
            2,
        )
    assert result is None
    assert [request.url.path for request in requests] == ["/generate", "/abort_request"]
    assert all(request.url.host == "engine" for request in requests)
    assert not owner.ledger.permits


def test_static_genrm_configuration_has_no_policy_weight_receiver():
    args = SimpleNamespace(
        genrm_model_path="judge-checkpoint",
        genrm_num_gpus_per_engine=2,
        genrm_num_gpus=4,
        genrm_engine_config={},
        genrm_sampling_config={"temperature": 0.2},
        num_gpus_per_node=4,
    )
    spec = static_role_spec(Role.GENRM, args, {"model_id": "judge"})
    model = spec.models[0]
    assert model.weight_source is WeightSource.STATIC
    assert not model.needs_dcs
    assert model.route_mode is RouteMode.DIRECT
    assert len(model.engine_groups[0].topology.replicas) == 2
    args.genrm_num_gpus = 3
    with pytest.raises(ValueError, match="complete replicas"):
        static_role_spec(Role.GENRM, args, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [Role.GENRM, Role.TEACHER])
async def test_static_requests_rotate_across_two_replicas(role):
    from relax.engine.inference.executor import InferenceRequestExecutor

    owner, remote = _owner(role)
    owner.pool.backend.nodes_per_engine = 1
    owner.pool.backend._engine_addr_and_ports[1] = {"host": "engine", "port": 16001}
    ports = []

    def respond(request):
        ports.append(request.url.port)
        return httpx.Response(200, json={"text": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        executor = InferenceRequestExecutor(role, remote, client)
        for _ in range(4):
            await executor.generate_json({"input_ids": [1]})
    assert ports == [16000, 16001, 16000, 16001]
    assert not owner.ledger.permits


def test_genrm_factory_assigns_disjoint_unequal_slices_and_rolls_back(monkeypatch):
    import ray

    import relax.distributed.ray.inference_manager as manager_module
    from relax.distributed.ray.placement_group import create_genrm_managers

    specs = {
        key: dict(model_path=key, num_gpus=gpus, num_gpus_per_engine=1, engine_config={}, sampling_config={})
        for key, gpus in (("a", 1), ("b", 2))
    }
    args = SimpleNamespace(colocate=False, _genrm_instances_resolved=specs)
    launched = []
    closed = []
    killed = []
    first = SimpleNamespace(dispose=SimpleNamespace(remote=lambda: closed.append("a")))

    def allocate(instance_args, role, binding, **options):
        launched.append((instance_args.genrm_model_path, options["bundle_offset"]))
        assert role == "genrm"
        if instance_args.genrm_model_path == "b":
            raise RuntimeError("second model cannot start")
        return first

    monkeypatch.setattr(manager_module, "create_static_inference_manager", allocate)
    monkeypatch.setattr(ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(ray, "kill", killed.append)
    with pytest.raises(RuntimeError, match="second model"):
        create_genrm_managers(args, "service-pg")
    assert launched == [("a", 0), ("b", 1)]
    assert closed == ["a"]
    assert killed == [first]


def test_teacher_factory_injects_public_owner_for_independent_requests(monkeypatch):
    import ray

    import relax.distributed.ray.inference_manager as manager_module
    from relax.utils.opd.opd_utils import create_managed_opd_teacher_manager

    teacher = SimpleNamespace(get_urls=SimpleNamespace(remote=lambda: ["http://engine/generate"]))
    captured = []

    def allocate(args, role, **options):
        captured.append((role, options["num_replicas"], options["gpus_per_replica"]))
        return teacher

    monkeypatch.setattr(manager_module, "create_static_inference_manager", allocate)
    monkeypatch.setattr(ray, "get", lambda ref: ref)
    args = SimpleNamespace(colocate=False)
    handle, urls = create_managed_opd_teacher_manager(args, num_replicas=2, gpus_per_replica=4)
    assert captured == [("teacher", 2, 4)]
    assert args._teacher_inference_manager_handle is handle is teacher
    assert urls == ["http://engine/generate"]
