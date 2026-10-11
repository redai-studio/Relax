# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Rollout discovery/serialization without loading training
backends."""

import ast
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.inference.manager import InferenceManager
from relax.inference.routing import engine_record


_RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


def methods(*names):
    path = Path(__file__).resolve().parents[2] / "relax/distributed/ray/rollout.py"
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutManager")
    body = [
        node
        for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    for node in body:
        node.decorator_list = []
    namespace = dict(
        ray=SimpleNamespace(get=lambda value, **kwargs: value),
        engine_record=engine_record,
        EngineGroupLifecycle=SimpleNamespace(ACTIVE="active"),
        _wrap_ipv6=lambda value: value,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def test_discovery_hides_followers_and_waits_for_actual_weight_commit():
    group = SimpleNamespace(
        worker_type="regular",
        all_engines=[SimpleNamespace(get_url=SimpleNamespace(remote=lambda: "http://head")), object()],
        nodes_per_engine=2,
        lifecycle_status="active",
        rank_offset=0,
        num_new_engines=2,
    )
    server = SimpleNamespace(engine_groups=[group], router_ip="router", router_port=8000)
    owner = InferenceManager("rollout", {"default": server})
    manager = SimpleNamespace(inference=owner, _is_weight_updating=False, _policy_weights_ready={"default": False})
    snapshot = methods("get_inference_snapshot")["get_inference_snapshot"]
    info = snapshot(manager)["models"]["default"]
    assert len(info["engines"]) == 1
    assert not info["engines"][0]["direct_eligible"]
    # Reconnection counters alone must never publish readiness.
    group.num_new_engines = 0
    assert snapshot(manager)["models"]["default"]["state"] != "ready"
    manager._policy_weights_ready["default"] = True
    assert snapshot(manager)["models"]["default"]["state"] == "ready"
    manager._is_weight_updating = True
    assert snapshot(manager)["models"]["default"]["state"] != "ready"


async def test_deferred_eval_cannot_offload_during_training_generation():
    namespace = methods("generate", "eval")
    events = []
    generation_started, generation_done = asyncio.Event(), asyncio.Event()

    async def generate(step):
        events.append("generate")
        generation_started.set()
        await generation_done.wait()
        events.append("published")

    async def evaluate(step):
        events.append("evaluate")

    manager = SimpleNamespace(
        args=SimpleNamespace(opd_teacher_defer=True),
        _deferred_execution_lock=asyncio.Lock(),
        _generate=generate,
        _eval=evaluate,
    )
    task = asyncio.create_task(namespace["generate"](manager, 0))
    await generation_started.wait()
    evaluation = asyncio.create_task(namespace["eval"](manager, 0))
    await asyncio.sleep(0)
    assert events == ["generate"]
    generation_done.set()
    await asyncio.gather(task, evaluation)
    assert events == ["generate", "published", "evaluate"]


def test_port_allocator_uses_actual_host_and_group_local_head(monkeypatch):
    if not _RAY_AVAILABLE:
        pytest.skip("requires ray")
    from relax.distributed.ray import inference_ports

    monkeypatch.setattr(inference_ports.ray, "get", lambda value, **kwargs: value)

    def engine(host):
        return SimpleNamespace(
            _get_current_node_ip_and_free_port=SimpleNamespace(
                remote=lambda start_port=10000, **kwargs: (host, start_port)
            )
        )

    addresses, _ = inference_ports.allocate_inference_ports(
        [(1, engine("node-a")), (2, engine("node-b"))], nodes_per_engine=2, base_port=15000, rank_offset=1
    )
    assert addresses[1]["dist_init_addr"] == addresses[2]["dist_init_addr"] == "node-a:15002"
    assert addresses[2]["host"] == "node-b"
    with pytest.raises(ValueError, match="every node"):
        inference_ports.allocate_inference_ports(
            [(2, engine("node-b"))], nodes_per_engine=2, base_port=15000, rank_offset=1
        )
