# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

from relax.distributed.coordination import RolloutOffloadBarrier
from relax.distributed.ray import rollout_worker
from relax.distributed.ray.inference_manager import InferenceManager
from relax.engine.inference.types import Role


def test_rollout_worker_binds_workload_and_local_scoring_to_same_owner(monkeypatch):
    owner = SimpleNamespace(
        begin_rollout=SimpleNamespace(remote=AsyncMock()),
        activate=SimpleNamespace(remote=AsyncMock()),
    )
    args = SimpleNamespace(tq_config={}, use_agentic_rollout=False, sglang_router_ip="router", sglang_router_port=1234)
    source, client = object(), object()
    workload = SimpleNamespace(generate=AsyncMock())
    constructor = Mock(return_value=workload)
    monkeypatch.setattr(rollout_worker, "RolloutWorkload", constructor)
    monkeypatch.setattr(rollout_worker, "init_tracking", Mock())
    monkeypatch.setattr(rollout_worker, "init_http_client", Mock())
    monkeypatch.setattr(rollout_worker.tq, "init", Mock())
    monkeypatch.setattr(rollout_worker.tq, "get_client", lambda: client)
    monkeypatch.setattr(rollout_worker, "_LOCAL_INFERENCE_MANAGER", None)
    try:
        import sglang.srt.constants  # noqa: F401
    except ImportError:
        # onload_kv reads the memory tags from sglang, which CPU CI does not install.
        constants = ModuleType("sglang.srt.constants")
        constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
        constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
        for name in ("sglang", "sglang.srt"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)
    worker = rollout_worker.RolloutWorker.__ray_metadata__.modified_class(args, source, owner)
    assert constructor.call_args.args[:3] == (args, source, client)
    assert rollout_worker.get_local_inference_manager() is owner
    port = constructor.call_args.args[3]

    async def run():
        await worker.generate(7)
        await port.resume_health_monitoring()
        await port.onload_kv()

    asyncio.run(run())
    workload.generate.assert_awaited_once_with(7)
    owner.begin_rollout.remote.assert_awaited_once_with()
    owner.activate.remote.assert_awaited_once_with(Role.ROLLOUT, tags=["kv_cache", "cuda_graph"])
    assert not hasattr(worker, "offload")
    assert not hasattr(worker, "recover_rollout_engines")


def test_inference_manager_preserves_recovery_guard_before_first_generation():
    manager = InferenceManager()
    pool = SimpleNamespace(recover_rollout_engines=Mock(), health_monitoring_resume=Mock())
    manager._rollout_pool = pool
    manager.rollout_operation("recover_rollout_engines")
    pool.recover_rollout_engines.assert_called_once_with(rollout_started=False)
    manager.begin_rollout()
    pool.health_monitoring_resume.assert_called_once_with()
    manager.rollout_operation("recover_rollout_engines", "policy")
    pool.recover_rollout_engines.assert_called_with("policy", rollout_started=True)


def test_rollout_offload_barrier_polls_inference_owner(monkeypatch):
    released = Mock(side_effect=[False, True])
    owner = SimpleNamespace(rollout_released=SimpleNamespace(remote=released))
    monkeypatch.setattr("relax.distributed.coordination.ray.get", lambda value: value)
    asyncio.run(RolloutOffloadBarrier(owner, poll_interval=0).wait_offloaded())
    assert released.call_count == 2
