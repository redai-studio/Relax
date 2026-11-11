# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise native child cleanup without starting Ray or loading a model."""

import ast
import asyncio
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def _method(filename: str, class_name: str, name: str):
    relative = filename if "/" in filename else f"relax/distributed/ray/{filename}"
    path = Path(__file__).parents[4] / relative
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    namespace = {"Any": Any, "asyncio": asyncio, "GPU_MEMORY_TYPE_KV_CACHE": "kv", "GPU_MEMORY_TYPE_CUDA_GRAPH": "cg"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _manager():
    engine = SimpleNamespace(shutdown_for_recovery=SimpleNamespace(remote=Mock(return_value="shutdown-ref")))
    manager = SimpleNamespace(
        servers={"default": SimpleNamespace(engine_groups=[SimpleNamespace(all_engines=[engine, None])])},
        rollout_engine_lock=object(),
        _weight_sync_lock=object(),
        _recovery_generate_active=False,
        _is_weight_updating=False,
        _eviction_monitor_thread=None,
        _stop_eviction_monitor=Mock(),
        _health_monitors=[],
        _recovery_router_processes=[],
        pg=object(),
    )
    return manager, engine


def _install_wait_all(monkeypatch, wait_all):
    module = ModuleType("relax.core.inference_recovery")
    module.wait_all = wait_all
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_inference_call_submits_all_ranks_without_observing_results():
    refs = [object(), object()]
    ranks = [SimpleNamespace(compute=SimpleNamespace(remote=Mock(return_value=ref))) for ref in refs]
    group = SimpleNamespace(_actor_handlers=ranks)
    call = _method("actor_group.py", "RayTrainGroup", "async_inference_call")

    assert call(group, "compute", 7, receive_weights=False) == refs
    for rank in ranks:
        rank.compute.remote.assert_called_once_with(7, receive_weights=False)


def test_recovery_children_include_native_engines_and_locks():
    manager, engine = _manager()
    children = _method("rollout.py", "RolloutManager", "get_recovery_children")(manager)
    assert children == [engine, manager.rollout_engine_lock, manager._weight_sync_lock]
    assert manager.pg not in children


def test_recovery_disposal_waits_before_stopping_owned_router(monkeypatch):
    manager, engine = _manager()
    events = []
    router = Mock(pid=123)
    router.is_alive.side_effect = [True, False, False]
    router.terminate.side_effect = lambda: events.append("router-stop")
    manager._recovery_router_processes = [router]
    original_pg = manager.pg

    def wait_all(refs, timeout):
        assert refs == ["shutdown-ref"] and timeout == 12
        events.append("all-engines-terminal")

    _install_wait_all(monkeypatch, wait_all)
    _method("rollout.py", "RolloutManager", "dispose_for_recovery")(manager, timeout=12)

    assert events == ["all-engines-terminal", "router-stop"]
    engine.shutdown_for_recovery.remote.assert_called_once_with()
    assert manager._recovery_router_processes == []
    assert manager.pg is original_pg


def test_recovery_disposal_propagates_engine_shutdown_failure(monkeypatch):
    manager, _ = _manager()
    router = Mock()
    manager._recovery_router_processes = [router]
    _install_wait_all(monkeypatch, Mock(side_effect=RuntimeError("rank shutdown failed")))

    with pytest.raises(RuntimeError, match="rank shutdown failed"):
        _method("rollout.py", "RolloutManager", "dispose_for_recovery")(manager)
    router.terminate.assert_not_called()
    assert manager._recovery_router_processes == [router]


@pytest.mark.parametrize("busy_field", ["_recovery_generate_active", "_is_weight_updating"])
def test_recovery_disposal_rejects_nonidle_manager(monkeypatch, busy_field):
    manager, engine = _manager()
    setattr(manager, busy_field, True)
    wait_all = Mock()
    _install_wait_all(monkeypatch, wait_all)

    with pytest.raises(RuntimeError, match="must be idle"):
        _method("rollout.py", "RolloutManager", "dispose_for_recovery")(manager)
    wait_all.assert_not_called()
    engine.shutdown_for_recovery.remote.assert_not_called()


def test_recovery_disposal_rejects_surviving_router(monkeypatch):
    manager, _ = _manager()
    router = Mock(pid=123)
    router.is_alive.return_value = True
    manager._recovery_router_processes = [router]
    _install_wait_all(monkeypatch, Mock())

    with pytest.raises(RuntimeError, match="Owned Router process 123 did not stop"):
        _method("rollout.py", "RolloutManager", "dispose_for_recovery")(manager)
    router.terminate.assert_called_once_with()
    router.kill.assert_called_once_with()
    assert manager._recovery_router_processes == [router]


def test_cancelled_generate_await_does_not_claim_thread_idle():
    manager, _ = _manager()
    manager.args = SimpleNamespace(ci_test=False, partial_rollout=False)
    manager.health_monitoring_resume = Mock()
    manager.generate_rollout = object()
    manager.data_source = object()
    manager.data_system_client = object()
    entered, release = threading.Event(), threading.Event()
    generate = _method("rollout.py", "RolloutManager", "generate")

    def generate_batch(*args, **kwargs):
        entered.set()
        assert release.wait(3)

    batch = _method("rollout.py", "RolloutManager", "_generate_batch")
    batch.__globals__["call_rollout_fn"] = generate_batch
    manager._generate_batch = batch.__get__(manager)

    async def run():
        task = asyncio.create_task(generate(manager, 0))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert manager._recovery_generate_active
        finally:
            release.set()

    asyncio.run(run())
    assert not manager._recovery_generate_active


def test_engine_shutdown_confirms_entire_captured_process_tree(monkeypatch):
    import psutil

    root, child = Mock(pid=100), Mock(pid=101)
    root.children.return_value = [child]
    monkeypatch.setattr(psutil, "Process", Mock(return_value=root))
    events = []
    shutdown = _method("relax/backends/sglang/sglang_engine.py", "SGLangEngine", "shutdown")
    kill = Mock(side_effect=lambda _pid: events.append("kill"))
    shutdown.__globals__.update(logger=Mock(), kill_process_tree=kill)
    engine = SimpleNamespace(
        args=SimpleNamespace(rollout_external=False),
        process=Mock(pid=100),
        server_host="engine",
        server_port=1,
        unregister_from_router=Mock(side_effect=lambda **_kwargs: events.append("unregister")),
    )
    engine.shutdown = lambda: shutdown(engine)

    def wait_procs(processes, timeout):
        assert processes == [root, child] and timeout == 7
        assert events == ["unregister", "kill"]
        return processes, []

    monkeypatch.setattr(psutil, "wait_procs", wait_procs)
    _method("relax/backends/sglang/sglang_engine.py", "SGLangEngine", "shutdown_for_recovery")(engine, timeout=7)
    root.children.assert_called_once_with(recursive=True)
    engine.unregister_from_router.assert_called_once_with(timeout=5.0)
    kill.assert_called_once_with(engine.process.pid)


def test_engine_shutdown_rejects_surviving_child(monkeypatch):
    import psutil

    root, child = Mock(pid=100), Mock(pid=101)
    root.children.return_value = [child]
    monkeypatch.setattr(psutil, "Process", Mock(return_value=root))
    monkeypatch.setattr(psutil, "wait_procs", Mock(return_value=([root], [child])))
    engine = SimpleNamespace(args=SimpleNamespace(rollout_external=False), process=Mock(pid=100), shutdown=Mock())

    with pytest.raises(RuntimeError, match="did not exit.*101"):
        _method("relax/backends/sglang/sglang_engine.py", "SGLangEngine", "shutdown_for_recovery")(engine)


def test_engine_shutdown_rejects_already_dead_server():
    engine = SimpleNamespace(args=SimpleNamespace(rollout_external=False), process=Mock(), shutdown=Mock())
    engine.process.is_alive.return_value = False
    with pytest.raises(RuntimeError, match="live, owned"):
        _method("relax/backends/sglang/sglang_engine.py", "SGLangEngine", "shutdown_for_recovery")(engine)
    engine.shutdown.assert_not_called()


def test_discard_pending_preserves_cursor_and_monotonic_sample_ids():
    source = SimpleNamespace(buffer=[[object()], [object()]], sample_offset=17, epoch_id=2, sample_index=93)
    discard = _method("relax/engine/rollout/data_source.py", "RolloutDataSourceWithBuffer", "discard_pending")
    assert discard(source) == 2
    assert source.buffer == []
    assert (source.sample_offset, source.epoch_id, source.sample_index) == (17, 2, 93)


def test_kv_onload_completes_before_generation_resumes(monkeypatch):
    events = []
    engine = SimpleNamespace(continue_generation=SimpleNamespace(remote=Mock(return_value="continue-ref")))
    group = SimpleNamespace(onload=Mock(return_value=["kv-ref"]))
    manager = SimpleNamespace(
        args=SimpleNamespace(inference_recovery=True, rollout_http_timeout=9),
        servers={"default": SimpleNamespace(engine_groups=[group])},
        rollout_engines=[engine],
        status="offload",
    )

    def wait_all(refs, timeout):
        assert timeout == 9
        if refs == ["kv-ref"]:
            engine.continue_generation.remote.assert_not_called()
            events.append("kv-loaded")
        else:
            assert refs == ["continue-ref"] and events == ["kv-loaded"]
            events.append("continued")

    _install_wait_all(monkeypatch, wait_all)
    asyncio.run(_method("rollout.py", "RolloutManager", "onload_kv")(manager))
    assert events == ["kv-loaded", "continued"]
    assert manager.status == "onload"


def test_kv_onload_failure_does_not_resume_generation(monkeypatch):
    engine = SimpleNamespace(continue_generation=SimpleNamespace(remote=Mock()))
    group = SimpleNamespace(onload=Mock(return_value=["kv-ref"]))
    manager = SimpleNamespace(
        args=SimpleNamespace(inference_recovery=True, rollout_http_timeout=9),
        servers={"default": SimpleNamespace(engine_groups=[group])},
        rollout_engines=[engine],
        status="offload",
    )
    _install_wait_all(monkeypatch, Mock(side_effect=RuntimeError("KV failed")))
    with pytest.raises(RuntimeError, match="KV failed"):
        asyncio.run(_method("rollout.py", "RolloutManager", "onload_kv")(manager))
    engine.continue_generation.remote.assert_not_called()
    assert manager.status == "offload"


@pytest.mark.parametrize("mode", ["complete", "pending", "cancelled-generation", "cancelled-write", "failed-write"])
def test_reset_data_window_drops_carryover_after_prefetch_is_terminal(monkeypatch, mode):
    class GenerateState:
        _instances = {}

    task = Mock()
    task.done.return_value = mode != "pending"
    task.cancelled.return_value = mode.startswith("cancelled")
    task.exception.return_value = RuntimeError("write failed") if mode == "failed-write" else None
    state = SimpleNamespace(
        pendings={task} if mode in {"pending", "cancelled-generation"} else set(),
        recovery_transfers=[task] if mode.endswith("write") else [],
        protected_pendings=set(),
        evaluating=0,
        prefetched_samples_ref="prefetch-ref",
        last_step_current_deficit=4,
        reset=Mock(),
    )
    GenerateState._instances[GenerateState] = state
    generate_module = ModuleType("relax.engine.rollout.sglang_rollout")
    generate_module.GenerateState = GenerateState
    utils_module = ModuleType("relax.utils.utils")
    utils_module.CURRENT_ROLLOUT_BATCH = [object()]
    monkeypatch.setitem(sys.modules, generate_module.__name__, generate_module)
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
    wait_all = Mock()
    _install_wait_all(monkeypatch, wait_all)
    manager, _ = _manager()

    reset = _method("rollout.py", "RolloutManager", "reset_data_window")
    if mode in {"pending", "cancelled-write", "failed-write"}:
        with pytest.raises(RuntimeError, match="not terminal|storage writes"):
            reset(manager, timeout=6)
        wait_all.assert_not_called()
        state.reset.assert_not_called()
        return
    reset(manager, timeout=6)
    wait_all.assert_called_once_with(["prefetch-ref"], timeout=6)
    state.reset.assert_called_once_with()
    assert state.prefetched_samples_ref is None and state.last_step_current_deficit == 0
    assert utils_module.CURRENT_ROLLOUT_BATCH == []
