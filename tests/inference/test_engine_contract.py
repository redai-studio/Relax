# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run the engine's real HTTP lifecycle methods without importing GPU
libraries."""

import ast
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests


def engine_contract():
    path = Path(__file__).resolve().parents[2] / "relax/backends/sglang/sglang_engine.py"
    source = ast.parse(path.read_text())
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "SGLangEngine")
    names = {
        "_make_request",
        "_require_dynamic_weights",
        "register_dcs",
        "drain",
        "release_memory_occupation",
        "resume_memory_occupation",
        "pause_generation",
        "continue_generation",
        "abort_requests",
        "shutdown",
        "_init_normal",
    }
    cls = ast.ClassDef(
        name="Engine",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names],
    )
    namespace = dict(
        requests=requests,
        time=time,
        logger=Mock(),
        _SGLANG_HTTP_ATTEMPT_TIMEOUT_S=1,
        _GENRM_OFFLOAD_RELEASE_TIMEOUT_S=1,
        _GENRM_OFFLOAD_DRAIN_TIMEOUT_S=1,
        _MIN_HTTP_TIMEOUT_S=0.01,
        _MAX_CONSECUTIVE_CONNECT_ERRORS=1,
        kill_process_tree=Mock(),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    engine = namespace["Engine"]()
    engine.role = "teacher"
    engine.node_rank = 0
    engine.server_host, engine.server_port = "host", 8000
    engine._loaded_memory_tags = {"weights", "kv_cache", "cuda_graph"}
    engine._generation_paused = False
    engine._memory_released = False
    return engine


def test_shutdown_before_init_and_after_cleanup_is_idempotent():
    engine = engine_contract()
    engine.args = SimpleNamespace(rollout_external=False)
    del engine.server_host, engine.server_port
    engine.shutdown()
    engine.shutdown.__globals__["kill_process_tree"].assert_not_called()
    engine.server_host, engine.server_port = "host", 8000
    engine.unregister_from_router = Mock()
    process = Mock(pid=123)
    process.is_alive.return_value = False
    engine.process = process
    engine.shutdown()
    engine.shutdown()
    engine.shutdown.__globals__["kill_process_tree"].assert_called_once_with(123)
    process.join.assert_called_once_with(timeout=10)
    assert engine.process is None


def test_shutdown_does_not_acknowledge_process_that_remains_alive():
    engine = engine_contract()
    engine.args = SimpleNamespace(rollout_external=False)
    engine.unregister_from_router = Mock()
    engine.process = Mock(pid=123)
    engine.process.is_alive.return_value = True
    with pytest.raises(RuntimeError, match="did not exit"):
        engine.shutdown()
    assert engine.process is not None


@pytest.mark.parametrize("kill_raises", [False, True])
def test_startup_failure_retains_process_until_confirmed_cleanup(monkeypatch, kill_raises):
    engine = engine_contract()
    engine.args = SimpleNamespace(rollout_external=False)
    engine.unregister_from_router = Mock()
    process = Mock(pid=123)
    process.is_alive.return_value = True
    namespace = engine._init_normal.__globals__
    namespace.update(
        multiprocessing=multiprocessing,
        Callable=Callable,
        ServerArgs=lambda **kwargs: SimpleNamespace(**kwargs, url=lambda: "http://host:8000"),
        Envs=SimpleNamespace(
            RELAX_OPTIMIZE_ROUTING_REPLAY=False, RELAX_OPD_PREEXPANDED_PATCH=False, RELAX_OPD_PER_POS_TOKEN_IDS=False
        ),
        _launch_server_with_patches=Mock(),
        _wait_server_healthy=Mock(side_effect=RuntimeError("startup unhealthy")),
    )
    path = Path(__file__).resolve().parents[2] / "relax/backends/sglang/sglang_engine.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "launch_server_process"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(multiprocessing, "Process", Mock(return_value=process))
    monkeypatch.setattr(multiprocessing, "set_start_method", Mock())
    kill = namespace["kill_process_tree"]
    if kill_raises:
        kill.side_effect = OSError("kill failed")
    with pytest.raises((OSError, RuntimeError)):
        engine._init_normal({"host": "host", "node_rank": 0, "api_key": None}, apply_policy_load_plan=False)
    process.start.assert_called_once()
    assert engine.process is process
    # Cleanup is retried through the same real shutdown method. A child
    # still alive must not be acknowledged as released.
    kill.side_effect = None
    with pytest.raises(RuntimeError, match="did not exit"):
        engine.shutdown()
    assert engine.process is process
    process.is_alive.return_value = False
    engine.shutdown()
    assert engine.process is None


def test_engine_repeated_full_partial_and_drain_activation_are_idempotent(monkeypatch):
    engine = engine_contract()
    offloaded = set()
    calls = []

    def post(url, json, timeout):
        endpoint = url.rsplit("/", 1)[-1]
        calls.append(endpoint)
        if endpoint == "release_memory_occupation":
            offloaded.update({"weights", "kv_cache", "cuda_graph"})
        if endpoint == "resume_memory_occupation":
            assert json["tags"], "SGLang treats empty tags as all tags"
            for tag in json["tags"]:
                offloaded.remove(tag)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200))
    engine.resume_memory_occupation()
    assert calls == []
    engine.drain()
    engine.resume_memory_occupation()
    assert "resume_memory_occupation" not in calls
    assert calls[-1] == "continue_generation"
    engine.release_memory_occupation()
    engine.release_memory_occupation()
    engine.resume_memory_occupation(["weights"])
    engine.resume_memory_occupation(["weights"])
    assert offloaded == {"kv_cache", "cuda_graph"}
    assert engine._generation_paused
    engine.resume_memory_occupation()
    engine.resume_memory_occupation()
    assert not offloaded
    assert calls.count("release_memory_occupation") == 1
    assert calls.count("resume_memory_occupation") == 2


@pytest.mark.parametrize("role", ["genrm", "teacher"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "update_weights_from_tensor",
        "init_weights_update_group",
        "send_weights_to_remote_instance",
        "load_lora_adapter",
        "post_process_weights",
    ],
)
def test_static_model_rejects_weight_mutations_before_network(monkeypatch, role, endpoint):
    engine = engine_contract()
    engine.role = role
    network = Mock()
    monkeypatch.setattr(requests, "post", network)
    with pytest.raises(RuntimeError, match="Static"):
        engine._make_request(endpoint)
    network.assert_not_called()


@pytest.mark.parametrize("role", ["genrm", "teacher"])
def test_static_model_rejects_dcs_registration_before_client_creation(role):
    engine = engine_contract()
    engine.role = role
    engine.args = SimpleNamespace(fully_async=True)
    with pytest.raises(RuntimeError, match="Static"):
        engine.register_dcs()


def test_failed_pause_never_releases_gpu_memory(monkeypatch):
    engine = engine_contract()
    network = Mock(side_effect=requests.ConnectionError("admission not fenced"))
    monkeypatch.setattr(requests, "post", network)
    with pytest.raises(requests.ConnectionError):
        engine.release_memory_occupation()
    assert network.call_count == 1
    assert network.call_args.args[0].endswith("/pause_generation")
    assert not engine._memory_released


def test_group_pipeline_override_preserves_gpu_width_and_local_node_rank():
    path = Path(__file__).resolve().parents[2] / "relax/backends/sglang/sglang_engine.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_server_args"
    )
    namespace = dict(
        os=os,
        is_s3_uri=lambda path: False,
        _to_local_gpu_id=lambda gpu: gpu,
        is_lora_enabled=lambda args: False,
        _enable_draft_weights_cpu_backup=lambda *args: False,
        _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS=[],
        ServerArgs=object,
        logger=Mock(),
        dataclasses=SimpleNamespace(
            fields=lambda cls: [SimpleNamespace(name=name) for name in ["tp_size", "pp_size", "node_rank"]]
        ),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(
        num_gpus_per_node=2,
        rollout_num_gpus_per_engine=4,
        hf_checkpoint="checkpoint",
        seed=1,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
    )
    kwargs, _ = namespace["_compute_server_args"](
        args, 1, "node:10000", 10001, "node", 10002, base_gpu_id=0, sglang_overrides={"pp_size": 2, "node_rank": 0}
    )
    assert kwargs["tp_size"] * kwargs["pp_size"] == 4
    assert kwargs["node_rank"] == 0


@pytest.mark.parametrize("offload", [False, True])
def test_genrm_memory_saver_disables_incompatible_default_prefill_graph(offload):
    path = Path(__file__).resolve().parents[2] / "relax/backends/sglang/sglang_engine.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_genrm_server_args"
    )
    namespace = dict(
        os=os,
        is_s3_uri=lambda path: False,
        _to_local_gpu_id=lambda gpu: gpu,
        _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS=[],
        ServerArgs=object,
        logger=Mock(),
        dataclasses=SimpleNamespace(
            fields=lambda cls: [
                SimpleNamespace(name=name) for name in ["enable_memory_saver", "cuda_graph_backend_prefill"]
            ]
        ),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(
        num_gpus_per_node=1,
        genrm_num_gpus_per_engine=2,
        genrm_model_path="checkpoint",
        genrm_engine_config={},
        seed=1,
        offload_rollout=offload,
        fp16=False,
    )
    kwargs, _ = namespace["_compute_genrm_server_args"](args, 0, "node:10000", 10001, "node", 10002, base_gpu_id=0)
    assert kwargs.get("cuda_graph_backend_prefill") == ("disabled" if offload else None)
