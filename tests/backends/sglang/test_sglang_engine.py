# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import dataclasses
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


pytest_plugins = ["tests.backends.sglang.test_router_registration"]


@pytest.mark.parametrize("node_rank", [0, 1])
def test_launch_server_process_returns_owned_process_and_only_head_waits(monkeypatch, sglang_engine_module, node_rank):
    module = sglang_engine_module
    events = []

    process = Mock()
    process.start.side_effect = lambda: events.append("start")
    process.is_alive.return_value = True
    multiprocessing = SimpleNamespace(set_start_method=Mock(), Process=Mock(return_value=process))

    wait = Mock(side_effect=lambda **_kwargs: events.append("wait"))
    monkeypatch.setattr(module, "multiprocessing", multiprocessing)
    monkeypatch.setattr(module, "_wait_server_healthy", wait)
    monkeypatch.setattr(module.os, "getpid", lambda: 123)
    monkeypatch.setattr(module.Envs, "RELAX_OPTIMIZE_ROUTING_REPLAY", False, raising=False)
    monkeypatch.setattr(module.Envs, "RELAX_OPD_PER_POS_TOKEN_IDS", False, raising=False)
    server_args = SimpleNamespace(node_rank=node_rank, url=lambda: "http://worker:8000", api_key="api_key")

    assert module.launch_server_process(server_args) is process

    multiprocessing.set_start_method.assert_called_once_with("spawn", force=True)
    multiprocessing.Process.assert_called_once_with(target=module._launch_server_with_patches, args=(server_args, 123))
    process.start.assert_called_once_with()
    if node_rank == 0:
        assert events == ["start", "wait"]
        wait.assert_called_once()
        assert wait.call_args.kwargs["base_url"] == "http://worker:8000"
        assert wait.call_args.kwargs["api_key"] == "api_key"
        assert wait.call_args.kwargs["is_process_alive"]()
        process.is_alive.assert_called_once_with()
    else:
        assert events == ["start"]
        wait.assert_not_called()


@pytest.mark.parametrize(
    ("routing_replay", "parent_alive"),
    [(False, True), (True, True), (False, False)],
    ids=["plain-launch", "patched-launch", "parent-exited"],
)
def test_server_child_installs_parent_guard_before_check_and_preserves_patches(
    monkeypatch, sglang_engine_module, routing_replay, parent_alive
):
    module = sglang_engine_module
    events = []
    http_server = ModuleType("sglang.srt.entrypoints.http_server")
    http_server.launch_server = Mock(side_effect=lambda *_args, **_kwargs: events.append("launch"))
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.http_server", http_server)
    monkeypatch.setattr(
        sys.modules["sglang.srt.utils"],
        "kill_itself_when_parent_died",
        lambda: events.append("guard"),
        raising=False,
    )

    def getppid():
        events.append("parent_check")
        return 123 if parent_alive else 1

    monkeypatch.setattr(module.os, "getppid", getppid)
    monkeypatch.setattr(module.Envs, "RELAX_OPTIMIZE_ROUTING_REPLAY", routing_replay, raising=False)
    server_args = SimpleNamespace()
    module._launch_server_with_patches(server_args, 123)

    if parent_alive:
        assert events == ["guard", "parent_check", "launch"]
        kwargs = {"run_scheduler_process_func": module._patched_run_scheduler_process} if routing_replay else {}
        http_server.launch_server.assert_called_once_with(server_args, **kwargs)
    else:
        assert events == ["guard", "parent_check"]
        http_server.launch_server.assert_not_called()


def _is_record_fields_import(node: ast.AST) -> bool:
    return isinstance(node, ast.Try) and any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "sglang.srt.arg_groups.arg_utils"
        and any(alias.name == "record_fields" for alias in statement.names)
        for statement in node.body
    )


@pytest.fixture()
def server_args_module() -> SimpleNamespace:
    """Run complete argument builders with the real SGLang ServerArgs."""
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.utils.logging_utils import get_logger
    from relax.utils.s3_model_loader import is_s3_uri

    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "relax/backends/sglang/sglang_engine.py"
    functions = {"_compute_server_args", "_compute_genrm_server_args", "_enable_draft_weights_cpu_backup"}
    imports = {"sglang.srt.arg_groups.arg_utils", "sglang.srt.server_args"}
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.ImportFrom) and node.module in imports)
        or _is_record_fields_import(node)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS"
                for target in node.targets
            )
        )
    ]
    peft_source = repo_root / "relax/utils/megatron_peft_utils.py"
    nodes.extend(
        node
        for node in ast.parse(peft_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "is_lora_enabled"
    )
    namespace = {
        "dataclasses": dataclasses,
        "os": os,
        "logger": get_logger(__name__),
        "is_s3_uri": is_s3_uri,
        "_to_local_gpu_id": lambda gpu_id: gpu_id,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("arg_utils_available", [False, True], ids=["missing-module", "missing-function"])
def test_record_fields_import_falls_back_to_dataclasses(
    monkeypatch: pytest.MonkeyPatch, arg_utils_available: bool
) -> None:
    source = Path(__file__).resolve().parents[3] / "relax/backends/sglang/sglang_engine.py"
    nodes = [node for node in ast.parse(source.read_text()).body if _is_record_fields_import(node)]
    assert len(nodes) == 1

    for name in ("sglang", "sglang.srt", "sglang.srt.arg_groups"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module_name = "sglang.srt.arg_groups.arg_utils"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name) if arg_utils_available else None)

    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["record_fields"] is dataclasses.fields


def test_compute_server_args_accepts_installed_server_args(server_args_module: SimpleNamespace) -> None:
    module = server_args_module

    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="/models/policy",
        seed=17,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_mem_fraction_static=0.6,
        sglang_attention_backend="flashinfer",
        sglang_not_a_server_field=True,
    )

    kwargs, _ = module._compute_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=1236,
        base_gpu_id=0,
        sglang_overrides={"mem_fraction_static": 0.7},
    )

    assert "not_a_server_field" not in kwargs
    server_args = module.ServerArgs(**kwargs)
    assert server_args.model_path == "/models/policy"
    assert server_args.tp_size == 2
    assert server_args.mem_fraction_static == 0.7
    assert server_args.attention_backend == "flashinfer"
    assert server_args.cuda_graph_backend_prefill == "disabled"


def test_compute_genrm_server_args_accepts_installed_server_args(server_args_module: SimpleNamespace) -> None:
    module = server_args_module

    args = SimpleNamespace(
        genrm_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        genrm_model_path="/models/genrm",
        seed=17,
        offload_rollout=False,
        genrm_engine_config={"attention_backend": "triton", "not_a_server_field": True},
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_mem_fraction_static=0.6,
        sglang_attention_backend="flashinfer",
        sglang_load_format="dummy",
    )

    kwargs, _ = module._compute_genrm_server_args(
        args,
        rank=0,
        dist_init_addr="127.0.0.1:1234",
        nccl_port=1235,
        host="127.0.0.1",
        port=1236,
        base_gpu_id=0,
    )

    assert "not_a_server_field" not in kwargs
    server_args = module.ServerArgs(**kwargs)
    assert server_args.model_path == "/models/genrm"
    assert server_args.tp_size == 2
    assert server_args.mem_fraction_static == 0.6
    assert server_args.attention_backend == "triton"
    assert server_args.load_format == "auto"


@pytest.mark.parametrize(
    ("enable_mtp_training", "speculative_algorithm", "overrides", "expected"),
    [
        (True, "EAGLE", None, False),
        (False, None, None, False),
        (False, "EAGLE", None, True),
        (False, None, {"speculative_algorithm": "EAGLE"}, True),
        (False, "EAGLE", {"speculative_algorithm": None}, False),
    ],
)
def test_draft_weights_cpu_backup_follows_mtp_and_speculative_config(
    server_args_module, enable_mtp_training, speculative_algorithm, overrides, expected
):
    args = SimpleNamespace(
        enable_mtp_training=enable_mtp_training,
        sglang_speculative_algorithm=speculative_algorithm,
    )

    assert server_args_module._enable_draft_weights_cpu_backup(args, overrides) is expected


def test_expert_sync_layout_uses_runtime_overrides(monkeypatch, sglang_engine_module):
    from unittest.mock import Mock

    module = sglang_engine_module
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.server_host, engine.server_port = "localhost", 1234
    response = Mock()
    response.json.return_value = {
        "model_path": "/test/runtime-model",
        "tp_size": 16,
        "ep_size": 16,
        "enable_eplb": True,
        "moe_dp_size": 1,
        "ignored_server_field": "ignored",
        "json_model_override_args": "{}",
    }
    get = Mock(return_value=response)
    config = Mock(return_value=SimpleNamespace(text_config=SimpleNamespace(num_experts=896)))
    misc = ModuleType("relax.utils.misc")
    misc.get_hf_config = config
    monkeypatch.setitem(sys.modules, misc.__name__, misc)
    monkeypatch.setattr(module.requests, "get", get)
    result = engine.get_expert_weight_sync_layout()
    assert result["enable_eplb"] is True
    assert result["num_experts"] == 896
    assert "ignored_server_field" not in result
    config.assert_called_once_with("/test/runtime-model")
    get.assert_called_once_with("http://localhost:1234/server_info", timeout=30)
    response.raise_for_status.assert_called_once()


def test_expert_sync_version_commit_does_not_abort_again(monkeypatch, sglang_engine_module):
    from unittest.mock import Mock

    engine = object.__new__(sglang_engine_module.SGLangEngine)
    request = Mock(return_value={"success": True})
    monkeypatch.setattr(engine, "_make_request", request)
    assert engine.update_weight_version("7") == {"success": True}
    request.assert_called_once_with("update_weight_version", {"new_version": "7", "abort_all_requests": False})
