# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import sys
from dataclasses import make_dataclass
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


@pytest.fixture
def expert_sync_engine_module(monkeypatch, sglang_engine_module):
    import relax.utils

    misc = ModuleType("relax.utils.misc")
    misc.get_hf_config = Mock()
    monkeypatch.setitem(sys.modules, misc.__name__, misc)
    monkeypatch.setattr(relax.utils, "misc", misc, raising=False)
    return sglang_engine_module


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
    enable_mtp_training, speculative_algorithm, overrides, expected
):
    pytest.importorskip("sglang.srt.server_args", exc_type=ImportError)

    from relax.backends.sglang.sglang_engine import _enable_draft_weights_cpu_backup

    args = SimpleNamespace(
        enable_mtp_training=enable_mtp_training,
        sglang_speculative_algorithm=speculative_algorithm,
    )

    assert _enable_draft_weights_cpu_backup(args, overrides) is expected


def test_expert_sync_layout_uses_runtime_overrides(monkeypatch, expert_sync_engine_module):
    from relax.utils import misc

    module = expert_sync_engine_module
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
        "custom_weight_loader": ["relax.backends.sglang.weight_loader.load_expert_slices"],
    }
    get = Mock(return_value=response)
    config = Mock(return_value=SimpleNamespace(text_config=SimpleNamespace(num_experts=896)))
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(misc, "get_hf_config", config)
    result = engine.get_expert_weight_sync_layout()
    assert result["enable_eplb"] is True
    assert result["num_experts"] == 896
    assert result["custom_weight_loader"] == ["relax.backends.sglang.weight_loader.load_expert_slices"]
    assert "ignored_server_field" not in result
    config.assert_called_once_with("/test/runtime-model")
    get.assert_called_once_with("http://localhost:1234/server_info", timeout=30)
    response.raise_for_status.assert_called_once()


def test_expert_sync_version_commit_does_not_abort_again(monkeypatch, sglang_engine_module):
    engine = object.__new__(sglang_engine_module.SGLangEngine)
    request = Mock(return_value={"success": True})
    monkeypatch.setattr(engine, "_make_request", request)
    assert engine.update_weight_version("7") == {"success": True}
    request.assert_called_once_with("update_weight_version", {"new_version": "7", "abort_all_requests": False})


@pytest.mark.parametrize(
    ("runtime", "accelerator", "local_engine", "expected_backend"),
    [
        ({}, "cuda", True, "triton"),
        ({"moe_runner_backend": "triton"}, "cuda", True, "triton"),
        ({"moe_runner_backend": "flashinfer_trtllm"}, "cuda", True, "flashinfer_trtllm"),
        ({"moe_runner_backend": "triton_kernel"}, "cuda", True, "triton_kernel"),
        ({"moe_runner_backend": "deep_gemm"}, "cuda", True, "deep_gemm"),
        ({"moe_runner_backend": "flashinfer_cutlass"}, "cuda", True, "flashinfer_cutlass"),
        ({"moe_runner_backend": "deep_gemm"}, "rocm", True, "deep_gemm"),
        ({"moe_a2a_backend": "deepep"}, "cuda", True, "triton"),
        ({"device": "cpu"}, "cuda", True, "auto"),
        ({}, "rocm", True, "auto"),
        (None, "cuda", True, "auto"),
        ({}, "cuda", False, "auto"),
        ({"moe_runner_backend": "deep_gemm"}, "cuda", False, "deep_gemm"),
        ({"moe_runner_backend": "flashinfer_cutlass"}, "cuda", False, "flashinfer_cutlass"),
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_expert_sync_auto_uses_scheduler_config_and_dtype_resolver(
    monkeypatch, expert_sync_engine_module, runtime, accelerator, local_engine, expected_backend, dtype
):
    from relax.utils import misc

    module = expert_sync_engine_module
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank, engine.worker_type = 0, "regular"
    engine.server_host, engine.server_port = "localhost", 1234
    if local_engine:
        engine.process = object()
    startup = dict(
        model_path="/test/model",
        dtype="auto",
        device="cuda",
        moe_runner_backend="auto",
        moe_a2a_backend="none",
        disable_shared_experts_fusion=True,
    )
    states = [] if runtime is None else [startup | runtime]
    response = Mock()
    response.json.return_value = startup | {"internal_states": states}
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))
    text_config = SimpleNamespace(num_experts=256, dtype=dtype)
    monkeypatch.setattr(misc, "get_hf_config", lambda _: SimpleNamespace(text_config=text_config))
    model_config = ModuleType("sglang.srt.configs.model_config")
    resolver = Mock(return_value=f"torch.{dtype}")
    model_config._get_and_verify_dtype = resolver
    monkeypatch.setitem(sys.modules, model_config.__name__, model_config)
    monkeypatch.setattr(module.device_utils, "get_accelerator_type", lambda: accelerator, raising=False)
    monkeypatch.setattr(module.device_utils, "AcceleratorType", SimpleNamespace(CUDA="cuda"), raising=False)

    layout = engine.get_expert_weight_sync_layout()

    assert layout["moe_runner_backend"] == expected_backend
    assert layout["is_cuda"] is (
        local_engine and accelerator == "cuda" and (runtime or {}).get("device", "cuda") == "cuda"
    )
    assert layout["dtype"] == dtype
    assert layout["disable_shared_experts_fusion"] is True
    resolver.assert_called_once_with(text_config, "auto")
    assert module.requests.get.call_count == (2 if runtime is None else 1)
    assert startup["dtype"] == startup["moe_runner_backend"] == "auto"


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float16"),
        ("disable_shared_experts_fusion", False),
        ("ep_size", 2),
        ("moe_dp_size", 2),
        ("dwdp_size", 2),
        ("elastic_ep_backend", "mooncake"),
    ],
)
def test_expert_sync_rejects_inconsistent_scheduler_layouts(monkeypatch, expert_sync_engine_module, field, value):
    module = expert_sync_engine_module
    engine = object.__new__(module.SGLangEngine)
    engine.node_rank, engine.worker_type = 0, "regular"
    engine.server_host, engine.server_port = "localhost", 1234
    startup = dict(model_path="/test/model", dtype="bfloat16", disable_shared_experts_fusion=True)
    response = Mock()
    response.json.return_value = startup | {"internal_states": [startup, startup | {field: value}]}
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))

    with pytest.raises(ValueError, match="schedulers disagree"):
        engine.get_expert_weight_sync_layout()


@pytest.mark.parametrize(
    ("enforce", "overrides", "expected_disabled"),
    [
        (False, None, True),
        (True, None, False),
        (False, {"disable_shared_experts_fusion": False}, False),
        (False, {"disable_shared_experts_fusion": True}, True),
        (False, {"enforce_shared_experts_fusion": True}, False),
    ],
)
def test_policy_shared_experts_default_respects_explicit_overrides(
    monkeypatch, sglang_engine_module, enforce, overrides, expected_disabled
):
    module = sglang_engine_module
    server_args = make_dataclass(
        "ServerArgs",
        [("disable_shared_experts_fusion", bool, False), ("enforce_shared_experts_fusion", bool, False)],
    )
    monkeypatch.setattr(module, "ServerArgs", server_args)
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu_id: gpu_id)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="/test/model",
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_disable_shared_experts_fusion=False,
        sglang_enforce_shared_experts_fusion=enforce,
    )

    result, _ = module._compute_server_args(
        args, 0, "localhost:8000", 8001, "localhost", 8002, base_gpu_id=0, sglang_overrides=overrides
    )

    assert result["disable_shared_experts_fusion"] is expected_disabled


@pytest.mark.parametrize(
    "existing", [None, [], ["custom.loader"], ["relax.backends.sglang.weight_loader.load_expert_slices"]]
)
def test_expert_slice_loader_registration_preserves_user_loaders(monkeypatch, sglang_engine_module, existing):
    module = sglang_engine_module
    monkeypatch.setattr(module, "ServerArgs", make_dataclass("ServerArgs", [("custom_weight_loader", list, None)]))
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda value: value)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="/test/model",
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
    )
    result, _ = module._compute_server_args(
        args,
        0,
        "localhost:8000",
        8001,
        "localhost",
        8002,
        base_gpu_id=0,
        sglang_overrides={"custom_weight_loader": existing},
    )
    expected = list(dict.fromkeys([*(existing or []), "relax.backends.sglang.weight_loader.load_expert_slices"]))
    assert result["custom_weight_loader"] == expected
    assert result["custom_weight_loader"] is not existing


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
def test_policy_pd_routing_replay_accepts_configured_topology(monkeypatch, sglang_engine_module, worker_type):
    module = sglang_engine_module
    monkeypatch.setattr(
        module,
        "ServerArgs",
        make_dataclass(
            "ServerArgs", [("disaggregation_mode", str, "null"), ("enable_return_routed_experts", bool, False)]
        ),
    )
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu_id: gpu_id)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        hf_checkpoint="/test/model",
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=True,
        sglang_disaggregation_mode=None,
        fp16=False,
    )
    result, _ = module._compute_server_args(
        args,
        0,
        "localhost:8000",
        8001,
        "localhost",
        8002,
        worker_type=worker_type,
        disaggregation_bootstrap_port=8003,
        base_gpu_id=0,
        sglang_overrides=None,
    )

    assert result["disaggregation_mode"] == worker_type
    assert result["enable_return_routed_experts"] is True


@pytest.mark.parametrize("engine_class,actual_mode", [("SGLangEngine", "prefill"), ("GenRMEngine", "decode")])
def test_external_pd_routing_replay_registers_native_worker(
    monkeypatch, sglang_engine_module, engine_class, actual_mode
):
    module = sglang_engine_module
    engine = object.__new__(getattr(module, engine_class))
    engine.args = SimpleNamespace(use_rollout_routing_replay=True)
    engine.rank, engine.worker_type = 0, "regular"
    engine.server_host, engine.server_port = "localhost", 8000
    engine._skip_router_registration = False
    engine.process = None
    engine.register_to_router = Mock()
    response = Mock()
    response.json.return_value = {"disaggregation_mode": actual_mode}
    monkeypatch.setattr(module.requests, "get", Mock(return_value=response))
    monkeypatch.setattr(module, "_wait_server_healthy", Mock())
    # Actual PD topology is accepted for both policy and independent GenRM engines.
    engine._init_external({}, external_engine_need_check_fields=[])
    engine.register_to_router.assert_called_once_with()


@pytest.mark.parametrize(
    "engine,worker,gpus,engine_gpus,cli,overrides,expected",
    [
        ("policy", "regular", 2, None, [], {}, "cuda_ipc"),
        ("policy", "prefill", 32, None, [], {}, None),
        ("policy", "decode", 32, None, [], {}, None),
        ("policy", "regular", 32, 2, [], {}, "cuda_ipc"),
        ("genrm", "regular", 32, 2, [], {}, "cuda_ipc"),
        ("genrm", "regular", 2, 32, [], {}, None),
        ("genrm", "regular", 2, 2, [], {"nnodes": 4}, None),
        ("genrm", "regular", 2, 2, [], {"mm_feature_transport": None}, None),
        ("policy", "regular", 2, None, ["--sglang-mm-feature-transport", "cpu"], {}, "cpu"),
        (
            "policy",
            "regular",
            32,
            None,
            ["--sglang-mm-feature-transport", "cuda_ipc"],
            {"mm_feature_transport": "cpu"},
            "cpu",
        ),
        ("policy", "regular", 32, None, ["--sglang-mm-feature-transport", "cuda_ipc"], {}, "cuda_ipc"),
    ],
)
def test_transport_respects_engine_topology_and_native_validation(
    request, monkeypatch, engine, worker, gpus, engine_gpus, cli, overrides, expected
):
    from relax.backends.sglang.arguments import ServerArgs, add_sglang_arguments

    # Keep the real CLI and SGLang validation; isolate only hardware/model discovery.
    parser = add_sglang_arguments(argparse.ArgumentParser(add_help=False))
    parsed = parser.parse_args(cli)
    args = SimpleNamespace(
        rollout_num_gpus_per_engine=gpus,
        genrm_num_gpus_per_engine=engine_gpus or 2,
        num_gpus_per_node=8,
        hf_checkpoint="policy-model",
        genrm_model_path="reward-model",
        genrm_engine_config=overrides,
        seed=42,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        sglang_mm_feature_transport=parsed.sglang_mm_feature_transport,
    )
    module = request.getfixturevalue("sglang_engine_module")
    monkeypatch.setattr(module, "ServerArgs", ServerArgs)
    monkeypatch.setattr(module, "_to_local_gpu_id", lambda gpu: gpu)
    compute = module._compute_genrm_server_args if engine == "genrm" else module._compute_server_args
    options = {} if engine == "genrm" else {"sglang_overrides": overrides, "num_gpus_per_engine": engine_gpus}
    actual, _ = compute(
        args,
        0,
        "localhost:1234",
        1235,
        "localhost",
        1236,
        worker_type=worker,
        disaggregation_bootstrap_port=1237,
        base_gpu_id=0,
        **options,
    )
    assert actual["mm_feature_transport"] == expected

    cfg = SimpleNamespace(
        **{"encoder_only": False, "tokenizer_worker_num": 1, "disaggregation_mode": "null", **actual}
    )
    monkeypatch.delenv("SGLANG_USE_CUDA_IPC_TRANSPORT", raising=False)
    if hasattr(ServerArgs, "_handle_multimodal_feature_transport"):
        cfg.keep_mm_feature_on_device = False
        validate_transport = ServerArgs._handle_multimodal_feature_transport
        monkeypatch.setitem(validate_transport.__globals__, "is_cuda", lambda: True)
        # 0.5.17's native auto policy chooses IPC for a regular single-node engine.
        native_default = "cuda_ipc" if cfg.nnodes == 1 and cfg.disaggregation_mode == "null" else "cpu"
    else:
        from sglang.srt.arg_groups import serving_hook

        monkeypatch.setattr(serving_hook, "resolving_view", lambda _: cfg)
        monkeypatch.setattr(serving_hook, "model_config_of", lambda _: SimpleNamespace(is_multimodal=True))
        monkeypatch.setattr(serving_hook, "get_platform", lambda: SimpleNamespace(is_cuda=True))
        monkeypatch.setattr(serving_hook, "is_mnnvl_fabric_device", lambda: False)
        monkeypatch.setattr(serving_hook, "declare_resolution", lambda _, __, **values: vars(cfg).update(values))
        validate_transport = serving_hook.handle_multimodal_feature_transport
        native_default = "cpu"
    if expected == "cuda_ipc" and actual["nnodes"] > 1:
        # Explicit incompatible user choices must still receive SGLang's error.
        with pytest.raises(ValueError, match="only supports a single node"):
            validate_transport(cfg)
    else:
        validate_transport(cfg)
        assert cfg.mm_feature_transport == (expected or native_default)
