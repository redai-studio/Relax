# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


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
    }
    get = Mock(return_value=response)
    config = Mock(return_value=SimpleNamespace(text_config=SimpleNamespace(num_experts=896)))
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(misc, "get_hf_config", config)
    result = engine.get_expert_weight_sync_layout()
    assert result["enable_eplb"] is True
    assert result["num_experts"] == 896
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
