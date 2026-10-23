# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Dependency-free execution boundary tests; no model construction or CUDA
initialization."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


_SOURCE = Path(__file__).resolve().parents[3] / "relax/models/deepseekv41/execution_config.py"
_SPEC = importlib.util.spec_from_file_location("v41_execution_config_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
configure_execution_runtime = _MODULE.configure_execution_runtime
validate_execution_config = _MODULE.validate_execution_config


def _provider(**changes: Any) -> SimpleNamespace:
    values = {
        "num_layers": 40,
        "num_attention_heads": 64,
        "o_groups": 8,
        "kv_lora_rank": 512,
        "qk_pos_emb_head_dim": 64,
        "csa_window_size": 128,
        "params_dtype": "torch.bfloat16",
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "pipeline_model_parallel_layout": None,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "context_parallel_size": 1,
        "cp_partition_mode": "zigzag",
        "dynamic_context_parallel": False,
        "variable_seq_lengths": True,
        "recompute_granularity": None,
        "recompute_modules": [],
        "v41_qkv_format": "bshd",
        "v41_allgather_cp": False,
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tp,sequence_parallel", [(1, False), (2, True), (8, True)])
def test_validate_execution_config_allows_tp_with_and_without_sp(tp: int, sequence_parallel: bool) -> None:
    validate_execution_config(_provider(tensor_model_parallel_size=tp, sequence_parallel=sequence_parallel))


@pytest.mark.parametrize("pp", [1, 2, 8])
def test_validate_execution_config_allows_uniform_pp(pp: int) -> None:
    validate_execution_config(_provider(pipeline_model_parallel_size=pp))


@pytest.mark.parametrize("recompute", [None, "full", "selective"])
def test_validate_execution_config_allows_standard_moe_recompute(recompute: str | None) -> None:
    validate_execution_config(_provider(recompute_granularity=recompute, recompute_modules=["moe", "mlp"]))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"v41_qkv_format": "thd"}, "packed THD"),
        ({"dynamic_context_parallel": True}, "fixed context"),
        ({"context_parallel_size": 0}, "positive"),
        ({"context_parallel_size": 2, "cp_partition_mode": "contiguous"}, "allgather-cp"),
        ({"context_parallel_size": 2}, "allgather-cp"),
        ({"pipeline_model_parallel_size": 0}, "positive"),
        ({"pipeline_model_parallel_size": 3}, "divisible by PP"),
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline"),
        ({"pipeline_model_parallel_layout": "Ett|ttL"}, "custom pipeline"),
        ({"pipeline_model_parallel_size": 2, "num_layers_in_first_pipeline_stage": 8}, "uniformly divided"),
        ({"pipeline_model_parallel_size": 2, "num_layers_in_last_pipeline_stage": 8}, "uniformly divided"),
        ({"pipeline_model_parallel_size": 2, "variable_seq_lengths": False}, "cross-layer carry"),
        ({"pipeline_model_parallel_size": 2, "overlap_moe_expert_parallel_comm": True}, "requires VPP"),
        ({"recompute_granularity": "selective", "recompute_modules": ["moe", "mhc"]}, "mhc"),
        ({"recompute_granularity": "selective", "recompute_modules": ["core_attn"]}, "core_attn"),
        ({"params_dtype": "torch.float16"}, "BF16"),
        ({"params_dtype": "torch.float32"}, "BF16"),
        ({"fp8": "e4m3"}, "BF16"),
        ({"fp4": "e2m1"}, "BF16"),
        ({"kv_lora_rank": 256}, "kv_lora_rank=512"),
        ({"qk_pos_emb_head_dim": 32}, "qk_pos_emb_head_dim=64"),
        ({"csa_window_size": 64}, "csa_window_size=128"),
        ({"tensor_model_parallel_size": 0}, "positive"),
        ({"tensor_model_parallel_size": 16}, "o_groups=8"),
        ({"tensor_model_parallel_size": 3}, "num_attention_heads=64"),
        ({"o_groups": 3}, "divisible by o_groups"),
    ],
)
def test_validate_execution_config_rejects_unsupported_paths(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_execution_config(_provider(**changes))


@pytest.mark.parametrize("cp_partition_mode", ["zigzag", "contiguous"])
def test_validate_execution_config_allows_allgather_cp_and_preserves_runtime_options(cp_partition_mode: str) -> None:
    provider = _provider(
        context_parallel_size=2,
        cp_partition_mode=cp_partition_mode,
        v41_allgather_cp=True,
        pipeline_model_parallel_size=2,
        recompute_granularity="full",
        optimizer_cpu_offload=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="deepep",
    )
    before = vars(provider).copy()
    validate_execution_config(provider)
    assert vars(provider) == before


@pytest.mark.parametrize(("pp", "recompute"), [(1, None), (2, "full")])
def test_configure_execution_runtime_forwards_existing_options(pp: int, recompute: str | None) -> None:
    provider = _provider(pipeline_model_parallel_size=pp, recompute_granularity=recompute, variable_seq_lengths=False)
    args = SimpleNamespace(hf_checkpoint="fixture-checkpoint", qkv_format="bshd", allgather_cp=False)
    configure_execution_runtime(provider, args)
    assert provider.v41_hf_checkpoint == args.hf_checkpoint
    assert provider.v41_qkv_format == "bshd"
    assert provider.v41_allgather_cp is False
    assert provider.variable_seq_lengths is True


def test_configure_execution_runtime_leaves_configuration_only_small_shapes_usable() -> None:
    provider = _provider(kv_lora_rank=32, qk_pos_emb_head_dim=16, csa_window_size=16, params_dtype=None)
    configure_execution_runtime(provider, SimpleNamespace())
    assert provider.v41_qkv_format == "bshd"
    assert provider.v41_hf_checkpoint is None


def test_configure_execution_runtime_rejects_dynamic_cp_even_if_generic_provider_cleared_flag() -> None:
    with pytest.raises(ValueError, match="fixed context"):
        configure_execution_runtime(
            _provider(dynamic_context_parallel=False), SimpleNamespace(dynamic_context_parallel=True)
        )


def test_configure_execution_runtime_does_not_drop_overlap_request() -> None:
    with pytest.raises(ValueError, match="requires VPP"):
        configure_execution_runtime(
            _provider(pipeline_model_parallel_size=2), SimpleNamespace(overlap_moe_expert_parallel_comm=True)
        )


@pytest.mark.parametrize("cp_size", [1, 2])
def test_configure_execution_runtime_allows_bshd_rollout_routing_replay(cp_size: int) -> None:
    provider = _provider(context_parallel_size=cp_size, v41_allgather_cp=cp_size > 1)
    configure_execution_runtime(provider, SimpleNamespace(use_rollout_routing_replay=True))
    assert provider.v41_qkv_format == "bshd"
    assert provider.cp_partition_mode == "zigzag"
    assert provider.v41_allgather_cp == (cp_size > 1)


def test_configure_execution_runtime_forwards_allgather_without_changing_native_cp_partition_mode() -> None:
    provider = _provider(context_parallel_size=2)
    configure_execution_runtime(
        provider,
        SimpleNamespace(qkv_format="bshd", allgather_cp=True, use_rollout_routing_replay=True),
    )
    assert provider.v41_allgather_cp is True
    assert provider.cp_partition_mode == "zigzag"


@pytest.mark.parametrize("field", ["hidden_dropout", "attention_dropout"])
def test_validate_execution_config_rejects_nonzero_dropout(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field}=0.0"):
        validate_execution_config(_provider(**{field: 0.1}))


def test_validate_execution_config_rejects_unported_training_cuda_graphs() -> None:
    with pytest.raises(NotImplementedError, match="CUDA-graph"):
        validate_execution_config(_provider(cuda_graph_impl="local"))
