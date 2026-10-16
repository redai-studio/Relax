# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tests.backends.megatron.test_model_provider_vpp import _bridge_args, _FakeProvider, _load_model_provider


@pytest.fixture(autouse=True)
def _configuration_without_bridge_import(monkeypatch):
    package = ModuleType("relax.models.kimi_k3")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "relax/models/kimi_k3")]
    monkeypatch.setitem(sys.modules, package.__name__, package)


def test_kimi_k3_provider_preserves_hf_architecture_and_overrides_runtime(monkeypatch):
    provider = _FakeProvider()
    architecture = {
        "num_layers": 8,
        "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1, 1],
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "rotary_scaling_factor": 1.0,
        "rotary_base": 50000,
        "moe_router_pre_softmax": True,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
        "moe_router_dtype": "fp32",
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_shared_expert_intermediate_size": 6144,
        "moe_router_topk": 8,
        "moe_router_num_groups": 4,
        "moe_router_group_topk": 2,
        "moe_router_topk_scaling_factor": 1.5,
        "moe_router_score_function": "sigmoid",
        "moe_ffn_hidden_size": 2048,
        "activation_func_clamp_shared_expert": None,
    }
    vars(provider).update(architecture, kimi_kda_layers=(1, 3, 5, 7), fp8=None)
    module, _ = _load_model_provider(monkeypatch, provider)
    args = _bridge_args(
        **dict.fromkeys(architecture),
        virtual_pipeline_model_parallel_size=None,
        variable_seq_lengths=False,
        fp8="hybrid",
    )
    args.num_layers = 2
    args.moe_layer_freq = 1
    args.kv_lora_rank = 32
    args.moe_router_score_function = "softmax"

    module.get_model_provider_func(args)

    assert {key: getattr(provider, key) for key in architecture} == architecture
    assert provider.variable_seq_lengths is True
    assert provider.tensor_model_parallel_size == args.tensor_model_parallel_size
    assert provider.pipeline_model_parallel_size == args.pipeline_model_parallel_size
    assert provider.fp8 == "hybrid"
    assert provider.params_dtype is torch.bfloat16
    assert provider.finalized
    assert args.multimodal_input_key_map == {"image_grid_thw": "grid_thws"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_parallel_size": 2, "allgather_cp": True}, "zigzag partitioning"),
        ({"context_parallel_size": 2, "cp_partition_mode": "contiguous"}, "zigzag partitioning"),
        ({"dynamic_context_parallel": True}, "context parallelism"),
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline"),
    ],
)
def test_kimi_k3_provider_rejects_unsupported_parallelism_before_finalize(monkeypatch, overrides, message):
    provider = _FakeProvider()
    provider.kimi_kda_layers = (1, 3)
    module, _ = _load_model_provider(monkeypatch, provider)
    args = _bridge_args(**{"virtual_pipeline_model_parallel_size": None, **overrides})

    with pytest.raises(ValueError, match=message):
        module.get_model_provider_func(args)

    assert not provider.finalized


def test_non_kimi_k3_provider_keeps_architecture_override_behavior(monkeypatch):
    provider = _FakeProvider()
    provider.kv_lora_rank = 512
    module, _ = _load_model_provider(monkeypatch, provider)

    args = _bridge_args(num_layers=2, moe_layer_freq=1, kv_lora_rank=32)
    module.get_model_provider_func(args)

    assert provider.num_layers == 2
    assert provider.moe_layer_freq == 1
    assert provider.kv_lora_rank == 32
    assert not hasattr(args, "multimodal_input_key_map")


def test_kimi_k3_provider_accepts_cp_pp_and_full_uniform_recompute(monkeypatch):
    provider = _FakeProvider()
    provider.kimi_kda_layers = (1, 3)
    provider.linear_cp_mode = "chunkwise"
    provider.recompute_granularity = None
    provider.recompute_method = None
    provider.recompute_num_layers = None
    module, _ = _load_model_provider(monkeypatch, provider)
    args = _bridge_args(
        context_parallel_size=2,
        linear_cp_mode="headwise",
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )

    module.get_model_provider_func(args)

    assert provider.context_parallel_size == 2
    assert provider.linear_cp_mode == "headwise"
    assert provider.pipeline_model_parallel_size == 2
    assert provider.virtual_pipeline_model_parallel_size is None
    assert (provider.recompute_granularity, provider.recompute_method, provider.recompute_num_layers) == (
        "full",
        "uniform",
        1,
    )
    assert provider.finalized
