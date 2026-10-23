# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Bridge provider construction contract, without GPU model allocation."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge.models.conversion.model_bridge")

from relax.models.deepseekv41 import model as model_module  # noqa: E402
from relax.models.deepseekv41.provider import DeepSeekV41ModelProvider  # noqa: E402


@pytest.fixture
def execution_provider(tmp_path: Path) -> DeepSeekV41ModelProvider:
    provider = DeepSeekV41ModelProvider(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_head_dim=448,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        csa_window_size=128,
        csa_compress_ratios=[0, 0],
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        recompute_granularity=None,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        variable_seq_lengths=True,
        vocab_size=33,
        seq_length=128,
        should_pad_vocab=True,
        make_vocab_size_divisible_by=8,
        position_embedding_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        rope_scaling=True,
        rope_scaling_factor=2.0,
        seq_len_interpolation_factor=1.5,
        scatter_embedding_sequence_parallel=True,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v41"}))
    header = json.dumps({"embed.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embed.weight": "model.safetensors"}})
    )
    provider.v41_hf_checkpoint = str(tmp_path)
    provider._pg_collection = SimpleNamespace(pp=object(), tp=object(), cp=object())
    return provider


def _capture_model(monkeypatch, provider, rank: int):
    calls = []

    def get_rank(group):
        assert group is provider._pg_collection.pp
        return rank

    def get_world_size(group):
        assert group is provider._pg_collection.pp
        return 2

    def construct(config, **kwargs):
        instance = SimpleNamespace(config=config, kwargs=kwargs)
        calls.append(instance)
        return instance

    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(model_module, "DeepSeekV41GPTModel", construct)
    return calls


@pytest.mark.parametrize("rank,pre,post", [(0, True, False), (1, False, True)])
def test_provider_real_fields_reach_native_gpt_constructor(execution_provider, monkeypatch, rank, pre, post) -> None:
    calls = _capture_model(monkeypatch, execution_provider, rank)
    instance = execution_provider.provide(vp_stage=0)
    assert calls == [instance]
    assert instance.config is not execution_provider
    assert instance.config._vp_stage == 0
    assert instance.config.moe_router_dtype == "fp32"
    assert instance.config.params_dtype is torch.bfloat16
    assert instance.kwargs["pg_collection"] is execution_provider._pg_collection
    assert instance.kwargs["pre_process"] is pre and instance.kwargs["post_process"] is post
    assert instance.kwargs["vp_stage"] == 0
    assert instance.kwargs["vocab_size"] == 48
    assert instance.kwargs["max_sequence_length"] == 128
    assert instance.kwargs["position_embedding_type"] == "rope"
    assert instance.kwargs["rotary_base"] == 10000
    assert instance.kwargs["rotary_percent"] == 1.0
    assert instance.kwargs["rope_scaling"] is True
    assert instance.kwargs["rope_scaling_factor"] == 2.0
    assert instance.kwargs["seq_len_interpolation_factor"] == 1.5
    assert instance.kwargs["scatter_embedding_sequence_parallel"] is True
    assert instance.kwargs["parallel_output"] is execution_provider.parallel_output
    assert instance.kwargs["fp16_lm_cross_entropy"] is execution_provider.fp16_lm_cross_entropy
    assert (
        instance.kwargs["share_embeddings_and_output_weights"]
        is execution_provider.share_embeddings_and_output_weights
    )
    assert "logit_dtype" not in instance.kwargs
    assert "transformer_layer_spec" not in instance.kwargs


def test_provider_repeated_calls_copy_config_and_respect_explicit_stage_flags(execution_provider, monkeypatch) -> None:
    _capture_model(monkeypatch, execution_provider, 0)
    first = execution_provider.provide(pre_process=False, post_process=True, vp_stage=None)
    second = execution_provider.provide(pre_process=True, post_process=False, vp_stage=None)
    assert first.config is not second.config
    first.config._cpu_offloading_context = "actor"
    assert second.config._cpu_offloading_context != "actor"
    assert not hasattr(first.config, "v41_runtime")
    assert not hasattr(second.config, "v41_runtime")
    assert not hasattr(execution_provider, "v41_runtime")
    assert first.kwargs["pre_process"] is False and first.kwargs["post_process"] is True
    assert second.kwargs["pre_process"] is True and second.kwargs["post_process"] is False


def test_provider_unpadded_vocab_is_preserved(execution_provider, monkeypatch) -> None:
    _capture_model(monkeypatch, execution_provider, 0)
    execution_provider.should_pad_vocab = False
    assert execution_provider.provide().kwargs["vocab_size"] == 33


def test_provider_missing_vocab_fails_before_model_allocation(execution_provider, monkeypatch) -> None:
    calls = _capture_model(monkeypatch, execution_provider, 0)
    execution_provider.vocab_size = None
    with pytest.raises(ValueError, match="vocab_size"):
        execution_provider.provide()
    assert calls == []


def test_provider_meta_construction_fails_before_model_allocation(execution_provider, monkeypatch) -> None:
    calls = _capture_model(monkeypatch, execution_provider, 0)
    execution_provider.init_model_with_meta_device = True
    with pytest.raises(NotImplementedError, match="meta-device"):
        execution_provider.provide()
    assert calls == []


def test_provider_original_quantized_checkpoint_fails_before_model_allocation(execution_provider, monkeypatch) -> None:
    calls = _capture_model(monkeypatch, execution_provider, 0)
    config_path = Path(execution_provider.v41_hf_checkpoint) / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "fp8", "expert_dtype": "fp4"}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="BF16 train/rollout checkpoint"):
        execution_provider.provide()
    assert calls == []
