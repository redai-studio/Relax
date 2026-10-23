# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU conversion tests, run with the Bridge/Transformers training-image
dependencies."""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("megatron.bridge.models.conversion.model_bridge")

from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)

from relax.models.deepseekv41.bridge import DeepSeekV41Bridge  # noqa: E402
from relax.models.deepseekv41.configuration import (  # noqa: E402
    ARCHITECTURE_KEYS,
    V41_CONFIG_FIELDS,
    DeepSeekV41Config,
    DeepSeekV41TextConfig,
    configure_runtime,
    normalize_text_config,
    register_hf_config,
)
from relax.models.deepseekv41.provider import DeepSeekV41ModelProvider  # noqa: E402


@pytest.fixture
def checkpoint_config() -> dict[str, Any]:
    """Original nested schema, reduced dimensions, and no external model-
    directory dependency."""
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 127,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 128,
            "hidden_size": 128,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 16,
            "q_lora_rank": 64,
            "o_lora_rank": 16,
            "o_groups": 2,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-20,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "tie_word_embeddings": False,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 64,
            },
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 16,
            "compress_ratios": [0, 2, 1, 1, 0],
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1, 2],
            "index_n_heads": 4,
            "index_head_dim": 32,
            "index_topk": 4,
            "candidate_source_layer_id": 2,
            "candidate_topk_blocks": 8,
            "candidate_block_size": 2,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1],
            "engram_num_embeddings": [64],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 32,
            "engram_n_heads": 2,
            "engram_head_dim": 16,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 32,
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [3],
        },
        "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": 32},
    }


def _flatten_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    flat = deepcopy(config["text_config"])
    flat.update(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
        pad_token_id=config["pad_token_id"],
        num_nextn_predict_layers=0,
        compress_ratios=flat["compress_ratios"][: flat["num_hidden_layers"]],
    )
    for key in tuple(flat):
        if key.startswith("dspark_"):
            del flat[key]
    return flat


@pytest.fixture
def bridge(checkpoint_config: dict[str, Any]) -> DeepSeekV41Bridge:
    result = DeepSeekV41Bridge()
    result.hf_config = DeepSeekV41Config(**deepcopy(checkpoint_config))
    return result


@pytest.mark.parametrize("flattened", [False, True])
def test_normalize_text_config_preserves_source_and_model_identity(
    checkpoint_config: dict[str, Any], flattened: bool
) -> None:
    raw = _flatten_checkpoint(checkpoint_config) if flattened else checkpoint_config
    config = DeepSeekV41Config(**deepcopy(raw))
    before = deepcopy(config.to_dict())

    text = normalize_text_config(config)

    assert config.to_dict() == before
    assert text["model_type"] == "deepseek_v41"
    assert text["architectures"] == ["DeepseekV41ForCausalLM"]
    assert (text["hidden_size"], text["num_hidden_layers"], text["head_dim"]) == (128, 4, 32)
    assert text["pad_token_id"] == 2
    text["engram_layer_ids"].append(3)
    assert config.to_dict() == before, "normalization must not alias nested source lists"
    if not flattened:
        assert text["quantization_config"] == raw["quantization_config"]
        assert config.text_config.model_type == "deepseek_v41_text"


def test_normalize_text_config_rejects_other_models() -> None:
    with pytest.raises(ValueError, match="deepseek_v41"):
        normalize_text_config({"model_type": "deepseek_v4"})


@pytest.mark.parametrize("flattened", [False, True])
def test_hf_config_registration_round_trip(checkpoint_config: dict[str, Any], tmp_path: Path, flattened: bool) -> None:
    register_hf_config(for_training=True)
    register_hf_config(for_training=True)  # Multiple actor imports must be harmless.
    raw = _flatten_checkpoint(checkpoint_config) if flattened else checkpoint_config
    config = DeepSeekV41Config(**deepcopy(raw))
    config.save_pretrained(tmp_path)

    loaded = transformers.AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    restored = pickle.loads(pickle.dumps(loaded))

    assert isinstance(loaded, DeepSeekV41Config)
    assert isinstance(restored, DeepSeekV41Config)
    assert restored.to_dict() == loaded.to_dict()
    assert loaded.model_type == "deepseek_v41"
    if not flattened:
        assert isinstance(loaded.text_config, DeepSeekV41TextConfig)
        assert isinstance(restored.text_config, DeepSeekV41TextConfig)
        assert loaded.text_config.engram_layer_ids == [1]
    assert normalize_text_config(loaded)["compress_ratios"] == raw.get(
        "compress_ratios", raw.get("text_config", {}).get("compress_ratios")
    )


def test_training_registration_restores_rollout_override_without_rebinding_config(
    monkeypatch: pytest.MonkeyPatch, checkpoint_config: dict[str, Any]
) -> None:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from relax.models.deepseekv41 import configuration

    class RolloutOnlyConfig(transformers.PretrainedConfig):
        model_type = "deepseek_v41"

    original = configuration.DeepSeekV41Config
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "deepseek_v41", RolloutOnlyConfig)
    register_hf_config()
    assert CONFIG_MAPPING["deepseek_v41"] is RolloutOnlyConfig
    assert configuration.DeepSeekV41Config is original
    register_hf_config(for_training=True)
    assert CONFIG_MAPPING["deepseek_v41"] is original
    config = original(**deepcopy(checkpoint_config))
    assert type(pickle.loads(pickle.dumps(config))) is original


def test_autobridge_selects_v41_provider_without_hf_model_instantiation(checkpoint_config: dict[str, Any]) -> None:
    config = DeepSeekV41Config(**checkpoint_config)
    auto_bridge = AutoBridge.from_hf_config(config)

    provider = auto_bridge.to_megatron_provider(load_weights=False)

    assert isinstance(provider, DeepSeekV41ModelProvider)
    assert provider.num_layers == 4
    assert provider.v41_kv_source_layer_ids == (1, 2)


@pytest.mark.parametrize("flattened", [False, True])
def test_provider_uses_text_architecture_without_mutating_hf_config(
    bridge: DeepSeekV41Bridge, checkpoint_config: dict[str, Any], flattened: bool
) -> None:
    raw = _flatten_checkpoint(checkpoint_config) if flattened else checkpoint_config
    source = DeepSeekV41Config(**deepcopy(raw))
    before = deepcopy(source.to_dict())

    provider = bridge.provider_bridge(SimpleNamespace(config=source))

    assert isinstance(provider, DeepSeekV41ModelProvider)
    assert source.to_dict() == before
    assert provider.v41_hf_config == before
    assert provider.num_layers == 4
    assert provider.hidden_size == 128
    assert provider.num_attention_heads == 4
    assert provider.num_moe_experts == 4
    assert provider.moe_ffn_hidden_size == 128
    assert provider.moe_router_topk == 2
    assert provider.moe_router_score_function == "sqrtsoftplus"
    assert provider.moe_router_topk_scaling_factor == 1.5
    assert provider.moe_router_dtype == "fp32"
    assert provider.q_lora_rank == 64
    assert provider.kv_lora_rank == 32
    assert provider.qk_head_dim == 16
    assert provider.qk_pos_emb_head_dim == 16
    assert provider.v_head_dim == 32
    assert (provider.o_groups, provider.o_lora_rank) == (2, 16)
    assert provider.csa_compress_ratios == [0, 2, 1, 1]
    assert provider.moe_layer_freq == [1, 1, 1, 1]
    assert provider.moe_n_hash_layers == 0
    assert provider.mtp_num_layers is None
    assert provider.experimental_attention_variant is None
    assert provider.rotary_percent == 1.0
    assert provider.hidden_dropout == 0.0
    assert provider.attention_dropout == 0.0
    assert provider.rotary_scaling_factor == 2
    assert (provider.mscale, provider.mscale_all_dim) == (1.0, 1.0)
    assert provider.v41_hc_eps == 1e-6
    text = normalize_text_config(source)
    for name in V41_CONFIG_FIELDS:
        expected = tuple(text[name]) if isinstance(text[name], list) else text[name]
        assert getattr(provider, f"v41_{name}") == expected
    assert {"num_layers", "q_lora_rank", "moe_ffn_hidden_size", "mtp_num_layers"} <= ARCHITECTURE_KEYS
    assert "moe_router_dtype" in ARCHITECTURE_KEYS


@pytest.mark.parametrize("flattened", [False, True])
def test_provider_export_keeps_v41_metadata_and_removes_mtp(
    bridge: DeepSeekV41Bridge, checkpoint_config: dict[str, Any], flattened: bool
) -> None:
    raw = _flatten_checkpoint(checkpoint_config) if flattened else checkpoint_config
    provider = bridge.provider_bridge(SimpleNamespace(config=DeepSeekV41Config(**deepcopy(raw))))
    before = deepcopy(provider.v41_hf_config)

    exported = bridge.megatron_to_hf_config(provider)

    assert provider.v41_hf_config == before
    assert exported["model_type"] == "deepseek_v41"
    assert exported["architectures"] == ["DeepseekV41ForCausalLM"]
    text = exported if flattened else exported["text_config"]
    assert text["model_type"] == ("deepseek_v41" if flattened else "deepseek_v41_text")
    assert text["compress_ratios"] == [0, 2, 1, 1]
    assert text["num_nextn_predict_layers"] == 0
    assert text["kv_source_layer_ids"] == [1, 2]
    assert text["engram_num_embeddings"] == [64]
    text["engram_layer_ids"].append(3)
    assert provider.v41_hf_config == before


@pytest.mark.parametrize("ratios", [[0, 2, 1], [0, 4, 1, 1], [0, 2, 128, 1]])
def test_provider_rejects_invalid_v41_compression_ratios(
    bridge: DeepSeekV41Bridge, checkpoint_config: dict[str, Any], ratios: list[int]
) -> None:
    checkpoint_config["text_config"]["compress_ratios"] = ratios
    source = DeepSeekV41Config(**checkpoint_config)
    with pytest.raises(ValueError, match="compression ratio"):
        bridge.provider_bridge(SimpleNamespace(config=source))


def test_provider_rejects_unsupported_execution_geometry(bridge: DeepSeekV41Bridge) -> None:
    provider = bridge.provider_bridge(SimpleNamespace(config=bridge.hf_config))
    with pytest.raises(ValueError, match="kernels require kv_lora_rank=512"):
        provider.provide(pre_process=True, post_process=True, vp_stage=None)
    with pytest.raises(ValueError, match="MTP/DSpark"):
        configure_runtime(provider, SimpleNamespace(mtp_num_layers=1))


def test_bridge_does_not_advertise_complete_hf_checkpoint_export(bridge: DeepSeekV41Bridge) -> None:
    assert bridge.SUPPORTS_HF_PRETRAINED_EXPORT is False


@pytest.mark.parametrize("method", ["get_model_config", "get_model"])
def test_default_builder_cannot_bypass_v41_execution_gate(bridge: DeepSeekV41Bridge, method: str) -> None:
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.model_bridge import ModelConfigNotSupportedError

    auto = AutoBridge.from_hf_config(bridge.hf_config)
    with pytest.raises(ModelConfigNotSupportedError):
        getattr(auto, method)()
    assert isinstance(auto.to_megatron_provider(load_weights=False), DeepSeekV41ModelProvider)


@pytest.mark.parametrize(
    ("megatron_name", "hf_name", "mapping_type"),
    [
        ("embedding.word_embeddings.weight", "embed.weight", None),
        ("output_layer.weight", "head.weight", None),
        ("decoder.final_layernorm.weight", "norm.weight", ReplicatedMapping),
        ("self_attention.linear_o_group_proj", "attn.wo_a.weight", ColumnParallelMapping),
        ("self_attention.core_attention.attn_sink", "attn.attn_sink", ColumnParallelMapping),
        ("self_attention.linear_proj.weight", "attn.wo_b.weight", RowParallelMapping),
        ("self_attention.core_attention.indexer.linear_wk.weight", "attn.indexer.wk.weight", ReplicatedMapping),
        ("self_attention.core_attention.indexer.k_norm.weight", "attn.indexer.k_norm.weight", ReplicatedMapping),
        (
            "self_attention.core_attention.indexer.linear_weights_proj.weight",
            "attn.indexer.weights_proj.weight",
            ReplicatedMapping,
        ),
        ("v41_engram.linear_wkv.weight", "engram.wkv.weight", ReplicatedMapping),
        ("v41_engram.q_weight", "engram.q_weight", ReplicatedMapping),
        ("v41_engram.k_weight", "engram.k_weight", ReplicatedMapping),
    ],
)
def test_mapping_matches_checkpoint_names_and_tp_layout(
    bridge: DeepSeekV41Bridge, megatron_name: str, hf_name: str, mapping_type: type | None
) -> None:
    if not megatron_name.startswith(("embedding.", "output_layer.", "decoder.")):
        megatron_name, hf_name = f"decoder.layers.1.{megatron_name}", f"layers.1.{hf_name}"
    mapping = bridge.mapping_registry().megatron_to_hf_lookup(megatron_name)
    assert mapping is not None
    assert mapping.hf_param == hf_name
    if mapping_type is not None:
        assert isinstance(mapping, mapping_type)


@pytest.mark.parametrize(
    "megatron_name",
    [
        "decoder.layers.1.mlp.experts.linear_fc1.weight2",
        "decoder.layers.1.mlp.experts.local_experts.2.linear_fc1.weight",
        "decoder.layers.1.mlp.shared_experts.linear_fc1.weight",
    ],
)
def test_gated_expert_mapping_preserves_gate_up_order(bridge: DeepSeekV41Bridge, megatron_name: str) -> None:
    mapping = bridge.mapping_registry().megatron_to_hf_lookup(megatron_name)
    assert isinstance(mapping, GatedMLPMapping)
    prefix = "layers.1.ffn.shared_experts" if ".shared_experts." in megatron_name else "layers.1.ffn.experts.2"
    assert mapping.hf_param == {"gate": f"{prefix}.w1.weight", "up": f"{prefix}.w3.weight"}


@pytest.mark.parametrize(
    "megatron_name",
    [
        "decoder.hc_head_fn",
        "decoder.hc_head_base",
        "decoder.hc_head_scale",
        "decoder.layers.1.mlp.router.tid2eid",
        "decoder.layers.1.v41_engram.embed.weight",
        "decoder.layers.1.v41_engram.embed.scale",
        "decoder.layers.1.self_attention.core_attention.compressor.ape",
        "decoder.layers.1.self_attention.core_attention.indexer.compressor.linear_wkv.weight",
        "mtp.layers.0.e_proj.weight",
    ],
)
def test_mapping_excludes_v4_only_frozen_and_auxiliary_weights(bridge: DeepSeekV41Bridge, megatron_name: str) -> None:
    assert bridge.mapping_registry().megatron_to_hf_lookup(megatron_name) is None


@pytest.mark.parametrize(
    ("site", "hf_scale"),
    [("self_attention_hyper_connection", "hc_attn_scale"), ("mlp_hyper_connection", "hc_ffn_scale")],
)
def test_hc_alpha_import_export_preserves_fp32_and_secondary_is_noop(
    bridge: DeepSeekV41Bridge, site: str, hf_scale: str
) -> None:
    registry = bridge.mapping_registry()
    scale = torch.tensor([1.0001, 1.0002, 1.0003], dtype=torch.float32)
    module = SimpleNamespace(
        alpha_pre=torch.nn.Parameter(torch.zeros(1)),
        alpha_post=torch.nn.Parameter(torch.zeros(1)),
        alpha_res=torch.nn.Parameter(torch.zeros(1)),
    )
    for index, suffix in enumerate(("pre", "post", "res")):
        mapping = registry.megatron_to_hf_lookup(f"decoder.layers.1.{site}.alpha_{suffix}")
        assert mapping is not None
        assert mapping.hf_param == f"layers.1.{hf_scale}"
        loaded = mapping.hf_to_megatron(scale, module)
        assert torch.equal(loaded, scale[index : index + 1])
        getattr(module, f"alpha_{suffix}").data.copy_(loaded)
        if index:
            assert mapping.megatron_to_hf(loaded, module) == {}
    primary = registry.megatron_to_hf_lookup(f"decoder.layers.1.{site}.alpha_pre")
    exported = primary.megatron_to_hf(module.alpha_pre.detach(), module)
    assert exported[f"layers.1.{hf_scale}"].dtype == torch.float32
    assert torch.equal(exported[f"layers.1.{hf_scale}"], scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bf16_and_fp32_import_are_not_requantized_or_downcast(bridge: DeepSeekV41Bridge, dtype: Any) -> None:
    weight = torch.tensor([[1.0001, -1.0002]], dtype=dtype)
    name = "layers.1.attn.wq_a.weight"
    assert bridge.maybe_modify_loaded_hf_weight(name, {name: weight}) is weight


def _scale_encoding(scale: Any, encoding: str) -> Any:
    if encoding == "float32":
        return scale
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("requires the target training-image Torch with UE8M0 support")
    encoded = scale.to(torch.float8_e8m0fnu)
    return encoded.view(torch.uint8) if encoding == "uint8" else encoded


@pytest.mark.parametrize("scale_encoding", ["float32", "e8m0", "uint8"])
def test_dense_fp8_import_uses_32_by_32_tiles(bridge: DeepSeekV41Bridge, scale_encoding: str) -> None:
    name = "layers.1.attn.wq_a.weight"
    weight = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 2.0], [4.0, 8.0]])
    state = {name: weight, "layers.1.attn.wq_a.scale": _scale_encoding(scales, scale_encoding)}

    loaded = bridge.maybe_modify_loaded_hf_weight(name, state)

    expected = scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1).bfloat16()
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, expected)


@pytest.mark.parametrize("weight_dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("scale_encoding", ["float32", "e8m0", "uint8"])
def test_mxfp4_import_matches_independent_nibble_decode(
    bridge: DeepSeekV41Bridge, weight_dtype: Any, scale_encoding: str
) -> None:
    name = "layers.1.ffn.experts.2.w1.weight"
    packed = (torch.arange(64, dtype=torch.uint8).reshape(2, 32) * 3 + 7).contiguous()
    scales = torch.tensor([[0.5, 2.0], [4.0, 8.0]])
    state = {
        name: packed.view(weight_dtype),
        "layers.1.ffn.experts.2.w1.scale": _scale_encoding(scales, scale_encoding),
    }
    # E2M1 low nibble is the first value; this reference does not call Bridge helpers.
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    unpacked = torch.stack((table[(packed & 15).long()], table[(packed >> 4).long()]), dim=-1).flatten(-2)
    expected = (unpacked * scales.repeat_interleave(32, dim=1)).bfloat16()

    loaded = bridge.maybe_modify_loaded_hf_weight(name, state)

    assert loaded.dtype == torch.bfloat16
    assert loaded.shape == (2, 64)
    assert torch.equal(loaded, expected)


def test_gated_import_dequantizes_gate_and_up_independently(bridge: DeepSeekV41Bridge) -> None:
    gate, up = "layers.1.ffn.experts.2.w1.weight", "layers.1.ffn.experts.2.w3.weight"
    state = {
        gate: torch.full((2, 16), 0x22, dtype=torch.uint8),
        gate.removesuffix(".weight") + ".scale": torch.ones((2, 1)),
        up: torch.full((2, 16), 0x33, dtype=torch.uint8),
        up.removesuffix(".weight") + ".scale": torch.full((2, 1), 2.0),
    }

    loaded = bridge.maybe_modify_loaded_hf_weight({"gate": gate, "up": up}, state)

    assert torch.equal(loaded["gate"], torch.ones((2, 32), dtype=torch.bfloat16))
    assert torch.equal(loaded["up"], torch.full((2, 32), 3.0, dtype=torch.bfloat16))


def test_quantized_import_rejects_missing_scale_and_frozen_engram(bridge: DeepSeekV41Bridge) -> None:
    name = "layers.1.attn.wq_a.weight"
    with pytest.raises(ValueError, match="Missing quantization scale"):
        bridge.maybe_modify_loaded_hf_weight(name, {name: torch.ones((32, 32), dtype=torch.float8_e4m3fn)})
    with pytest.raises(ValueError, match="Frozen Engram"):
        bridge.maybe_modify_loaded_hf_weight("layers.1.engram.embed.weight", {})


def test_missing_router_bias_is_initialized_and_not_exported(bridge: DeepSeekV41Bridge) -> None:
    name = "layers.1.ffn.gate.bias"
    bias = bridge.maybe_modify_loaded_hf_weight(name, {})
    assert bias.dtype == torch.float32
    assert torch.equal(bias, torch.zeros(4))
    assert bridge.maybe_modify_converted_hf_weight(SimpleNamespace(weight_dtype=None), {name: bias}, {}) == {}


def test_explicit_bf16_export_does_not_requantize(bridge: DeepSeekV41Bridge) -> None:
    name = "layers.1.attn.wq_a.weight"
    weight = torch.ones((64, 64), dtype=torch.bfloat16)
    state = {name: weight.to(torch.float8_e4m3fn), "layers.1.attn.wq_a.scale": torch.ones((2, 2))}

    exported = bridge.maybe_modify_converted_hf_weight(
        SimpleNamespace(weight_dtype=torch.bfloat16), {name: weight}, state
    )

    assert list(exported) == [name]
    assert exported[name] is weight


@pytest.mark.parametrize("checkpoint_kind", ["dense_fp8", "expert_fp8", "expert_mxfp4"])
@pytest.mark.parametrize("scale_encoding", ["float32", "e8m0", "uint8"])
def test_quantized_export_restores_source_geometry_and_round_trips(
    bridge: DeepSeekV41Bridge, checkpoint_kind: str, scale_encoding: str
) -> None:
    packed = checkpoint_kind == "expert_mxfp4"
    name = "layers.1.attn.wq_a.weight" if checkpoint_kind == "dense_fp8" else "layers.1.ffn.experts.2.w1.weight"
    scale_key = name.removesuffix(".weight") + ".scale"
    weight = torch.full((2, 64) if packed else (64, 64), 4.0, dtype=torch.bfloat16)
    source_scale = _scale_encoding(torch.ones((2, 2)), scale_encoding)
    source_weight = torch.full((2, 32), 0x11, dtype=torch.uint8) if packed else weight.to(torch.float8_e4m3fn)
    source = {name: source_weight, scale_key: source_scale}

    exported = bridge.maybe_modify_converted_hf_weight(SimpleNamespace(weight_dtype=None), {name: weight}, source)

    assert set(exported) == {name, scale_key}
    assert exported[name].shape == source_weight.shape
    assert exported[scale_key].shape == source_scale.shape
    assert exported[scale_key].dtype == source_scale.dtype
    if not packed:
        assert exported[name].dtype == torch.float8_e4m3fn
    assert torch.equal(bridge.maybe_modify_loaded_hf_weight(name, exported), weight)


def test_v41_provider_finalize_does_not_apply_v4_tp_restriction(bridge: DeepSeekV41Bridge) -> None:
    provider = bridge.provider_bridge(SimpleNamespace(config=bridge.hf_config))
    provider.tensor_model_parallel_size = 2
    provider.finalize()
    assert provider.tensor_model_parallel_size == 2
    assert provider.csa_compress_ratios == [0, 2, 1, 1]
    assert provider.experimental_attention_variant is None
