# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU architecture/key coverage tests; these do not simulate Megatron
loading."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration


SOURCE = Path(__file__).resolve().parents[3] / "relax/models/qwen2_vl/configuration.py"
spec = importlib.util.spec_from_file_location("qwen2_vl_configuration_under_test", SOURCE)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


def tiny_config() -> Qwen2VLConfig:
    return Qwen2VLConfig(
        text_config={
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 24,
            "num_heads": 2,
            "mlp_ratio": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=26,
        vision_end_token_id=27,
        tie_word_embeddings=True,
    )


def test_reads_qwen2_vl_own_config_and_tied_embeddings():
    config = tiny_config()
    text, sections = configuration.validate_qwen2_vl_config(config)
    assert text is config.text_config
    assert sections == [1, 1, 2]
    assert config.tie_word_embeddings


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "qwen3_vl", "qwen2"])
def test_rejects_other_model_architectures(model_type):
    with pytest.raises(ValueError, match="model_type=qwen2_vl"):
        configuration.validate_qwen2_vl_config(SimpleNamespace(model_type=model_type))


@pytest.mark.parametrize("sections", [None, [1, 2], [1, 0, 3], [1, 1, 1]])
def test_rejects_missing_or_incomplete_rotary_sections(sections):
    config = tiny_config()
    config.text_config.rope_parameters["mrope_section"] = sections
    with pytest.raises(ValueError, match="mrope_section"):
        configuration.validate_qwen2_vl_config(config)


def test_rejects_unimplemented_sliding_window():
    config = tiny_config()
    config.text_config.use_sliding_window = True
    with pytest.raises(ValueError, match="sliding-window"):
        configuration.validate_qwen2_vl_config(config)


def test_supports_original_and_nested_hf_checkpoint_names():
    assert configuration.checkpoint_prefixes({"model.embed_tokens.weight"}) == ("model", "visual")
    assert configuration.checkpoint_prefixes({"model.language_model.embed_tokens.weight"}) == (
        "model.language_model",
        "model.visual",
    )
    with pytest.raises(ValueError, match="embedding key"):
        configuration.checkpoint_prefixes({"unrelated.weight"})


def test_mappings_cover_every_parameter_of_actual_hf_qwen2_vl():
    model = Qwen2VLForConditionalGeneration(tiny_config())
    names = set(model.state_dict())
    text, vision = configuration.checkpoint_prefixes(names)
    specs = configuration.parameter_mapping_specs(text, vision)
    patterns = []
    for _, _, target in specs:
        targets = target.values() if isinstance(target, dict) else (target,)
        for name in targets:
            expression = re.escape(name).replace(r"\*\*", ".+").replace(r"\*", r"\d+")
            patterns.append(re.compile(f"^{expression}$"))
    assert all(any(pattern.fullmatch(name) for pattern in patterns) for name in names)
    for pattern in patterns:
        assert any(pattern.fullmatch(name) for name in names), pattern.pattern
    # Qwen2-VL vision includes its own LayerNorm/MLP weights, not Qwen2.5's tower.
    assert any("visual.blocks.0.norm1.weight" in name for name in names)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight


def test_mappings_include_qkv_bias_and_gated_mlp_transforms():
    specs = configuration.parameter_mapping_specs("model", "visual")
    transforms = {name: (kind, target) for kind, name, target in specs}
    kind, target = transforms["language_model.decoder.layers.*.self_attention.linear_qkv.bias"]
    assert kind == "qkv"
    assert target == {name: f"model.layers.*.self_attn.{name}_proj.bias" for name in ("q", "k", "v")}
    assert transforms["language_model.decoder.layers.*.mlp.linear_fc1.weight"][0] == "gated_mlp"


def test_mappings_load_hf_norm_weights_into_standard_te_projections():
    targets = {name: target for _, name, target in configuration.parameter_mapping_specs("model", "visual")}
    prefix = "language_model.decoder.layers.*"
    assert targets[f"{prefix}.self_attention.linear_qkv.layer_norm_weight"] == "model.layers.*.input_layernorm.weight"
    assert targets[f"{prefix}.mlp.linear_fc1.layer_norm_weight"] == "model.layers.*.post_attention_layernorm.weight"
    assert targets["language_model.decoder.final_layernorm.weight"] == "model.norm.weight"
