# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real HF vision/position checks with a recording language boundary.

The recording language module does not validate Megatron attention or optimizer
updates. Those require the separate GPU integration test.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import Qwen2VLForConditionalGeneration

from tests.models.qwen2_vl.test_configuration import tiny_config


class _LanguageBoundary(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.tokens = reference.model.language_model.embed_tokens
        self.calls = []

    def embedding(self, input_ids, position_ids=None):
        return self.tokens(input_ids).transpose(0, 1)

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["decoder_input"]


@pytest.fixture
def model_class(monkeypatch):
    class _MegatronBoundary(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    module = ModuleType("megatron.core.transformer.module")
    module.MegatronModule = _MegatronBoundary
    monkeypatch.setitem(sys.modules, module.__name__, module)
    packed = ModuleType("megatron.bridge.training.utils.packed_seq_utils")
    packed.unpack_mcore_thd_tensor_for_position_ids = None
    packed.repack_mcore_thd_position_ids = None
    monkeypatch.setitem(sys.modules, packed.__name__, packed)
    path = Path(__file__).resolve().parents[3] / "relax/models/qwen2_vl/model.py"
    spec = importlib.util.spec_from_file_location("qwen2_vl_model_under_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.Qwen2VLModel


def build_model(model_class):
    hf_config = tiny_config()
    reference = Qwen2VLForConditionalGeneration(hf_config).eval()
    language = _LanguageBoundary(reference)
    config = SimpleNamespace(
        vision_config=hf_config.vision_config,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
        vision_end_token_id=hf_config.vision_end_token_id,
        share_embeddings_and_output_weights=True,
        provide_language_model=lambda **kwargs: language,
    )
    model = model_class(config).eval()
    model.visual.load_state_dict(reference.model.visual.state_dict(), strict=True)
    return model, reference, language


def test_text_embeddings_and_positions_match_actual_hf(model_class):
    model, reference, language = build_model(model_class)
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        output = model(tokens)
    torch.testing.assert_close(
        output.transpose(0, 1), reference.model.language_model.embed_tokens(tokens), rtol=0, atol=0
    )
    expected, _ = reference.model.get_rope_index(tokens, mm_token_type_ids=torch.zeros_like(tokens))
    torch.testing.assert_close(language.calls[-1]["position_ids"], expected, rtol=0, atol=0)


def test_packed_text_resets_positions_per_sequence_without_multimodal_unpack(model_class):
    model, _, language = build_model(model_class)
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 3, 5, 8]), cu_seqlens_q_padded=None)
    model(torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]), packed_seq_params=packed)
    expected = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2]).view(1, 1, -1).expand(3, 1, -1)
    torch.testing.assert_close(language.calls[-1]["position_ids"], expected, rtol=0, atol=0)


def test_actual_qwen2_vision_features_and_image_positions_match_hf(model_class):
    model, reference, language = build_model(model_class)
    tokens = torch.tensor([[1, 26, 28, 27, 2]])
    grid = torch.tensor([[1, 2, 2]])
    pixels = torch.randn(4, 3 * 2 * 2 * 2)
    with torch.no_grad():
        output = model(tokens, pixel_values=pixels, image_grid_thw=grid)
        features = reference.model.get_image_features(pixels, grid, return_dict=True).pooler_output
        features = torch.cat(features, dim=0)
        expected = reference.model.language_model.embed_tokens(tokens).clone()
        expected[:, 2] = features
    torch.testing.assert_close(output.transpose(0, 1), expected, rtol=0, atol=0)
    token_types = torch.zeros_like(tokens)
    token_types[tokens == 28] = 1
    positions, _ = reference.model.get_rope_index(tokens, token_types, image_grid_thw=grid)
    torch.testing.assert_close(language.calls[-1]["position_ids"], positions, rtol=0, atol=0)


def test_rejects_vision_token_count_mismatch(model_class):
    model, _, _ = build_model(model_class)
    with pytest.raises(RuntimeError, match="counts differ"):
        model(torch.tensor([[1, 2, 3]]), pixel_values=torch.randn(4, 24), image_grid_thw=torch.tensor([[1, 2, 2]]))


def test_rejects_padded_batches_and_incremental_inference(model_class):
    model, _, _ = build_model(model_class)
    with pytest.raises(ValueError, match="packed"):
        model(torch.tensor([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="inference"):
        model(torch.tensor([[1, 2]]), inference_context=object())


def test_freeze_keeps_language_and_projector_selection_explicit(model_class):
    model, _, _ = build_model(model_class)
    model.freeze(False, True, False)
    assert all(not value.requires_grad for value in model.visual.blocks.parameters())
    assert all(value.requires_grad for value in model.visual.merger.parameters())
    assert all(value.requires_grad for value in model.language_model.parameters())
