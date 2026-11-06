# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image insertion, gradient flow and external-loss interface without model
weights."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def packed_vl_module(monkeypatch):
    upstream = ModuleType("megatron.bridge.models.gemma_vl.modeling_gemma4_vl")
    upstream.Gemma4VLModel = torch.nn.Module
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    path = Path(__file__).parents[4] / "relax/models/gemma4/multimodal_model.py"
    spec = importlib.util.spec_from_file_location("_gemma4_packed_vl_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(module, bidirectional="vision"):
    model = module.PackedGemma4VLModel()
    model.config = SimpleNamespace(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=True,
        _pg_collection=SimpleNamespace(tp=object()),
        image_token_id=99,
        text_config=SimpleNamespace(pad_token_id=0, use_bidirectional_attention=bidirectional),
    )

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(16, 4)

        def forward(self, input_ids, position_ids):
            return self.word_embeddings(input_ids).transpose(0, 1)

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding()
            self.config = SimpleNamespace(scale_embeddings_by_hidden_size=True, hidden_size=4)
            self.received = None

        def forward(self, *, fp32_output=True, **kwargs):
            kwargs["fp32_output"] = fp32_output
            self.received = kwargs
            return kwargs["decoder_input"]

    model.language_model = LanguageModel()
    model.vision_tower = torch.nn.Linear(3, 4, bias=False)
    model.embed_vision = torch.nn.Linear(4, 4, bias=False)
    model.get_image_features = lambda pixels, **kwargs: model.embed_vision(model.vision_tower(pixels))
    return model


@pytest.mark.parametrize("bidirectional", ["vision", None])
@pytest.mark.parametrize("accepts_fp32_output", [True, False])
def test_packed_image_features_are_inserted_before_one_sp_scatter(
    packed_vl_module, monkeypatch, bidirectional, accepts_fp32_output
):
    torch.manual_seed(44)
    model = _model(packed_vl_module, bidirectional)
    if not accepts_fp32_output:

        def native_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            loss_mask,
            runtime_gather_output,
            packed_seq_params,
        ):
            model.language_model.received = {"packed_seq_params": packed_seq_params, "loss_mask": loss_mask}
            return decoder_input

        model.language_model.forward = native_forward
    ids = torch.tensor([[1, 99, 99, 2, 3, 99, 4, 5]])
    pixels = torch.randn(3, 3)
    params = SimpleNamespace(qkv_format="thd", cu_seqlens_q_cpu=[0, 4, 8])
    scatters = []

    def scatter(embeddings, group):
        assert group is model.config._pg_collection.tp
        with torch.no_grad():
            projected = model.get_image_features(pixels)
            torch.testing.assert_close(embeddings[[1, 2, 5], 0], projected)
            expected_text = model.language_model.embedding.word_embeddings(ids[0, [0, 3, 4, 6, 7]]) * 2
            torch.testing.assert_close(embeddings[[0, 3, 4, 6, 7], 0], expected_text)
        scatters.append(embeddings.shape)
        return embeddings

    monkeypatch.setattr(packed_vl_module, "scatter_to_sequence_parallel_region", scatter)
    output = model(ids, pixel_values=pixels, packed_seq_params=params, fp32_output=False)
    assert scatters == [torch.Size([8, 1, 4])]
    if accepts_fp32_output:
        assert model.language_model.received["fp32_output"] is False
    assert model.language_model.received["loss_mask"] is None
    forwarded = model.language_model.received["packed_seq_params"]
    assert not hasattr(params, "gemma4_vision_group_ids")
    if bidirectional == "vision":
        assert forwarded is not params
        assert forwarded.gemma4_vision_group_ids.tolist() == [-1, 1, 1, -1, -1, 2, -1, -1]
    else:
        assert forwarded is params
    output.square().sum().backward()
    for weight in (model.vision_tower.weight, model.embed_vision.weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    assert model.language_model.embedding.word_embeddings.weight.grad[0].count_nonzero() == 0


@pytest.mark.parametrize(
    "change,kwargs,error,match",
    [
        ({"context_parallel_size": 2}, {}, NotImplementedError, "CP1 and PP1"),
        ({"pipeline_model_parallel_size": 2}, {}, NotImplementedError, "CP1 and PP1"),
        ({}, {"packed_seq_params": None}, ValueError, "THD sequence boundaries"),
        ({}, {"input_features": torch.zeros(1)}, NotImplementedError, "audio"),
        ({}, {"labels": torch.zeros(1)}, ValueError, "external loss"),
    ],
)
def test_packed_image_adapter_rejects_unsupported_paths(packed_vl_module, change, kwargs, error, match):
    model = _model(packed_vl_module)
    for name, value in change.items():
        setattr(model.config, name, value)
    inputs = {"input_ids": torch.tensor([[1, 2]]), "packed_seq_params": SimpleNamespace(qkv_format="thd")}
    inputs.update(kwargs)
    with pytest.raises(error, match=match):
        model(**inputs)
