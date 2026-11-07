# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression coverage for SenseNova's checkpoint-configured MLP head."""

import json

import torch
from safetensors.torch import save_file
from torch import nn

from relax.models.sensenova import pipeline
from relax.models.sensenova.model import SenseNovaU1DiT


def small_model_config():
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "pad_token_id": 0,
        "vision_hidden_size": 16,
        "patch_size": 2,
        "use_pixel_head": False,
        "noise_scale_max_value": 8.0,
    }


def test_checkpoint_config_selects_mlp_head_and_loads_weights(tmp_path, monkeypatch):
    config = small_model_config()
    original = SenseNovaU1DiT(**config)
    checkpoint_config = {
        "llm_config": {key: value for key, value in config.items() if key not in {"vision_hidden_size"}},
        "vision_config": {"hidden_size": config["vision_hidden_size"]},
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    save_file(
        {name: value.contiguous() for name, value in original.state_dict().items()}, tmp_path / "model.safetensors"
    )
    tokenizer = object()
    monkeypatch.setattr(pipeline.Qwen2Tokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)

    loaded = pipeline.SenseNovaU1ImagePipeline.from_pretrained(tmp_path, torch_dtype=torch.float32)

    assert loaded.tokenizer is tokenizer
    assert loaded.dit.use_pixel_head is False
    assert loaded.dit.noise_scale_max_value == 8.0
    assert loaded.dit.fm_modules["fm_head"][2].weight.shape == (48, 4096)
    assert loaded.dit.state_dict().keys() == original.state_dict().keys()
    for name, expected in original.state_dict().items():
        assert torch.equal(loaded.dit.state_dict()[name], expected), name


def test_mlp_forward_and_gradient_match_reference_formula():
    model = SenseNovaU1DiT(**small_model_config())

    class Backbone(nn.Module):
        def forward(self, inputs_embeds, **kwargs):
            return inputs_embeds, None

    model.language_model.model = Backbone()
    inputs = torch.arange(128, dtype=torch.float32).reshape(1, 4, 32).div(128).requires_grad_()
    actual = model(inputs, None, None, (8, 8))
    head = model.fm_modules["fm_head"]
    reference = torch.nn.functional.linear(
        torch.nn.functional.gelu(torch.nn.functional.linear(inputs, head[0].weight, head[0].bias)),
        head[2].weight,
        head[2].bias,
    )
    assert torch.equal(actual, reference)
    assert actual.shape == (1, 4, 48)
    actual_grad = torch.autograd.grad(actual.square().sum(), (inputs, *head.parameters()), retain_graph=True)
    reference_grad = torch.autograd.grad(reference.square().sum(), (inputs, *head.parameters()))
    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_grad, reference_grad))
