# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SenseNova-U1.5 native I2I RL adapter contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relax.models.sensenova.adapter import SenseNovaU1ImageEditAdapter
from relax.models.sensenova.model import SenseNovaU1DiT, SenseNovaU1VisionEncoder
from tests.cpu.models.test_sensenova_rollout import _pack, _pair


def test_sensenova_full_boundary_freezes_understanding_and_unlisted_generation_modules():
    model = torch.nn.Module()
    model.q_proj_mot_gen = torch.nn.Linear(3, 3)
    model.q_proj = torch.nn.Linear(3, 3)
    model.norm_mot_gen = torch.nn.LayerNorm(3)
    model.vision_model_mot_gen = torch.nn.Linear(3, 3)
    adapter = SenseNovaU1ImageEditAdapter()
    selected = adapter.configure_full_training(model)
    assert selected == ["q_proj_mot_gen.weight", "q_proj_mot_gen.bias"]
    assert not model.q_proj.weight.requires_grad
    assert not model.norm_mot_gen.weight.requires_grad
    assert not model.vision_model_mot_gen.weight.requires_grad
    model.q_proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="boundary disagrees"):
        adapter.validate_full_trainable_boundary(model)


class _FakeDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_calls = 0
        self.replay_kwargs = None

    def forward(self, **kwargs):
        if "prefix_inputs_embeds" in kwargs:
            assert kwargs["prefix_inputs_embeds"] is not None
            self.prefix_calls += 1
            return "prefix-cache", torch.zeros(1)
        self.replay_kwargs = kwargs
        return torch.full_like(kwargs["replay_latents"], 3.0)


def _batch():
    _, packed = _pack(*_pair())
    return {key: value.unsqueeze(0) if key.startswith("cond_") else value for key, value in packed.items()}


def test_sensenova_prepare_replay_computes_prefix_once():
    batch = _batch()
    model = _FakeDiT()
    prepared = SenseNovaU1ImageEditAdapter().prepare_replay_batch(model, batch)
    assert model.prefix_calls == 1
    assert prepared["_sensenova_prefix_cache"] == "prefix-cache"
    assert SenseNovaU1ImageEditAdapter().prepare_replay_batch(model, prepared) is prepared
    assert model.prefix_calls == 1
    other = _FakeDiT()
    rebuilt = SenseNovaU1ImageEditAdapter().prepare_replay_batch(other, prepared)
    assert rebuilt is not prepared and other.prefix_calls == 1


class _ScalarEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.ModuleList([torch.nn.Linear(1, 1)])

    def forward(self, value):
        return value[:, None].expand(-1, 4)


def test_sensenova_timestep_embedding_keeps_per_sample_values():
    fake = SimpleNamespace(
        fm_modules=torch.nn.ModuleDict({"timestep_embedder": _ScalarEmbedder()}),
        add_noise_scale_embedding=False,
    )
    embedded = SenseNovaU1DiT.embed_timestep(
        fake,
        timestep=torch.tensor([0.2, 0.8]),
        noise_scale=torch.tensor([1.0, 1.0]),
        num_tokens=3,
        batch_size=2,
    )
    assert embedded.shape == (2, 3, 4)
    assert torch.all(embedded[0] == 0.2)
    assert torch.all(embedded[1] == 0.8)


@pytest.mark.parametrize("latent_value", [1.0009765625, 1.0078125])
def test_fp16_replay_matches_native_compute_residual_without_mutating_storage(latent_value):
    class Head(torch.nn.Module):
        def forward(self, hidden):
            return torch.ones(hidden.shape[0], 3072, 1, 1, dtype=torch.bfloat16)

    vision = SenseNovaU1VisionEncoder(hidden_size=8, llm_hidden_size=4, max_position_embeddings_vision=32).bfloat16()
    feature_inputs = []

    def extract(inputs, grid):
        feature_inputs.append(inputs)
        # Actually run the BF16 convolution without an autocast context.
        return vision(inputs, grid_hw=grid)

    fake = SimpleNamespace(
        patch_size=16,
        merge_size=2,
        fm_modules={"vision_model_mot_gen": vision, "fm_head": Head()},
        extract_gen_feature=extract,
        embed_timestep=lambda *args, **kwargs: torch.zeros(1, 1, 4, dtype=torch.bfloat16),
        language_model=SimpleNamespace(model=lambda **kwargs: (kwargs["inputs_embeds"], None)),
    )
    latent = torch.full((1, 3, 32, 32), latent_value, dtype=torch.float16)
    original = latent.clone()
    velocity = SenseNovaU1DiT.forward(
        fake,
        replay_latents=latent,
        replay_timestep=torch.tensor([700.0]),
        indexes_image=torch.zeros(3, 1, dtype=torch.long),
        past_key_values=object(),
    )
    assert feature_inputs[0].dtype == torch.bfloat16
    assert velocity.dtype == torch.float32
    torch.testing.assert_close(latent, original, rtol=0, atol=0)
    torch.testing.assert_close(
        velocity,
        ((original.bfloat16() - torch.ones_like(original, dtype=torch.bfloat16)).float() / 0.7).bfloat16().float(),
        rtol=0,
        atol=0,
    )
    assert bool(torch.count_nonzero(velocity)) == (latent_value == 1.0078125)
