# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Frozen embedding reuse preserves objective, gradients and RNG."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from relax.models.qwen_image.sft_adapter import QwenImageEditSFTModel


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return (hidden_states * self.weight + encoder_hidden_states.mean(),)


class FakePipe:
    def __init__(self):
        self.transformer = FakeTransformer()
        self.text_encoder = torch.nn.Linear(1, 1).requires_grad_(False)
        self.vae = torch.nn.Linear(1, 1).requires_grad_(False)
        self.calls = 0
        self.vae_calls = 0
        self.image_processor = SimpleNamespace(preprocess=lambda image, **kwargs: torch.ones(1, 16, 4, 4))

    def encode_prompt(self, prompt, image, device):
        self.calls += 1
        return torch.tensor([[[len(prompt) / 10.0]]], device=device), None

    def _encode_vae_image(self, pixels, generator):
        self.vae_calls += 1
        return pixels

    @staticmethod
    def _pack_latents(latents, batch, channels, height, width):
        return (
            latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, -1, channels * 4)
        )


@pytest.fixture(params=["cpu", "shm"])
def storage(request, monkeypatch, tmp_path):
    monkeypatch.setenv("RELAX_CONDITION_CACHE_DEVICE", request.param)
    monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path / "shm"))
    return request.param


def make_model(monkeypatch, budget):
    monkeypatch.setenv("RELAX_CONDITION_CACHE_BYTES", str(budget))
    return QwenImageEditSFTModel(FakePipe(), "a" * 64)


def data(prompt="edit"):
    return {"prompt": prompt, "image": Image.new("RGB", (32, 32)), "edit_image": [Image.new("RGB", (32, 32))]}


def test_cold_warm_loss_grad_and_rng_equal(monkeypatch, storage):
    online = make_model(monkeypatch, 0)
    cached = make_model(monkeypatch, 1 << 20)
    expected = None
    for model in [online, cached, cached]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(19)
        loss = model(data())
        loss.backward()
        if storage == "shm" and model is cached:
            assert cached.condition_cache.flush()
        current = (loss.detach().clone(), model.transformer.weight.grad.clone(), torch.get_rng_state().clone())
        if expected is None:
            expected = current
        else:
            for left, right in zip(expected, current):
                assert torch.equal(left, right)
    assert cached.pipe.calls == 1
    # VAE argmax latents are deterministic and reused; noise/sigma stay online.
    assert cached.pipe.vae_calls == 2
    assert cached.condition_cache_stats()["global_hits"] == 1
    cached.close_condition_cache()
    online.close_condition_cache()


def test_target_pixels_invalidate_vae_latents(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20)
    model.encode_conditions(data())
    changed = data()
    changed["image"].putpixel((0, 0), (255, 0, 0))
    model.encode_conditions(changed)
    assert model.pipe.calls == 2
    assert model.pipe.vae_calls == 4
    model.close_condition_cache()


def test_effective_prompt_and_image_content_invalidate(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20)
    model.encode_conditions(data())
    model.encode_conditions(data("different"))
    changed = data()
    changed["edit_image"][0].putpixel((0, 0), (255, 0, 0))
    model.encode_conditions(changed)
    assert model.pipe.calls == 3
    model.close_condition_cache()


def test_trainable_encoder_bypasses_and_close_releases(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20)
    model.encode_conditions(data())
    if storage == "shm":
        assert model.condition_cache.flush()
    model.text_encoder.requires_grad_(True)
    model.encode_conditions(data())
    assert model.pipe.calls == 2
    assert model.condition_cache.stats()["entries"] == 0
    model.close_condition_cache()
    assert model.condition_cache.stats()["bytes"] == 0


def test_frozen_modules_remain_eval(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20).train()
    assert model.transformer.training
    assert not model.text_encoder.training
    assert not model.vae.training
    model.close_condition_cache()


def test_encoder_inplace_update_invalidates(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20)
    model.encode_conditions(data())
    with torch.no_grad():
        model.text_encoder.weight.add_(1)
    model.encode_conditions(data())
    assert model.pipe.calls == 2
    model.close_condition_cache()


def test_real_tiny_qwen_dit_geometry_and_backward(monkeypatch):
    """Exercise pinned Diffusers' real transformer with the SFT packing
    path."""
    from diffusers import QwenImageTransformer2DModel

    monkeypatch.setenv("RELAX_CONDITION_CACHE_BYTES", "0")
    pipe = FakePipe()
    pipe.transformer = QwenImageTransformer2DModel(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
    ).to(dtype=torch.bfloat16)
    pipe.encode_prompt = lambda **kwargs: (torch.full((1, 4, 32), 0.1, dtype=torch.bfloat16), None)
    model = QwenImageEditSFTModel(pipe, "a" * 64)
    loss = model(data())
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.transformer.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_trainable_condition_bypass_does_not_detach_gradient(monkeypatch, storage):
    model = make_model(monkeypatch, 1 << 20)
    model.text_encoder.requires_grad_(True)
    model.pipe.encode_prompt = lambda **kwargs: (model.text_encoder.weight.reshape(1, 1, 1), None)
    embeds = model.encode_conditions(data())["embeds"]
    embeds.sum().backward()
    assert model.text_encoder.weight.grad is not None
    assert model.condition_cache.stats()["entries"] == 0
    model.close_condition_cache()
