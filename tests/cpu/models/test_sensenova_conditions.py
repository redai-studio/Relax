# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SenseNova's real pipeline runner must preserve RNG and borrowed KV
storage."""

import pytest
import torch
from PIL import Image
from transformers.cache_utils import DynamicCache

from relax.models.frozen_conditions import FrozenConditionCache
from relax.models.sensenova.adapter import SenseNovaSFTAdapter, SenseNovaSFTModel
from relax.models.sensenova.pipeline import SenseNovaU1ImagePipeline, SenseNovaU1ImageUnit_PromptEmbedder


def test_sensenova_cached_prefix_preserves_rng_and_generation_gradient(monkeypatch):
    # Isolate heavyweight prefix encoding; use the production unit runner,
    # NoiseInitializer, input preparation, scheduler and flow-matching loss.
    pipe = SenseNovaU1ImagePipeline(device="cpu", torch_dtype=torch.float32)
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.dit = torch.nn.Module()
    pipe.dit.weight = torch.nn.Parameter(torch.tensor(0.3))
    pipe.dit.compute_noise_scale = lambda height, width: 1.0
    pipe.units = [unit for unit in pipe.units if type(unit).__name__ != "SenseNovaU1ImageUnit_EditImageEmbedder"]

    def prefix(self, pipe, **kwargs):
        cache = DynamicCache()
        cache.update(torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 0.25), 0)
        return {"past_key_values": cache, "indexes_image": torch.ones(3, 1, dtype=torch.long)}

    monkeypatch.setattr(SenseNovaU1ImageUnit_PromptEmbedder, "process", prefix)
    pipe.model_fn = lambda dit, latents, past_key_values, **kwargs: (
        latents * dit.weight + past_key_values.layers[0].values.mean()
    )
    model = SenseNovaSFTModel(pipe, prefix_gradient_checkpointing=False)
    data = {"prompt": "edit", "image": Image.new("RGB", (32, 32), "red"), "edit_image": []}

    def evaluate(enabled):
        pipe.dit.weight.grad = None
        torch.manual_seed(27)
        sample = (
            data
            if not enabled
            else {**data, "condition_request": model.condition_cache.lookup(model.condition_key(data))}
        )
        loss = model(sample)
        loss.backward()
        return loss.detach(), pipe.dit.weight.grad.clone(), torch.get_rng_state()

    baseline = evaluate(False)
    model.condition_cache = FrozenConditionCache(1024, 2)
    cold = evaluate(True)
    stored = model.condition_cache.lookup(model.condition_key(data)).tensors
    before = {key: value.clone() for key, value in stored.items()}
    hot = evaluate(True)
    torch.testing.assert_close(cold, baseline, rtol=0, atol=0)
    torch.testing.assert_close(hot, baseline, rtol=0, atol=0)
    torch.testing.assert_close(stored, before, rtol=0, atol=0)


def test_shared_prefix_parameter_cannot_be_hidden_by_generation_alias():
    pipe = torch.nn.Module()
    pipe.dit = torch.nn.Module()
    shared = torch.nn.Linear(2, 2, bias=False)
    # Register generation first: default duplicate filtering would miss the
    # trainable understanding alias below and admit an unsafe cache.
    pipe.dit.q_proj_mot_gen = shared
    pipe.dit.q_proj = shared
    model = SenseNovaSFTModel(pipe)
    adapter = SenseNovaSFTAdapter()
    adapter.prepare_for_sharding(model)
    assert not model.prefix_frozen
    with pytest.raises(ValueError, match="frozen"):
        adapter.validate_condition_cache(model)
