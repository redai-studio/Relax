# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""PickScore scorer: score formula and CPU-runtime CUDA guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relax.engine.rewards import generative
from relax.engine.rewards.generative import BaseGenerativeScorer, RewardRequest
from relax.engine.rewards.pickscore import PickScoreScorer, pickscore_from_embeds
from relax.engine.rollout import native_generation


def test_pickscore_from_embeds_aligned_vs_orthogonal():
    logit_scale = torch.tensor(1.0)  # exp(1) ~ 2.718
    aligned_t = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    aligned_i = aligned_t.clone()
    aligned = pickscore_from_embeds(aligned_t, aligned_i, logit_scale)
    ortho_i = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    ortho = pickscore_from_embeds(aligned_t, ortho_i, logit_scale)
    assert (aligned > ortho).all()
    assert torch.allclose(ortho, torch.zeros(2), atol=1e-6)


def test_pickscore_cpu_runtime_disables_cuda():
    cpu_args = SimpleNamespace(reward_runtime="cpu", reward_model_path=None)
    scorer = PickScoreScorer(cpu_args)
    assert scorer.device.type == "cpu"


def test_base_scorer_onload_offload_no_model():
    # A CPU scorer with no model must not crash on lifecycle calls.
    class _Empty(BaseGenerativeScorer):
        required_tracks = ("image",)

    s = _Empty(SimpleNamespace(reward_runtime="cpu"))
    s.onload()
    s.offload()
    assert s.device.type == "cpu"


def test_pickscore_rejects_missing_image_track():
    scorer = PickScoreScorer(SimpleNamespace(reward_runtime="cpu", reward_model_path=None))
    with pytest.raises(ValueError, match="no required 'image' artifact"):
        scorer.score_batch([RewardRequest(prompt="p", outputs=[])])


def test_pickscore_rejects_corrupt_image(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    request = RewardRequest(prompt="p", outputs=[{"track": "image", "uri": str(path)}])
    scorer = PickScoreScorer(SimpleNamespace(reward_runtime="cpu", reward_model_path=None))
    with pytest.raises(ValueError, match="failed to load required 'image' artifact"):
        scorer.score_batch([request])


def test_reward_request_resolves_and_releases_an_in_memory_image():
    uri, _h, _w, sha256 = native_generation.cache_candidate_image(torch.rand(3, 8, 12), "t2i", 0, 0, 0)
    sample = SimpleNamespace(
        prompt="p",
        multimodal_inputs=None,
        metadata={},
        train_metadata={
            "manifest": {"outputs": [{"track": "image", "uri": uri, "sha256": sha256}]},
        },
    )

    request = generative._build_reward_request(None, sample)

    assert request["memory_images"][uri].shape == (8, 12, 3)
    generative._release_memory_images([request])
    assert native_generation.get_cached_candidate_image(uri) is None
