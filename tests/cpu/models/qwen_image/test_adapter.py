# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qwen-Image adapter: geometry, request, packing, replay, name map."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from relax.models.generative import GenerativeModelAdapter
from relax.models.qwen_image.adapter import QwenImageAdapter, latent_grid, qwen_image_sigmas


def test_latent_grid_even():
    lh, lw = latent_grid(384, 384)
    assert lh % 2 == 0 and lw % 2 == 0
    assert (lh, lw) == (2 * (384 // 16), 2 * (384 // 16))


def test_latent_grid_non_square():
    assert latent_grid(384, 512) == (2 * (384 // 16), 2 * (512 // 16))


def test_qwen_image_sigmas_match_dynamic_shift_schedule():
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
        shift_terminal=0.02,
    )
    image_seq_len = (384 // 8 // 2) * (384 // 8 // 2)
    mu = image_seq_len * ((0.9 - 0.5) / (8192 - 256)) + (0.5 - ((0.9 - 0.5) / (8192 - 256)) * 256)
    scheduler.set_timesteps(num_inference_steps=12, sigmas=torch.linspace(1.0, 1.0 / 12, 12).tolist(), mu=mu)

    sigmas = torch.tensor(qwen_image_sigmas(12, 384, 384))
    assert torch.allclose(sigmas, scheduler.sigmas, atol=1e-7)
    assert sigmas[-2].item() == pytest.approx(0.02, abs=1e-7)
    assert sigmas[-1].item() == 0.0


def test_adapter_satisfies_protocol():
    assert isinstance(QwenImageAdapter(), GenerativeModelAdapter)


def test_adapter_supports_only_t2i():
    assert QwenImageAdapter.supported_tasks == ("t2i",)


@pytest.mark.parametrize("revision", ["feature-revision", None])
def test_load_train_model_uses_configured_revision(monkeypatch, revision):
    captured = {}

    class _Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, **kwargs)
            return object()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.QwenImageTransformer2DModel = _Loader
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    QwenImageAdapter().load_train_model(
        SimpleNamespace(model_path="model", model_revision=revision, fsdp_param_dtype="bf16")
    )
    assert captured["path"] == "model"
    if revision is None:
        assert "revision" not in captured
    else:
        assert captured["revision"] == revision


def test_build_rollout_request_t2i():
    a = QwenImageAdapter()
    sampling = {"num_inference_steps": 12, "eta": 0.7, "height": 384, "width": 384, "sde_indices": [0, 1, 2]}
    sample = SimpleNamespace(prompt="a cat", metadata={"task": "t2i"}, multimodal_inputs=None)
    req = a.build_rollout_request(sample, sampling, seed=5)
    # RolloutRequest schema (SGLang io_struct)
    assert req["num_inference_steps"] == 12
    assert req["rollout_sde_step_indices"] == [0, 1, 2]
    assert req["rollout"] is True and req["rollout_return_dit_trajectory"] is True
    assert req["rollout_return_denoising_env"] is True
    assert req["rollout_noise_level"] == 0.7
    assert req["num_outputs_per_prompt"] == 1
    assert req["true_cfg_scale"] == 1.0
    assert req["extra_sampling_params"]["sigmas"] == list(qwen_image_sigmas(12, 384, 384)[:-1])
    assert req["extra_sampling_params"]["initial_noise_latent_shape"] == [16, *latent_grid(384, 384)]
    assert "image_path" not in req


def test_build_rollout_request_rejects_cfg():
    """CFG rollout with a single-forward replay would train a different
    policy."""
    a = QwenImageAdapter()
    sample = SimpleNamespace(prompt="a cat", metadata={"task": "t2i"}, multimodal_inputs=None)
    with pytest.raises(ValueError, match="guidance_scale"):
        a.build_rollout_request(sample, {"guidance_scale": 3.5}, seed=1)


def test_validate_rollout_response():
    a = QwenImageAdapter()
    good = {"trajectory_latents": [1], "timesteps": [1], "sde_indices": [0], "height": 384, "width": 384}
    a.validate_rollout_response(good)
    for missing in ("trajectory_latents", "timesteps", "sde_indices", "height", "width"):
        with pytest.raises(ValueError, match=missing):
            a.validate_rollout_response({k: v for k, v in good.items() if k != missing})


def _response(seq: int, height: int = 384, width: int = 384, t: int = 4):
    return {
        "trajectory_latents": torch.randn(t + 1, seq, 64),
        "timesteps": torch.linspace(1000, 0, t + 1),
        "sde_indices": torch.tensor([1, 2]),
        "height": height,
        "width": width,
        "denoising_env": {"pos_cond_kwargs": {"encoder_hidden_states": [torch.randn(12, 3584)], "txt_seq_lens": [12]}},
        "rollout_log_probs": torch.randn(4),
    }


def test_pack_trajectory_shapes():
    a = QwenImageAdapter()
    seq = 576  # packed trajectory: [T+1, seq, 64] (real SGLang schema), 384x384
    resp = _response(seq)
    packed = a.pack_trajectory(resp)
    assert packed["image_x_t"].shape == (2, seq, 64)
    assert packed["image_x_next"].shape == (2, seq, 64)
    assert packed["cond_encoder_hidden_states"].shape == (12, 3584)
    assert packed["cond_txt_seq_lens"].tolist() == [12]
    assert torch.allclose(packed["sigmas"], torch.linspace(1000, 0, 5) / 1000.0)
    assert packed["rollout_old_logp"].shape == (4,)
    assert packed["image_grid"].tolist() == list(latent_grid(384, 384))


def test_pack_trajectory_accepts_the_last_transition():
    response = _response(576)
    response["timesteps"] = response["timesteps"][:-1]
    response["sde_indices"] = [3]

    packed = QwenImageAdapter().pack_trajectory(response)

    assert packed["image_x_t"].shape[0] == 1
    assert torch.equal(packed["image_x_t"][0], response["trajectory_latents"][3])
    assert torch.equal(packed["image_x_next"][0], response["trajectory_latents"][4])


def test_pack_trajectory_drops_negative_conditioning():
    """No CFG replay ⇒ the negative embeds are dead weight in the sidecar."""
    a = QwenImageAdapter()
    resp = _response(576)
    resp["denoising_env"]["neg_cond_kwargs"] = {"encoder_hidden_states": [torch.randn(12, 3584)]}
    assert not any(k.startswith("cond_neg") for k in a.pack_trajectory(resp))


@pytest.mark.parametrize("index", [-1, 4])
def test_pack_trajectory_rejects_invalid_sde_index(index):
    response = _response(576)
    response["sde_indices"] = [index]
    with pytest.raises(ValueError, match="outside valid transitions"):
        QwenImageAdapter().pack_trajectory(response)


class _FakeQwenTransformer:
    """Records img_shapes and returns hidden_states (identity velocity)."""

    def __init__(self) -> None:
        self.img_shapes = None

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        img_shapes,
        txt_seq_lens,
        return_dict,
    ):
        self.img_shapes = img_shapes
        return (hidden_states,)


def _replay_batch(grid, seq, b=2):
    return {
        "sde_indices": torch.tensor([0, 1]),
        "sigmas": torch.linspace(1, 0, 3),
        "image_x_t": torch.randn(b, 2, seq, 64),
        "image_x_next": torch.randn(b, 2, seq, 64),
        "image_grid": torch.tensor([list(grid)] * b, dtype=torch.long),
        "cond_encoder_hidden_states": torch.randn(12, 3584),
        "cond_txt_seq_lens": torch.tensor([12]),
    }


def test_replay_transition_t2i_packed_shape():
    a = QwenImageAdapter()
    b, seq = 2, 576  # packed latents [B, K, seq, 64] for 384x384
    model = _FakeQwenTransformer()
    out = a.replay_transition(model, _replay_batch(latent_grid(384, 384), seq, b), step_index=1)
    assert out.shape == (b, seq, 64)  # packed tensor, no unpack, no track dict
    assert model.img_shapes == [[(1, 24, 24)]] * b


def test_replay_transition_non_square_grid():
    """384x512 has seq=768; round(sqrt(768))**2 == 784 — guessing would be
    wrong."""
    a = QwenImageAdapter()
    grid = latent_grid(384, 512)
    seq = (grid[0] // 2) * (grid[1] // 2)
    model = _FakeQwenTransformer()
    out = a.replay_transition(model, _replay_batch(grid, seq), step_index=1)
    assert out.shape == (2, seq, 64)
    assert model.img_shapes == [[(1, 24, 32)]] * 2


def test_replay_transition_accepts_the_last_sigma():
    batch = _replay_batch(latent_grid(384, 384), 576)
    batch["sde_indices"] = torch.tensor([1, 2])

    out = QwenImageAdapter().replay_transition(_FakeQwenTransformer(), batch, step_index=2)

    assert out.shape == (2, 576, 64)


def test_replay_transition_grid_sequence_mismatch_raises():
    a = QwenImageAdapter()
    batch = _replay_batch(latent_grid(384, 512), 576)  # grid says 768 tokens
    with pytest.raises(ValueError, match="tokens"):
        a.replay_transition(_FakeQwenTransformer(), batch, step_index=1)


@pytest.mark.parametrize("step_index", [-1, 3])
def test_replay_transition_rejects_invalid_step_index(step_index):
    batch = _replay_batch(latent_grid(384, 384), 576)
    with pytest.raises(ValueError, match="outside valid transitions"):
        QwenImageAdapter().replay_transition(_FakeQwenTransformer(), batch, step_index=step_index)


def test_weight_name_map_strips_prefix():
    a = QwenImageAdapter()
    assert a.weight_name_map("transformer.blocks.0.attn.to_q.weight") == "blocks.0.attn.to_q.weight"
    assert a.weight_name_map("blocks.0.x") == "blocks.0.x"
