# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for online Qwen-Image SFT with real tiny-module autograd.

The pipeline retains text encoding, posterior sampling and independent noise
and timesteps. It does not replace the pretrained GPU smoke test.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from relax.models.qwen_image import sft_adapter
from relax.models.qwen_image.sft_adapter import (
    QwenImageSFTAdapter,
    QwenImageSFTModel,
    flow_matching_loss,
    pack_latents,
    unpack_latents,
)


class TinyConfig(SimpleNamespace):
    def to_dict(self):
        return {key: value for key, value in vars(self).items() if not key.startswith("_")}


class TinyTokenizer:
    def __call__(self, prompts, *, max_length, padding, truncation, return_tensors):
        assert padding is False and truncation is True and return_tensors == "pt"
        assert len(prompts) == 1
        ids = torch.tensor([[ord(char) + 1 for char in prompts[0]][:max_length]], dtype=torch.int64)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 4)
        self.dropout = torch.nn.Dropout(0.5)
        self.register_buffer("conditioning_offset", torch.tensor(0.0), persistent=False)
        self.config = TinyConfig(hidden_size=4, rope_theta=10000, _attn_implementation="sdpa")
        self.calls = 0

    def forward(self, input_ids, attention_mask, **_kwargs):
        self.calls += 1
        # A prefix-dependent result exercises the entire effective token input.
        embeds = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(hidden_states=(embeds.cumsum(dim=1) / 16 + self.conditioning_offset,))


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(joint_attention_dim=4)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 8) / 8)
        self.lora_B = torch.nn.Parameter(torch.randn(8, 2) / 8)

    def forward(self, hidden_states, encoder_hidden_states, timestep, **_kwargs):
        residual = hidden_states.float() @ self.lora_A.T @ self.lora_B.T
        condition = encoder_hidden_states.float().mean(dim=(1, 2), keepdim=True)
        prediction = hidden_states.float() / 4 + residual + condition + timestep.view(-1, 1, 1) / 8
        return (prediction.to(hidden_states.dtype),)


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.config = SimpleNamespace(latents_mean=[0.1234567, -0.2345678], latents_std=[0.7654321, 1.234567])
        self.samples = []

    def encode(self, pixels):
        base = torch.nn.functional.avg_pool2d(pixels.squeeze(2).float(), 8)[:, :2].unsqueeze(2)
        base = (base * self.scale).to(pixels.dtype)

        def sample():
            value = base + torch.randn_like(base) / 8
            self.samples.append(value.detach().clone())
            return value

        return SimpleNamespace(latent_dist=SimpleNamespace(sample=sample))


class TinyPipeline:
    def __init__(self, dtype):
        self.text_encoder = TinyTextEncoder()
        self.transformer = TinyTransformer()
        self.vae = TinyVAE()
        self.tokenizer = TinyTokenizer()
        self.prompt_template_encode = "<{}>"
        self.prompt_template_encode_start_idx = 1
        self.tokenizer_max_length = 12
        self.dtype = dtype
        self.reference_calls = 0
        self.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=10),
            timesteps=torch.arange(10, 0, -1, dtype=torch.float32) * 100,
            sigmas=torch.linspace(1, 0, 11),
        )

    def encode_prompt(self, prompt, device, max_sequence_length):
        self.reference_calls += 1
        tokens = self.tokenizer(
            [self.prompt_template_encode.format(prompt)],
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        states = self.text_encoder(**{name: value.to(device) for name, value in tokens.items()}).hidden_states[-1]
        start = self.prompt_template_encode_start_idx
        return states[:, start : start + max_sequence_length].to(self.dtype).contiguous(), None


def _model(dtype=torch.bfloat16):
    torch.manual_seed(31)
    model = QwenImageSFTModel(TinyPipeline(dtype), {"max_sequence_length": 4}, dtype=dtype)
    model.train()
    return model


def _sample(prompt="red boat", offset=0):
    pixels = (np.arange(16 * 16 * 3).reshape(16, 16, 3) + offset).astype(np.uint8)
    return {"prompt": prompt, "image": Image.fromarray(pixels)}


def _pack_oracle(value):
    return torch.stack(
        [
            value[:, :, 0, top : top + 2, left : left + 2].flatten(start_dim=1)
            for top in range(0, value.shape[-2], 2)
            for left in range(0, value.shape[-1], 2)
        ],
        dim=1,
    )


def test_pack_latents_matches_explicit_channel_patch_order():
    value = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 1, 4, 6)
    assert torch.equal(pack_latents(value), _pack_oracle(value))
    with pytest.raises(ValueError, match="one frame and even"):
        pack_latents(torch.zeros(1, 2, 1, 3, 4))


def test_unpack_latents_matches_explicit_spatial_indices_and_preserves_gradient():
    batch, channels, height, width = 2, 3, 4, 6
    packed = torch.arange(batch * channels * height * width, dtype=torch.float32).reshape(batch, -1, channels * 4)
    packed.requires_grad_(True)
    actual = unpack_latents(packed, height, width)
    for sample in range(batch):
        for channel in range(channels):
            for top in range(height):
                for left in range(width):
                    token = (top // 2) * (width // 2) + left // 2
                    feature = channel * 4 + (top % 2) * 2 + left % 2
                    assert actual[sample, channel, 0, top, left] == packed[sample, token, feature]
    coefficients = torch.arange(actual.numel(), dtype=torch.float32).reshape_as(actual) / 7
    (actual * coefficients).sum().backward()
    assert torch.equal(packed.grad, _pack_oracle(coefficients))


def test_flow_objective_matches_independent_velocity_oracle():
    clean = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 2, 4) / 7
    noise = torch.flip(clean, [-1]) - 0.3
    sigma = torch.tensor(0.4).reshape(1, 1, 1, 1, 1)
    prompt = torch.ones(1, 3, 4) / 8
    captured = {}

    class Predictor(torch.nn.Module):
        def forward(self, **kwargs):
            captured.update(kwargs)
            return (kwargs["hidden_states"] / 3,)

    loss = flow_matching_loss(Predictor(), clean, prompt, noise, sigma, torch.tensor([400.0]))
    noisy = clean * 0.6 + noise * 0.4
    expected = ((noisy / 3 - (noise - clean)) ** 2).reshape(1, -1).mean(1).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured["hidden_states"], _pack_oracle(noisy), rtol=0, atol=0)
    assert captured["img_shapes"] == [[(1, 1, 2)]]
    assert torch.equal(captured["timestep"], torch.tensor([0.4]))
    assert captured["encoder_hidden_states_mask"] is None


def test_flow_loss_reduces_spatial_fp32_elements_in_reference_order():
    generator = torch.Generator().manual_seed(1)
    spatial = torch.randn(1, 4, 1, 8, 10, generator=generator)
    spatial = spatial * torch.logspace(-3, 3, spatial.numel()).reshape_as(spatial)
    packed = _pack_oracle(spatial)

    class Predictor(torch.nn.Module):
        def forward(self, **_kwargs):
            return (packed,)

    zero = torch.zeros_like(spatial)
    actual = flow_matching_loss(
        Predictor(), zero, torch.zeros(1, 1, 4), zero, torch.tensor(0.5), torch.tensor([500.0])
    )
    reference = spatial.square().reshape(1, -1).mean(1).mean()
    assert torch.equal(actual, reference)
    assert not torch.equal(reference, packed.square().mean()), "Fixture must distinguish packed reduction rounding"


@pytest.mark.parametrize("prompt", ["", "red boat", "abcdefghijklmnopqrstuv-long-caption"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fast_text_encode_matches_pipeline_reference_for_effective_context(prompt, dtype):
    model = _model(dtype=dtype)
    initial_rng = torch.get_rng_state().clone()
    actual = model._condition({"prompt": prompt})
    reference, mask = model.pipe.encode_prompt(prompt, model.device, model.config["max_sequence_length"])
    assert mask is None and actual.dtype == reference.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(initial_rng, torch.get_rng_state())


def test_train_mode_keeps_condition_encoder_and_vae_in_eval():
    model = _model()
    model.eval()
    model.train()
    assert model.training and model.transformer.training
    assert not model.text_encoder.training and not model.vae.training
    assert all(
        not parameter.requires_grad for module in (model.text_encoder, model.vae) for parameter in module.parameters()
    )


@pytest.mark.parametrize("placement", [None, "sharded", "replicated"])
def test_text_encoder_placement_preserves_registration_and_conditioning(placement):
    model = _model(dtype=torch.float32)
    if placement is not None:
        model.config["text_encoder_placement"] = placement
    encoder = model.text_encoder
    names = list(model.state_dict())
    parameters = dict(model.named_parameters())
    expected = model._condition(_sample())
    QwenImageSFTAdapter().prepare_for_sharding(model)
    assert model.text_encoder is encoder and model.pipe.text_encoder is encoder
    assert list(model.state_dict()) == names
    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    assert model._fsdp_replicated_modules == ((encoder,) if placement == "replicated" else ())
    assert encoder._fsdp_fine_grained is (placement != "replicated")
    assert torch.equal(model._condition(_sample()), expected)


@pytest.mark.parametrize("case", ["dtype", "device", "train_mode", "trainable", "vae_trainable", "policy"])
def test_replicated_text_encoder_rejects_invalid_placement_contract(case):
    model = _model(dtype=torch.float32)
    model.config["text_encoder_placement"] = "replicated"
    if case == "dtype":
        model.text_encoder.to(dtype=torch.bfloat16)
        match = "device and compute dtype"
    elif case == "device":
        model.device = torch.device("meta")
        match = "device and compute dtype"
    elif case == "train_mode":
        model.text_encoder.train()
        match = "eval mode"
    elif case in ("trainable", "vae_trainable"):
        (model.text_encoder if case == "trainable" else model.vae).requires_grad_(True)
        match = "confined to the DiT"
    else:
        model.config["text_encoder_placement"] = "automatic"
        match = "sharded or replicated"
    with pytest.raises(ValueError, match=match):
        QwenImageSFTAdapter().prepare_for_sharding(model)
    assert not getattr(model, "_fsdp_replicated_modules", ())


def test_vae_normalization_uses_fp32_statistics_before_bf16_rounding(monkeypatch):
    model = _model()
    captured = {}

    def inspect(_transformer, clean, _prompt, _noise, _sigma, _timestep):
        captured["clean"] = clean
        return clean.float().sum()

    monkeypatch.setattr(sft_adapter, "flow_matching_loss", inspect)
    model(_sample())
    sample = model.vae.samples[-1]
    mean = torch.tensor(model.vae.config.latents_mean, dtype=torch.float32).reshape(1, -1, 1, 1, 1)
    inverse_std = torch.tensor(model.vae.config.latents_std, dtype=torch.float32).reciprocal().reshape(1, -1, 1, 1, 1)
    expected = ((sample.float() - mean) * inverse_std).to(torch.bfloat16)
    assert torch.equal(captured["clean"], expected)
    rounded_early = (sample - mean.to(torch.bfloat16)) * inverse_std.to(torch.bfloat16)
    assert not torch.equal(expected, rounded_early), "Fixture must expose early-rounding target error"


@pytest.mark.parametrize("time", [997, 513, 257])
def test_scheduler_timestep_is_divided_in_fp32_before_transformer_compute(monkeypatch, time):
    model = _model()
    model.pipe.scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1),
        timesteps=torch.tensor([time], dtype=torch.float32),
        sigmas=torch.tensor([0.7, 0.0]),
    )
    observed = {}
    forward = model.transformer.forward

    def record(**kwargs):
        observed["timestep"] = kwargs["timestep"].detach().clone()
        return forward(**kwargs)

    monkeypatch.setattr(model.transformer, "forward", record)
    model(_sample())
    expected = torch.tensor([time], dtype=torch.float32) / 1000
    assert observed["timestep"].dtype == torch.float32
    assert torch.equal(observed["timestep"], expected)
    rounded_early = (torch.tensor([time], dtype=torch.bfloat16) / 1000).float()
    assert not torch.equal(rounded_early, expected), "Fixture must expose early BF16 timestep rounding"


def test_random_crop_and_caption_dropout_follow_original_rng_stream(tmp_path):
    source = Image.fromarray(np.arange(24 * 48 * 3).reshape(24, 48, 3).astype(np.uint8))
    source.save(tmp_path / "target.png")
    config = {"height": 16, "width": 16, "random_crop": True, "caption_dropout": 0.5}
    row = {"image": "target.png", "prompt": "a boat"}
    torch.manual_seed(77)
    top = int(torch.randint(1, (1,)))
    left = int(torch.randint(17, (1,)))
    drop = bool(torch.rand(()) < 0.5)
    expected_rng = torch.get_rng_state().clone()
    expected = source.resize((32, 16), Image.Resampling.BILINEAR).crop((left, top, left + 16, top + 16))
    torch.manual_seed(77)
    actual = QwenImageSFTAdapter().prepare_sample(row, tmp_path, config)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    np.testing.assert_array_equal(np.asarray(actual["image"]), np.asarray(expected))
    assert actual["prompt"] == ("" if drop else "a boat")


def test_deterministic_preprocessing_does_not_consume_training_rng(tmp_path):
    _sample()["image"].save(tmp_path / "target.png")
    state = torch.get_rng_state().clone()
    sample = QwenImageSFTAdapter().prepare_sample(
        {"image": "target.png", "prompt": "boat"}, tmp_path, {"height": 16, "width": 16}
    )
    assert sample["image"].size == (16, 16) and sample["prompt"] == "boat"
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("placement", ["sharded", "replicated"])
def test_qwen_rejects_text_placement_changes_after_sharding(placement):
    model = _model(torch.float32)
    model.config["text_encoder_placement"] = placement
    QwenImageSFTAdapter().prepare_for_sharding(model)
    model.config["text_encoder_placement"] = "replicated" if placement == "sharded" else "sharded"
    model.text_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="Changing text_encoder_placement requires a new model"):
        model(_sample())
    assert model.text_encoder.calls == 0


def test_online_effective_batch_matches_pipeline_text_reference_and_fp32_update():
    models = [_model(torch.bfloat16), _model(torch.bfloat16)]
    reference = models[1]
    reference._condition = lambda data: reference.pipe.encode_prompt(
        data["prompt"], reference.device, reference.config["max_sequence_length"]
    )[0]
    results = []
    for model in models:
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        before = {name: value.detach().clone() for name, value in parameters.items()}
        optimizer = torch.optim.AdamW(parameters.values(), lr=1e-3, foreach=False)
        torch.manual_seed(73)
        losses = []
        # Repeated conditions must still encode and independently sample each micro.
        for _ in range(2):
            loss = model(_sample()) / 2
            loss.backward()
            losses.append(loss.detach())
        gradients = {name: value.grad.detach().clone() for name, value in parameters.items()}
        optimizer.step()
        updates = {name: value.detach() - before[name] for name, value in parameters.items()}
        assert model.text_encoder.calls == len(model.vae.samples) == 2
        assert not torch.equal(model.vae.samples[0], model.vae.samples[1])
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in updates.values())
        assert all(torch.count_nonzero(value) > 0 for value in updates.values())
        results.append((torch.stack(losses), gradients, updates, torch.get_rng_state().clone()))
    actual, expected = results
    assert torch.equal(actual[0], expected[0])
    for index in (1, 2):
        assert actual[index].keys() == expected[index].keys()
        for name in actual[index]:
            assert torch.equal(actual[index][name], expected[index][name]), name
    assert torch.equal(actual[3], expected[3])


def test_replicated_text_encoder_unfreeze_fails_before_encoding():
    model = _model(dtype=torch.float32)
    model.config["text_encoder_placement"] = "replicated"
    QwenImageSFTAdapter().prepare_for_sharding(model)
    model.text_encoder.embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Replicated Qwen conditioning is frozen-only"):
        model(_sample())
    assert model.text_encoder.calls == 0
    assert model.text_encoder.embedding.weight.grad is None
