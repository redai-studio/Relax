# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real CPU autograd/cache tests for the Qwen SFT contract with tiny modules.

These do not replace the pretrained GPU comparison. The small pipeline retains
text encoding, posterior sampling, noise/timestep sampling and trainable low-
rank updates so public cache and full-effective-batch verification run without
mocks.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from relax.engine.sft.conditioning import ConditioningRuntime
from relax.engine.sft.conditioning_verification import verify_update
from relax.models.qwen_image import sft_adapter
from relax.models.qwen_image.sft_adapter import (
    QwenImageSFTAdapter,
    QwenImageSFTModel,
    flow_matching_loss,
    pack_latents,
    unpack_latents,
)
from relax.utils.conditioning_cache import build_key


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


class TinyPosteriorVAE(TinyVAE):
    """Deterministic encoder with the actual Diffusers posterior/sampler."""

    def __init__(self):
        from diffusers.configuration_utils import FrozenDict

        super().__init__()
        self.config = FrozenDict(vars(self.config))
        self.use_tiling = self.use_slicing = False
        self.calls = self.clear_calls = 0
        self.fail_encode = False
        self._feat_cache = []
        self.posterior = None
        self.posterior_memory_format = torch.contiguous_format
        self.register_buffer("offset", torch.tensor(0.0))

    def clear_cache(self):
        self.clear_calls += 1
        self._feat_cache = []

    def encode(self, pixels):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        self.calls += 1
        base = torch.nn.functional.avg_pool2d(pixels.squeeze(2).float(), 8)[:, :2].unsqueeze(2)
        base = (base * self.scale + self.offset).to(pixels.dtype)
        self._feat_cache = [base]
        if self.fail_encode:
            raise RuntimeError("VAE encode interrupted")
        parameters = torch.cat((base, torch.full_like(base, -4.0)), dim=1)
        parameters = torch.empty_like(parameters, memory_format=self.posterior_memory_format).copy_(parameters)
        self.posterior = DiagonalGaussianDistribution(parameters)
        self.clear_cache()
        return SimpleNamespace(latent_dist=self.posterior)


def _posterior_model(dtype=torch.bfloat16, *, placement="sharded"):
    torch.manual_seed(31)
    pipeline = TinyPipeline(dtype)
    pipeline.text_encoder.to(dtype)
    pipeline.vae = TinyPosteriorVAE().to(dtype)
    pipeline.vae_scale_factor = 8
    model = QwenImageSFTModel(
        pipeline,
        {"max_sequence_length": 4, "cache_vae_posterior": True, "text_encoder_placement": placement},
        dtype=dtype,
    )
    model.train()
    return model


def _sample(prompt="red boat", offset=0):
    pixels = (np.arange(16 * 16 * 3).reshape(16, 16, 3) + offset).astype(np.uint8)
    return {"prompt": prompt, "image": Image.fromarray(pixels)}


@pytest.fixture
def cache_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_CONDITION_CACHE", "1")
    monkeypatch.setenv("RELAX_CONDITION_CACHE_MAX_BYTES", str(1024**2))
    monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "0")
    monkeypatch.setenv("RELAX_CONDITION_VERIFY_SNAPSHOT", "0")
    monkeypatch.delenv("RELAX_CONDITION_CACHE_RUN_DIR", raising=False)
    runtimes = []

    def create(model):
        from safetensors.torch import save_file

        directory = tmp_path / f"model-{len(runtimes)}"
        directory.mkdir()
        save_file(dict(model.text_encoder.state_dict()), directory / "model.safetensors")
        runtime = ConditioningRuntime(
            model=model,
            model_path=str(directory),
            device=torch.device("cpu"),
            group=None,
            rank=0,
            world_size=1,
            namespace=f"qwen-tiny-{len(runtimes)}",
            config={"model": "tiny-qwen-contract"},
            metadata_sha256="tiny-data-v1",
            global_batch_size=2,
            start_step=0,
            num_steps=2,
        )
        runtimes.append(runtime)
        return runtime

    yield create
    for runtime in reversed(runtimes):
        runtime.close(0, status="cancelled")


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


def test_condition_cache_preserves_every_bf16_bit_and_independent_reference(cache_runtime):
    model = _model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    initial_rng = torch.get_rng_state().clone()
    runtime.force_online = True
    reference = model._condition(_sample())
    assert model.pipe.reference_calls == 1
    runtime.force_online = False
    actual = model._condition(_sample())
    assert model.text_encoder.calls == 1
    assert reference.dtype == actual.dtype == torch.bfloat16
    assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert runtime._step["encoding_calls"] == runtime._step["effective_hit"] == 1
    actual.zero_()
    assert torch.equal(model._condition(_sample()), reference), "Consumer mutation must not modify stored data"


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


@pytest.mark.parametrize("fault", ["dtype", "shape", "fields"])
def test_invalid_cached_condition_is_recomputed_without_changing_tensor_contract(cache_runtime, fault):
    model = _model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    data = _sample()
    expected = model._condition(data)
    key = runtime._conditioning_key(model.conditioning_spec(data))
    runtime.cache.invalidate(key)
    invalid = expected.float() if fault == "dtype" else expected[:, :1] if fault == "shape" else expected
    runtime.cache.put(key, {"unexpected" if fault == "fields" else "prompt_embeds": invalid})
    actual = model._condition(data)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)
    assert runtime._step["cache_errors"] == 1
    assert runtime._step["effective_hit"] == 0
    assert model.text_encoder.calls == 2


@pytest.mark.parametrize("field", ["prompt", "template", "prefix", "context", "length", "dtype", "rope", "attention"])
def test_effective_condition_dependencies_invalidate_key(field):
    model = _model()
    data = _sample()
    reference = build_key(model.conditioning_spec(data).dependencies)
    if field == "prompt":
        data["prompt"] = "blue boat"
    elif field == "template":
        model.pipe.prompt_template_encode = "[{}]"
    elif field == "prefix":
        model.pipe.prompt_template_encode_start_idx = 2
    elif field == "context":
        model.pipe.tokenizer_max_length = 11
    elif field == "length":
        model.config["max_sequence_length"] = 3
    elif field == "dtype":
        model.compute_dtype = torch.float32
    elif field == "rope":
        model.text_encoder.config.rope_theta = 20000
    else:
        model.text_encoder.config._attn_implementation = "eager"
    assert build_key(model.conditioning_spec(data).dependencies) != reference


def test_truncated_equivalent_prompts_share_effective_tokens_and_key():
    model = _model()
    first = model.conditioning_spec({"prompt": "abcdefghijklmnop-first"})
    second = model.conditioning_spec({"prompt": "abcdefghijklmnop-second"})
    assert first.dependencies == second.dependencies
    assert len(first.dependencies["input_ids"][0]) == 13
    assert first.dependencies["encoder_context_length"] == 12


def test_hf_encoder_config_integer_label_keys_have_canonical_cache_identity():
    model = _model()
    model.text_encoder.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    model.text_encoder.config.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    spec = model.conditioning_spec(_sample())
    assert isinstance(spec.dependencies["encoder_config"], str)
    assert json.loads(spec.dependencies["encoder_config"])["id2label"] == {"0": "LABEL_0", "1": "LABEL_1"}
    reference = build_key(spec.dependencies)
    model.text_encoder.config.id2label = {1: "LABEL_1", 0: "LABEL_0"}
    assert build_key(model.conditioning_spec(_sample()).dependencies) == reference
    model.text_encoder.config.id2label[0] = "OTHER"
    assert build_key(model.conditioning_spec(_sample()).dependencies) != reference


def test_deterministic_attention_controls_are_cache_dependencies():
    model = _model()
    data = _sample()
    reference = build_key(model.conditioning_spec(data).dependencies)
    model.config["deterministic_attention"] = True
    assert build_key(model.conditioning_spec(data).dependencies) != reference
    model.config["deterministic_attention"] = False
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(not previous)
        assert build_key(model.conditioning_spec(data).dependencies) != reference
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("mutation", ["weight", "trainable", "train_mode"])
def test_frozen_encoder_mutation_disables_reuse_and_keeps_gradients(cache_runtime, mutation):
    model = _model(dtype=torch.float32)
    runtime = cache_runtime(model)
    runtime.sample(0)
    original = model._condition(_sample())
    if mutation == "weight":
        with torch.no_grad():
            model.text_encoder.embedding.weight.add_(1)
    elif mutation == "trainable":
        model.text_encoder.embedding.weight.requires_grad_(True)
    else:
        model.text_encoder.train()
    runtime.sample(1)
    result = model._condition(_sample())
    assert runtime._cache_guard_disabled
    assert model.text_encoder.calls == 2
    assert runtime._step["effective_hit"] == 0
    if mutation == "weight":
        assert not torch.equal(result, original)
    elif mutation == "trainable":
        result.sum().backward()
        assert torch.count_nonzero(model.text_encoder.embedding.weight.grad)
        with pytest.raises(ValueError, match="confined to the DiT"):
            QwenImageSFTAdapter().prepare_for_sharding(model)


@pytest.mark.parametrize("posterior_bundle", [False, True])
@pytest.mark.parametrize("mutation", ["inplace", "replace"])
def test_text_buffer_mutation_disables_old_text_and_joint_cache(cache_runtime, posterior_bundle, mutation):
    model = (_posterior_model if posterior_bundle else _model)(dtype=torch.float32)
    runtime = cache_runtime(model)
    runtime.sample(0)
    data = _sample()
    original = model._condition(data)
    assert torch.equal(model._condition(data), original)
    assert runtime._step["effective_hit"] == 1
    key = runtime._conditioning_key(model.conditioning_spec(data))
    if mutation == "inplace":
        model.text_encoder.conditioning_offset.add_(1)
    else:
        model.text_encoder.conditioning_offset = torch.ones_like(model.text_encoder.conditioning_offset)
    # This nonpersistent rotary-like state is absent from the checkpoint and
    # effective-input key; the live tensor identity/version guard must catch it.
    assert "conditioning_offset" not in model.text_encoder.state_dict()
    assert runtime._conditioning_key(model.conditioning_spec(data)) == key
    runtime.sample(1)
    actual = model._condition(data)
    assert runtime._cache_guard_disabled
    assert "text_encoder.buffers.conditioning_offset" in runtime._cache_guard_reason
    assert model.text_encoder.calls == 2 and runtime._step["effective_hit"] == 1
    assert torch.equal(actual, original + 1)
    assert not torch.equal(actual, original)
    if posterior_bundle:
        assert model.vae.calls == 2


def test_train_mode_keeps_condition_encoder_and_vae_in_eval():
    model = _model()
    model.eval()
    model.train()
    assert model.training and model.transformer.training
    assert not model.text_encoder.training and not model.vae.training
    assert model.conditioning_is_frozen()


@pytest.mark.parametrize("placement", [None, "sharded", "replicated"])
def test_text_encoder_placement_preserves_registration_and_conditioning(placement):
    model = _model(dtype=torch.float32)
    if placement is not None:
        model.config["text_encoder_placement"] = placement
    encoder = model.text_encoder
    names = list(model.state_dict())
    parameters = dict(model.named_parameters())
    original_key = build_key(model.conditioning_spec(_sample()).dependencies)
    QwenImageSFTAdapter().prepare_for_sharding(model)
    assert model.text_encoder is encoder and model.pipe.text_encoder is encoder
    assert list(model.state_dict()) == names
    assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
    assert model._fsdp_replicated_modules == ((encoder,) if placement == "replicated" else ())
    assert encoder._fsdp_fine_grained is (placement != "replicated")
    assert build_key(model.conditioning_spec(_sample()).dependencies) == original_key


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


def test_replicated_text_encoder_unfreeze_fails_before_cache_or_encoding(cache_runtime):
    model = _model(dtype=torch.float32)
    model.config["text_encoder_placement"] = "replicated"
    QwenImageSFTAdapter().prepare_for_sharding(model)
    runtime = cache_runtime(model)
    runtime.sample(0)
    model._condition(_sample())
    calls = model.text_encoder.calls
    model.text_encoder.embedding.weight.requires_grad_(True)
    runtime.sample(1)
    with pytest.raises(ValueError, match="Replicated Qwen conditioning is frozen-only"):
        model._condition(_sample())
    assert model.text_encoder.calls == calls
    assert model.text_encoder.embedding.weight.grad is None


def test_vae_posterior_noise_and_timesteps_remain_online_with_same_rng(cache_runtime, monkeypatch):
    model = _model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    model._condition(_sample())
    captured = []
    objective = sft_adapter.flow_matching_loss

    def record(transformer, clean, prompt, noise, sigma, timestep):
        captured.append(tuple(value.detach().clone() for value in (clean, noise, sigma, timestep)))
        return objective(transformer, clean, prompt, noise, sigma, timestep)

    monkeypatch.setattr(sft_adapter, "flow_matching_loss", record)
    states, results = [], []
    for force_online in (True, False):
        runtime.force_online = force_online
        torch.manual_seed(73)
        results.append(torch.stack([model(_sample()) for _ in range(4)]))
        states.append(torch.get_rng_state().clone())
    assert len(model.vae.samples) == len(captured) == 8
    assert torch.equal(results[0], results[1]) and torch.equal(states[0], states[1])
    for reference, actual in zip(captured[:4], captured[4:]):
        assert all(torch.equal(left, right) for left, right in zip(reference, actual))
    assert not torch.equal(captured[0][0], captured[1][0])
    assert not torch.equal(captured[0][1], captured[1][1])
    assert len({float(item[2].flatten()[0]) for item in captured[:4]}) > 1


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


@pytest.mark.parametrize(
    "posterior_bundle,memory_format",
    [(False, None), (True, torch.contiguous_format), (True, torch.channels_last_3d)],
)
def test_public_verifier_compares_complete_two_sample_batch_and_fp32_update(
    cache_runtime, tmp_path, posterior_bundle, memory_format
):
    model = (_posterior_model if posterior_bundle else _model)(dtype=torch.float32)
    if posterior_bundle:
        model.vae.posterior_memory_format = memory_format
    runtime = cache_runtime(model)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([parameter for _, parameter in parameters], lr=0.01)
    samples = [_sample("red boat"), _sample("blue bird", offset=17)]

    def forward(sample):
        runtime.sample(samples.index(sample))
        return model(sample)

    report = verify_update(
        model,
        optimizer,
        parameters,
        samples,
        forward,
        lambda value: setattr(runtime, "force_online", value),
        device=torch.device("cpu"),
        group=None,
        gloo_group=None,
        loss_scale=0.5,
        rank=0,
        output_dir=tmp_path / "verification",
        relative_tolerance=0.0,
        cache_statistics=lambda: {
            "effective_hit": runtime._step["effective_hit"],
            "encoding_calls": runtime._step["encoding_calls"],
        },
    )
    assert report["passed"] and report["cache_branch_verified"]
    assert report["local_samples"] == 2
    assert all(parameter.dtype == torch.float32 for _, parameter in parameters)
    assert len(optimizer.state) == len(parameters)


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
    model = _model()
    assert (
        model.conditioning_spec(actual).dependencies
        == model.conditioning_spec({"prompt": actual["prompt"]}).dependencies
    )


def test_deterministic_preprocessing_does_not_consume_training_rng(tmp_path):
    _sample()["image"].save(tmp_path / "target.png")
    state = torch.get_rng_state().clone()
    sample = QwenImageSFTAdapter().prepare_sample(
        {"image": "target.png", "prompt": "boat"}, tmp_path, {"height": 16, "width": 16}
    )
    assert sample["image"].size == (16, 16) and sample["prompt"] == "boat"
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("posterior_bundle", [False, True])
def test_online_and_cached_normal_micro_loops_preserve_random_augmentation_order(
    cache_runtime, tmp_path, posterior_bundle
):
    source = Image.fromarray(np.arange(24 * 48 * 3).reshape(24, 48, 3).astype(np.uint8))
    source.save(tmp_path / "target.png")
    config = {"height": 16, "width": 16, "random_crop": True, "caption_dropout": 0.5}
    row = {"image": "target.png", "prompt": "red boat"}
    model = _posterior_model() if posterior_bundle else _model()
    runtime = cache_runtime(model)
    adapter = QwenImageSFTAdapter()
    runs, states = [], []
    for force_online in (True, False):
        runtime.force_online = force_online
        torch.manual_seed(137)
        presentations = []
        for micro in range(6):
            runtime.sample(micro)
            data = adapter.prepare_sample(row, tmp_path, config, index=micro)
            # Preserve the actual trainer ordering: augmentation, forward's VAE
            # posterior/noise/timestep, then the next presentation's augmentation.
            loss = model(data)
            presentations.append((data["prompt"], np.array(data["image"]), loss.detach().clone()))
        runs.append(presentations)
        states.append(torch.get_rng_state().clone())
    assert torch.equal(states[0], states[1])
    for reference, actual in zip(runs[0], runs[1]):
        assert reference[0] == actual[0]
        np.testing.assert_array_equal(reference[1], actual[1])
        assert torch.equal(reference[2], actual[2])
    assert {presentation[0] for presentation in runs[0]} == {"red boat", ""}
    assert runtime._step["encoding_calls"] == runtime._step["effective_hit"] == 6


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last_3d])
@pytest.mark.parametrize("degenerate", [False, True])
def test_joint_bundle_preserves_posterior_bits_and_original_cold_object(
    cache_runtime, dtype, memory_format, degenerate
):
    model = _posterior_model(dtype)
    model.vae.posterior_memory_format = memory_format
    runtime = cache_runtime(model)
    runtime.sample(0)
    data = _sample()
    if degenerate:
        data["image"] = data["image"].resize((8, 8))
    initial_rng = torch.get_rng_state().clone()
    runtime.force_online = True
    cold = model._conditioning_features(data)
    assert cold["posterior"] is model.vae.posterior
    parameters = cold["posterior"].parameters
    assert parameters.is_contiguous(memory_format=memory_format)
    if degenerate:
        # Both formats describe the same element order when spatial axes are 1.
        assert parameters.is_contiguous() and parameters.is_contiguous(memory_format=torch.channels_last_3d)
    else:
        assert parameters.is_contiguous() is (memory_format == torch.contiguous_format)
    field = "posterior_parameters" if parameters.is_contiguous() else "posterior_parameters_channels_last_3d"
    key = runtime._conditioning_key(model.conditioning_spec(data))
    with runtime.cache.lookup(key) as lease:
        assert set(lease.data) == {"prompt_embeds", field}
        assert lease.data[field].is_contiguous(), "The real public store canonicalizes both layouts"
    runtime.force_online = False
    hot = model._conditioning_features(data)
    assert hot["posterior"] is not cold["posterior"]
    assert hot["posterior"].parameters.is_contiguous(memory_format=memory_format)
    if not degenerate:
        assert hot["posterior"].parameters.stride() == parameters.stride()
    elif memory_format == torch.channels_last_3d:
        assert hot["posterior"].parameters.stride() != parameters.stride()
    assert model.text_encoder.calls == model.vae.calls == 1
    assert runtime._step["encoding_calls"] == runtime._step["effective_hit"] == 1
    assert torch.equal(initial_rng, torch.get_rng_state()), "Neither branch may sample while resolving"
    for reference, actual in (
        (cold["prompt_embeds"], hot["prompt_embeds"]),
        (cold["posterior"].parameters, hot["posterior"].parameters),
    ):
        assert reference.dtype == actual.dtype == dtype
        assert torch.equal(reference.reshape(-1).view(torch.uint8), actual.reshape(-1).view(torch.uint8))
    # Sampling and the next random draw retain logical element correspondence,
    # including the ambiguous-format singleton case.
    sampled, noises, states = [], [], []
    for features in (cold, hot):
        torch.manual_seed(73)
        sampled.append(features["posterior"].sample())
        noises.append(torch.randn_like(sampled[-1]))
        states.append(torch.get_rng_state().clone())
    assert torch.equal(sampled[0], sampled[1]) and torch.equal(noises[0], noises[1])
    assert torch.equal(states[0], states[1])
    for actual in (hot["prompt_embeds"], hot["posterior"].parameters):
        actual.zero_()
    fresh = model._conditioning_features(data)
    assert torch.equal(fresh["posterior"].parameters, cold["posterior"].parameters)
    assert torch.equal(fresh["prompt_embeds"], cold["prompt_embeds"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("memory_format", [torch.contiguous_format, torch.channels_last_3d])
def test_joint_bundle_samples_noise_and_timestep_online_with_identical_rng(
    cache_runtime, monkeypatch, dtype, memory_format
):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = _posterior_model(dtype)
    model.vae.posterior_memory_format = memory_format
    runtime = cache_runtime(model)
    runtime.sample(0)
    captured = []
    sampled = []
    objective = sft_adapter.flow_matching_loss
    original_sample = DiagonalGaussianDistribution.sample

    def sample(posterior, *args, **kwargs):
        value = original_sample(posterior, *args, **kwargs)
        sampled.append(value.detach().clone())
        return value

    def record(transformer, clean, prompt, noise, sigma, timestep):
        captured.append(tuple(value.detach().clone() for value in (clean, noise, sigma, timestep)))
        return objective(transformer, clean, prompt, noise, sigma, timestep)

    monkeypatch.setattr(sft_adapter, "flow_matching_loss", record)
    monkeypatch.setattr(DiagonalGaussianDistribution, "sample", sample)
    states, losses = [], []
    for force_online in (True, False):
        runtime.force_online = force_online
        torch.manual_seed(73)
        losses.append(torch.stack([model(_sample()) for _ in range(4)]))
        states.append(torch.get_rng_state().clone())
    assert model.vae.calls == model.text_encoder.calls == 4
    assert runtime._step["encoding_calls"] == runtime._step["effective_hit"] == 4
    assert torch.equal(losses[0], losses[1]) and torch.equal(states[0], states[1])
    for reference, actual in zip(sampled[:4], sampled[4:]):
        assert torch.equal(reference, actual) and reference.stride() == actual.stride()
    for reference, actual in zip(captured[:4], captured[4:]):
        assert all(torch.equal(left, right) for left, right in zip(reference, actual))
        assert all(left.stride() == right.stride() for left, right in zip(reference, actual))
    assert not torch.equal(captured[0][0], captured[1][0])
    assert not torch.equal(captured[0][1], captured[1][1])


@pytest.mark.parametrize("field", ["pixels", "shape", "prompt", "vae_config", "dtype", "scale_factor"])
def test_joint_bundle_dependencies_cover_actual_augmented_image_and_vae(field):
    model = _posterior_model()
    data = _sample()
    reference = model.conditioning_spec(data)
    if field == "pixels":
        data["image"].putpixel((0, 0), (123, 45, 67))
    elif field == "shape":
        data["image"] = data["image"].resize((32, 16))
    elif field == "prompt":
        data["prompt"] = "blue bird"
    elif field == "vae_config":
        model.vae.config.latents_mean[0] += 0.25
    elif field == "dtype":
        model.compute_dtype = torch.float32
    else:
        model.pipe.vae_scale_factor = 16
    assert model.conditioning_spec(data).name == reference.name == "qwen_image.text_vae_posterior.v2"
    assert build_key(model.conditioning_spec(data).dependencies) != build_key(reference.dependencies)


def test_joint_bundle_reuses_image_content_and_effective_tokens_not_filenames(cache_runtime):
    model = _posterior_model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    first = _sample("abcdefghijklmnop-first")
    first["filename"] = "first.png"
    same = _sample("abcdefghijklmnop-second")
    same["filename"] = "second.png"
    assert model.conditioning_spec(first).dependencies == model.conditioning_spec(same).dependencies
    model._condition(first)
    model._condition(same)
    assert runtime._step["effective_hit"] == 1
    # A different image is a distinct bundle even when the effective text is identical.
    model._condition(_sample("abcdefghijklmnop-third", offset=7))
    assert model.vae.calls == model.text_encoder.calls == 2


@pytest.mark.parametrize("fault", ["dtype", "shape", "fields", "both_layouts", "missing", "corrupt"])
def test_bad_joint_bundle_falls_back_to_online_without_leases_or_rng_changes(cache_runtime, fault):
    model = _posterior_model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    data = _sample()
    expected = model._conditioning_features(data)
    key = runtime._conditioning_key(model.conditioning_spec(data))
    if fault == "missing":
        runtime.cache.invalidate(key)
    elif fault == "corrupt":
        digest = hashlib.sha256(key.encode()).hexdigest()
        (runtime.cache.directory / f"{digest}.safetensors").write_bytes(b"corrupt")
    else:
        runtime.cache.invalidate(key)
        parameters = expected["posterior"].parameters
        invalid = parameters.float() if fault == "dtype" else parameters[:, :1] if fault == "shape" else parameters
        payload = {
            "prompt_embeds": expected["prompt_embeds"],
            "wrong" if fault == "fields" else "posterior_parameters": invalid,
        }
        if fault == "both_layouts":
            payload["posterior_parameters_channels_last_3d"] = parameters
        runtime.cache.put(key, payload)
    initial_rng = torch.get_rng_state().clone()
    actual = model._conditioning_features(data)
    assert torch.equal(actual["posterior"].parameters, expected["posterior"].parameters)
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert model.vae.calls == model.text_encoder.calls == 2
    assert runtime._step["effective_hit"] == 0
    assert runtime.cache.stats()["leased_bytes"] == 0


@pytest.mark.parametrize("layout", ["transpose", "holes"])
def test_joint_bundle_rejects_unsupported_posterior_layout_before_storage(cache_runtime, monkeypatch, layout):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

    model = _posterior_model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    original_encode = model.vae.encode

    def encode(pixels):
        parameters = original_encode(pixels).latent_dist.parameters
        if layout == "transpose":
            parameters = parameters.transpose(-2, -1)
        else:
            parameters = parameters.repeat_interleave(2, dim=-1)[..., ::2]
        assert not parameters.is_contiguous()
        assert not parameters.is_contiguous(memory_format=torch.channels_last_3d)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(parameters))

    monkeypatch.setattr(model.vae, "encode", encode)
    initial_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="only contiguous or channels_last_3d"):
        model(_sample())
    assert runtime.cache.stats()["entries"] == 0
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert model.vae._feat_cache == []


def test_joint_bundle_guard_exposes_parameter_and_buffer_stride_without_readback():
    model = _posterior_model()
    tensor = torch.zeros(1, 2, 1, 2, 2)
    model.vae.register_parameter("layout_parameter", torch.nn.Parameter(tensor.clone(), requires_grad=False))
    model.vae.register_buffer("layout_buffer", tensor.clone())
    before = model.conditioning_state()["parameters"]
    model.vae.to(memory_format=torch.channels_last_3d)
    after = model.conditioning_state()["parameters"]
    for name in ("vae.parameters.layout_parameter", "vae.buffers.layout_buffer"):
        assert before[name]["stride"] != after[name]["stride"]
        assert before[name]["shape"] == after[name]["shape"]


@pytest.mark.parametrize("mutation", ["weight", "buffer", "train_mode", "trainable"])
def test_joint_bundle_guard_covers_frozen_vae_and_refuses_old_posteriors(cache_runtime, mutation):
    model = _posterior_model(dtype=torch.float32)
    runtime = cache_runtime(model)
    runtime.sample(0)
    expected = model._conditioning_features(_sample())
    if mutation in ("weight", "buffer"):
        with torch.no_grad():
            (model.vae.scale if mutation == "weight" else model.vae.offset).add_(1)
    elif mutation == "train_mode":
        model.vae.train()
    else:
        model.vae.scale.requires_grad_(True)
    runtime.sample(1)
    assert runtime._cache_guard_disabled
    if mutation == "trainable":
        with pytest.raises(ValueError, match="Replicated Qwen VAE is frozen-only"):
            model(_sample())
        assert model.vae.calls == 1
    else:
        actual = model._conditioning_features(_sample())
        assert model.vae.calls == 2
        if mutation != "train_mode":
            assert not torch.equal(actual["posterior"].parameters, expected["posterior"].parameters)
    assert runtime._step["effective_hit"] == 0


@pytest.mark.parametrize("placement", ["sharded", "replicated"])
def test_joint_bundle_composes_replica_ownership_and_collective_contract(placement):
    model = _posterior_model(placement=placement)
    state = model.state_dict()
    QwenImageSFTAdapter().prepare_for_sharding(model)
    expected = (model.text_encoder, model.vae) if placement == "replicated" else (model.vae,)
    assert model._fsdp_replicated_modules == expected
    assert model._conditioning_collective_free_modules == (expected if placement == "replicated" else ())
    assert model.text_encoder._fsdp_fine_grained is (placement == "sharded")
    assert model.state_dict().keys() == state.keys()


@pytest.mark.parametrize("case", ["dtype", "device", "train_mode"])
def test_joint_bundle_validates_resident_vae_device_dtype_and_mode(case):
    model = _posterior_model()
    if case == "dtype":
        model.vae.float()
    elif case == "device":
        model.device = torch.device("meta")
    else:
        model.vae.train()
    with pytest.raises(ValueError, match="Replicated VAE"):
        QwenImageSFTAdapter().prepare_for_sharding(model)
    assert not getattr(model, "_fsdp_replicated_modules", ())


@pytest.mark.parametrize("mode", ["use_tiling", "use_slicing", "change_policy"])
def test_joint_bundle_rejects_unsupported_or_changed_execution_policy(mode):
    model = _posterior_model()
    if mode == "change_policy":
        model.config["cache_vae_posterior"] = False
    else:
        setattr(model.vae, mode, True)
    with pytest.raises(ValueError, match="requires a new model|tiling and slicing disabled"):
        model(_sample())
    assert model.vae.calls == model.text_encoder.calls == 0


@pytest.mark.parametrize("posterior_bundle", [False, True])
@pytest.mark.parametrize("placement", ["sharded", "replicated"])
def test_qwen_rejects_text_placement_changes_after_sharding(posterior_bundle, placement):
    model = _posterior_model(dtype=torch.float32, placement=placement) if posterior_bundle else _model(torch.float32)
    model.config["text_encoder_placement"] = placement
    QwenImageSFTAdapter().prepare_for_sharding(model)
    model.config["text_encoder_placement"] = "replicated" if placement == "sharded" else "sharded"
    model.text_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="Changing text_encoder_placement requires a new model"):
        model(_sample())
    assert model.text_encoder.calls == 0


@pytest.mark.parametrize("status", ["cancelled", "error", "complete"])
def test_joint_bundle_releases_temporal_cache_and_storage_across_runs(cache_runtime, status):
    model = _posterior_model()
    runtime = cache_runtime(model)
    runtime.sample(0)
    model.vae.fail_encode = True
    with pytest.raises(RuntimeError, match="VAE encode interrupted"):
        model(_sample())
    assert model.vae.clear_calls == 1 and model.vae._feat_cache == []
    assert runtime.cache.stats()["entries"] == 0
    model.vae.fail_encode = False
    owned = model._conditioning_features(_sample())
    reference = owned["posterior"].parameters.clone()
    directory = runtime.cache.directory
    assert runtime.cache.stats()["entries"] == 1
    runtime.close(2 if status == "complete" else 0, status=status)
    assert model.conditioning_cache is None and not list(directory.glob("*.safetensors"))
    assert torch.equal(owned["posterior"].parameters, reference)
    assert model.vae._feat_cache == []
    second = cache_runtime(model)
    second.sample(0)
    model(_sample())
    assert second._step["encoding_calls"] == 1 and second._step["effective_hit"] == 0
    second.close(0, status="cancelled")
    assert not list(second.cache.directory.glob("*.safetensors"))


def test_online_effective_batch_matches_pipeline_text_reference_and_fp32_update():
    models = [_model(torch.bfloat16), _model(torch.bfloat16)]
    reference = models[1]
    reference._conditioning_features = lambda data: {
        "prompt_embeds": reference.pipe.encode_prompt(
            data["prompt"], reference.device, reference.config["max_sequence_length"]
        )[0]
    }
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
