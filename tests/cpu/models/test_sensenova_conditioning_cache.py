# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Frozen conditioning reuse, gradient boundaries and RNG regression tests."""

import copy
import json

import pytest
import torch


class _Tokenizer:
    special_tokens_map = {"eos_token": "<end>"}

    def __init__(self):
        self.added = {}
        self.vocabulary = {"edit": 0, "<end>": 1}
        self.init_kwargs = {"normalizer": "original"}
        self.normalizer = "identity"
        self.bpe_ranks = {}
        self.special_ids = {"<img>": 6, "<IMG_CONTEXT>": 7}
        self.calls = 0

    def get_vocab(self):
        return {**self.vocabulary, **self.added}

    def get_added_vocab(self):
        return dict(self.added)

    def convert_tokens_to_ids(self, token):
        return self.special_ids[token]

    def __call__(self, text, return_tensors=None, **kwargs):
        self.calls += 1
        # Ignore the system template and synthetic image run when determining
        # the effective caption. Empty-caption dropout must remain observable.
        caption = text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0]
        token = "edit" if "edit" in caption else "<end>"
        token_id = self.added.get(token, self.vocabulary[token])
        if self.normalizer == "changed":
            token_id = (token_id + 1) % 6
        token_id = (token_id + sum(self.bpe_ranks.values())) % 6
        ids = [token_id]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}


class _MemoryResolver:
    """A storage-free resolver exercising the public model-side contract."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.entries = {}
        self.computes = 0
        self.force_online = False

    def resolve(self, spec, compute, restore):
        key = json.dumps(spec.dependencies, sort_keys=True)
        if self.enabled and not self.force_online and spec.cacheable and key in self.entries:
            return restore(self.entries[key])
        self.computes += 1
        features = compute()
        if self.enabled and spec.cacheable:
            self.entries[key] = {name: value.detach().clone() for name, value in features.items()}
        return restore(features)


@pytest.fixture
def tiny_conditioning_model(monkeypatch):
    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova import adapter as sft
    from relax.models.sensenova.model import SenseNovaU1Model

    sft.configure_attention("torch")

    class Pipe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device, self.torch_dtype = torch.device("cpu"), torch.float32
            self.tokenizer = _Tokenizer()
            self.dit = torch.nn.Module()
            self.dit.patch_size, self.dit.downsample_ratio, self.dit.merge_size = 16, 0.5, 2
            self.dit.vision_model = torch.nn.Linear(3, 32, bias=False)
            self.dit.language_model = torch.nn.Module()
            self.dit.language_model.model = SenseNovaU1Model(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=8,
                pad_token_id=0,
            )
            self.units = tuple(range(5))
            self.prefix_calls = 0
            self.prefix_prepared = []
            self.random_events = []
            self.eval().requires_grad_(False)
            for name, parameter in self.dit.language_model.model.named_parameters():
                if "_mot_gen" in name:
                    parameter.requires_grad_(True)

        def unit_runner(self, unit, pipe, shared, positive, negative):
            if unit == 1:
                pixels = torch.tensor(list(shared["edit_image"][0].getpixel((0, 0))), dtype=torch.float32)
                shared["vision"] = self.dit.vision_model(pixels.reshape(1, 1, 3) / 255)
            elif unit == 2:
                self.prefix_calls += 1
                prepared = shared.get("prepared_prefix")
                self.prefix_prepared.append(prepared is not None)
                ids = (
                    prepared["input_ids"]
                    if prepared is not None
                    else self.tokenizer(positive["prompt"], return_tensors="pt")["input_ids"]
                )
                indexes = prepared["indexes"] if prepared is not None else torch.zeros(3, 1, dtype=torch.long)
                _, cache = self.dit.language_model.model(
                    inputs_embeds=self.dit.language_model.model.embed_tokens(ids) + shared["vision"],
                    indexes=indexes,
                    use_cache=True,
                    update_cache=True,
                )
                positive.update(past_key_values=cache, indexes_image=torch.zeros(3, 1, dtype=torch.long))
                negative.update(positive)
            elif unit == 3:
                # This draw must still happen on cache hits, even though the
                # training loss will draw its own timestep/noise afterwards.
                self.random_events.append(torch.randn(1))
            elif unit == 4:
                shared["input_latents"] = torch.ones(1, 1, 32)
            return shared, positive, negative

    def loss(pipe, inputs):
        timestep = torch.randint(0, 1000, (1,))
        noise = torch.randn_like(inputs["input_latents"])
        pipe.random_events.extend([timestep, noise])
        generated, _ = pipe.dit.language_model.model(
            inputs_embeds=noise,
            image_gen_indicators=torch.ones(1, 1, dtype=torch.bool),
            indexes=inputs["indexes_image"],
            past_key_values=inputs["past_key_values"],
            use_cache=True,
            update_cache=False,
        )
        return (generated.float() - inputs["input_latents"]).square().mean()

    monkeypatch.setattr(sft, "flow_matching_loss", loss)
    model = sft.SenseNovaSFTModel(Pipe(), prefix_gradient_checkpointing=False)
    data = {
        "image": Image.new("RGB", (32, 32), (90, 80, 70)),
        "edit_image": [Image.new("RGB", (32, 32), (10, 20, 30))],
        "prompt": "edit",
    }
    return model, data


@pytest.fixture
def public_conditioning_runtime(tmp_path, monkeypatch, tiny_conditioning_model):
    """Real tiny vision/decoder and adapter through the production cache."""
    from safetensors.torch import save_file

    from relax.engine.sft.conditioning import ConditioningRuntime
    from relax.models.sensenova.model import SenseNovaU1VisionEncoder

    model, data = tiny_conditioning_model
    vision = (
        SenseNovaU1VisionEncoder(hidden_size=8, llm_hidden_size=32, patch_size=16, max_position_embeddings_vision=8)
        .eval()
        .requires_grad_(False)
    )
    # An ordinary registered buffer must also participate in the guard, even
    # though this extra fixture buffer does not affect the vision calculation.
    vision.register_buffer("guard_fixture", torch.zeros(2))
    model.pipe.dit.vision_model = vision
    original_unit_runner = model.pipe.unit_runner

    def unit_runner(unit, pipe, shared, positive, negative):
        if unit != 1:
            return original_unit_runner(unit, pipe, shared, positive, negative)
        pixels = torch.tensor(list(shared["edit_image"][0].getdata()), dtype=torch.float32)
        pixels = pixels.reshape(2, 16, 2, 16, 3).permute(0, 2, 4, 1, 3).reshape(4, -1) / 255
        shared["vision"] = vision(pixel_values=pixels, grid_hw=torch.tensor([[2, 2]])).unsqueeze(0)
        return shared, positive, negative

    model.pipe.unit_runner = unit_runner
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()}, checkpoint / "model.safetensors"
    )
    monkeypatch.setenv("RELAX_CONDITION_CACHE", "1")
    monkeypatch.setenv("RELAX_CONDITION_CACHE_MAX_BYTES", str(4 * 1024**2))
    monkeypatch.setenv("RELAX_CONDITION_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("RELAX_CONDITION_CACHE_RUN_DIR", raising=False)
    monkeypatch.setenv("RELAX_CONDITION_PREFETCH_DEPTH", "0")
    runtime = ConditioningRuntime(
        model=model,
        model_path=str(checkpoint),
        device=torch.device("cpu"),
        group=None,
        rank=0,
        world_size=1,
        namespace="sensenova-mutation-test",
        config={},
        metadata_sha256="fixed-tiny-image",
        global_batch_size=1,
        start_step=0,
        num_steps=3,
    )
    try:
        yield model, data, runtime
    finally:
        runtime.close(3, status="complete")


def _public_forward(model, data, runtime):
    torch.manual_seed(42)
    runtime.begin_step()
    runtime.sample(0)  # The production entrypoint that actually runs the guard.
    with torch.no_grad():
        loss = model(data)
    return loss, torch.get_rng_state().clone()


def test_public_cache_lazy_rope_materialization_keeps_hot_hits(public_conditioning_runtime):
    model, data, runtime = public_conditioning_runtime
    vision = model.pipe.dit.vision_model.embeddings
    rotary = model.pipe.dit.language_model.model.layers[0].self_attn.rotary_emb
    assert vision.cos_cached_x is None and rotary.inv_freq is None
    spec_before = model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True)
    expected = _public_forward(model, data, runtime)
    assert runtime._conditioning_snapshot is not None
    assert vision.cos_cached_x is not None and rotary.inv_freq is not None
    assert vision.conditioning_rope_cache_is_pristine() and rotary.conditioning_rope_cache_is_pristine()
    vision.to(device="cpu")  # A no-op device placement must not poison reuse.
    actual = _public_forward(model, data, runtime)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual, strict=True))
    assert runtime._step["effective_hit"] == 1 and runtime._step["encoding_calls"] == 0
    assert not runtime._cache_guard_disabled
    assert model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True) == spec_before


@pytest.mark.parametrize(
    "mutation",
    [
        "attention_scaling",
        "rope_theta",
        "rms_epsilon",
        "attention_scale",
        "mode",
        "inv_freq_inplace",
        "inv_freq_replaced",
        "vision_rope_inplace",
        "vision_rope_replaced",
        "ordinary_buffer_inplace",
        "ordinary_buffer_replaced",
        "vision_rope_dtype",
        "parameter_layout",
    ],
)
def test_public_cache_rejects_changed_nonparameter_conditioning(public_conditioning_runtime, mutation):
    model, data, runtime = public_conditioning_runtime
    captured = []
    original_restore = model._restore_conditioning

    def restore(features, *args):
        captured.append({name: value.clone() for name, value in features.items()})
        return original_restore(features, *args)

    model._restore_conditioning = restore
    _public_forward(model, data, runtime)
    old_prefix = captured[-1]
    attention = model.pipe.dit.language_model.model.layers[0].self_attn
    vision = model.pipe.dit.vision_model
    if mutation == "attention_scaling":
        attention.rotary_emb.attention_scaling *= 2
    elif mutation == "rope_theta":
        attention.rotary_emb.rope_theta *= 2
    elif mutation == "rms_epsilon":
        attention.q_norm.variance_epsilon *= 100
    elif mutation == "attention_scale":
        attention.scaling *= 2
    elif mutation == "mode":
        attention.train(not attention.training)
    elif mutation == "inv_freq_inplace":
        attention.rotary_emb.inv_freq.add_(0.1)
    elif mutation == "inv_freq_replaced":
        attention.rotary_emb.inv_freq = attention.rotary_emb.inv_freq + 0.1
    elif mutation == "vision_rope_inplace":
        vision.embeddings.cos_cached_x.add_(0.1)
    elif mutation == "vision_rope_replaced":
        vision.embeddings.cos_cached_x = vision.embeddings.cos_cached_x + 0.1
    elif mutation == "ordinary_buffer_inplace":
        vision.guard_fixture.add_(1)
    elif mutation == "ordinary_buffer_replaced":
        vision.guard_fixture = torch.ones(2)
    elif mutation == "vision_rope_dtype":
        vision.embeddings.cos_cached_x = vision.embeddings.cos_cached_x.double()
    elif mutation == "parameter_layout":
        parameter = vision.embeddings.patch_embedding.weight
        before = (id(parameter), parameter._version, tuple(parameter.stride()))
        vision.embeddings.to(memory_format=torch.channels_last)
        parameter = vision.embeddings.patch_embedding.weight
        assert (id(parameter), parameter._version) == before[:2]
        assert tuple(parameter.stride()) != before[2]
    actual = _public_forward(model, data, runtime)
    current_prefix = captured[-1]
    assert runtime._cache_guard_disabled
    assert runtime._step["effective_hit"] == 0 and runtime._step["encoding_calls"] == 1
    runtime.force_online = True
    expected = _public_forward(model, data, runtime)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))
    assert all(torch.equal(current_prefix[name], value) for name, value in captured[-1].items())
    if mutation == "attention_scaling":
        # The r02 acceptance probe: guard must run after a real warm prefix,
        # and the stale prefix genuinely differs from the current encoder.
        assert not torch.equal(old_prefix["kv.0.key"], current_prefix["kv.0.key"])


def test_generation_only_update_keeps_public_conditioning_hot(public_conditioning_runtime):
    model, data, runtime = public_conditioning_runtime
    _public_forward(model, data, runtime)
    with torch.no_grad():
        model.pipe.dit.language_model.model.layers[0].self_attn.q_proj_mot_gen.weight.add_(0.1)
    actual = _public_forward(model, data, runtime)
    assert not runtime._cache_guard_disabled
    assert runtime._step["effective_hit"] == 1 and runtime._step["encoding_calls"] == 0
    runtime.force_online = True
    expected = _public_forward(model, data, runtime)
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


def test_rope_known_rebuilds_are_allowed_but_do_not_rebaseline_mutations(public_conditioning_runtime):
    model, data, runtime = public_conditioning_runtime
    _public_forward(model, data, runtime)
    vision = model.pipe.dit.vision_model.embeddings
    rotary = model.pipe.dit.language_model.model.layers[0].self_attn.rotary_emb
    for name in ("cos_cached_x", "sin_cached_x", "cos_cached_y", "sin_cached_y"):
        setattr(vision, name, None)
    rotary.inv_freq = None
    # Exercise the same reconstruction helpers used after a device change.
    vision._ensure_rope_cache(torch.device("cpu"))
    rotary._ensure_inv_freq(torch.device("cpu"))
    _public_forward(model, data, runtime)
    assert not runtime._cache_guard_disabled and runtime._step["effective_hit"] == 1
    vision.cos_cached_x.add_(0.1)
    vision._ensure_rope_cache(torch.device("cpu"))  # Already resident: must not bless external writes.
    vision.to(device="cpu")
    assert not vision.conditioning_rope_cache_is_pristine()
    _public_forward(model, data, runtime)
    assert runtime._cache_guard_disabled and runtime._step["encoding_calls"] == 1


def test_prefetch_spec_matches_effective_forward_key_without_rng_or_encoding(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    model.pipe.check_resize_height_width = lambda height, width: (height, width)
    before = torch.get_rng_state().clone()
    spec = model.conditioning_prefetch_spec(data)
    assert spec == model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True)
    assert torch.equal(before, torch.get_rng_state())
    assert model.pipe.prefix_calls == 0
    data["prompt"] = "different caption"
    assert model.conditioning_prefetch_spec(data).dependencies != spec.dependencies


def test_lora_fsdp_conditioning_boundaries_follow_online_and_hit_module_calls(tiny_conditioning_model, monkeypatch):
    """Exercise the real wrap collector and hooks used by fully_shard.

    CPU hooks stand in for device all-gathers: an online miss invokes vision
    and text embedding, a warm hit still runs generation but invokes neither,
    and neither SFT path ever invokes the separately owned vocabulary head.
    """
    from relax.backends.fsdp.runtime import fsdp2_wrap, resolve_fine_grained_modules
    from relax.models.sensenova.adapter import SenseNovaSFTAdapter

    model, data = tiny_conditioning_model
    dit = model.pipe.dit
    dit.language_model.lm_head = torch.nn.Linear(32, 8, bias=False).requires_grad_(False)
    SenseNovaSFTAdapter().prepare_for_sharding(model)
    # Keep this CPU fixture in FP32; the marker collection does not depend on
    # floating-point dtype. GPU tests exercise the actual BF16 FSDP policy.
    model.pipe.to(dtype=torch.float32)
    conditioning_modules = (dit.vision_model, dit.language_model.model.embed_tokens, dit.language_model.lm_head)
    assert set(resolve_fine_grained_modules(model)) == set(conditioning_modules)
    wrapped, calls, handles = [], [], []

    def shard(module, **kwargs):
        wrapped.append(module)
        handles.append(module.register_forward_pre_hook(lambda current, args: calls.append(current)))

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", shard)
    fsdp2_wrap(model)
    assert wrapped[:3] == list(conditioning_modules)
    assert wrapped[-1] is model
    assert len(wrapped) == len(dit.language_model.model.layers) + 4
    model.conditioning_cache = _MemoryResolver()
    try:
        model(data)  # Cold miss populates the cache through the online modules.
        assert calls.count(dit.vision_model) == calls.count(dit.language_model.model.embed_tokens) == 1
        assert dit.language_model.lm_head not in calls
        calls.clear()
        model(data).backward()
        assert model in calls
        assert not any(module in calls for module in conditioning_modules)
        assert all(layer in calls for layer in dit.language_model.model.layers)
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        calls.clear()
        model.conditioning_cache.enabled = False
        model(data)
        assert calls.count(dit.vision_model) == calls.count(dit.language_model.model.embed_tokens) == 1
        assert dit.language_model.lm_head not in calls
    finally:
        for handle in handles:
            handle.remove()


def test_actual_prefix_trainability_controls_reuse(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    shared = {"height": 32, "width": 32}
    assert model.conditioning_is_frozen()
    spec = model.conditioning_spec(data, shared, enabled=True)
    generation = model.pipe.dit.language_model.model.layers[0].self_attn.q_proj_mot_gen.weight
    with torch.no_grad():
        generation.add_(1)
    assert model.conditioning_spec(data, shared, enabled=True) == spec
    model.training_mode = "full"
    assert model.conditioning_is_frozen()  # The mode name alone is not the boundary.
    prefix = model.pipe.dit.language_model.model.layers[0].self_attn.q_proj.weight
    prefix.requires_grad_(True)
    assert not model.conditioning_spec(data, shared, enabled=True).cacheable
    prefix.requires_grad_(False)
    model.pipe.dit.vision_model.weight.requires_grad_(True)
    assert not model.conditioning_is_frozen()


def test_conditioning_state_tracks_prefix_mutation_but_excludes_generation(tiny_conditioning_model):
    model, _ = tiny_conditioning_model
    original = model.conditioning_state()
    generation = model.pipe.dit.language_model.model.layers[0].self_attn.q_proj_mot_gen.weight
    with torch.no_grad():
        generation.add_(1)
    assert model.conditioning_state() == original
    with torch.no_grad():
        model.pipe.dit.vision_model.weight.add_(1)
    changed = model.conditioning_state()
    assert changed["frozen"]
    assert changed["parameters"]["vision.weight"]["version"] > original["parameters"]["vision.weight"]["version"]
    assert (
        changed["parameters"]["vision.weight"]["local_version"]
        > original["parameters"]["vision.weight"]["local_version"]
    )


@pytest.mark.parametrize("cache_enabled", [None, False, True])
def test_lora_mode_unfrozen_prefix_retains_conditioning_gradients(tiny_conditioning_model, cache_enabled):
    model, data = tiny_conditioning_model
    model.pipe.dit.vision_model.weight.requires_grad_(True)
    prefix = model.pipe.dit.language_model.model.layers[0].self_attn.k_proj.weight
    prefix.requires_grad_(True)
    if cache_enabled is not None:
        model.conditioning_cache = _MemoryResolver(enabled=cache_enabled)
    model(data).backward()
    assert model.training_mode == "lora"
    assert model.pipe.dit.vision_model.weight.grad is not None
    assert torch.count_nonzero(model.pipe.dit.vision_model.weight.grad) > 0
    assert prefix.grad is not None
    assert torch.count_nonzero(prefix.grad) > 0
    if cache_enabled is not None:
        assert not model.conditioning_cache.entries


def test_conditioning_dependencies_use_effective_contents_and_order(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    shared = {"height": 32, "width": 32}

    def dependencies():
        return model.conditioning_spec(data, shared, enabled=True).dependencies

    original = dependencies()
    data["edit_image"][0].putpixel((0, 0), (1, 2, 3))
    assert dependencies() != original
    data["edit_image"].append(data["image"].copy())
    ordered = dependencies()
    data["edit_image"].reverse()
    assert dependencies() != ordered
    original = dependencies()
    data["prompt"] = "a different caption"
    assert dependencies() != original
    original = dependencies()
    shared["width"] = 64
    assert dependencies() != original
    original = dependencies()
    model.pipe.torch_dtype = torch.bfloat16
    assert dependencies() != original
    data["prompt"] = "edit"
    original = dependencies()
    model.pipe.tokenizer.added["unused"] = 2
    assert dependencies() == original
    model.pipe.tokenizer.vocabulary["edit"] = 3
    assert dependencies() != original
    original = dependencies()
    model.pipe.tokenizer.normalizer = "changed"
    assert dependencies() != original
    original = dependencies()
    model.pipe.dit.patch_size = 8
    assert dependencies() != original


def test_preprocessing_normalization_changes_invalidate_effective_input_key(tiny_conditioning_model, monkeypatch):
    from relax.models.sensenova import processing

    model, data = tiny_conditioning_model
    original = model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True)
    monkeypatch.setattr(processing, "IMAGENET_MEAN", (0.4, 0.5, 0.6))
    assert model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True) != original


def test_online_profile_skips_cache_key_hashing(tiny_conditioning_model, monkeypatch):
    model, data = tiny_conditioning_model

    def forbidden(*args, **kwargs):
        raise AssertionError("Online baseline must not pay cache-key hashing costs")

    monkeypatch.setattr(data["edit_image"][0], "tobytes", forbidden)
    monkeypatch.setattr(model.pipe.tokenizer, "get_vocab", forbidden)
    assert model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=False).dependencies == {}


def test_effective_tokenizer_inputs_cover_mutations_without_full_vocabulary_scan(tiny_conditioning_model, monkeypatch):
    model, data = tiny_conditioning_model
    tokenizer = model.pipe.tokenizer

    def forbidden(*args, **kwargs):
        raise AssertionError("Effective-input key must not scan full tokenizer state")

    class Backend:
        to_str = forbidden

    tokenizer.backend_tokenizer = Backend()
    monkeypatch.setattr(tokenizer, "get_vocab", forbidden)
    monkeypatch.setattr(tokenizer, "get_added_vocab", forbidden)

    def spec():
        return model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True)

    original = spec()
    assert original.name == "sensenova.prefix.v3"
    tokenizer.added["unused"] = 3
    tokenizer.init_kwargs["nested"] = {"rules": ["unused"]}
    assert spec() == original  # Metadata unused by this input cannot affect its prefix.
    tokenizer.init_kwargs["nested"]["rules"].append("also unused")
    assert spec() == original
    tokenizer.vocabulary["edit"] = 2
    changed = spec()
    assert changed != original
    tokenizer.normalizer = "changed"
    normalized = spec()
    assert normalized != changed
    tokenizer.bpe_ranks[("e", "d")] = 1
    assert spec() != normalized
    original = spec()
    tokenizer.special_ids["<img>"] = 5
    assert spec() != original


def test_unchanged_effective_tokenizer_inputs_preserve_actual_frozen_prefix(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    model.conditioning_cache = _MemoryResolver(enabled=False)
    features = []
    original_export = model._export_conditioning

    def capture(positive, *, detach):
        result = original_export(positive, detach=detach)
        features.append({name: value.clone() for name, value in result.items()})
        return result

    model._export_conditioning = capture
    model(data)
    before = model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True)
    model.pipe.tokenizer.vocabulary["unrelated"] = 4
    model.pipe.tokenizer.init_kwargs["normalizer"] = {"unused": [1, 2, 3]}
    model(data)
    assert model.conditioning_spec(data, {"height": 32, "width": 32}, enabled=True) == before
    assert all(torch.equal(features[0][name], features[1][name]) for name in features[0])


def test_prepared_forward_tokenizes_once_and_force_online_uses_legacy(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    model.conditioning_cache = _MemoryResolver()
    model(data)
    assert model.pipe.tokenizer.calls == 1
    assert model.pipe.prefix_prepared == [True]
    model.conditioning_cache.force_online = True
    model(data)
    assert model.pipe.tokenizer.calls == 3  # One key preparation, one independent legacy reference.
    assert model.pipe.prefix_prepared == [True, False]


def test_synthetic_random_crop_and_caption_dropout_keep_effective_keys_and_rng(tiny_conditioning_model):
    """Exercise externally prepared augmentations; the real adapter uses center
    crops."""
    from PIL import Image

    model, data = tiny_conditioning_model
    online, cached = copy.deepcopy(model), copy.deepcopy(model)
    cached.conditioning_cache = _MemoryResolver()
    source = Image.new("RGB", (33, 32))
    source.putdata([(x * 7, y * 7, (x + y) * 3) for y in range(32) for x in range(33)])

    def prepared(offset, drop):
        return {
            **data,
            "edit_image": [source.crop((offset, 0, offset + 32, 32))],
            "prompt": "" if drop else data["prompt"],
        }

    variants = [prepared(offset, drop) for offset in (0, 1) for drop in (False, True)]
    keys = [
        json.dumps(
            model.conditioning_spec(sample, {"height": 32, "width": 32}, enabled=True).dependencies, sort_keys=True
        )
        for sample in variants
    ]
    assert len(set(keys)) == 4
    with torch.no_grad():
        for sample in variants:
            cached(sample)
    cached.pipe.prefix_calls = 0

    def run(current):
        torch.manual_seed(41)
        losses, effective_inputs = [], []
        with torch.no_grad():
            for _ in range(16):
                offset, drop = int(torch.randint(0, 2, ())), bool(torch.rand(()) < 0.5)
                sample = prepared(offset, drop)
                effective_inputs.append((offset, sample["prompt"]))
                losses.append(current(sample))
        return torch.stack(losses), effective_inputs, torch.get_rng_state().clone()

    expected, expected_inputs, expected_rng = run(online)
    actual, actual_inputs, actual_rng = run(cached)
    assert actual_inputs == expected_inputs
    assert torch.equal(actual, expected)
    assert torch.equal(actual_rng, expected_rng)
    assert online.pipe.prefix_calls == 16 and cached.pipe.prefix_calls == 0


def _update(model, data):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    losses = []
    for _ in range(2):
        loss = model(data)
        losses.append(loss.detach())
        (loss / 2).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    updates = {name: parameter.detach() - before[name] for name, parameter in model.named_parameters()}
    return torch.stack(losses), gradients, updates


@pytest.mark.parametrize("reference", ["legacy", "prepared", "forced_online"])
def test_warm_reuse_preserves_rng_batch_gradients_and_fp32_updates(tiny_conditioning_model, reference):
    model, data = tiny_conditioning_model
    online, cached = copy.deepcopy(model), copy.deepcopy(model)
    if reference != "legacy":
        online.conditioning_cache = _MemoryResolver(enabled=reference == "forced_online")
        online.conditioning_cache.force_online = reference == "forced_online"
    cached.conditioning_cache = _MemoryResolver()
    cached(data)  # Build outside the fixed numerical comparison, without updates.
    snapshot = {name: value.clone() for name, value in next(iter(cached.conditioning_cache.entries.values())).items()}
    cached.pipe.random_events.clear()
    cached.pipe.prefix_calls = 0
    torch.manual_seed(42)
    expected = _update(online, data)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    actual = _update(cached, data)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(online.pipe.random_events) == len(cached.pipe.random_events) == 6
    assert all(torch.equal(x, y) for x, y in zip(online.pipe.random_events, cached.pipe.random_events, strict=True))
    assert torch.equal(actual[0], expected[0])
    for actual_values, expected_values in zip(actual[1:], expected[1:], strict=True):
        assert actual_values.keys() == expected_values.keys()
        for name in expected_values:
            assert actual_values[name].dtype == torch.float32
            assert torch.equal(actual_values[name], expected_values[name]), name
    assert online.pipe.prefix_calls == 2
    assert cached.pipe.prefix_calls == 0
    stored = next(iter(cached.conditioning_cache.entries.values()))
    assert all(torch.equal(stored[name], value) for name, value in snapshot.items())


def test_uncacheable_forward_retains_full_conditioning_gradients(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    model.training_mode = "full"
    model.pipe.requires_grad_(True)
    online, resolved = copy.deepcopy(model), copy.deepcopy(model)
    resolved.conditioning_cache = _MemoryResolver()
    torch.manual_seed(13)
    expected = _update(online, data)
    torch.manual_seed(13)
    actual = _update(resolved, data)
    assert not resolved.conditioning_cache.entries
    assert "pipe.dit.vision_model.weight" in actual[1]
    for actual_values, expected_values in zip(actual[1:], expected[1:], strict=True):
        assert actual_values.keys() == expected_values.keys()
        for name in expected_values:
            assert torch.equal(actual_values[name], expected_values[name]), name


@pytest.mark.parametrize("damage", ["missing", "dtype", "indexes", "shape"])
def test_restore_rejects_incompatible_payload(tiny_conditioning_model, damage):
    model, data = tiny_conditioning_model
    model.conditioning_cache = _MemoryResolver()
    model(data)
    features = dict(next(iter(model.conditioning_cache.entries.values())))
    if damage == "missing":
        features.pop("kv.0.key")
    elif damage == "dtype":
        features["kv.0.key"] = features["kv.0.key"].to(torch.bfloat16)
    elif damage == "indexes":
        features["indexes_image"] = torch.zeros(3, 2, dtype=torch.long)
    else:
        features["kv.0.key"] = features["kv.0.key"][:, :, :0, :]
    with pytest.raises(ValueError, match="Cached SenseNova"):
        model._restore_conditioning(features, 32, 32)


def test_restore_creates_fresh_wrapper_without_tensor_clones(tiny_conditioning_model):
    model, data = tiny_conditioning_model
    model.conditioning_cache = _MemoryResolver()
    model(data)
    features = next(iter(model.conditioning_cache.entries.values()))
    left = model._restore_conditioning(features, 32, 32)
    right = model._restore_conditioning(features, 32, 32)
    assert left["past_key_values"] is not right["past_key_values"]
    assert left["past_key_values"].is_initialized
    assert left["past_key_values"].get_seq_length() == features["kv.0.key"].shape[2]
    assert left["past_key_values"].layers[0].keys.data_ptr() == features["kv.0.key"].data_ptr()
    assert right["indexes_image"].data_ptr() == features["indexes_image"].data_ptr()


@pytest.mark.parametrize("size", [(32, 32), (65, 513), (513, 65), (1100, 800)])
def test_size_only_reference_geometry_matches_native_pixel_preprocessing(size):
    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova.processing import image_resize_geometry, load_image_native

    image = Image.new("RGB", size, (11, 22, 33))
    height, width = image_resize_geometry(*size, min_pixels=256**2, max_pixels=512**2)
    pixels, grid = load_image_native(image, min_pixels=256**2, max_pixels=512**2)
    assert grid.tolist() == [[height // 16, width // 16]]
    assert pixels.shape == ((height // 16) * (width // 16), 3 * 16**2)


def test_real_prompt_units_prepared_inputs_equal_independent_legacy_inputs(monkeypatch):
    """Use production pixel/query/prompt units and a real tiny frozen decoder.

    The reference constructs the pre-v2 conversation explicitly, tokenizes
    separately, and derives positions from native pixel preprocessing. It does
    not consume the prepared helper's query, IDs, indexes or geometry.
    """
    import re
    from types import SimpleNamespace

    pytest.importorskip("torchvision")
    from PIL import Image

    from relax.models.sensenova.adapter import SenseNovaSFTModel, configure_attention
    from relax.models.sensenova.model import SenseNovaU1Model
    from relax.models.sensenova.pipeline import (
        SenseNovaU1ImageUnit_EditImageEmbedder as Edit,
    )
    from relax.models.sensenova.pipeline import (
        SenseNovaU1ImageUnit_PromptEmbedder as Prompt,
    )
    from relax.models.sensenova.processing import (
        IMG_START_TOKEN,
        NON_THINK_PREFIX,
        SYSTEM_MESSAGE_FOR_GEN,
        build_conversation_prompt,
        build_image_token_block,
        build_thw_indexes,
        create_block_causal_mask,
        load_image_native,
    )

    configure_attention("torch")

    class Tokenizer(_Tokenizer):
        def __call__(self, text, return_tensors=None, **kwargs):
            self.calls += 1
            ids = []
            for part in re.split(r"(<IMG_CONTEXT>|<img>|</img>)", text):
                if part:
                    ids.append({"<img>": 6, "<IMG_CONTEXT>": 7, "</img>": 5}.get(part, 0))
            return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_size, self.downsample_ratio, self.merge_size = 16, 0.5, 2
            self.vision_model = torch.nn.Linear(3, 16, bias=False)
            self.language_model = torch.nn.Module()
            self.language_model.model = SenseNovaU1Model(
                vocab_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                pad_token_id=0,
            )
            self.received = []

        def embed_tokens(self, ids):
            return self.language_model.model.embed_tokens(ids)

        def extract_und_feature(self, pixels, grids):
            features, offset = [], 0
            for height, width in grids.tolist():
                count = height * width
                values = pixels[offset : offset + count].reshape(height, width, 3, 16**2).mean(-1)
                values = values.reshape(height // 2, 2, width // 2, 2, 3).mean((1, 3))
                features.append(self.vision_model(values.reshape(-1, 3)))
                offset += count
            return torch.cat(features)

        def encode_prefix(self, **kwargs):
            self.received.append(
                {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in kwargs.items()}
            )
            hidden, cache = self.language_model.model(
                input_ids=kwargs["input_ids"],
                inputs_embeds=kwargs["inputs_embeds"],
                indexes=kwargs["indexes"],
                attention_mask=kwargs["attention_mask"],
                use_cache=True,
                update_cache=True,
            )
            return cache, hidden

    pipe = SimpleNamespace(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        tokenizer=Tokenizer(),
        dit=DiT().eval().requires_grad_(False),
        scheduler=SimpleNamespace(training=True),
        load_models_to_device=lambda _: None,
    )
    refs = [Image.new("RGB", (64, 96), (11, 22, 33)), Image.new("RGB", (96, 64), (34, 45, 56))]
    # Preparing a key cannot resize or normalize pixels.
    with monkeypatch.context() as patch:
        patch.setattr(Image.Image, "resize", lambda *a, **kw: pytest.fail("Preparation processed pixels"))
        rng = torch.get_rng_state().clone()
        prepared = Prompt.prepare_sft_inputs(pipe, "edit", refs)
        assert torch.equal(torch.get_rng_state(), rng)
    native = [load_image_native(image, min_pixels=512**2, max_pixels=2048**2) for image in refs]
    legacy_pixels, legacy_grid = torch.cat([x[0] for x in native]), torch.cat([x[1] for x in native])
    legacy_prompt = "Image-1:<image>\nImage-2:<image>\nedit"
    legacy_query = build_conversation_prompt(
        legacy_prompt,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text=NON_THINK_PREFIX + IMG_START_TOKEN,
    )
    for grid in legacy_grid:
        legacy_query = legacy_query.replace("<image>", build_image_token_block(int(grid.prod() * 0.25)), 1)
    legacy_ids = pipe.tokenizer(legacy_query, return_tensors="pt")["input_ids"]
    legacy_indexes = build_thw_indexes(legacy_ids[0], 6, 7, legacy_grid, merge_size=2)
    assert prepared["query"] == legacy_query
    assert torch.equal(prepared["input_ids"], legacy_ids)
    assert torch.equal(prepared["indexes"], legacy_indexes)
    assert torch.equal(prepared["grid_hw"], legacy_grid)
    assert prepared["special_token_ids"] == {"image_start": 6, "image_context": 7}
    effective = Edit().process(pipe, refs, prepared)
    assert torch.equal(effective["edit_pixel_values"], legacy_pixels)
    assert torch.equal(effective["edit_grid_hw"], legacy_grid)

    def encode(preparation):
        return Prompt().process(
            pipe,
            "edit",
            True,
            32,
            32,
            effective["edit_pixel_values"],
            effective["edit_grid_hw"],
            False,
            False,
            False,
            prepared_prefix=preparation,
        )

    with torch.no_grad():
        legacy = encode(None)
        calls = pipe.tokenizer.calls
        # Even a later tokenizer edit cannot make compute disagree with the
        # already-keyed context selection and token/index tensors.
        pipe.tokenizer.special_ids["<IMG_CONTEXT>"] = 4
        current = encode(prepared)
    assert pipe.tokenizer.calls == calls  # The cache miss consumes the keyed IDs, without a second tokenization.
    expected, actual = pipe.dit.received
    assert torch.equal(expected["indexes"], legacy_indexes)
    assert torch.equal(expected["attention_mask"], create_block_causal_mask(legacy_indexes[0]))
    for name in ("inputs_embeds", "indexes", "attention_mask"):
        assert torch.equal(actual[name], expected[name]), name
    assert expected["input_ids"] is actual["input_ids"] is None  # Vision replacement embeds were passed.
    left = SenseNovaSFTModel._export_conditioning(legacy, detach=True)
    right = SenseNovaSFTModel._export_conditioning(current, detach=True)
    assert all(torch.equal(left[name], right[name]) for name in left)
