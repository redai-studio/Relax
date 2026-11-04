# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU correctness/lifetime tests; no model downloads or CUDA are required."""

import copy
from dataclasses import replace
from datetime import timedelta

import pytest
import torch
from PIL import Image

from relax.models.condition_cache import (
    ConditionFeatureCache,
    FrozenConditionSpec,
    all_ranks_hit,
    condition_cache_key,
)
from relax.models.sensenova import adapter as sense


SPEC = FrozenConditionSpec("weights-sha", "resize-v1", "torch.float32", "kv-v1", "adapter-v1")


def test_content_key_covers_types_order_dtype_and_every_dependency():
    content = {"prompt": "edit", "images": [b"pixels-a", b"pixels-b"], "size": (32, 64)}
    key = condition_cache_key(SPEC, content)
    assert key == condition_cache_key(SPEC, dict(reversed(list(content.items()))))
    for field in SPEC.__dataclass_fields__:
        assert key != condition_cache_key(replace(SPEC, **{field: "changed"}), content)
    for changed in (
        {**content, "prompt": "new caption after dropout"},
        {**content, "images": list(reversed(content["images"]))},
        {**content, "images": [b"different crop", b"pixels-b"]},
        {**content, "size": (64, 32)},
    ):
        assert key != condition_cache_key(SPEC, changed)
    assert condition_cache_key(SPEC, torch.ones(2)) != condition_cache_key(SPEC, torch.ones(2).double())
    assert condition_cache_key(SPEC, [1, 2]) != condition_cache_key(SPEC, (1, 2))
    assert condition_cache_key(SPEC, True) != condition_cache_key(SPEC, 1)
    with pytest.raises(ValueError, match="Non-finite"):
        condition_cache_key(SPEC, float("nan"))
    with pytest.raises(TypeError, match="Unsupported"):
        condition_cache_key(SPEC, object())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool])
def test_tensor_tree_roundtrip_is_exact_private_and_preserves_dtype(dtype):
    original = torch.tensor([[0, 1], [1, 0]], dtype=dtype).t()
    payload = {"kv": [(original, original.clone())], "metadata": (3, "tag", None)}
    cache = ConditionFeatureCache(100_000)
    assert cache.put("a", payload)
    original.zero_()
    result = cache.get("a")
    assert result["kv"][0][0].dtype == dtype
    assert torch.equal(result["kv"][0][0], torch.tensor([[0, 1], [1, 0]], dtype=dtype).t())
    result["kv"][0][0].zero_()
    result["kv"].append((torch.ones(1), torch.ones(1)))
    second = cache.get("a")
    assert len(second["kv"]) == 1
    assert second["kv"][0][0].count_nonzero() == 2
    assert cache.stats()["read_tensor_bytes"] == original.numel() * original.element_size() * 4


def test_lru_byte_entry_bounds_oversize_and_replacement():
    payload = {"x": torch.arange(16)}
    sizing = ConditionFeatureCache(100_000)
    assert sizing.put("a", payload)
    one_size = sizing.stats()["bytes"]
    cache = ConditionFeatureCache(one_size * 2, max_entries=2)
    assert cache.put("a", payload) and cache.put("b", payload)
    assert cache.get("a") is not None
    assert cache.put("c", payload)
    assert cache.keys() == ("a", "c")
    assert cache.get("b") is None
    assert cache.stats()["bytes"] <= cache.max_bytes
    assert cache.put("a", payload)
    assert len(cache.keys()) == 2
    assert not cache.put("huge", torch.zeros(100_000))
    assert len(cache.keys()) == 2
    assert cache.stats()["oversized"] == 1
    cache.clear()
    assert cache.stats()["bytes"] == 0 and not cache.keys()
    assert cache.put("a", payload)
    cache.close()
    cache.close()
    assert cache.stats()["closed"] and cache.stats()["bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("a")


def test_frozen_features_only_and_corrupted_entry_falls_back():
    cache = ConditionFeatureCache(100_000)
    with pytest.raises(ValueError, match="detached"):
        cache.put("unsafe", torch.ones(1, requires_grad=True))
    assert cache.put("safe", {"x": torch.ones(2)})
    # Simulate memory corruption without any public alias to the stored tree.
    cache._entries["safe"].payload["x"].add_(1)
    assert not cache.contains("safe")
    assert cache.get("safe") is None
    assert cache.stats()["corruptions"] == 1
    assert cache.stats()["bytes"] == 0


def test_inference_mode_payloads_are_owned_and_readable():
    cache = ConditionFeatureCache(10_000)
    with torch.inference_mode():
        source = torch.arange(4)
        assert cache.put("inference", {"x": source})
        assert cache.contains("inference")
        assert torch.equal(cache.get("inference")["x"], source)
    assert torch.equal(cache.get("inference")["x"], source)


def test_consensus_requires_all_ranks_and_an_explicit_group(monkeypatch):
    import torch.distributed as dist

    assert all_ranks_hit(True)
    assert not all_ranks_hit(False)
    group = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda supplied: 2 if supplied is group else 1)
    monkeypatch.setattr(dist, "get_backend", lambda supplied: "gloo")
    calls = []

    def reduce_miss(flag, op, group):
        calls.append((op, group))
        flag.zero_()

    monkeypatch.setattr(dist, "all_reduce", reduce_miss)
    assert not all_ranks_hit(True, group)
    assert calls == [(dist.ReduceOp.MIN, group)]


class _PrefixUnit:
    frozen_condition_schema = "toy-prefix-v1"

    @staticmethod
    def pack_frozen_conditions(outputs):
        return {"prefix": outputs["prefix"]}

    @staticmethod
    def unpack_frozen_conditions(payload):
        return payload


class _ToyPipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.dit = torch.nn.Module()
        self.dit.understanding = torch.nn.Linear(3, 3, bias=False).requires_grad_(False)
        self.dit.fm_modules = torch.nn.Linear(3, 3, bias=False)
        self.units = [_PrefixUnit(), "noise"]
        self.calls = 0

    def unit_runner(self, unit, pipe, shared, positive, negative):
        if isinstance(unit, _PrefixUnit):
            self.calls += 1
            pixel = shared["edit_image"][0].getpixel((0, 0))
            features = torch.tensor(pixel, dtype=torch.float32) / 255 + len(positive["prompt"]) / 10
            result = {"prefix": self.dit.understanding(features)}
            return shared, {**positive, **result}, {**negative, **result}
        # Mirrors an existing stochastic unit that must NOT be skipped on hit.
        return {**shared, "noise_unit": torch.randn(3)}, positive, negative


def _toy_loss(pipe, inputs):
    timestep = torch.randint(5, ())
    noise = torch.randn(3)
    predicted = pipe.dit.fm_modules(inputs["prefix"] + inputs["noise_unit"] + noise + timestep)
    return (predicted - 0.25).square().mean()


def _data(prompt="edit", color=(20, 40, 60), size=(32, 32)):
    return {"image": Image.new("RGB", size), "edit_image": [Image.new("RGB", size, color)], "prompt": prompt}


def _toy_model(cached=True):
    return sense.SenseNovaSFTModel(
        _ToyPipe(),
        condition_cache=ConditionFeatureCache(100_000) if cached else None,
        condition_identity="verified-assets" if cached else None,
    )


def test_cached_online_full_batch_loss_grad_delta_and_rng_are_exact(monkeypatch):
    monkeypatch.setattr(sense, "flow_matching_loss", _toy_loss)
    torch.manual_seed(14)
    base = _toy_model(False)
    initial = copy.deepcopy(base.state_dict())
    rng = torch.get_rng_state().clone()
    data = [_data(), _data("caption"), _data(), _data("caption")]

    def update(cached, warm):
        model = _toy_model(cached)
        model.load_state_dict(initial)
        if warm:
            for sample in data:
                model(sample)
        torch.set_rng_state(rng)
        optimizer = torch.optim.AdamW(model.pipe.dit.fm_modules.parameters(), lr=0.01)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        losses = []
        for sample in data:
            loss = model(sample) / len(data)
            losses.append(loss.detach())
            loss.backward()
        grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
        optimizer.step()
        delta = {name: p.detach() - before[name] for name, p in model.named_parameters() if p.requires_grad}
        return model, torch.stack(losses), grads, delta, torch.get_rng_state()

    reference = update(False, False)
    for warm in (False, True):
        actual = update(True, warm)
        assert torch.equal(reference[1], actual[1])
        for index in (2, 3):
            assert reference[index].keys() == actual[index].keys()
            assert all(torch.equal(value, actual[index][key]) for key, value in reference[index].items())
        assert torch.equal(reference[4], actual[4])
        assert actual[0].condition_cache_stats()["global_hits"] >= 2
        assert actual[0].pipe.calls == 2


def test_effective_pixels_caption_shape_and_weights_invalidate(monkeypatch):
    monkeypatch.setattr(sense, "flow_matching_loss", _toy_loss)
    model = _toy_model()
    model(_data())
    model(_data())
    assert model.pipe.calls == 1
    for sample in (_data("new caption"), _data(color=(21, 40, 60)), _data(size=(32, 64))):
        model(sample)
    assert model.pipe.calls == 4
    with torch.no_grad():
        model.pipe.dit.understanding.weight.add_(1)
    model(_data())
    assert model.pipe.calls == 5
    assert model.condition_cache_stats()["invalidations"] == 1
    model.load_state_dict(model.state_dict())
    assert not model.condition_cache.keys()
    model(_data())
    model.pipe.dit.understanding.requires_grad_(True)
    model(_data())
    assert model.condition_cache_stats()["bypasses"] == 1
    assert not model.condition_cache.keys()


def test_encoder_backend_and_autocast_are_key_dependencies(monkeypatch):
    model = _toy_model()
    data, unit = _data(), _PrefixUnit()
    key = model._prefix_key(data, unit)
    monkeypatch.setenv("RELAX_ATTENTION_IMPLEMENTATION", "torch")
    torch_key = model._prefix_key(data, unit)
    assert key != torch_key
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch_key != model._prefix_key(data, unit)


def test_random_prefix_bypasses_and_exception_releases(monkeypatch):
    monkeypatch.setattr(sense, "flow_matching_loss", _toy_loss)
    model = _toy_model()
    model.pipe.dit.dropout = torch.nn.Dropout(0.1)
    model(_data())
    assert model.condition_cache_stats()["bypasses"] == 1
    model.pipe.dit.dropout.eval()
    model(_data())
    assert model.condition_cache.keys()

    def fail(pipe, inputs):
        raise RuntimeError("cancelled stage")

    monkeypatch.setattr(sense, "flow_matching_loss", fail)
    with pytest.raises(RuntimeError, match="cancelled stage"):
        model(_data())
    assert model.condition_cache.stats()["closed"]
    assert model.condition_cache.stats()["bytes"] == 0


def test_full_training_never_accepts_frozen_prefix_cache():
    with pytest.raises(ValueError, match="LoRA"):
        sense.SenseNovaSFTModel(
            _ToyPipe(), training_mode="full", condition_cache=ConditionFeatureCache(100_000), condition_identity="sha"
        )


def test_external_asset_identity_is_validated_and_fallback_hashes_content(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAX_CONDITION_CACHE_IDENTITY", raising=False)
    (tmp_path / "model-001.safetensors").write_bytes(b"original weights")
    (tmp_path / "tokenizer.json").write_text("{}")
    first = sense._condition_encoder_identity(str(tmp_path))
    (tmp_path / "model-001.safetensors").write_bytes(b"changed weights")
    assert first != sense._condition_encoder_identity(str(tmp_path))
    monkeypatch.setenv("RELAX_CONDITION_CACHE_IDENTITY", "a" * 64)
    assert sense._condition_encoder_identity(str(tmp_path)) == "a" * 64
    monkeypatch.setenv("RELAX_CONDITION_CACHE_IDENTITY", "model-name")
    with pytest.raises(ValueError, match="SHA256"):
        sense._condition_encoder_identity(str(tmp_path))


def test_condition_cache_environment_is_registered_and_typed(monkeypatch):
    from relax.utils.env import Envs, known_env_names, validate_env

    values = {
        "RELAX_CONDITION_CACHE_BYTES": "4096",
        "RELAX_CONDITION_CACHE_DEVICE": "cpu",
        "RELAX_CONDITION_CACHE_IDENTITY": "a" * 64,
    }
    monkeypatch.delenv("RELAX_ATTENTION_IMPLEMENTATION", raising=False)
    monkeypatch.delenv("RELAX_DETERMINISTIC_ATTENTION", raising=False)
    assert set(values) <= known_env_names()
    validate_env({**values, "RELAX_STRICT_ENV": "1"})
    monkeypatch.setenv("RELAX_CONDITION_CACHE_BYTES", "4096")
    assert Envs.RELAX_CONDITION_CACHE_BYTES == 4096
    with pytest.raises(ValueError, match="RELAX_CONDITION_CACHE_BYTES"):
        validate_env({"RELAX_CONDITION_CACHE_BYTES": "typo"})


def test_real_dynamic_cache_is_rebuilt_without_mutable_aliases():
    from transformers.cache_utils import DynamicCache

    from relax.models.sensenova.pipeline import SenseNovaU1ImageUnit_PromptEmbedder

    original = DynamicCache()
    original.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 0)
    outputs = {"past_key_values": original, "indexes_image": torch.arange(6).reshape(3, 2)}
    cache = ConditionFeatureCache(100_000)
    unit = SenseNovaU1ImageUnit_PromptEmbedder()
    cache.put("prefix", unit.pack_frozen_conditions(outputs))
    first = unit.unpack_frozen_conditions(cache.get("prefix"))
    second = unit.unpack_frozen_conditions(cache.get("prefix"))
    assert first["past_key_values"] is not second["past_key_values"]
    assert torch.equal(original.layers[0].keys, first["past_key_values"].layers[0].keys)
    assert torch.equal(outputs["indexes_image"], first["indexes_image"])
    first["past_key_values"].update(torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)
    assert first["past_key_values"].get_seq_length() == 4
    assert second["past_key_values"].get_seq_length() == 3
    assert original.get_seq_length() == 3


def _gloo_cache_worker(rank, rendezvous, output):
    """One rank has an entry the other lacks: both must encode, then both
    hit."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        cache = ConditionFeatureCache(100_000)
        if rank == 0:
            cache.put("sample", {"kv": torch.ones(2)})
        encoded = 0
        decisions = []
        for _ in range(3):
            hit = all_ranks_hit(cache.contains("sample"), dist.group.WORLD)
            decisions.append(hit)
            if not hit:
                # Stands in for a sharded encoder collective. A local-only hit
                # branch here would leave rank 1 hanging in this all_reduce.
                feature = torch.ones(2)
                dist.all_reduce(feature, group=dist.group.WORLD)
                cache.put("sample", {"kv": feature})
                encoded += 1
            else:
                assert torch.equal(cache.get("sample")["kv"], torch.full((2,), 2.0))
        torch.save({"decisions": decisions, "encodes": encoded}, f"{output}-{rank}.pt")
        cache.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_two_real_cpu_ranks_agree_on_partial_hit_without_collective_deadlock(tmp_path):
    import torch.multiprocessing as mp

    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_gloo_cache_worker, args=(rendezvous, str(tmp_path / "result")), nprocs=2, join=True)
    for rank in range(2):
        result = torch.load(tmp_path / f"result-{rank}.pt", weights_only=True)
        assert result == {"decisions": [False, True, True], "encodes": 1}


class _TinyTokenizer:
    def __call__(self, text, **kwargs):
        count = text.count("<IMG_CONTEXT>")
        return {"input_ids": torch.tensor([[7, 1] + [2] * count + [3, 4]])}

    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<img>": 1, "<IMG_CONTEXT>": 2, "<|im_end|>": 4}[token]


def test_native_small_sensenova_units_and_flow_loss_cache_are_exact(monkeypatch):
    """Real native pipeline/DiT math at tiny dimensions, not pretrained GPU
    proof."""
    from relax.models.sensenova.model import SenseNovaU1DiT
    from relax.models.sensenova.pipeline import SenseNovaU1ImagePipeline

    monkeypatch.setenv("RELAX_ATTENTION_IMPLEMENTATION", "torch")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(29)
        pipe = SenseNovaU1ImagePipeline(device="cpu", torch_dtype=torch.float32)
        pipe.dit = SenseNovaU1DiT(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vision_hidden_size=8,
            pad_token_id=0,
        )
        pipe.tokenizer = _TinyTokenizer()
        pipe.scheduler.set_timesteps(10, training=True)
        pipe.freeze_except([])
        # Only a generation-side parameter needs gradients for this small CPU
        # integration check. The actual LoRA/FSDP recipe remains a GPU gate.
        pipe.dit.fm_modules["fm_head"].conv2.weight.requires_grad_(True)
        initial = copy.deepcopy(pipe.state_dict())
        rng = torch.get_rng_state().clone()

        def update(cached):
            local_pipe = copy.deepcopy(pipe)
            local_pipe.load_state_dict(initial)
            model = sense.SenseNovaSFTModel(
                local_pipe,
                condition_cache=ConditionFeatureCache(4 * 1024**2) if cached else None,
                condition_identity="small-native-fixture" if cached else None,
            )
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            before = parameters[0].detach().clone()
            torch.set_rng_state(rng)
            losses = []
            for _ in range(2):
                loss = model(_data()) / 2
                losses.append(loss.detach())
                loss.backward()
            grad = parameters[0].grad.clone()
            optimizer.step()
            result = (torch.stack(losses), grad, parameters[0].detach() - before, torch.get_rng_state())
            stats = model.condition_cache_stats()
            model.close_condition_cache()
            return result, stats

        online, online_stats = update(False)
        cached, cached_stats = update(True)
        assert all(torch.equal(expected, actual) for expected, actual in zip(online, cached))
        assert online_stats["prefix_encodes"] == 2
        assert cached_stats["prefix_encodes"] == 1 and cached_stats["global_hits"] == 1
    finally:
        torch.set_num_threads(previous_threads)
