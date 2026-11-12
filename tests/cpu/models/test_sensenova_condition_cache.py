# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer

from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.engine.sft.condition_cache import condition_key
from relax.engine.sft.condition_tracker import ConditionTracker, summarize_baseline_dir
from relax.models.sensenova.adapter import SenseNovaSFTAdapter, SenseNovaSFTModel, configure_attention
from relax.models.sensenova.model import SenseNovaU1DiT
from relax.models.sensenova.pipeline import SenseNovaU1ImagePipeline
from relax.models.sensenova.processing import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN


def build_model(tmp_path: Path) -> SenseNovaSFTModel:
    configure_attention("torch")
    tokens = sorted(ByteLevel.alphabet()) + [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
    ]
    vocab = dict(zip(tokens, range(len(tokens)), strict=True))
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[])
    tokenizer.add_special_tokens({"additional_special_tokens": tokens[-5:]})
    pipe = SenseNovaU1ImagePipeline(device="cpu", torch_dtype=torch.float32)
    pipe.tokenizer = tokenizer
    pipe.dit = SenseNovaU1DiT(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        pad_token_id=0,
        vision_hidden_size=16,
        max_position_embeddings_vision=2048,
    )
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.freeze_except([])
    for name, parameter in pipe.dit.named_parameters():
        if "_mot_gen" in name:
            parameter.requires_grad_(True)
    root = SenseNovaSFTModel(pipe)
    root._condition_config = {"max_pixels": 1048576}
    root._condition_model_path = str(tmp_path)
    return root


@pytest.fixture
def model(tmp_path) -> SenseNovaSFTModel:
    return build_model(tmp_path)


def sample() -> dict:
    return {
        "prompt": "edit",
        "image": Image.new("RGB", (32, 32), "blue"),
        "edit_image": [Image.new("RGB", (32, 32), "red")],
    }


def test_condition_input_identity_and_managed_tokenizer_update(model: SenseNovaSFTModel) -> None:
    adapter = SenseNovaSFTAdapter()
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})
    data = sample()

    def identity(value: dict) -> str:
        return condition_key(cache.specs[0], adapter.prepare_condition_request(model, value)["prefix_kv"].metadata, 0)

    original = identity(data)
    assert identity(copy.deepcopy(data)) == original
    assert identity({**data, "prompt": "another"}) != original
    assert identity({**data, "image": Image.new("RGB", (64, 32))}) != original
    data["edit_image"][0].putpixel((16, 16), (0, 0, 0))
    assert identity(data) != original
    data = sample()
    data["edit_image"].append(Image.new("RGB", (32, 32), "green"))
    ordered = identity(data)
    data["edit_image"].reverse()
    assert identity(data) != ordered
    previous = model._condition_text_identity
    cache.bump_encoder_version()
    model.pipe.tokenizer.add_tokens(["new-token"])
    cache.prepare(sample())
    assert model._condition_text_identity != previous
    cache.end_micro()
    for name in (
        "random_crop",
        "random_flip",
        "random_augment",
        "caption_dropout_prob",
        "reasoning",
        "is_noisy_input",
    ):
        assert not adapter.prepare_condition_request(model, {**sample(), name: True})["prefix_kv"].eligible
    cache.close()


def test_reused_prefix_shares_cached_tensors_and_detects_inplace_writes(model: SenseNovaSFTModel) -> None:
    cache = ConditionCacheBinding(model, SenseNovaSFTAdapter(), torch.device("cpu"), {})
    data = sample()
    consumed = []
    model_fn = model.pipe.model_fn

    def consume(**kwargs: object) -> torch.Tensor:
        condition = cache.values.get("prefix_kv")
        if condition is not None:
            layers = kwargs["past_key_values"].layers
            assert len(layers) == len(condition["past_key_values"])
            for layer, (key, value) in zip(layers, condition["past_key_values"], strict=True):
                assert layer.keys is key and layer.values is value
            assert kwargs["indexes_image"] is condition["indexes_image"]
            consumed.append(kwargs["past_key_values"])
        return model_fn(**kwargs)

    model.pipe.model_fn = consume
    for _ in range(2):
        cache.decide([cache.prepare(data)])
        model(data)
        cache.end_micro()
    assert len(consumed) == 1
    with torch.no_grad():
        consumed[0].layers[0].keys.add_(1)
    with pytest.raises(ValueError, match="metadata changed"):
        cache.prepare(data)
    assert cache.stats()["entries"] == 0
    cache.end_micro()
    cache.close()


def test_cache_preserves_loss_complete_gradients_updates_and_rng(model: SenseNovaSFTModel) -> None:
    initial = copy.deepcopy(model)
    data = sample()
    results = []
    for enabled in (False, True):
        current = copy.deepcopy(initial)
        torch.manual_seed(192)
        before_bind_rng = torch.get_rng_state().clone()
        cache = ConditionCacheBinding(current, SenseNovaSFTAdapter(), torch.device("cpu"), {}) if enabled else None
        assert torch.equal(before_bind_rng, torch.get_rng_state())
        tracker = ConditionTracker(0, 1, 2, 2, Path(model._condition_model_path) / str(enabled), warmup_steps=0)
        current.pipe.tracker = current.pipe.dit.tracker = tracker
        parameters = [parameter for parameter in current.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        history = []
        for step in range(3):
            tracker.begin_step(step, torch.device("cpu"))
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros(())
            before = [parameter.detach().clone() for parameter in parameters]
            for micro in range(2):
                tracker.begin_micro(
                    micro,
                    0,
                    {"prompt": data["prompt"], "image": "target", "edit_image": ["reference"]},
                    model._condition_config,
                    "lora",
                    32,
                    32,
                )
                if cache is not None:
                    state = cache.prepare(data)
                    cache.decide([state])
                loss = current(data)
                (loss / 2).backward()
                loss_sum += loss.detach() / 2
                del loss
                if cache is not None:
                    cache.end_micro()
            gradients = [parameter.grad.clone() for parameter in parameters]
            optimizer.step()
            updates = [parameter.detach() - previous for parameter, previous in zip(parameters, before, strict=True)]
            history.append((loss_sum, gradients, updates, torch.get_rng_state().clone()))
            if cache is not None:
                tracker.condition_cache_stats = cache.stats()
            tracker.end_step(step, 0, 1)
        summary = tracker.save_summary()
        assert summary["encoding_stats"]["prefix_encode"]["calls"] == (1 if enabled else 6)
        assert summary["encoding_stats"]["ref_vision_encode"]["calls"] == (1 if enabled else 6)
        assert summary["encoding_stats"]["gen_vision_encode"]["calls"] == 6
        assert summary["transfer_stats"]["prefix_input_ids"]["total_transfer_calls"] == (1 if enabled else 6)
        if cache is not None:
            assert cache.stats()["effective_reuses"] == 5
            assert cache.generations["prefix_kv"] == 0
            assert cache.stats()["charged_bytes"] <= cache.cache.max_bytes
            aggregate = summarize_baseline_dir(tracker.output_dir)
            assert aggregate["condition_cache"]["0"]["effective_reuses"] == 5
            assert aggregate["condition_cache"]["0"]["local_hits"] == 5
            cache.close()
        results.append(history)
    for online, cached in zip(*results, strict=True):
        assert torch.equal(online[0], cached[0])
        assert torch.equal(online[3], cached[3])
        for reference, actual in zip(online[1] + online[2], cached[1] + cached[2], strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
