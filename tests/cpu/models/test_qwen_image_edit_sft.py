# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import validate_i2i_sft_config
from relax.backends.fsdp.condition_cache import ConditionCacheBinding
from relax.engine.sft.condition_cache import condition_key
from relax.engine.sft.condition_tracker import ConditionTracker, summarize_baseline_dir
from relax.models.qwen_image.scheduler import QwenImageTrainingSchedule
from relax.models.qwen_image.sft_adapter import QWEN_IMAGE_EDIT_TRACKER_SCHEMA, QwenImageEditSFTAdapter
from tests.cpu.models.qwen_image_edit_helpers import CONFIG, FIXTURES, build_adapter, build_model, small_sample


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_training_schedule_bf16_lookup_and_weighting() -> None:
    schedule = QwenImageTrainingSchedule()
    assert schedule.timesteps.shape == (1000,)
    quantized = schedule.timesteps.to(torch.bfloat16)
    indices = [schedule.lookup_index(timestep).item() for timestep in quantized]
    assert sum(index != actual for index, actual in enumerate(indices)) == 624
    assert schedule.training_weight(quantized[0]).item() == 0.0
    sample = torch.ones(1, 16, 4, 4)
    noise = torch.zeros_like(sample)
    assert torch.equal(schedule.training_target(sample, noise, quantized[500]), noise - sample)
    noisy = schedule.add_noise(sample, noise, quantized[500])
    assert torch.isfinite(noisy).all()


def test_first_parallel_prepare_sample(tmp_path: Path) -> None:
    adapter = build_adapter()
    rows = [{"prompt": f"change {i}", "image": f"input_{i}.png", "edit_image": [f"input_{i}.png"]} for i in (0, 1)]
    # 默认关闭缓存与统计时，不计算样本摘要和统计键。
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda row: adapter.prepare_sample(row, FIXTURES, CONFIG), rows))
    for row, actual in zip(rows, results, strict=True):
        expected = adapter.prepare_sample(row, FIXTURES, CONFIG)
        for omitted in ("text_inputs_digest", "digests", "tracker_keys", "sample_preprocess_ms"):
            assert omitted not in actual
        for field in actual["text_inputs"]:
            assert torch.equal(actual["text_inputs"][field], expected["text_inputs"][field])
        assert torch.equal(actual["target_pixels"], expected["target_pixels"])

    # 仅开启缓存时仅生成缓存所需摘要，不生成统计键。
    cache_only = adapter.prepare_sample(rows[0], FIXTURES, {**CONFIG, "condition_cache_enabled": True})
    assert "text_inputs_digest" in cache_only and "digests" in cache_only
    assert "tracker_keys" not in cache_only and "sample_preprocess_ms" not in cache_only

    # 仅开启统计时生成统计键与预处理计时，不冗余保存缓存字段。
    tracked = adapter.prepare_sample(rows[0], FIXTURES, {**CONFIG, "baseline_stats_dir": str(tmp_path)})
    assert "tracker_keys" in tracked and "sample_preprocess_ms" in tracked
    assert "text_inputs_digest" not in tracked and "digests" not in tracked


@pytest.mark.parametrize("count", [1, 2])
def test_text_encoding_uses_3d_image_positions(count: int) -> None:
    adapter, model = build_adapter(), build_model()
    data = small_sample(adapter, count=count)
    positions = []

    def capture(module, args):
        positions.append(args[1].clone())

    hook = model.text_encoder.model.language_model.rotary_emb.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model.text_encoder.model(**data["text_inputs"], use_cache=False)
            model.text_encoder.model.rope_deltas = None
            model.text_encoder.model(
                **{k: v for k, v in data["text_inputs"].items() if k != "mm_token_type_ids"}, use_cache=False
            )
    finally:
        hook.remove()
    assert not torch.equal(positions[0], positions[1])


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cache_continuous_training_preserves_gradients_updates_rng(mode: str) -> None:
    histories = []
    for enabled in (False, True):
        adapter, model = build_adapter(), build_model(mode)
        samples = [small_sample(adapter, i) for i in (0, 1)]
        parameters = [p for p in model.parameters() if p.requires_grad]
        assert all(p.dtype == torch.float32 for p in parameters)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {}) if enabled else None
        torch.manual_seed(1991)
        history = []
        for step in range(4):
            data = samples[step % 2]
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in parameters]
            if cache is not None:
                cache.decide([cache.prepare(data)])
            loss = model(data)
            loss.backward()
            gradients = [p.grad.clone() if p.grad is not None else None for p in parameters]
            optimizer.step()
            history.append(
                [
                    loss.detach(),
                    *gradients,
                    *(p.detach() - old for p, old in zip(parameters, before, strict=True)),
                    torch.get_rng_state(),
                ]
            )
            if cache is not None:
                cache.end_micro()
                if step == 1:
                    saved_stats = cache.stats()
                    stats_copy = copy.deepcopy(saved_stats)
        if cache is not None:
            assert saved_stats == stats_copy
            stats = cache.stats()
            assert stats["effective_reuses"] == 2
            assert stats["restore_ms"] > 0 and stats["build_host_ms"] > 0
            for counter in adapter.condition_skip_counters:
                assert stats[counter] == 2 and saved_stats[counter] == 0
            cache.close()
        histories.append(history)
    for online, cached in zip(*histories, strict=True):
        for expected, actual in zip(online, cached, strict=True):
            if expected is None:
                assert actual is None
                continue
            assert torch.isfinite(actual).all()
            assert torch.equal(expected, actual)


def test_cache_identity_versions_and_shared_tensors() -> None:
    adapter, model = build_adapter(), build_model()
    data = small_sample(adapter)
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})

    def identity(sample):
        request = adapter.prepare_condition_request(model, sample)["edit_inputs"]
        return condition_key(cache.specs[0], request.metadata, 0)

    original = identity(data)
    assert original == identity(copy.deepcopy(data))
    for field in ("prompt", "fixed_pixels_budget"):
        changed = copy.deepcopy(data)
        changed[field] = "new prompt" if field == "prompt" else 2048
        assert identity(changed) != original
    for field in ("target", "vae", "vl"):
        changed = copy.deepcopy(data)
        digest = changed["digests"][field]
        (digest if isinstance(digest, dict) else digest[0])["sha256"] = "changed"
        assert identity(changed) != original
    for field, value in (("max_pixels", 8192), ("deterministic_attention", True)):
        model._condition_config[field] = value
        assert identity(data) != original
        model._condition_config = dict(CONFIG)
    model.torch_dtype = torch.bfloat16
    assert identity(data) != original
    model.torch_dtype = torch.float32
    model.text_encoder.config._attn_implementation = "eager"
    assert identity(data) != original
    model.text_encoder.config._attn_implementation = "sdpa"
    cache.version_tags["edit_inputs"] = "encoder-v2"
    assert identity(data) != original
    cache.version_tags["edit_inputs"] = None
    adapter.processor.tokenizer.add_tokens(["extra-token"])
    cache.bump_encoder_version()
    cache.decide([cache.prepare(data)])
    assert identity(data) != original
    model(data).backward()
    cache.end_micro()
    cache.decide([cache.prepare(data)])
    stored = cache.values["edit_inputs"]
    consumed = []
    predict = model.predict

    def capture(noisy, condition, timestep):
        consumed.append(condition)
        return predict(noisy, condition, timestep)

    model.predict = capture
    model(data).backward()
    assert consumed[0]["prompt_embeds"] is stored["prompt_embeds"]
    assert consumed[0]["target_latents"] is stored["target_latents"]
    assert consumed[0]["edit_latents"][0] is stored["edit_latents"][0]
    cache.end_micro()
    stored["prompt_embeds"].add_(1)
    with pytest.raises(ValueError, match="changed"):
        cache.prepare(data)
    assert cache.stats()["entries"] == 0
    cache.close()


def test_prefetched_processor_inputs_keep_distinct_cache_identity() -> None:
    adapter, model = build_adapter(), build_model()
    row = {"prompt": "change the color", "image": "input_0.png", "edit_image": ["input_0.png"]}
    cached_config = {**CONFIG, "condition_cache_enabled": True}
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})
    prepared = adapter.prepare_sample(row, FIXTURES, cached_config)
    adapter.processor.image_processor.rescale_factor *= 0.5
    cache.bump_encoder_version()
    cache.decide([cache.prepare(prepared)])
    old_condition = model.encode_conditions(prepared)
    cache.put("edit_inputs", old_condition)
    cache.end_micro()

    current = adapter.prepare_sample(row, FIXTURES, cached_config)
    expected = model.encode_conditions(current)
    assert not torch.equal(prepared["text_inputs"]["pixel_values"], current["text_inputs"]["pixel_values"])
    assert not torch.equal(old_condition["prompt_embeds"], expected["prompt_embeds"])
    cache.decide([cache.prepare(current)])
    assert not cache.values
    model(current).backward()
    cache.end_micro()
    cache.decide([cache.prepare(current)])
    actual = cache.values["edit_inputs"]
    assert torch.equal(actual["prompt_embeds"], expected["prompt_embeds"])
    assert torch.equal(actual["target_latents"], expected["target_latents"])
    cache.end_micro()
    cache.close()


def test_native_attention_mask_and_training_boundaries() -> None:
    model = build_model()
    inputs = {
        "hidden_states": torch.randn(1, 8, 64),
        "encoder_hidden_states": torch.randn(1, 11, 32),
        "timestep": torch.tensor([0.5]),
        "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
        "return_dict": False,
    }
    one = model.transformer(**inputs, encoder_hidden_states_mask=None)[0]
    two = model.transformer(**inputs, encoder_hidden_states_mask=torch.ones(1, 11, dtype=torch.bool))[0]
    assert torch.equal(one, two)
    actor = SimpleNamespace(
        model=model,
        adapter=QwenImageEditSFTAdapter(),
        _trainable_mode="full",
        args=Namespace(loss_type="sft", generation_task="i2i"),
    )
    FSDPTrainRayActor._assert_trainable_boundary(actor)
    parameter = next(model.text_encoder.parameters())
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen modules"):
        FSDPTrainRayActor._assert_trainable_boundary(actor)
    parameter.requires_grad_(False)
    next(model.transformer.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="all transformer params trainable"):
        FSDPTrainRayActor._assert_trainable_boundary(actor)
    lora = build_model("lora")
    assert all(
        name.startswith("transformer.") and ".lora_" in name for name, p in lora.named_parameters() if p.requires_grad
    )
    assert not lora.text_encoder.training and not lora.vae.training


def test_configuration_validation() -> None:
    def arguments(config):
        return Namespace(
            loss_type="sft",
            task_type="causal_lm",
            generation_task="i2i",
            train_backend="fsdp",
            model_path="/model",
            prompt_data="/data.json",
            i2i_sft_config={"data_root": "/data", "dataset_size": 2, **config},
            global_batch_size=2,
            rollout_batch_size=2,
            fsdp_trainable_mode="full",
            lora_rank=0,
            num_rollout=2,
            num_epoch=None,
            seed=42,
            num_updates_per_batch=1,
            fsdp_lr_scheduler="constant",
            fsdp_param_dtype="bf16",
            fsdp_reduce_dtype="fp32",
            fsdp_master_dtype=None,
            model_adapter_path="relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter",
        )

    default = arguments({})
    validate_i2i_sft_config(default)
    valid = arguments({"transformer_gradient_checkpointing": True})
    validate_i2i_sft_config(valid)
    assert default.i2i_sft_config == valid.i2i_sft_config
    assert valid.i2i_sft_config["transformer_gradient_checkpointing"] is True
    assert "prefix_gradient_checkpointing" not in valid.i2i_sft_config
    with pytest.raises(ValueError, match="prefix_gradient_checkpointing"):
        validate_i2i_sft_config(arguments({"prefix_gradient_checkpointing": False}))
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="transformer_gradient_checkpointing"):
            validate_i2i_sft_config(arguments({"transformer_gradient_checkpointing": value}))
    with pytest.raises(ValueError, match="unknown|Unknown|Unsupported"):
        validate_i2i_sft_config(arguments({"unknown": True}))


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_adapter_loads_component_directories(tmp_path: Path, mode: str) -> None:
    from relax.backends.fsdp.lora import inject_lora_adapter

    source = build_model()
    source.transformer.save_pretrained(tmp_path / "transformer")
    source.text_encoder.save_pretrained(tmp_path / "text_encoder")
    source.vae.save_pretrained(tmp_path / "vae")
    shutil.copytree(FIXTURES / "processor", tmp_path / "processor")
    adapter = QwenImageEditSFTAdapter()
    loaded = adapter.load_train_model(
        Namespace(
            model_path=str(tmp_path),
            fsdp_trainable_mode=mode,
            i2i_sft_config={**CONFIG, "attention_implementation": "torch"},
        )
    )
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    if mode == "lora":
        inject_lora_adapter(loaded, rank=2, alpha=2, target_modules=adapter.lora_target_modules)
    adapter.prepare_for_sharding(loaded)
    adapter.on_device_ready(loaded, torch.device("cpu"))
    assert loaded.transformer.is_gradient_checkpointing
    assert all(
        parameter.device.type == "cpu" and parameter.dtype == torch.bfloat16 for parameter in loaded.parameters()
    )
    if mode == "full":
        assert all(
            parameter.requires_grad == name.startswith("transformer.") for name, parameter in loaded.named_parameters()
        )
    else:
        assert any(p.requires_grad for p in loaded.parameters())
        assert all(
            name.startswith("transformer.") and ".lora_" in name
            for name, parameter in loaded.named_parameters()
            if parameter.requires_grad
        )
    data = small_sample(adapter)
    loss = loaded(data)
    assert torch.isfinite(loss)
    loss.backward()


class _LegacyCustomSFTAdapter:
    family = "legacy_custom"
    supported_tasks = ("i2i",)
    lora_target_modules = ("linear",)

    def load_train_model(self, args):
        return torch.nn.Linear(2, 2)

    def prepare_for_sharding(self, model):
        return None

    def on_device_ready(self, model, device):
        return None

    def prepare_sample(self, row, root, config, index=0):
        return row


def test_legacy_custom_adapter_compatibility() -> None:
    legacy_path = f"{__name__}._LegacyCustomSFTAdapter"
    args = Namespace(
        loss_type="sft",
        task_type="causal_lm",
        generation_task="i2i",
        train_backend="fsdp",
        model_path="/model",
        prompt_data="/data.json",
        i2i_sft_config={"data_root": "/data", "dataset_size": 2},
        global_batch_size=2,
        rollout_batch_size=2,
        fsdp_trainable_mode="lora",
        lora_rank=4,
        lora_alpha=4,
        num_rollout=2,
        num_epoch=None,
        seed=42,
        num_updates_per_batch=1,
        fsdp_lr_scheduler="constant",
        fsdp_param_dtype="bf16",
        fsdp_reduce_dtype="fp32",
        fsdp_master_dtype=None,
        model_adapter_path=legacy_path,
    )
    validate_i2i_sft_config(args)
    assert args.i2i_sft_config["prefix_gradient_checkpointing"] is True

    bad_args = Namespace(
        **{
            **vars(args),
            "i2i_sft_config": {"data_root": "/data", "dataset_size": 2, "transformer_gradient_checkpointing": True},
        }
    )
    with pytest.raises(ValueError, match="transformer_gradient_checkpointing"):
        validate_i2i_sft_config(bad_args)

    linear = torch.nn.Linear(2, 2)
    actor = SimpleNamespace(
        model=linear,
        adapter=_LegacyCustomSFTAdapter(),
        _trainable_mode="full",
        args=Namespace(loss_type="sft", generation_task="i2i"),
    )
    FSDPTrainRayActor._assert_trainable_boundary(actor)
    linear.bias.requires_grad_(False)
    with pytest.raises(ValueError, match="all transformer params trainable"):
        FSDPTrainRayActor._assert_trainable_boundary(actor)


def test_tracker_qwen_schema_across_ranks_and_cli(tmp_path: Path) -> None:
    adapter, model = build_adapter(), build_model()
    data = small_sample(adapter, count=2)
    config = {**CONFIG, "conditioning_images": 2}
    trackers = [
        ConditionTracker(
            rank=rank,
            world_size=2,
            micro_batches=1,
            dataset_size=2,
            output_dir=tmp_path,
            warmup_steps=0,
            schema=QWEN_IMAGE_EDIT_TRACKER_SCHEMA,
        )
        for rank in (0, 1)
    ]
    for tracker in trackers:
        model.tracker = tracker
        for step in range(2):
            tracker.begin_step(step, torch.device("cpu"))
            tracker.begin_micro(0, tracker.rank, {"_prepared": data}, config, "full", 32, 32)
            condition = model.encode_conditions(data)
            tracker.end_step(step, 0.001, 0.01)
        result = tracker.save_summary()
        for kind in ("prompt_encode", "target_vae_encode"):
            assert result["encoding_stats"][kind]["repeated_calls"] == 1
        assert result["encoding_stats"]["ref_vae_encode"]["repeated_calls"] == 2
        assert result["encoding_stats"]["ref_vae_encode"]["encoded_images"] == 4
        assert result["feature_tensor_bytes"]["prompt_embeds_bytes"]["max_bytes"] == condition["prompt_embeds"].nbytes
        assert result["feature_tensor_bytes"]["edit_latents_bytes"]["max_bytes"] == sum(
            tensor.nbytes for tensor in condition["edit_latents"]
        )
        assert (
            result["feature_tensor_bytes"]["target_latents_bytes"]["max_bytes"] == condition["target_latents"].nbytes
        )
        with pytest.raises(ValueError, match="Unknown encode"):
            tracker.record_encode("undefined", 1)
        with pytest.raises(ValueError, match="Unknown feature"):
            tracker.record_encode("prompt_encode", 1, {"undefined": 1})
    summary = summarize_baseline_dir(tmp_path)
    assert summary["tracker_schema"] == QWEN_IMAGE_EDIT_TRACKER_SCHEMA.name
    assert summary["encoding_stats"]["prompt_encode"]["global_repeated_calls"] == 3
    assert summary["encoding_stats"]["ref_vae_encode"]["global_repeated_calls"] == 6
    assert summary["repeat_evidence"]["global_repeat_count"] == 3
    expected = (tmp_path / "baseline_summary.json").read_bytes()
    subprocess.run(
        [sys.executable, "-m", "relax.engine.sft.condition_tracker", str(tmp_path)], check=True, capture_output=True
    )
    assert (tmp_path / "baseline_summary.json").read_bytes() == expected


def test_qwen_dual_ref_skip_counter_and_vl_h2d_repeat_identity(tmp_path: Path) -> None:
    # 双参考图缓存命中时，skipped_ref_vae_encodes 须按参考图数量累加（加 2）。
    adapter, model = build_adapter(), build_model()
    model._condition_config = {**CONFIG, "conditioning_images": 2}
    dual = small_sample(adapter, index=0, count=2)
    cache = ConditionCacheBinding(model, adapter, torch.device("cpu"), {})
    cache.decide([cache.prepare(dual)])
    model(dual).backward()
    cache.end_micro()
    cache.decide([cache.prepare(dual)])
    model(dual).backward()
    cache.end_micro()
    stats = cache.stats()
    assert stats["skipped_prompt_encodes"] == 1
    assert stats["skipped_ref_vae_encodes"] == 2
    assert stats["skipped_target_vae_encodes"] == 1
    cache.close()

    # 同一参考图搭配不同指令时，prompt_key 不同而 ref_vl_pixels_key 相同，重复视觉 H2D 须被识别。
    track_dir = tmp_path / "vl_h2d"
    track_cfg = {**CONFIG, "baseline_stats_dir": str(track_dir)}
    row_a = {"prompt": "turn sky blue", "image": "input_0.png", "edit_image": ["input_1.png"]}
    row_b = {"prompt": "add warm sunset", "image": "input_2.png", "edit_image": ["input_1.png"]}
    sample_a = adapter.prepare_sample(row_a, FIXTURES, track_cfg)
    sample_b = adapter.prepare_sample(row_b, FIXTURES, track_cfg)
    assert sample_a["tracker_keys"]["prompt_key"] != sample_b["tracker_keys"]["prompt_key"]
    assert sample_a["tracker_keys"]["ref_vl_pixels_key"] == sample_b["tracker_keys"]["ref_vl_pixels_key"]

    tracker = ConditionTracker(
        rank=0,
        world_size=1,
        micro_batches=2,
        dataset_size=2,
        output_dir=track_dir,
        warmup_steps=0,
        schema=QWEN_IMAGE_EDIT_TRACKER_SCHEMA,
    )
    model.tracker = tracker
    tracker.begin_step(0, torch.device("cpu"))
    for micro, sample in enumerate((sample_a, sample_b)):
        tracker.begin_micro(micro, micro, {"_prepared": sample}, track_cfg, "full", 32, 32, is_padding=False)
        model.encode_conditions(sample)
    tracker.end_step(0, 0.001, 0.01)
    summary = tracker.save_summary()
    assert summary["transfer_stats"]["prompt_text_inputs"]["repeated_transfer_calls"] == 0
    assert summary["transfer_stats"]["ref_vl_pixel_values"]["repeated_transfer_calls"] == 1
    assert (
        summary["transfer_stats"]["ref_vl_pixel_values"]["repeated_transfer_bytes"]
        == sample_b["text_inputs"]["pixel_values"].nbytes
    )
