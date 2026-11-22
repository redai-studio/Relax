# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import shutil
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp.actor import FSDPTrainRayActor
from relax.backends.fsdp.arguments import validate_i2i_sft_config
from relax.models.qwen_image.scheduler import QwenImageTrainingSchedule
from relax.models.qwen_image.sft_adapter import QwenImageEditSFTAdapter
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


def test_first_parallel_prepare_sample() -> None:
    adapter = build_adapter()
    rows = [{"prompt": f"change {i}", "image": f"input_{i}.png", "edit_image": [f"input_{i}.png"]} for i in (0, 1)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda row: adapter.prepare_sample(row, FIXTURES, CONFIG), rows))
    for row, actual in zip(rows, results, strict=True):
        expected = adapter.prepare_sample(row, FIXTURES, CONFIG)
        for field in actual["text_inputs"]:
            assert torch.equal(actual["text_inputs"][field], expected["text_inputs"][field])
        assert torch.equal(actual["target_pixels"], expected["target_pixels"])


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
            generation_task="i2i",
            train_backend="fsdp",
            model_path="/model",
            prompt_data="/data.json",
            i2i_sft_config={"data_root": "/data", "dataset_size": 2, **config},
            global_batch_size=2,
            rollout_batch_size=2,
            fsdp_trainable_mode="full",
            num_rollout=2,
            num_epoch=None,
            seed=42,
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
