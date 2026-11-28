# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Small real Diffusers VAE/DiT: cache parity, RNG and LoRA export scaling."""

from types import SimpleNamespace

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from relax.models.frozen_conditions import FrozenConditionCache
from relax.models.qwen_image.edit_sft import QwenImageEditSFTAdapter, QwenImageEditSFTModel


class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 16)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(hidden_states=[self.embedding(input_ids)])


def tiny_model():
    vae = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
    )
    transformer = QwenImageTransformer2DModel(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    pipe = QwenImageEditPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        transformer=transformer,
        text_encoder=TextEncoder(),
        tokenizer=None,
        processor=None,
    )
    pipe.processor = lambda **kwargs: BatchFeature({"input_ids": torch.tensor([[1, 2, 3, 4]])})
    pipe.prompt_template_encode_start_idx = 1
    model = QwenImageEditSFTModel(pipe, torch.float32)
    model.text_encoder.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.train()
    return model


def test_prefetch_preserves_prepared_images_order_and_rng(tmp_path):
    import json
    import random

    import numpy as np

    from relax.engine.sft.image import ImageBatchSampler, ImageEditDataset
    from relax.engine.sft.image_prefetch import ImagePrefetch

    rows = []
    for index, color in enumerate(("red", "blue", "green")):
        name = f"{index}.png"
        Image.new("RGB", (96, 64), color).save(tmp_path / name)
        rows.append({"prompt": color, "image": name, "edit_image": [name]})
    metadata = tmp_path / "train.json"
    metadata.write_text(json.dumps(rows))
    args = SimpleNamespace(
        prompt_data=str(metadata),
        i2i_sft_config={"data_root": str(tmp_path), "dataset_size": 3, "conditioning_images": 1, "max_pixels": 4096},
    )
    dataset = ImageEditDataset(args, QwenImageEditSFTAdapter())
    sampler = ImageBatchSampler(3, 2, 42)
    expected = [dataset[index] for step in range(4) for index in sampler.batch(step)]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    prefetch = ImagePrefetch(dataset, sampler, 0, 1, 0, 4, depth=2, workers=2)
    try:
        actual = [
            prefetch.get(step, micro, index) for step in range(4) for micro, index in enumerate(sampler.batch(step))
        ]
    finally:
        prefetch.close()
    for online, prefetched in zip(expected, actual, strict=True):
        assert online["prompt"] == prefetched["prompt"]
        for left, right in zip(
            [online["image"], *online["edit_image"]], [prefetched["image"], *prefetched["edit_image"]]
        ):
            assert (left.mode, left.size, left.tobytes()) == (right.mode, right.size, right.tobytes())
    assert random.getstate() == rng[0]
    np.testing.assert_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])


def test_real_dit_vae_cache_preserves_loss_gradient_update_and_rng():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        model = tiny_model()
        data = {
            "prompt": "edit",
            "image": Image.new("RGB", (32, 32), "red"),
            "edit_image": [Image.new("RGB", (32, 32), "blue")],
        }
        weights = {name: value.detach().clone() for name, value in model.transformer.named_parameters()}

        def step(cached):
            model.transformer.load_state_dict(weights)
            optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=1e-3, foreach=False)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            sample = data
            if cached:
                sample = {**data, "condition_request": model.condition_cache.lookup(model.condition_key(data))}
            loss = model(sample)
            loss.backward()
            gradients = {
                name: p.grad.clone() for name, p in model.transformer.named_parameters() if p.grad is not None
            }
            optimizer.step()
            updates = {name: p.detach() - weights[name] for name, p in model.transformer.named_parameters()}
            return loss.detach(), gradients, updates, torch.get_rng_state()

        baseline = step(False)
        model.condition_cache = FrozenConditionCache(1_000_000, 2)
        cold, hot = step(True), step(True)
        for actual in (cold, hot):
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        assert model.condition_cache.encodes == model.condition_cache.hits == 1
        assert all(p.grad is None for module in (model.text_encoder, model.vae) for p in module.parameters())
        assert model.training and not model.vae.training and not model.text_encoder.training
    finally:
        torch.set_num_threads(old_threads)


def test_export_loads_into_original_dit_with_nonunit_alpha(tmp_path, monkeypatch):
    from relax.backends.fsdp import checkpoint
    from relax.backends.fsdp.lora import inject_lora_adapter, lora_metadata_dict

    torch.manual_seed(5)
    transformer = QwenImageTransformer2DModel(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    inject_lora_adapter(transformer, rank=2, alpha=6, target_modules=["attn.to_q"])
    state = {
        f"transformer.{name}": p.detach().clone() for name, p in transformer.named_parameters() if p.requires_grad
    }
    contract = {
        "trainable_mode": "lora",
        "lora": lora_metadata_dict(rank=2, alpha=6, dropout=0, target_modules=["attn.to_q"]),
    }
    committed = tmp_path / "checkpoint"
    committed.mkdir()
    (committed / "COMMITTED").touch()
    monkeypatch.setattr(checkpoint, "_load_dcp_model_state", lambda path: state)
    output = tmp_path / "export"
    QwenImageEditSFTAdapter.export_checkpoint(committed, output, contract)
    transformer.delete_adapters("default")
    transformer.load_lora_adapter(output, prefix="transformer", use_safetensors=True)
    layer = transformer.transformer_blocks[0].attn.to_q
    assert layer.scaling["default_0"] == 3
    for name, parameter in transformer.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, state[f"transformer.{name.replace('default_0', 'default')}"])
