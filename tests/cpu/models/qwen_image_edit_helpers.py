# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import atexit
import tempfile
from collections import UserDict
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2TokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from relax.backends.fsdp.lora import inject_lora_adapter
from relax.models.qwen_image.sft_adapter import (
    IMAGE_TEMPLATE,
    PROMPT_TEMPLATE,
    TEXT_INPUT_FIELDS,
    QwenImageEditSFTAdapter,
    QwenImageEditSFTModel,
    _digest_text_inputs,
    image_digest,
    load_image,
    preprocess_pixels,
)


def _create_test_fixtures(directory: Path) -> Path:
    specials = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()) + specials)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, pad_token="<|endoftext|>", eos_token="<|im_end|>")
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            min_pixels=3136, max_pixels=12845056, patch_size=14, temporal_patch_size=2, merge_size=2
        ),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(
            min_pixels=3136, max_pixels=12845056, patch_size=14, temporal_patch_size=2, merge_size=2
        ),
    )
    processor.save_pretrained(directory / "processor")
    for index, (width, height) in enumerate(((65, 49), (79, 113), (19, 35))):
        rgb = (np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) * (index + 7) % 256).astype(
            np.uint8
        )
        Image.fromarray(rgb).save(directory / f"input_{index}.png")
    return directory


_fixture_directory = tempfile.TemporaryDirectory(prefix="qwen-image-edit-sft-test-")
atexit.register(_fixture_directory.cleanup)
FIXTURES = _create_test_fixtures(Path(_fixture_directory.name))
CONFIG = {"conditioning_images": 1, "max_pixels": 4096, "seed": 42, "fixed_pixels_tiers": None}


def build_adapter() -> QwenImageEditSFTAdapter:
    adapter = QwenImageEditSFTAdapter()
    adapter.configure_processor(Qwen2VLProcessor.from_pretrained(FIXTURES / "processor"), FIXTURES / "processor")
    return adapter


def build_model(mode: str = "full", dtype: torch.dtype = torch.float32) -> QwenImageEditSFTModel:
    torch.manual_seed(1729)
    tokenizer = build_adapter().processor.tokenizer
    transformer = QwenImageTransformer2DModel(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
    )
    transformer.set_attention_backend("native")
    text_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
            "max_position_embeddings": 4096,
            "use_cache": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": [0],
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    text_config._attn_implementation = "sdpa"
    text_encoder = Qwen2_5_VLForConditionalGeneration(text_config)
    vae = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=16,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
    )
    model = QwenImageEditSFTModel(transformer, text_encoder, vae, training_mode=mode, torch_dtype=dtype).to(dtype)
    if mode == "lora":
        inject_lora_adapter(model, rank=2, alpha=2, target_modules=QwenImageEditSFTAdapter.lora_target_modules)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(std=0.01)
        model.to(dtype)
    model._condition_config = dict(CONFIG)
    model._condition_model_path = str(FIXTURES)
    model.train()
    return model


def small_sample(adapter: QwenImageEditSFTAdapter, index: int = 0, count: int = 1) -> UserDict:
    target = load_image(FIXTURES / f"input_{index}.png", 4096).resize((32, 32))
    edits = [load_image(FIXTURES / f"input_{i}.png", 4096).resize((32, 32)) for i in range(count)]
    prompt = f"change the color {index}"
    text = PROMPT_TEMPLATE.format("".join(IMAGE_TEMPLATE.format(i + 1) for i in range(count)) + prompt)
    outputs = adapter.processor(text=[text], images=edits, padding=True, return_tensors="pt")
    return UserDict(
        {
            "prompt": prompt,
            "fixed_pixels_budget": None,
            "image": target,
            "edit_image": edits,
            "target_pixels": preprocess_pixels(target),
            "edit_pixels": tuple(preprocess_pixels(image) for image in edits),
            "text_inputs": {name: outputs[name] for name in TEXT_INPUT_FIELDS},
            "text_inputs_digest": _digest_text_inputs(outputs),
            "digests": {
                "target": image_digest(target),
                "vae": [image_digest(x) for x in edits],
                "vl": [image_digest(x) for x in edits],
            },
            "edit_shapes": tuple((1, 2, 2) for _ in edits),
            "tracker_keys": {
                "prompt_key": f"prompt-{index}",
                "ref_vl_pixels_key": f"vl-{count}",
                "ref_vae_image_keys": [f"ref-{i}" for i in range(count)],
                "target_image_key": f"target-{index}",
            },
            "sample_preprocess_ms": 0.1,
        }
    )
