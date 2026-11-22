# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import math
import os
import threading
from collections import UserDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from relax.engine.sft.image import image_tier_budget, load_sft_image
from relax.models.qwen_image.scheduler import QwenImageTrainingSchedule


PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
IMAGE_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
DROP_TOKENS = 64
TEXT_INPUT_FIELDS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
LORA_TARGETS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_out.0",
    "attn.to_add_out",
    "img_mlp.net.2",
    "img_mod.1",
    "txt_mlp.net.2",
    "txt_mod.1",
)


def load_image(path: Path, max_pixels: int, fixed_pixels: int | None = None) -> Any:
    """Load an RGB image, resize to the pixel budget and center-crop to 16
    px."""
    return load_sft_image(path, max_pixels, patch_size=16, fixed_pixels=fixed_pixels)


def resize_condition(image: Any, area: int) -> Any:
    """Resize a conditioning image to ``area`` with dimensions rounded to
    32."""
    ratio = image.width / image.height
    width = math.sqrt(area * ratio)
    height = width / ratio
    size = (round(width / 32) * 32, round(height / 32) * 32)
    if min(size) <= 0:
        raise ValueError("Condition image aspect ratio produces an empty dimension")
    return image.resize(size)


def preprocess_pixels(image: Any) -> torch.Tensor:
    """Normalize an RGB PIL image to ``[-1, 1]`` BF16 BCHW CPU tensor."""
    pixels = torch.from_numpy(np.array(image, dtype=np.float32)).to(torch.bfloat16)
    pixels = pixels * (2 / 255)
    pixels = pixels + (-1)
    return pixels.permute(2, 0, 1).unsqueeze(0)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack ``[B, C, H, W]`` VAE latents into ``[B, (H//2)*(W//2), C*4]``."""
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height // 2 * (width // 2), channels * 4)
    )


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack ``[B, (H//2)*(W//2), C*4]`` DiT tokens back to ``[B, C, H,
    W]``."""
    batch, _, channels = latents.shape
    return (
        latents.reshape(batch, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, height, width)
    )


class QwenImageEditSFTModel(torch.nn.Module):
    """FSDP root module combining Qwen-Image DiT, frozen VL encoder and VAE."""

    _no_split_modules = ("QwenImageTransformerBlock", "Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock")

    def __init__(
        self,
        transformer: torch.nn.Module,
        text_encoder: torch.nn.Module,
        vae: torch.nn.Module,
        *,
        training_mode: str = "lora",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if transformer.config.guidance_embeds:
            raise ValueError("Qwen image-edit SFT requires guidance_embeds=False")
        self.transformer, self.text_encoder, self.vae = transformer, text_encoder, vae
        self.training_mode = training_mode
        self.torch_dtype = torch_dtype
        self.device = next(transformer.parameters()).device
        self.schedule = QwenImageTrainingSchedule()
        text_encoder.requires_grad_(False).eval()
        vae.requires_grad_(False).eval()
        transformer.requires_grad_(training_mode == "full").train()
        for path in (
            "text_encoder.model.language_model.embed_tokens",
            "text_encoder.lm_head",
            "text_encoder.model.visual.patch_embed",
            "text_encoder.model.visual.merger",
            "vae.encoder",
            "vae.decoder",
        ):
            self.get_submodule(path)._fsdp_fine_grained = True

    def train(self, mode: bool = True) -> QwenImageEditSFTModel:
        """Set training mode on the transformer while keeping encoders in
        eval."""
        super().train(mode)
        self.text_encoder.eval()
        self.vae.eval()
        return self

    def encode_vae(self, pixels: torch.Tensor) -> torch.Tensor:
        """Deterministically encode ``[1, 3, H, W]`` pixels with ``vae``
        mode."""
        latent = self.vae.encode(pixels.unsqueeze(2)).latent_dist.mode()
        mean = (
            torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
            .to(device=latent.device, dtype=self.torch_dtype)
            .view(1, -1, 1, 1, 1)
        )
        inv_std = (
            (1 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32))
            .to(device=latent.device, dtype=self.torch_dtype)
            .view(1, -1, 1, 1, 1)
        )
        return ((latent - mean) * inv_std).squeeze(2)

    @torch.no_grad()
    def encode_conditions(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Compute prompt embeddings, packed edit latents and target
        latents."""
        inputs = {name: tensor.to(self.device) for name, tensor in data["text_inputs"].items()}
        hidden = self.text_encoder.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        prompt_embeds = torch.stack([hidden[0, DROP_TOKENS:]]).to(self.torch_dtype)
        edit_latents = tuple(
            pack_latents(self.encode_vae(pixels.to(device=self.device, dtype=self.torch_dtype)))
            for pixels in data["edit_pixels"]
        )
        target_latents = self.encode_vae(data["target_pixels"].to(device=self.device, dtype=self.torch_dtype))
        return {"prompt_embeds": prompt_embeds, "edit_latents": edit_latents, "target_latents": target_latents}

    def predict(self, noisy: torch.Tensor, condition: Mapping[str, Any], timestep: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity over packed target and edit latent tokens."""
        height, width = noisy.shape[-2:]
        shapes = [(1, height // 2, width // 2), *condition["edit_shapes"]]
        hidden = torch.cat([pack_latents(noisy), *condition["edit_latents"]], dim=1)
        prediction = self.transformer(
            hidden_states=hidden,
            encoder_hidden_states=condition["prompt_embeds"],
            encoder_hidden_states_mask=None,
            timestep=timestep / 1000,
            img_shapes=[shapes],
            guidance=None,
            return_dict=False,
        )[0]
        return unpack_latents(prediction[:, : height // 2 * (width // 2)], height, width)

    def flow_matching_loss(
        self,
        condition: Mapping[str, Any],
        *,
        timestep_id: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted FP32 MSE flow-matching loss for one micro-batch."""
        if timestep_id is None:
            timestep_id = torch.randint(0, self.schedule.num_train_timesteps, (1,))
        scheduler_timestep = self.schedule.timesteps[timestep_id].to(self.torch_dtype)
        clean = condition["target_latents"]
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.schedule.add_noise(clean, noise, scheduler_timestep)
        target = self.schedule.training_target(clean, noise, scheduler_timestep)
        prediction = self.predict(noisy, condition, scheduler_timestep.to(self.device))
        return torch.nn.functional.mse_loss(prediction.float(), target.float()) * self.schedule.training_weight(
            scheduler_timestep
        )

    def forward(
        self,
        data: Mapping[str, Any],
        *,
        timestep_id: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode frozen conditions and compute the SFT loss."""
        condition = {**self.encode_conditions(data), "edit_shapes": data["edit_shapes"]}
        # DiffSynth's NoiseInitializer consumes one RNG draw before the loss draws training noise.
        torch.randn(condition["target_latents"].shape, device=self.device, dtype=self.torch_dtype)
        return self.flow_matching_loss(condition, timestep_id=timestep_id, noise=noise)


class QwenImageEditSFTAdapter:
    """Qwen-Image-Edit model adapter for the shared image-edit SFT trainer."""

    family = "qwen_image_edit"
    supported_tasks = ("i2i",)
    lora_target_modules = LORA_TARGETS
    lora_task_type = "FEATURE_EXTRACTION"
    image_tier_policy = True
    cpu_only_prepare_sample = True
    frozen_module_prefixes = ("text_encoder.", "vae.")

    def __init__(self) -> None:
        self.processor = None
        self.processor_lock = threading.Lock()
        self._processor_path = None

    @staticmethod
    def validate_sft_config(config: dict[str, Any]) -> list[str]:
        """Validate Qwen-Image-Edit options in ``i2i_sft_config``."""
        errors: list[str] = []
        if "prefix_gradient_checkpointing" in config:
            errors.append("Qwen-Image-Edit does not support prefix_gradient_checkpointing")
        config.setdefault("transformer_gradient_checkpointing", True)
        if type(config["transformer_gradient_checkpointing"]) is not bool:
            errors.append("transformer_gradient_checkpointing must be boolean")
        return errors

    def configure_processor(self, processor: Any, source: Path) -> None:
        """Bind a torchvision-backed ``Qwen2VLProcessor`` and initialize its
        padding state."""
        from PIL import Image
        from transformers.image_processing_backends import TorchvisionBackend

        if not isinstance(processor.image_processor, TorchvisionBackend):
            raise ValueError("Qwen image-edit SFT requires the torchvision image processor")
        self.processor = processor
        self._processor_path = source
        with self.processor_lock:
            output = processor(
                text=[PROMPT_TEMPLATE.format(IMAGE_TEMPLATE.format(1))],
                images=[Image.new("RGB", (32, 32))],
                padding=True,
                return_tensors="pt",
            )
            self._validate_text_inputs(output)

    @staticmethod
    def _validate_text_inputs(inputs: Mapping[str, torch.Tensor]) -> None:
        if set(TEXT_INPUT_FIELDS) - inputs.keys():
            raise ValueError(f"Processor must return {TEXT_INPUT_FIELDS}")
        mask = inputs["attention_mask"]
        if mask.device.type != "cpu" or mask.shape[0] != 1 or not bool(mask.all()):
            raise ValueError("Qwen image-edit SFT requires one unpadded CPU text sample")
        if inputs["input_ids"].shape[1] <= DROP_TOKENS:
            raise ValueError("Qwen image-edit prompt must contain tokens after the system prefix")

    def load_train_model(self, args: Any) -> QwenImageEditSFTModel:
        """Load diffusers/transformers components and configure attention."""
        from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

        config = args.i2i_sft_config
        implementation = config["attention_implementation"]
        deterministic = config.get("deterministic_attention", False)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # full 和 LoRA 均先在 CPU 加载，再通过 fsdp2_wrap 分片到各 CUDA 设备，避免分片前在每张 GPU 驻留完整权重。
        load_options = {"torch_dtype": torch.bfloat16, "device_map": {"": torch.device("cpu")}}
        transformer = QwenImageTransformer2DModel.from_pretrained(
            args.model_path, subfolder="transformer", **load_options
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            subfolder="text_encoder",
            attn_implementation="flash_attention_2" if implementation == "flash_attention_2" else "sdpa",
            **load_options,
        )
        vae = AutoencoderKLQwenImage.from_pretrained(args.model_path, subfolder="vae", **load_options)
        processor = Qwen2VLProcessor.from_pretrained(args.model_path, subfolder="processor")
        self.configure_processor(processor, Path(args.model_path) / "processor")
        if implementation == "flash_attention_2":
            from .attention import QwenImageFlashAttnProcessor

            transformer.set_attn_processor(QwenImageFlashAttnProcessor(deterministic))
        elif implementation == "torch":
            transformer.set_attention_backend("native")
        else:
            raise ValueError(f"Unsupported Qwen attention implementation: {implementation}")
        if config.get("transformer_gradient_checkpointing", True):
            transformer.enable_gradient_checkpointing()
        return QwenImageEditSFTModel(transformer, text_encoder, vae, training_mode=args.fsdp_trainable_mode)

    def prepare_for_sharding(self, model: QwenImageEditSFTModel) -> None:
        """Verify LoRA targets and cast the root module to BF16 before FSDP."""
        invalid = [
            name for name, _ in model.named_parameters() if "lora_" in name and not name.startswith("transformer.")
        ]
        if invalid:
            raise ValueError(f"Qwen LoRA parameters must belong to transformer: {invalid[:5]}")
        model.to(dtype=torch.bfloat16)

    def on_device_ready(self, model: QwenImageEditSFTModel, device: torch.device) -> None:
        """Record the local execution device after FSDP wrapping."""
        model.device = device

    def prepare_sample(
        self,
        row: dict[str, Any],
        root: Path,
        config: dict[str, Any],
        index: int = 0,
    ) -> UserDict[str, Any]:
        """Preprocess target and conditioning inputs into a CPU
        ``UserDict``."""
        if self.processor is None:
            raise RuntimeError("Load the Qwen processor before preparing samples")
        if len(row["edit_image"]) != config["conditioning_images"]:
            raise ValueError("Condition image count does not match conditioning_images")
        budget = image_tier_budget(config, index)
        image = load_image(root / row["image"], config["max_pixels"], budget)
        edit_images = [load_image(root / path, config["max_pixels"], budget) for path in row["edit_image"]]
        vae_images = [resize_condition(edit_image, 1024 * 1024) for edit_image in edit_images]
        vl_images = [resize_condition(edit_image, 384 * 384) for edit_image in vae_images]
        text = PROMPT_TEMPLATE.format(
            "".join(IMAGE_TEMPLATE.format(i + 1) for i in range(len(vl_images))) + row["prompt"]
        )
        with self.processor_lock:
            text_inputs = self.processor(text=[text], images=vl_images, padding=True, return_tensors="pt")
        self._validate_text_inputs(text_inputs)
        return UserDict(
            {
                "prompt": row["prompt"],
                "fixed_pixels_budget": budget,
                "image": image,
                "edit_image": edit_images,
                "target_pixels": preprocess_pixels(image),
                "edit_pixels": tuple(preprocess_pixels(x) for x in vae_images),
                "text_inputs": {name: text_inputs[name] for name in TEXT_INPUT_FIELDS},
                "edit_shapes": tuple((1, x.height // 16, x.width // 16) for x in vae_images),
            }
        )
