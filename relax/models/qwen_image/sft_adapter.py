# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import UserDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from relax.engine.sft.condition_cache import ConditionFieldSpec, ConditionRequest
from relax.engine.sft.condition_tracker import EncodeSpec, EvidenceSpec, TrackerSchema, TransferSpec
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


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _digest_text_inputs(inputs: Mapping[str, torch.Tensor]) -> str:
    return _digest_json(
        {
            name: {
                "shape": tuple(inputs[name].shape),
                "dtype": str(inputs[name].dtype),
                "sha256": hashlib.sha256(inputs[name].contiguous().view(torch.uint8).numpy()).hexdigest(),
            }
            for name in TEXT_INPUT_FIELDS
        }
    )


def image_digest(image: Any) -> dict[str, Any]:
    """Return the mode, dimensions and pixel SHA256 digest of a PIL image."""
    return {"mode": image.mode, "size": image.size, "sha256": hashlib.sha256(image.tobytes()).hexdigest()}


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


def _qwen_micro_keys(
    row: dict[str, Any], config: dict[str, Any], budget: int | None, height: int, width: int, training_mode: str
) -> dict[str, Any]:
    data = row["_prepared"]
    return {**data["tracker_keys"], "eligible": True}


QWEN_IMAGE_EDIT_TRACKER_SCHEMA = TrackerSchema(
    name="relax.models.qwen_image.sft_adapter.QWEN_IMAGE_EDIT_TRACKER_SCHEMA",
    stage_order=(
        "data_wait_ms",
        "sample_preprocess_ms",
        "prompt_encode_ms",
        "ref_vae_encode_ms",
        "target_vae_encode_ms",
        "dit_forward_ms",
        "backward_ms",
        "optimizer_ms",
        "train_ms",
    ),
    host_stages=("sample_preprocess_ms",),
    micro_gpu_stages=(
        "prompt_encode_ms",
        "ref_vae_encode_ms",
        "target_vae_encode_ms",
        "dit_forward_ms",
        "backward_ms",
    ),
    key_fields=("prompt_key", "ref_vae_image_keys", "target_image_key"),
    micro_keys=_qwen_micro_keys,
    transfers=(
        TransferSpec("prompt_text_inputs", "prompt_key"),
        TransferSpec("ref_vl_pixel_values", "prompt_key"),
        TransferSpec("ref_vae_pixels", "ref_vae_image_keys", per_image=True),
        TransferSpec("target_pixels", "target_image_key"),
    ),
    encodes=(
        EncodeSpec(
            "prompt_encode",
            "prompt_key",
            "prompt_encode_ms",
            feature_bytes=("prompt_embeds_bytes",),
            repeat_flag="prompt_repeated",
        ),
        EncodeSpec(
            "ref_vae_encode",
            "ref_vae_image_keys",
            "ref_vae_encode_ms",
            feature_bytes=("edit_latents_bytes",),
            items_field="encoded_images",
            repeat_flag="ref_vae_repeated",
        ),
        EncodeSpec(
            "target_vae_encode",
            "target_image_key",
            "target_vae_encode_ms",
            feature_bytes=("target_latents_bytes",),
            repeat_flag="target_repeated",
        ),
    ),
    host_repeat_stages=(),
    condition_stages=("prompt_encode_ms", "ref_vae_encode_ms", "target_vae_encode_ms"),
    evidence=EvidenceSpec(
        "prompt_key",
        ("target_image_key",),
        "condition_h2d",
        ("prompt_text_inputs", "ref_vl_pixel_values", "ref_vae_pixels", "target_pixels"),
        ("prompt_encode_ms", "ref_vae_encode_ms", "target_vae_encode_ms"),
    ),
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
        self.tracker = None
        self._condition_cache = None
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
        tracker = self.tracker
        text_inputs = data["text_inputs"]
        if tracker is not None:
            tracker.record_transfer(
                "prompt_text_inputs", tuple(text_inputs[name] for name in TEXT_INPUT_FIELDS if name != "pixel_values")
            )
            tracker.record_transfer("ref_vl_pixel_values", text_inputs["pixel_values"])
        with tracker.gpu_stage("prompt_encode_ms") if tracker is not None else nullcontext():
            inputs = {name: tensor.to(self.device) for name, tensor in text_inputs.items()}
            hidden = self.text_encoder.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            prompt_embeds = torch.stack([hidden[0, DROP_TOKENS:]]).to(self.torch_dtype)
        if tracker is not None:
            tracker.record_encode("prompt_encode", 1, {"prompt_embeds_bytes": prompt_embeds.nbytes})
        edit_latents = []
        for index, pixels in enumerate(data["edit_pixels"]):
            if tracker is not None:
                tracker.record_transfer("ref_vae_pixels", pixels, image_index=index)
            with tracker.gpu_stage("ref_vae_encode_ms") if tracker is not None else nullcontext():
                latent = pack_latents(self.encode_vae(pixels.to(device=self.device, dtype=self.torch_dtype)))
            edit_latents.append(latent)
            if tracker is not None:
                tracker.record_encode(
                    "ref_vae_encode",
                    1,
                    {"edit_latents_bytes": latent.nbytes},
                    key=data["tracker_keys"]["ref_vae_image_keys"][index],
                )
        if tracker is not None:
            tracker.record_transfer("target_pixels", data["target_pixels"])
        with tracker.gpu_stage("target_vae_encode_ms") if tracker is not None else nullcontext():
            target_latents = self.encode_vae(data["target_pixels"].to(device=self.device, dtype=self.torch_dtype))
        if tracker is not None:
            tracker.record_encode("target_vae_encode", 1, {"target_latents_bytes": target_latents.nbytes})
        return {"prompt_embeds": prompt_embeds, "edit_latents": tuple(edit_latents), "target_latents": target_latents}

    def predict(self, noisy: torch.Tensor, condition: Mapping[str, Any], timestep: torch.Tensor) -> torch.Tensor:
        """Predict flow velocity over packed target and edit latent tokens."""
        height, width = noisy.shape[-2:]
        shapes = [(1, height // 2, width // 2), *condition["edit_shapes"]]
        hidden = torch.cat([pack_latents(noisy), *condition["edit_latents"]], dim=1)
        tracker = self.tracker
        with tracker.gpu_stage("dit_forward_ms") if tracker is not None else nullcontext():
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
        """Run condition encoding or cache reuse, then compute the SFT loss."""
        if self.tracker is not None:
            self.tracker.add_host_time("sample_preprocess_ms", data["sample_preprocess_ms"])
        cache = self._condition_cache
        condition = cache.values.get("edit_inputs") if cache is not None else None
        started = time.monotonic() if cache is not None else 0.0
        if condition is None:
            condition = self.encode_conditions(data)
            if cache is not None:
                cache.counters["build_host_ms"] += (time.monotonic() - started) * 1000
                cache.put("edit_inputs", condition)
        else:
            cache.counters["restore_ms"] += (time.monotonic() - started) * 1000
            for name in QwenImageEditSFTAdapter.condition_skip_counters:
                cache.counters[name] += 1
        condition = {**condition, "edit_shapes": data["edit_shapes"]}
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
    condition_skip_counters = ("skipped_prompt_encodes", "skipped_ref_vae_encodes", "skipped_target_vae_encodes")

    def __init__(self) -> None:
        self.processor = None
        self.processor_lock = threading.Lock()
        self._processor_path = None
        self._text_identity = None

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

    @staticmethod
    def tracker_schema() -> TrackerSchema:
        """Return the condition tracker schema for Qwen-Image-Edit SFT."""
        return QWEN_IMAGE_EDIT_TRACKER_SCHEMA

    def bind_tracker(self, model: QwenImageEditSFTModel, tracker: Any | None) -> None:
        """Attach or clear the condition tracker on ``model``."""
        model.tracker = tracker

    def configure_processor(self, processor: Any, source: Path) -> None:
        """Bind a torchvision-backed ``Qwen2VLProcessor`` and cache its
        identity."""
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
            self._text_identity = self._processor_identity()

    @staticmethod
    def _validate_text_inputs(inputs: Mapping[str, torch.Tensor]) -> None:
        if set(TEXT_INPUT_FIELDS) - inputs.keys():
            raise ValueError(f"Processor must return {TEXT_INPUT_FIELDS}")
        mask = inputs["attention_mask"]
        if mask.device.type != "cpu" or mask.shape[0] != 1 or not bool(mask.all()):
            raise ValueError("Qwen image-edit SFT requires one unpadded CPU text sample")
        if inputs["input_ids"].shape[1] <= DROP_TOKENS:
            raise ValueError("Qwen image-edit prompt must contain tokens after the system prefix")

    def _processor_identity(self) -> dict[str, Any]:
        import torchvision

        tokenizer = self.processor.tokenizer
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self._processor_path.iterdir())
            if path.is_file()
        }
        return {
            "files": files,
            "backend": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "special_tokens": tokenizer.special_tokens_map,
            "added_vocab": tokenizer.get_added_vocab(),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
            "split_special_tokens": tokenizer.split_special_tokens,
            "image_processor_class": type(self.processor.image_processor).__qualname__,
            "image_processor": self.processor.image_processor.to_dict(),
            "torchvision": torchvision.__version__,
            "return_mm_token_type_ids": True,
            "prompt_template": PROMPT_TEMPLATE,
            "image_template": IMAGE_TEMPLATE,
            "drop_tokens": DROP_TOKENS,
        }

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
        model = QwenImageEditSFTModel(transformer, text_encoder, vae, training_mode=args.fsdp_trainable_mode)
        model._condition_config = dict(config)
        model._condition_model_path = str(Path(args.model_path).resolve())
        return model

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
        started = time.perf_counter()
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
            text_identity = self._text_identity
        self._validate_text_inputs(text_inputs)
        text_inputs_digest = _digest_text_inputs(text_inputs)
        digests = {
            "target": image_digest(image),
            "vae": [image_digest(x) for x in vae_images],
            "vl": [image_digest(x) for x in vl_images],
        }
        data = UserDict(
            {
                "prompt": row["prompt"],
                "fixed_pixels_budget": budget,
                "image": image,
                "edit_image": edit_images,
                "target_pixels": preprocess_pixels(image),
                "edit_pixels": tuple(preprocess_pixels(x) for x in vae_images),
                "text_inputs": {name: text_inputs[name] for name in TEXT_INPUT_FIELDS},
                "text_inputs_digest": text_inputs_digest,
                "digests": digests,
                "edit_shapes": tuple((1, x.height // 16, x.width // 16) for x in vae_images),
                "tracker_keys": {
                    "prompt_key": _digest_json([row["prompt"], digests["vl"], text_identity, text_inputs_digest]),
                    "ref_vae_image_keys": [_digest_json(x) for x in digests["vae"]],
                    "target_image_key": _digest_json(digests["target"]),
                },
            }
        )
        data["sample_preprocess_ms"] = (time.perf_counter() - started) * 1000
        return data

    def condition_field_specs(self, model: QwenImageEditSFTModel) -> tuple[ConditionFieldSpec, ...]:
        """Declare the single cacheable ``edit_inputs`` condition field."""
        return (
            ConditionFieldSpec(
                name="edit_inputs",
                fields=("prompt_embeds", "edit_latents", "target_latents"),
                dependency_paths=("text_encoder", "vae"),
                input_dependencies=("prompt", "edit_images", "target_image"),
                preprocessing_dependencies=("preprocessing",),
                max_tensors=2 + model._condition_config["conditioning_images"],
                max_depth=2,
            ),
        )

    def bind_condition_cache(self, model: QwenImageEditSFTModel, cache: Any | None) -> None:
        """Attach ``cache`` and capture frozen text and adapter identities."""
        model._condition_cache = cache
        if cache is None:
            model._condition_text_identity = None
            model._condition_adapter_identity = None
            return
        with self.processor_lock:
            self._text_identity = self._processor_identity()
            model._condition_text_identity = self._text_identity
        active = []
        for prefix in ("text_encoder", "vae"):
            for name, module in model.get_submodule(prefix).named_modules():
                if getattr(module, "peft_config", None):
                    adapters = module.active_adapters
                    active.append(
                        {"module": f"{prefix}.{name}", "active": adapters() if callable(adapters) else adapters}
                    )
        model._condition_adapter_identity = active

    def prepare_condition_request(
        self,
        model: QwenImageEditSFTModel,
        data: Mapping[str, Any],
    ) -> dict[str, ConditionRequest]:
        """Build the cache key metadata for one prepared micro-batch."""
        config = model._condition_config
        metadata = {
            "weights": {
                "source": model._condition_model_path,
                "active_adapters": model._condition_adapter_identity,
                "adapter_version": model._condition_cache.version_tags["edit_inputs"],
            },
            "text": {"processor": model._condition_text_identity, "inputs": data["text_inputs_digest"]},
            "prompt": data["prompt"],
            "edit_images": {"vae": data["digests"]["vae"], "vl": data["digests"]["vl"]},
            "target_image": data["digests"]["target"],
            "preprocessing": {
                "max_pixels": config["max_pixels"],
                "fixed_pixels_tiers": config.get("fixed_pixels_tiers"),
                "fixed_pixels_budget": data["fixed_pixels_budget"],
                "division": 16,
                "condition_areas": (1024 * 1024, 384 * 384),
                "condition_rounding": 32,
                "resize_order": "crop_then_vae_then_vl",
                "crop_interpolation": "BILINEAR",
                "condition_interpolation": "PIL_default_BICUBIC",
                "height": data["image"].height,
                "width": data["image"].width,
            },
            "precision": {
                "compute": str(model.torch_dtype),
                "fields": str(model.torch_dtype),
                "attention": model.text_encoder.config._attn_implementation,
                "deterministic_attention": config.get("deterministic_attention", False),
                "latents_mean": model.vae.config.latents_mean,
                "latents_std": model.vae.config.latents_std,
            },
        }
        return {"edit_inputs": ConditionRequest(metadata)}
