# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qwen-Image-Edit LoRA SFT with reusable frozen text/vision conditions.

Uses Diffusers Edit conditioning and latent geometry. VAE encoding and flow
noise sampling run on every training sample.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from typing import Any

import torch

from relax.models.condition_cache import (
    ConditionFeatureCache,
    FrozenConditionSpec,
    all_ranks_hit,
    condition_cache_key,
)


def _processor_cache_identity(processor: Any) -> str:
    """Preserve ProcessorMixin's live repr without copying discarded
    tokenizers.

    Legacy to_dict deep-copies every attribute before dropping components and
    chat_template. Exclude only those same fields before that copy; component
    reprs are still evaluated on the original objects on every call. Custom
    serializers and newer APIs that retain image configs use the native repr.
    """
    from inspect import signature

    from transformers.processing_utils import ProcessorMixin

    cls = type(processor)
    if (
        not isinstance(processor, ProcessorMixin)
        or "legacy_serialization" not in signature(ProcessorMixin.to_dict).parameters
        or cls.__repr__ is not ProcessorMixin.__repr__
        or cls.__str__ is not object.__str__
        or cls.to_dict is not ProcessorMixin.to_dict
        or cls.to_json_string is not ProcessorMixin.to_json_string
        or getattr(processor.to_dict, "__func__", None) is not ProcessorMixin.to_dict
        or getattr(processor.to_json_string, "__func__", None) is not ProcessorMixin.to_json_string
        or getattr(cls, "__copy__", None) is not None
        or cls.__reduce_ex__ is not object.__reduce_ex__
    ):
        return str(processor)
    components = "\n".join(f"- {name}: {repr(getattr(processor, name))}" for name in processor.attributes)
    excluded = set(cls.attributes) | {"tokenizer", "qformer_tokenizer", "protein_tokenizer", "chat_template"}
    config_view = copy(processor)
    config_view.__dict__ = {key: value for key, value in vars(processor).items() if key not in excluded}
    return f"{cls.__name__}:\n{components}\n\n{config_view.to_json_string()}"


class QwenImageEditSFTModel(torch.nn.Module):
    """Root encompasses conditioning and trainable DiT for FSDP ordering."""

    _no_split_modules = ("QwenImageTransformerBlock", "Qwen2_5_VLDecoderLayer")

    def __init__(self, pipe: Any, identity: str) -> None:
        super().__init__()
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.device = torch.device("cpu")
        self.condition_cache_group = None
        self._encoder_versions = None
        self._condition_metrics = dict(prefix_encodes=0, global_hits=0, global_misses=0, bypasses=0)
        self.condition_cache = ConditionFeatureCache(
            int(os.environ.get("RELAX_CONDITION_CACHE_BYTES", "0")),
            storage_device=os.environ.get("RELAX_CONDITION_CACHE_DEVICE", "cpu"),
        )
        self.condition_spec = FrozenConditionSpec(
            encoder_identity=identity,
            preprocessing_version=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dtype="bfloat16",
            feature_schema="qwen-edit-text-vision-embeds-mask-v1",
            adapter_version="qwen-image-edit-sft-v1",
        )

        self.register_load_state_dict_post_hook(lambda module, incompatible: module.clear_condition_cache())

    def clear_condition_cache(self) -> None:
        self.condition_cache.clear()

    def close_condition_cache(self) -> None:
        self.condition_cache.close()

    def condition_cache_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.condition_cache.max_bytes > 0,
            **self.condition_cache.stats(),
            **self._condition_metrics,
        }

    def train(self, mode: bool = True) -> QwenImageEditSFTModel:
        super().train(mode)
        # Frozen encoder dropout must stay disabled in both online and cache.
        self.text_encoder.eval()
        self.vae.eval()
        return self

    def encode_conditions(self, data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Content is the actual post-resize image and effective prompt."""
        cache_on = self.condition_cache.max_bytes > 0 and not any(
            p.requires_grad for p in self.text_encoder.parameters()
        )
        key = None
        value = None
        versions = tuple((name, parameter._version) for name, parameter in self.text_encoder.named_parameters())
        if self._encoder_versions is not None and versions != self._encoder_versions:
            self.condition_cache.clear()
        self._encoder_versions = versions
        cache_on = all_ranks_hit(cache_on, self.condition_cache_group)
        if not cache_on:
            self.condition_cache.clear()
            self._condition_metrics["bypasses"] += 1
        if cache_on:
            image = data["edit_image"][0]
            key = condition_cache_key(
                self.condition_spec,
                {
                    "prompt": data["prompt"],
                    "size": list(image.size),
                    "mode": image.mode,
                    "pixels": image.tobytes(),
                    "template": getattr(self.pipe, "prompt_template_encode", ""),
                    "template_start": getattr(self.pipe, "prompt_template_encode_start_idx", 0),
                    "encoder_attention": getattr(
                        getattr(self.text_encoder, "config", None), "_attn_implementation", None
                    ),
                    "encoder_dtype": str(next(self.text_encoder.parameters()).dtype),
                    "autocast": torch.is_autocast_enabled("cuda"),
                    "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                    "processor": _processor_cache_identity(getattr(self.pipe, "processor", "")),
                },
            )
        # All ranks must enter or skip the frozen encoder's FSDP collectives.
        if cache_on and all_ranks_hit(self.condition_cache.contains(key), self.condition_cache_group):
            self._condition_metrics["global_hits"] += 1
            value = self.condition_cache.get(key, device=self.device)
            return value["embeds"], value["mask"]
        self._condition_metrics["global_misses"] += int(cache_on)
        self._condition_metrics["prefix_encodes"] += 1
        context = (
            torch.no_grad() if not any(p.requires_grad for p in self.text_encoder.parameters()) else nullcontext()
        )
        with context:
            embeds, mask = self.pipe.encode_prompt(
                prompt=data["prompt"],
                image=data["edit_image"][0],
                device=self.device,
            )
        if cache_on:
            self.condition_cache.put(
                key, {"embeds": embeds.detach(), "mask": mask.detach() if mask is not None else None}
            )
        return embeds, mask

    def _image_latents(self, image: Any) -> torch.Tensor:
        pixels = self.pipe.image_processor.preprocess(image, height=image.height, width=image.width)
        pixels = pixels.unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)
        # Native Edit pipeline uses argmax/mode for image latents. No stochastic
        # VAE or training noise is cached or skipped on a condition-cache hit.
        return self.pipe._encode_vae_image(pixels, generator=None)

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        embeds, mask = self.encode_conditions(data)
        context = torch.no_grad() if not any(p.requires_grad for p in self.vae.parameters()) else nullcontext()
        with context:
            clean = self._image_latents(data["image"])
            condition = self._image_latents(data["edit_image"][0])
        # Rectified flow, uniform sigma training, target velocity epsilon-x0.
        # Same exact sampling position in A/B; no sampling in cache callbacks.
        noise = torch.randn_like(clean)
        sigma = torch.rand((clean.shape[0],), device=clean.device, dtype=torch.float32)
        expanded = sigma.reshape(-1, 1, 1, 1, 1)
        noisy = ((1 - expanded) * clean.float() + expanded * noise.float()).to(clean.dtype)
        pack = self.pipe._pack_latents
        batch, channels, _, height, width = clean.shape
        noisy_packed = pack(noisy.squeeze(2), batch, channels, height, width)
        ch, cw = condition.shape[-2:]
        condition_packed = pack(condition.squeeze(2), batch, channels, ch, cw)
        predicted = self.transformer(
            hidden_states=torch.cat((noisy_packed, condition_packed), dim=1),
            timestep=sigma.to(dtype=noisy_packed.dtype),
            guidance=None,
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=mask,
            img_shapes=[[(1, height // 2, width // 2), (1, ch // 2, cw // 2)]] * batch,
            txt_seq_lens=[embeds.shape[1]] * batch,
            return_dict=False,
        )[0][:, : noisy_packed.shape[1]]
        target = pack((noise - clean).squeeze(2), batch, channels, height, width)
        return torch.nn.functional.mse_loss(predicted.float(), target.float())


class QwenImageEditSFTAdapter:
    family = "qwen_image_edit"
    supported_tasks = ("i2i",)
    lora_task_type = "FEATURE_EXTRACTION"
    lora_target_modules = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    )

    def load_train_model(self, args: Any) -> QwenImageEditSFTModel:
        from diffusers import QwenImageEditPipeline

        if args.fsdp_trainable_mode != "lora":
            raise ValueError("Qwen-Image-Edit SFT supports fsdp_trainable_mode='lora'")
        # Load on CPU first; actor/runner owns movement and FSDP wrapping.
        pipe = QwenImageEditPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        for module in (pipe.transformer, pipe.text_encoder, pipe.vae):
            module.requires_grad_(False)
        pipe.transformer.enable_gradient_checkpointing()
        identity = os.environ.get("RELAX_CONDITION_CACHE_IDENTITY", "")
        if not identity:
            # Verified asset manifest identity is mandatory for cache runs.
            if int(os.environ.get("RELAX_CONDITION_CACHE_BYTES", "0")):
                raise ValueError("A verified model identity is required before enabling condition cache")
            identity = "cache-disabled"
        return QwenImageEditSFTModel(pipe, identity)

    def prepare_for_sharding(self, model: QwenImageEditSFTModel) -> None:
        model.to(dtype=torch.bfloat16)

    def on_device_ready(self, model: QwenImageEditSFTModel, device: torch.device) -> None:
        model.device = device

    def prepare_sample(
        self, row: dict[str, Any], root: Path, config: dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        from relax.models.sensenova.adapter import SenseNovaSFTAdapter

        if len(row["edit_image"]) != 1:
            raise ValueError("Qwen-Image-Edit initial release supports exactly one condition image")
        return {
            "prompt": row["prompt"],
            "image": SenseNovaSFTAdapter.load_image(root / row["image"], config["max_pixels"]),
            "edit_image": [SenseNovaSFTAdapter.load_image(root / row["edit_image"][0], config["max_pixels"])],
        }
