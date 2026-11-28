# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qwen-Image-Edit supervision through Relax's image SFT trainer.

Conditioning and reference latent geometry follow Diffusers v0.37.0's
QwenImageEditPipeline. Target posterior sampling and the rectified-flow
objective follow its Qwen-Image DreamBooth training recipe (uniform timestep
density, unweighted loss). Diffusers remains the model implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from relax.models.frozen_conditions import FrozenConditionCache, image_condition_key
from relax.models.qwen_image.adapter import QwenImageAdapter


class QwenImageEditSFTModel(torch.nn.Module):
    """Keep every parameter access inside the FSDP forward root."""

    _no_split_modules = ("QwenImageTransformerBlock",)

    def __init__(self, pipeline: Any, dtype: torch.dtype) -> None:
        super().__init__()
        # DiffusionPipeline is not an nn.Module: explicitly register each
        # component once. The pipeline keeps references to these same objects.
        self.transformer = pipeline.transformer
        self.text_encoder = pipeline.text_encoder
        self.vae = pipeline.vae
        self.pipeline = pipeline
        self.compute_dtype = dtype
        self.device = torch.device("cpu")
        self.condition_cache: FrozenConditionCache | None = None

    def train(self, mode: bool = True) -> QwenImageEditSFTModel:
        super().train(mode)
        self.text_encoder.eval()
        self.vae.eval()
        return self

    def encode_condition(self, data: dict[str, Any]) -> torch.Tensor:
        # One unpadded sample has no masked tokens. Extracting its hidden
        # states directly avoids Diffusers' GPU -> CPU length/mask checks.
        inputs = self.pipeline.processor(
            text=[self.pipeline.prompt_template_encode.format(data["prompt"])],
            images=data["edit_image"][0],
            padding=False,
            return_tensors="pt",
        )
        output = self.text_encoder(
            **inputs.to(self.device), output_hidden_states=True, use_cache=False, logits_to_keep=1
        )
        start = self.pipeline.prompt_template_encode_start_idx
        return output.hidden_states[-1][:, start:].to(self.compute_dtype).contiguous()

    def condition_key(self, data: dict[str, Any]) -> bytes:
        return image_condition_key(data, version="qwen-image-edit-condition-v1", dtype=self.compute_dtype)

    def image_latents(self, image: Any, *, sample: bool) -> torch.Tensor:
        pixels = self.pipeline.image_processor.preprocess(image, height=image.height, width=image.width)
        pixels = pixels.unsqueeze(2).to(device=self.device, dtype=self.compute_dtype)
        try:
            posterior = self.vae.encode(pixels).latent_dist
        finally:
            # The temporal VAE owns intermediate feature caches, including
            # when encode raises before its normal cleanup.
            self.vae.clear_cache()
        latents = posterior.sample() if sample else posterior.mode()
        # Training targets use the official DreamBooth FP32 normalization;
        # reference conditioning follows the edit pipeline's latent dtype.
        dtype = torch.float32 if sample else latents.dtype
        mean = latents.new_tensor(self.vae.config.latents_mean, dtype=dtype).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.vae.config.latents_std, dtype=dtype).view(1, -1, 1, 1, 1)
        return ((latents.to(dtype) - mean) / std).to(self.compute_dtype)

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        with torch.no_grad():
            request = data.get("condition_request")
            cached = None if request is None else request.tensors
            if cached is None:
                with torch.profiler.record_function("QwenImageEditPromptEncoder"):
                    embeddings = self.encode_condition(data)
            else:
                embeddings = cached["embeddings"]
                self.condition_cache.hits += 1
            # Keep posterior sampling in the same position in the RNG stream.
            with torch.profiler.record_function("QwenImageEditTargetVAE"):
                clean = self.image_latents(data["image"], sample=True)
            if cached is None:
                with torch.profiler.record_function("QwenImageEditReferenceVAE"):
                    reference = self.image_latents(data["edit_image"][0], sample=False)
            else:
                reference = cached["reference"]
            if request is not None and cached is None:
                self.condition_cache.encodes += 1
                self.condition_cache.put(request.key, {"embeddings": embeddings, "reference": reference})
        noise = torch.randn_like(clean)
        scheduler = self.pipeline.scheduler
        index = (torch.rand((clean.shape[0],)) * scheduler.config.num_train_timesteps).long()
        timestep = scheduler.timesteps[index].to(self.device)
        sigma = scheduler.sigmas[index].to(device=self.device, dtype=clean.dtype).view(-1, 1, 1, 1, 1)
        noisy = (1 - sigma) * clean + sigma * noise
        batch, channels, _, height, width = clean.shape
        ref_height, ref_width = reference.shape[-2:]
        pack = self.pipeline._pack_latents
        target_tokens = pack(noisy, batch, channels, height, width)
        reference_tokens = pack(reference, batch, channels, ref_height, ref_width)
        prediction = self.transformer(
            hidden_states=torch.cat((target_tokens, reference_tokens), dim=1),
            encoder_hidden_states=embeddings,
            encoder_hidden_states_mask=None,
            timestep=timestep / 1000,
            guidance=None,
            img_shapes=[[(1, height // 2, width // 2), (1, ref_height // 2, ref_width // 2)]] * batch,
            return_dict=False,
        )[0][:, : target_tokens.shape[1]]
        # Compare in spatial layout, matching the reference FP32 reduction.
        prediction = self.pipeline._unpack_latents(
            prediction, data["image"].height, data["image"].width, self.pipeline.vae_scale_factor
        )
        return torch.nn.functional.mse_loss(prediction.float(), (noise - clean).float())


class QwenImageEditSFTAdapter:
    family = "qwen_image_edit"
    supported_tasks = ("i2i",)
    lora_target_modules = QwenImageAdapter.lora_target_modules
    lora_task_type = QwenImageAdapter.lora_task_type

    def load_train_model(self, args: Any) -> QwenImageEditSFTModel:
        from diffusers import QwenImageEditPipeline

        config = args.i2i_sft_config
        if args.fsdp_trainable_mode != "lora" or config["conditioning_images"] != 1:
            raise ValueError("Qwen-Image-Edit SFT requires DiT LoRA and one reference image")
        if config.get("fixed_pixels_tiers"):
            raise ValueError("Qwen-Image-Edit SFT uses max_pixels, not SenseNova fixed pixel tiers")
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.fsdp_param_dtype]
        pipeline = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=True)
        for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
            module.requires_grad_(False)
        pipeline.text_encoder.set_attn_implementation(
            "sdpa" if config["attention_implementation"] == "torch" else "flash_attention_2"
        )
        if config["deterministic_attention"]:
            import os

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            pipeline.transformer.set_attention_backend("native")
        if config["prefix_gradient_checkpointing"]:
            pipeline.transformer.enable_gradient_checkpointing()
        model = QwenImageEditSFTModel(pipeline, dtype)
        model.train()
        return model

    def prepare_for_sharding(self, model: QwenImageEditSFTModel) -> None:
        self.validate_condition_cache(model)
        # Gather the text encoder once per encode. VAE.encode is called
        # directly, so its parameters must stay on the FSDP root.
        model.text_encoder._fsdp_fine_grained = True

    def on_device_ready(self, model: QwenImageEditSFTModel, device: torch.device) -> None:
        model.device = device

    def validate_condition_cache(self, model: QwenImageEditSFTModel) -> None:
        if any(p.requires_grad for module in (model.text_encoder, model.vae) for p in module.parameters()):
            raise ValueError("Qwen condition reuse requires frozen text encoder and VAE")

    @staticmethod
    def export_checkpoint(checkpoint_dir: Path, output: Path, contract: dict[str, Any]) -> None:
        from diffusers import QwenImageEditPipeline
        from peft import LoraConfig

        from relax.backends.fsdp.checkpoint import _load_dcp_model_state
        from relax.backends.fsdp.lora import peft_adapter_state_dict

        if not (checkpoint_dir / "COMMITTED").is_file() or contract["trainable_mode"] != "lora":
            raise ValueError("Qwen-Image-Edit export requires a committed LoRA checkpoint")
        if output.exists():
            raise FileExistsError(output)
        state = _load_dcp_model_state(str(checkpoint_dir / "fsdp"))
        renamed = peft_adapter_state_dict(list(state.items()))
        prefix = "base_model.model.transformer."
        if not state or len(renamed) != len(state) or any(not key.startswith(prefix) for key in renamed):
            raise ValueError("Checkpoint must contain only transformer LoRA parameters")
        lora = contract["lora"]
        QwenImageEditPipeline.save_lora_weights(
            str(output),
            transformer_lora_layers={key.removeprefix(prefix): value.contiguous() for key, value in renamed.items()},
            transformer_lora_adapter_metadata=LoraConfig(
                r=lora["rank"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"],
                bias=lora["bias"],
            ).to_dict(),
        )

    def prepare_sample(
        self, row: dict[str, Any], root: Path, config: dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        from relax.engine.sft.image import load_sft_image

        return {
            "prompt": row["prompt"],
            "image": load_sft_image(root / row["image"], config["max_pixels"], patch_size=32),
            "edit_image": [load_sft_image(root / row["edit_image"][0], config["max_pixels"], patch_size=32)],
        }
