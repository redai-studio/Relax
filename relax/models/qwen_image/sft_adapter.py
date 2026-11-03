# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qwen-Image LoRA SFT with deterministic frozen conditioning reuse.

The public SFT trainer owns optimization, sharding and cache storage. This
adapter owns only model math and effective conditioning dependencies. Target
VAE sampling, diffusion noise and timesteps always execute online.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from relax.models.generative_sft import ConditioningResolver, ConditioningSpec
from relax.models.qwen_image.adapter import QwenImageAdapter


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack the singleton frame into the existing Qwen 2x2 DiT geometry."""
    batch, channels, frames, height, width = latents.shape
    if frames != 1 or height % 2 or width % 2:
        raise ValueError("Qwen-Image SFT requires one frame and even latent dimensions")
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // 4, channels * 4)
    )


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Restore the official spatial layout before reducing the SFT loss."""
    batch, _, channels = latents.shape
    return (
        latents.reshape(batch, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, 1, height, width)
    )


def flow_matching_loss(
    transformer: torch.nn.Module,
    clean: torch.Tensor,
    prompt_embeds: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Uniformly weighted rectified-flow objective in the official layout."""
    noisy = (1 - sigma) * clean + sigma * noise
    prediction = transformer(
        hidden_states=pack_latents(noisy),
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=None,  # Singleton unpadded text, as in encode_prompt.
        timestep=timestep / 1000,
        img_shapes=[[(1, clean.shape[-2] // 2, clean.shape[-1] // 2)]],
        return_dict=False,
    )[0]
    prediction = unpack_latents(prediction, clean.shape[-2], clean.shape[-1])
    # Match the reference's spatial FP32 reduction order, not just the
    # mathematically equivalent mean over the packed token/channel layout.
    return (prediction.float() - (noise - clean).float()).square().reshape(clean.shape[0], -1).mean(1).mean()


class QwenImageSFTModel(torch.nn.Module):
    """All parameter use stays inside one FSDP root forward.

    The frozen text encoder is a separate sharding boundary: a cache hit skips
    its collectives as well as its compute. The shared resolver makes that
    decision unanimously across the FSDP group.
    """

    _no_split_modules = ("QwenImageTransformerBlock",)

    def __init__(self, pipeline: Any, config: dict[str, Any], dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.transformer = pipeline.transformer
        self.text_encoder = pipeline.text_encoder
        self.vae = pipeline.vae
        self.pipe = pipeline  # DiffusionPipeline is not an nn.Module.
        self.config = dict(config)
        self.device = torch.device("cpu")
        self.compute_dtype = dtype
        self.conditioning_cache: ConditioningResolver | None = None
        self.conditioning_reuse_disabled = False
        self.conditioning_reuse_disabled_reason: str | None = None
        self.text_encoder.eval().requires_grad_(False)
        self.vae.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> QwenImageSFTModel:
        super().train(mode)
        # Freezing parameters alone does not disable dropout. Even when the
        # trainable DiT enters train(), frozen conditioning must stay in eval.
        self.text_encoder.eval()
        self.vae.eval()
        return self

    def conditioning_is_frozen(self) -> bool:
        return not self.text_encoder.training and not any(p.requires_grad for p in self.text_encoder.parameters())

    def conditioning_state(self) -> dict[str, Any]:
        parameters = {}
        tensors = list(self.text_encoder.named_parameters())
        tensors.extend((f"text_encoder.buffers.{name}", value) for name, value in self.text_encoder.named_buffers())
        for name, parameter in tensors:
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            parameters[name] = {
                "identity": id(parameter),
                "version": parameter._version,
                "local_version": local._version,
                "requires_grad": parameter.requires_grad,
                "dtype": str(parameter.dtype),
                "shape": tuple(parameter.shape),
                "stride": tuple(parameter.stride()),
            }
        return {"frozen": self.conditioning_is_frozen(), "parameters": parameters}

    def _prepare_conditioning(self, prompt: str) -> dict[str, torch.Tensor]:
        # Keep the official 1024-token context even when only the first 512
        # hidden states are returned. Truncating encoder input earlier changes
        # the conditioning contract for long prompts.
        tokens = self.pipe.tokenizer(
            [self.pipe.prompt_template_encode.format(prompt)],
            max_length=self.pipe.tokenizer_max_length + self.pipe.prompt_template_encode_start_idx,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        prepared = {name: tokens[name] for name in ("input_ids", "attention_mask")}
        if any(value.device.type != "cpu" for value in prepared.values()):
            raise ValueError("Conditioning dependencies must be prepared on CPU")
        if prepared["input_ids"].shape[0] != 1 or not bool(prepared["attention_mask"].all()):
            raise ValueError("Qwen-Image SFT encodes one unpadded caption at a time")
        return prepared

    def conditioning_spec(
        self,
        data: dict[str, Any],
        *,
        prepared: dict[str, torch.Tensor] | None = None,
        enabled: bool = True,
    ) -> ConditioningSpec:
        cacheable = self.conditioning_is_frozen()
        dependencies: dict[str, Any] = {}
        if enabled and cacheable:
            import json

            prepared = self._prepare_conditioning(data["prompt"]) if prepared is None else prepared
            dependencies = {
                "input_ids": prepared["input_ids"].tolist(),
                "attention_mask": prepared["attention_mask"].tolist(),
                "template": self.pipe.prompt_template_encode,
                "drop_prefix_tokens": self.pipe.prompt_template_encode_start_idx,
                "encoder_context_length": self.pipe.tokenizer_max_length,
                "max_sequence_length": self.config["max_sequence_length"],
                "dtype": str(self.compute_dtype),
                "encoder_class": type(self.text_encoder).__name__,
                # HF configs can contain integer id2label keys. Canonical JSON
                # keeps the full configuration without weakening the public
                # cache's string-key-only dependency contract.
                "encoder_config": json.dumps(self.text_encoder.config.to_dict(), sort_keys=True, allow_nan=False),
                "attention_implementation": self.text_encoder.config._attn_implementation,
                "deterministic_attention": self.config.get("deterministic_attention", False),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "hidden_layer": -1,
            }
        # Checkpoint content identity and mutation guards are supplied by the
        # public runtime; filenames and image identities never authorize reuse.
        return ConditioningSpec("qwen_image.text.v1", dependencies, cacheable)

    def conditioning_prefetch_spec(self, data: dict[str, Any]) -> ConditioningSpec:
        return self.conditioning_spec(data)

    def _restore_conditioning(
        self, features: dict[str, torch.Tensor], sequence_length: int
    ) -> dict[str, torch.Tensor]:
        if set(features) != {"prompt_embeds"}:
            raise ValueError("Cached Qwen-Image conditioning fields differ from the model")
        embeds = features["prompt_embeds"]
        expected = (1, sequence_length, self.transformer.config.joint_attention_dim)
        if embeds.shape != expected or embeds.dtype != self.compute_dtype or embeds.device != self.device:
            raise ValueError("Cached Qwen-Image text embedding has an invalid shape, dtype or device")
        return features

    def _condition(self, data: dict[str, Any]) -> torch.Tensor:
        return self._conditioning_features(data)["prompt_embeds"]

    def _encode_posterior(self, pixels: torch.Tensor) -> Any:
        try:
            return self.vae.encode(pixels.to(device=self.device, dtype=self.compute_dtype)).latent_dist
        except BaseException:
            # Qwen's successful encode clears its temporal convolution caches,
            # but an exception can bypass that cleanup. Do not retain GPU views.
            self.vae.clear_cache()
            raise

    def _conditioning_features(self, data: dict[str, Any]) -> dict[str, Any]:
        import time

        placement = self.config.get("text_encoder_placement", "sharded")
        if placement != getattr(self, "_text_encoder_placement", placement):
            raise ValueError("Changing text_encoder_placement requires a new model and FSDP ownership")
        if placement == "replicated" and any(parameter.requires_grad for parameter in self.text_encoder.parameters()):
            raise ValueError(
                "Replicated Qwen conditioning is frozen-only; training its parameters requires a new model"
            )
        started = time.perf_counter()
        prepared = self._prepare_conditioning(data["prompt"])
        resolver = self.conditioning_cache
        spec = self.conditioning_spec(data, prepared=prepared, enabled=bool(resolver and resolver.enabled))
        start = self.pipe.prompt_template_encode_start_idx
        length = min(prepared["input_ids"].shape[1] - start, self.config["max_sequence_length"])
        if length <= 0:
            raise ValueError("Prompt template produced no conditioning tokens")
        if resolver is not None and hasattr(resolver, "record_prepare"):
            resolver.record_prepare(time.perf_counter() - started)

        def compute() -> dict[str, torch.Tensor]:
            if resolver is not None:
                resolver.record_input_transfer(
                    sum(value.numel() * value.element_size() for value in prepared.values())
                )
            with torch.no_grad() if spec.cacheable else nullcontext():
                if resolver is not None and getattr(resolver, "force_online", False):
                    # Independent, original Diffusers path for verification.
                    embeds, mask = self.pipe.encode_prompt(
                        prompt=data["prompt"],
                        device=self.device,
                        max_sequence_length=self.config["max_sequence_length"],
                    )
                    if mask is not None:
                        raise ValueError("Singleton reference prompt unexpectedly contains padding")
                else:
                    output = self.text_encoder(
                        **{name: value.to(self.device) for name, value in prepared.items()},
                        output_hidden_states=True,
                        use_cache=False,
                        logits_to_keep=1,  # Hidden states do not need vocabulary logits for every token.
                    )
                    embeds = output.hidden_states[-1][:, start : start + length].to(self.compute_dtype)
            return {"prompt_embeds": embeds.contiguous()}

        def restore(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return self._restore_conditioning(features, length)

        return restore(compute()) if resolver is None else resolver.resolve(spec, compute, restore)

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        import numpy as np

        features = self._conditioning_features(data)
        prompt_embeds = features["prompt_embeds"]
        with torch.no_grad():
            pixels = torch.from_numpy(np.array(data["image"], dtype=np.float32)).permute(2, 0, 1)
            pixels = pixels.div(127.5).sub(1).unsqueeze(0).unsqueeze(2)
            posterior = self._encode_posterior(pixels)
            # Always sample here, on the original RNG stream, for every micro.
            clean = posterior.sample()
            del posterior
            # The official training recipe normalizes in FP32 and only then
            # rounds to the training dtype. BF16 statistics change the target.
            mean = clean.new_tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
            inv_std = clean.new_tensor(self.vae.config.latents_std, dtype=torch.float32).reciprocal()
            clean = ((clean - mean) * inv_std.view(1, -1, 1, 1, 1)).to(self.compute_dtype)
        noise = torch.randn_like(clean)
        # Same uniform density and scheduler lookup as Diffusers' SFT recipe.
        index = (torch.rand((1,), device="cpu") * self.pipe.scheduler.config.num_train_timesteps).long()
        # Divide the FP32 scheduler timestep before the DiT casts it. Rounding
        # t to BF16 before t/1000 changes the model's time embedding.
        timestep = self.pipe.scheduler.timesteps[index].to(device=self.device)
        sigma = self.pipe.scheduler.sigmas[index].to(device=self.device, dtype=self.compute_dtype).view(1, 1, 1, 1, 1)
        return flow_matching_loss(self.transformer, clean, prompt_embeds, noise, sigma, timestep)


class QwenImageSFTAdapter:
    family = "qwen_image"
    supported_tasks = ("t2i",)
    lora_target_modules = QwenImageAdapter.lora_target_modules
    lora_task_type = QwenImageAdapter.lora_task_type

    def load_train_model(self, args: Any) -> QwenImageSFTModel:
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

        from relax.engine.sft.runtime import get_image_sft_config

        config = get_image_sft_config(args)
        if args.fsdp_trainable_mode != "lora":
            raise ValueError("Qwen-Image SFT currently supports DiT LoRA training")
        if config.get("deterministic_attention", False):
            import os

            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.fsdp_param_dtype]
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_path, subfolder="scheduler", shift=3.0)
        pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=dtype, scheduler=scheduler)
        pipe.to(device)
        pipe.text_encoder.set_attn_implementation("sdpa")
        pipe.transformer.requires_grad_(False)
        if config.get("gradient_checkpointing", True):
            pipe.transformer.enable_gradient_checkpointing()
        model = QwenImageSFTModel(pipe, config, dtype=dtype)
        model.device = device
        model.train()
        return model

    def prepare_for_sharding(self, model: QwenImageSFTModel) -> None:
        if any(p.requires_grad for p in model.text_encoder.parameters()) or any(
            p.requires_grad for p in model.vae.parameters()
        ):
            raise ValueError("Qwen-Image SFT LoRA targets must be confined to the DiT")
        placement = model.config.get("text_encoder_placement", "sharded")
        if placement not in ("sharded", "replicated"):
            raise ValueError("text_encoder_placement must be sharded or replicated")
        replicas = []
        if placement == "replicated":
            replicas.append(("text encoder", model.text_encoder))
        for name, module in replicas:
            # The native ignored-parameter policy keeps normal module ownership,
            # state names and mutation guards. It does not move/cast parameters.
            if any(
                parameter.device != model.device or parameter.dtype != model.compute_dtype
                for parameter in module.parameters()
            ):
                raise ValueError(f"Replicated {name} must already use the model device and compute dtype")
            if module.training:
                raise ValueError(f"Replicated {name} must be in eval mode")
        # Gather a sharded encoder once per online encode, without per-layer boundaries.
        model.text_encoder._fsdp_fine_grained = placement == "sharded"
        model._fsdp_replicated_modules = tuple(module for _, module in replicas)
        model._text_encoder_placement = placement

    def on_device_ready(self, model: QwenImageSFTModel, device: torch.device) -> None:
        model.device = device

    def prepare_sample(
        self, row: dict[str, Any], root: Path, config: dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        from PIL import Image, ImageOps
        from torchvision.transforms import InterpolationMode, functional

        with Image.open(root / row["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        height, width = config["height"], config["width"]
        scale = max(height / image.height, width / image.width)
        image = functional.resize(
            image,
            [max(height, round(image.height * scale)), max(width, round(image.width * scale))],
            interpolation=InterpolationMode.BILINEAR,
        )
        if config.get("random_crop", False):
            top = int(torch.randint(image.height - height + 1, (1,)))
            left = int(torch.randint(image.width - width + 1, (1,)))
            image = functional.crop(image, top, left, height, width)
        else:
            image = functional.center_crop(image, (height, width))
        prompt = row["prompt"]
        dropout = config.get("caption_dropout", 0)
        if dropout > 0 and bool(torch.rand(()) < dropout):
            prompt = ""
        return {"prompt": prompt, "image": image}

    @staticmethod
    def export_checkpoint(checkpoint_dir: Path, output: Path, contract: dict[str, Any]) -> None:
        import json

        from diffusers import QwenImagePipeline

        from relax.backends.fsdp.checkpoint import _load_dcp_model_state

        if torch.distributed.is_initialized():
            raise RuntimeError("Export in a separate CPU process after checkpoint commit")
        if not (checkpoint_dir / "COMMITTED").is_file() or output.exists():
            raise ValueError("Export requires a committed checkpoint and a new output directory")
        if json.loads((checkpoint_dir / "adapter_contract.json").read_text()) != contract:
            raise ValueError("Export contract differs from the committed checkpoint")
        rank = contract["lora"]["rank"]
        alpha = contract["lora"]["alpha"]
        if (
            contract["trainable_mode"] != "lora"
            or type(rank) is not int
            or rank <= 0
            or type(alpha) not in (int, float)
            or alpha != rank
        ):
            raise ValueError("Qwen-Image export requires DiT LoRA with alpha == rank")
        state = _load_dcp_model_state(str(checkpoint_dir / "fsdp"))
        if not state:
            raise ValueError("Qwen-Image checkpoint has no LoRA parameters")
        pairs: dict[str, set[str]] = {}
        for name, tensor in state.items():
            kind = next((part for part in ("lora_A", "lora_B") if name.endswith(f".{part}.default.weight")), None)
            if kind is None or not name.startswith("transformer.") or tensor.ndim != 2:
                raise ValueError(f"Unexpected Qwen-Image adapter tensor: {name}")
            if tensor.shape[0 if kind == "lora_A" else 1] != rank:
                raise ValueError(f"Adapter rank mismatch: {name}")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Adapter weights must be finite floating-point tensors: {name}")
            pairs.setdefault(name.rsplit(f".{kind}.", 1)[0], set()).add(kind)
        if any(kinds != {"lora_A", "lora_B"} for kinds in pairs.values()):
            raise ValueError("Incomplete Qwen-Image LoRA A/B pair")
        QwenImagePipeline.save_lora_weights(
            str(output),
            transformer_lora_layers={
                name.removeprefix("transformer.").replace(".default.", "."): tensor.contiguous()
                for name, tensor in state.items()
            },
        )
        (output / "relax_adapter_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
