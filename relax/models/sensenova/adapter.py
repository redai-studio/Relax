# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax-native SenseNova model, data processing and SFT objective.

Import this module only inside training workers. No DiffSynth training module,
Accelerator, or DeepSpeed training loop is used here.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from relax.models.frozen_conditions import FrozenConditionCache, image_condition_key


LORA_TARGETS = (
    "q_proj_mot_gen",
    "k_proj_mot_gen",
    "v_proj_mot_gen",
    "o_proj_mot_gen",
    "mlp_mot_gen.gate_proj",
    "mlp_mot_gen.up_proj",
    "mlp_mot_gen.down_proj",
)


def configure_attention(implementation: str, deterministic: bool = False) -> None:
    """Pin the process-local kernel; optional deterministic backward for
    parity."""
    import os

    if implementation not in ("flash_attention_2", "torch"):
        raise ValueError("Unsupported SenseNova attention implementation")
    os.environ["RELAX_ATTENTION_IMPLEMENTATION"] = implementation
    os.environ["RELAX_DETERMINISTIC_ATTENTION"] = "1" if deterministic else "0"
    if deterministic:
        # Masked prefix attention uses Torch SDPA, and vision uses cuDNN.
        # FA2's flag alone does not make full-parameter backward repeatable.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flow_matching_loss(
    pipe: Any,
    inputs: dict[str, Any],
    *,
    timestep_id: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match FlowMatchSFTLoss; injected noise is already noise-scale
    adjusted."""
    if timestep_id is None:
        timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,))
    # Keep scheduler lookup on CPU. The model sees the same quantized timestep,
    # without copying a CUDA scalar back to the host three times per forward.
    scheduler_timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype)
    timestep = scheduler_timestep.to(device=pipe.device)
    clean = inputs["input_latents"]
    if noise is None:
        noise = torch.randn_like(clean) * inputs.get("noise_scale", 1.0)
    noisy = pipe.scheduler.add_noise(clean, noise, scheduler_timestep)
    target = pipe.scheduler.training_target(clean, noise, scheduler_timestep)
    prediction = pipe.model_fn(dit=pipe.dit, **{**inputs, "latents": noisy}, timestep=timestep)
    return torch.nn.functional.mse_loss(prediction.float(), target.float()) * pipe.scheduler.training_weight(
        scheduler_timestep
    )


class SenseNovaSFTModel(torch.nn.Module):
    """Root FSDP forward encompasses BOTH prefix and generation parameter use.

    Wrapping just dit.forward would leave embedding/extract/encode_prefix calls
    outside the root all-gather hooks. Decoder layers are sharded separately;
    generation checkpointing stays in the reference model (cache updates off).
    """

    _no_split_modules = ("SenseNovaU1DecoderLayer",)

    def __init__(self, pipe: Any, training_mode: str = "lora", prefix_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.pipe = pipe
        self.training_mode = training_mode
        self.prefix_gradient_checkpointing = prefix_gradient_checkpointing
        self.condition_cache: FrozenConditionCache | None = None
        self.prefix_frozen = training_mode == "lora"
        if training_mode == "full":
            self._mark_fine_grained_modules()

    def _mark_fine_grained_modules(self) -> None:
        from .model import SenseNovaU1Attention, SenseNovaU1ForCausalLM, SenseNovaU1MLP

        for module in self.pipe.modules():
            if isinstance(module, SenseNovaU1Attention):
                projections = (
                    module.q_proj,
                    module.q_proj_mot_gen,
                    module.o_proj,
                    module.o_proj_mot_gen,
                )
                for projection in projections:
                    projection._fsdp_fine_grained = True
            elif isinstance(module, SenseNovaU1MLP):
                for projection in (module.gate_proj, module.up_proj, module.down_proj):
                    projection._fsdp_fine_grained = True
            elif isinstance(module, SenseNovaU1ForCausalLM):
                # These two tensors are about 1.16 GiB each in U1. Leaving
                # them on the root keeps both all-gathered across the decoder.
                module.model.embed_tokens._fsdp_fine_grained = True
                module.lm_head._fsdp_fine_grained = True

    def condition_key(self, data: dict[str, Any]) -> bytes:
        return image_condition_key(data, version="sensenova-prefix-v1", dtype=self.pipe.torch_dtype)

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        from .pipeline import SenseNovaU1ImageUnit_EditImageEmbedder, SenseNovaU1ImageUnit_PromptEmbedder

        image = data["image"]
        shared = {
            "input_image": image,
            "edit_image": data["edit_image"],
            "height": image.height,
            "width": image.width,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.prefix_gradient_checkpointing,
            "use_gradient_checkpointing_offload": False,
        }
        inputs = (shared, {"prompt": data["prompt"], "is_positive": True}, {"prompt": "", "is_positive": False})
        request = data.get("condition_request")
        cached = None if request is None else request.tensors
        # Custom prefix LoRA must retain gradients, even in LoRA training mode.
        with torch.no_grad() if self.prefix_frozen else nullcontext():
            for unit in self.pipe.units:
                if cached is not None and isinstance(unit, SenseNovaU1ImageUnit_EditImageEmbedder):
                    continue
                if cached is not None and isinstance(unit, SenseNovaU1ImageUnit_PromptEmbedder):
                    from transformers.cache_utils import DynamicCache

                    prefix = DynamicCache()
                    for layer in range((len(cached) - 1) // 2):
                        prefix.update(cached[f"key.{layer}"], cached[f"value.{layer}"], layer)
                    inputs[1].update(past_key_values=prefix, indexes_image=cached["indexes_image"])
                    self.condition_cache.hits += 1
                    continue
                with torch.profiler.record_function(type(unit).__name__):
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                if request is not None and isinstance(unit, SenseNovaU1ImageUnit_PromptEmbedder):
                    tensors = {"indexes_image": inputs[1]["indexes_image"]}
                    for index, layer in enumerate(inputs[1]["past_key_values"].layers):
                        tensors[f"key.{index}"] = layer.keys
                        tensors[f"value.{index}"] = layer.values
                    self.condition_cache.encodes += 1
                    self.condition_cache.put(request.key, tensors)
        return flow_matching_loss(self.pipe, {**inputs[0], **inputs[1]})


class SenseNovaSFTAdapter:
    """SenseNova model policy for the shared image-edit SFT trainer."""

    family = "sensenova"
    supported_tasks = ("i2i",)
    lora_target_modules = LORA_TARGETS
    lora_task_type = "FEATURE_EXTRACTION"

    def load_train_model(self, args: Any) -> SenseNovaSFTModel:
        from .pipeline import SenseNovaU1ImagePipeline

        config = args.i2i_sft_config
        configure_attention(
            config.get("attention_implementation", "flash_attention_2"), config.get("deterministic_attention", False)
        )
        mode = args.fsdp_trainable_mode
        device = torch.device("cpu") if mode == "full" else torch.device("cuda", torch.cuda.current_device())
        pipe = SenseNovaU1ImagePipeline.from_pretrained(
            model_path=args.model_path,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        pipe.scheduler.set_timesteps(1000, training=True)
        pipe.freeze_except(["dit"] if mode == "full" else [])
        return SenseNovaSFTModel(
            pipe,
            training_mode=mode,
            prefix_gradient_checkpointing=config.get("prefix_gradient_checkpointing", True),
        )

    def prepare_for_sharding(self, model: SenseNovaSFTModel) -> None:
        # Match the reference BF16 adapter initialization before FP32 masters.
        model.pipe.to(dtype=torch.bfloat16)
        model.prefix_frozen = not any(
            parameter.requires_grad
            for name, parameter in model.pipe.dit.named_parameters(remove_duplicate=False)
            if not name.startswith(("fm_modules.", "language_model.lm_head."))
            and not any(part.endswith("_mot_gen") for part in name.split("."))
        )

    def validate_condition_cache(self, model: SenseNovaSFTModel) -> None:
        if not model.prefix_frozen:
            raise ValueError("SenseNova condition reuse requires frozen vision and understanding parameters")

    def on_device_ready(self, model: SenseNovaSFTModel, device: torch.device) -> None:
        # Never move the already sharded parameters through Module.to().
        model.pipe.device = device

    def prepare_sample(
        self, row: dict[str, Any], root: Path, config: dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        from relax.engine.sft.image import image_tier_budget

        budget = image_tier_budget(config, index)
        return {
            "prompt": row["prompt"],
            "image": self.load_image(root / row["image"], config["max_pixels"], fixed_pixels=budget),
            "edit_image": [
                self.load_image(root / path, config["max_pixels"], fixed_pixels=budget) for path in row["edit_image"]
            ],
        }

    @staticmethod
    def export_checkpoint(checkpoint_dir: Path, output: Path, contract: dict[str, Any]) -> None:
        from .export import export_adapter, export_full

        if contract["trainable_mode"] == "lora":
            export_adapter(checkpoint_dir, output)
        else:
            export_full(checkpoint_dir, Path(contract["model_path"]), output)

    @staticmethod
    def load_image(path: Path, max_pixels: int, fixed_pixels: int | None = None) -> Any:
        from relax.engine.sft.image import load_sft_image

        from .processing import PATCH_SIZE

        return load_sft_image(path, max_pixels, patch_size=PATCH_SIZE, fixed_pixels=fixed_pixels)
