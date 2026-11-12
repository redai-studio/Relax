# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Relax-native SenseNova model, data processing and SFT objective.

Import this module only inside training workers. No DiffSynth training module,
Accelerator, or DeepSpeed training loop is used here.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from relax.engine.sft.condition_cache import ConditionFieldSpec, ConditionRequest, condition_key


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
        if implementation != "flash_attention_2":
            raise ValueError("Deterministic attention validation currently requires FlashAttention 2")
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
        self._condition_cache = None
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
        # Only generation projections have adapters. Prefix/vision are frozen;
        # never checkpoint the mutable prefix cache in this LoRA-only path.
        with torch.no_grad() if self.training_mode == "lora" else nullcontext():
            for unit in self.pipe.units:
                cache = self._condition_cache
                condition = cache.values.get("prefix_kv") if cache is not None else None
                if condition is not None and isinstance(unit, SenseNovaU1ImageUnit_EditImageEmbedder):
                    continue
                if condition is not None and isinstance(unit, SenseNovaU1ImageUnit_PromptEmbedder):
                    from transformers.cache_utils import DynamicCache, DynamicLayer

                    started = time.monotonic()
                    # 直接设置层属性，共享缓存中的 K/V Tensor
                    restored = DynamicCache()
                    for key, value in condition["past_key_values"]:
                        layer = DynamicLayer()
                        layer.lazy_initialization(key, value)
                        layer.keys, layer.values = key, value
                        restored.layers.append(layer)
                    fields = {"past_key_values": restored, "indexes_image": condition["indexes_image"]}
                    inputs[1].update(fields)
                    inputs[2].update(fields)
                    cache.counters["restore_ms"] += (time.monotonic() - started) * 1000
                    cache.counters["skipped_prefix_encodes"] += 1
                    cache.counters["skipped_vision_encodes"] += bool(data["edit_image"])
                    continue
                started = time.monotonic() if cache is not None else 0
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                if cache is not None and isinstance(
                    unit, (SenseNovaU1ImageUnit_EditImageEmbedder, SenseNovaU1ImageUnit_PromptEmbedder)
                ):
                    cache.counters["build_host_ms"] += (time.monotonic() - started) * 1000
                if cache is not None and isinstance(unit, SenseNovaU1ImageUnit_PromptEmbedder):
                    prefix = inputs[1]["past_key_values"]
                    cache.put(
                        "prefix_kv",
                        {
                            "past_key_values": tuple((layer.keys, layer.values) for layer in prefix.layers),
                            "indexes_image": inputs[1]["indexes_image"],
                        },
                    )
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
        model = SenseNovaSFTModel(
            pipe,
            training_mode=mode,
            prefix_gradient_checkpointing=config.get("prefix_gradient_checkpointing", True),
        )
        model._condition_config = dict(config)
        model._condition_model_path = str(Path(args.model_path).resolve())
        return model

    def condition_field_specs(self, model: SenseNovaSFTModel) -> tuple[ConditionFieldSpec, ...]:
        root = "pipe.dit.language_model.model"
        paths = ["pipe.dit.vision_model", f"{root}.embed_tokens", f"{root}.norm"]
        layers = model.pipe.dit.language_model.model.layers
        for index in range(len(layers)):
            layer = f"{root}.layers.{index}"
            paths.extend(f"{layer}.{name}" for name in ("input_layernorm", "post_attention_layernorm", "mlp"))
            paths.extend(
                f"{layer}.self_attn.{name}"
                for name in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "q_norm",
                    "q_norm_hw",
                    "k_norm",
                    "k_norm_hw",
                    "rotary_emb",
                    "rotary_emb_hw",
                )
            )
        return (
            ConditionFieldSpec(
                name="prefix_kv",
                fields=("past_key_values", "indexes_image"),
                dependency_paths=tuple(paths),
                input_dependencies=("prompt", "is_positive", "images"),
                preprocessing_dependencies=("preprocessing",),
                max_tensors=2 * len(layers) + 1,
            ),
        )

    def bind_condition_cache(self, model: SenseNovaSFTModel, cache: Any | None) -> None:
        model._condition_cache = cache
        if cache is None:
            model._condition_text_identity = None
            model._condition_adapter_identity = None
            return
        from .processing import IMAGE_PLACEHOLDER, IMG_START_TOKEN, NON_THINK_PREFIX, SYSTEM_MESSAGE_FOR_GEN

        model.pipe.dit.vision_model.embeddings._ensure_rope_cache(torch.device(model.pipe.device))
        tokenizer = model.pipe.tokenizer
        model._condition_adapter_identity = [
            {"module": name, "active": list(module.active_adapters)}
            for name, module in model.named_modules()
            if hasattr(module, "active_adapters")
        ]
        files = {}
        source = Path(model._condition_model_path)
        for name in (*tokenizer.vocab_files_names.values(), "tokenizer_config.json", "special_tokens_map.json"):
            path = source / name
            if path.is_file():
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = {
            "weights": model._condition_model_path,
            "text": {
                "files": files,
                # 完整 tokenizer JSON：词表、merges、normalizer、pre-tokenizer、post-processor 与新增 token
                "backend": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest(),
                "special_tokens": tokenizer.special_tokens_map,
                "added_vocab": tokenizer.get_added_vocab(),
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
                "model_max_length": tokenizer.model_max_length,
                "split_special_tokens": tokenizer.split_special_tokens,
                "system_message": SYSTEM_MESSAGE_FOR_GEN,
                "append_text": NON_THINK_PREFIX + IMG_START_TOKEN,
                "placeholder": IMAGE_PLACEHOLDER,
                "think_mode": False,
            },
            "precision": str(model.pipe.torch_dtype),
            "prompt": "",
            "is_positive": True,
            "images": [],
            "preprocessing": {},
        }
        model._condition_text_identity = condition_key(self.condition_field_specs(model)[0], metadata, 0)

    def prepare_condition_request(self, model: SenseNovaSFTModel, data: dict[str, Any]) -> dict[str, ConditionRequest]:
        config = model._condition_config
        random_fields = (
            "random_crop",
            "random_flip",
            "random_augment",
            "caption_dropout_prob",
            "reasoning",
            "is_noisy_input",
        )
        eligible = model.training_mode == "lora" and not any(
            config.get(name) or data.get(name) for name in random_fields
        )
        if not eligible:
            return {"prefix_kv": ConditionRequest({}, eligible=False)}
        images = []
        for image in data["edit_image"]:
            if image.mode != "RGB":
                raise ValueError("SenseNova condition identity requires RGB images")
            images.append(
                {"mode": image.mode, "size": image.size, "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
            )
        height, width = model.pipe.check_resize_height_width(data["image"].height, data["image"].width)
        dit = model.pipe.dit
        metadata = {
            "weights": {
                "source": model._condition_model_path,
                "active_adapters": model._condition_adapter_identity,
                "adapter_version": model._condition_cache.version_tags["prefix_kv"],
            },
            "prompt": data["prompt"],
            "is_positive": True,
            "images": images,
            "preprocessing": {
                "max_pixels": config["max_pixels"],
                "fixed_pixels_tiers": config.get("fixed_pixels_tiers"),
                "fixed_pixels_budget": data.get("fixed_pixels_budget"),
                "min_pixels": 512 * 512,
                "max_condition_pixels": min(2048 * 2048, (4096 * 4096) // len(images)),
                "upscale": False,
                "patch_size": dit.patch_size,
                "downsample_ratio": dit.downsample_ratio,
                "height": height,
                "width": width,
                "merge_size": dit.merge_size,
            },
            "text": model._condition_text_identity,
            "precision": {
                "compute": str(model.pipe.torch_dtype),
                "past_key_values": str(model.pipe.torch_dtype),
                "indexes_image": str(torch.int64),
                "attention": os.environ.get("RELAX_ATTENTION_IMPLEMENTATION", "flash_attention_2"),
                "deterministic_attention": os.environ.get("RELAX_DETERMINISTIC_ATTENTION", "0"),
            },
        }
        return {"prefix_kv": ConditionRequest(metadata)}

    def prepare_for_sharding(self, model: SenseNovaSFTModel) -> None:
        # Match the reference BF16 adapter initialization before FP32 masters.
        model.pipe.to(dtype=torch.bfloat16)

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
            "fixed_pixels_budget": budget,
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
        from PIL import Image
        from torchvision.transforms import InterpolationMode, functional

        from .processing import PATCH_SIZE

        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if fixed_pixels is not None and fixed_pixels > 0:
            # Bi-directional rescale to the pixel budget (small images are
            # upscaled), keeping the aspect ratio; mirrors the inference-side
            # fixed-pixels tier semantics.
            scale = (width * height / fixed_pixels) ** 0.5
            height, width = int(round(height / scale)), int(round(width / scale))
        elif width * height > max_pixels:
            scale = (width * height / max_pixels) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height, width = height // PATCH_SIZE * PATCH_SIZE, width // PATCH_SIZE * PATCH_SIZE
        if fixed_pixels is not None and fixed_pixels > 0:
            # A fixed budget is approximate after patch alignment. Keep at least
            # one patch per axis, including for tiny budgets and extreme ratios.
            height, width = max(PATCH_SIZE, height), max(PATCH_SIZE, width)
        if min(height, width) <= 0:
            raise ValueError(f"Image is smaller than patch size: {path}")
        old_width, old_height = image.size
        scale = max(width / old_width, height / old_height)
        image = functional.resize(
            image, (round(old_height * scale), round(old_width * scale)), interpolation=InterpolationMode.BILINEAR
        )
        return functional.center_crop(image, (height, width))
