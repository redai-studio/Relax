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

from relax.models.generative_sft import ConditioningResolver, ConditioningSpec


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

    def __init__(
        self,
        pipe: Any,
        training_mode: str = "lora",
        prefix_gradient_checkpointing: bool = True,
        fsdp_branch_groups: bool = False,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.training_mode = training_mode
        self.prefix_gradient_checkpointing = prefix_gradient_checkpointing
        self.fsdp_branch_groups = fsdp_branch_groups
        self.conditioning_cache: ConditioningResolver | None = None
        self.conditioning_reuse_disabled = False
        self.conditioning_reuse_disabled_reason: str | None = None
        if training_mode == "full" and not fsdp_branch_groups:
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

    def _conditioning_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        dit = self.pipe.dit
        parameters = [(f"vision.{name}", parameter) for name, parameter in dit.vision_model.named_parameters()]
        parameters.extend(
            (f"prefix.{name}", parameter)
            for name, parameter in dit.language_model.model.named_parameters()
            if "_mot_gen" not in name
        )
        return parameters

    def _conditioning_modules(self) -> list[tuple[str, torch.nn.Module]]:
        dit = self.pipe.dit
        return [(f"vision.{name}", module) for name, module in dit.vision_model.named_modules()] + [
            (f"prefix.{name}", module)
            for name, module in dit.language_model.model.named_modules()
            if "_mot_gen" not in name
        ]

    def _conditioning_semantic(self) -> dict[str, Any]:
        """Actual non-tensor forward settings, not just checkpoint config."""
        from .model import (
            SenseNovaU1Attention,
            SenseNovaU1RMSNorm,
            SenseNovaU1RotaryEmbedding,
            SenseNovaU1VisionEmbeddings,
        )

        fields = (
            (SenseNovaU1RMSNorm, ("variance_epsilon",)),
            (
                SenseNovaU1RotaryEmbedding,
                ("rope_head_dim", "rope_theta", "max_seq_len_cached", "original_max_seq_len", "attention_scaling"),
            ),
            (
                SenseNovaU1Attention,
                ("head_dim", "num_attention_heads", "num_key_value_heads", "num_key_value_groups", "scaling"),
            ),
            (
                SenseNovaU1VisionEmbeddings,
                (
                    "embed_dim",
                    "llm_embed_dim",
                    "downsample_factor",
                    "patch_size",
                    "rope_dim_part",
                    "max_position_embeddings_vision",
                    "rope_theta_vision",
                ),
            ),
            (torch.nn.Conv2d, ("kernel_size", "stride", "padding", "dilation", "groups", "padding_mode")),
            (torch.nn.GELU, ("approximate",)),
            (torch.nn.SiLU, ("inplace",)),
            (torch.nn.Dropout, ("p", "inplace", "training")),
        )
        return {
            name: {"training": module.training, **{field: getattr(module, field) for field in names}}
            for name, module in self._conditioning_modules()
            for kind, names in fields
            if isinstance(module, kind)
        }

    def _conditioning_buffers(self) -> list[tuple[str, torch.Tensor]]:
        from .model import SenseNovaU1RotaryEmbedding, SenseNovaU1VisionEmbeddings

        buffers = []
        for name, module in self._conditioning_modules():
            derived = ()
            if isinstance(module, (SenseNovaU1RotaryEmbedding, SenseNovaU1VisionEmbeddings)):
                # Their helpers record only tensors created by deterministic
                # lazy initialization/device rebuilding. Check those records
                # before excluding their changing identities from the guard.
                if not module.conditioning_rope_cache_is_pristine():
                    raise ValueError(f"{name} derived RoPE tensors changed outside materialization")
                if isinstance(module, SenseNovaU1VisionEmbeddings):
                    derived = ("cos_cached_x", "sin_cached_x", "cos_cached_y", "sin_cached_y")
            buffers.extend(
                (f"{name}.buffers.{field}", tensor)
                for field, tensor in module.named_buffers(recurse=False)
                if field not in derived
            )
        return buffers

    def conditioning_is_frozen(self) -> bool:
        """Check the understanding path, excluding the changing generation
        LoRA.

        Prefix encoding calls the understanding projections and vision tower;
        it never calls ``*_mot_gen`` modules. Parameter metadata remains safe
        to inspect after FSDP wrapping and does not synchronize the device.
        """
        parameters = self._conditioning_parameters()
        return bool(parameters) and not any(parameter.requires_grad for _, parameter in parameters)

    def conditioning_state(self) -> dict[str, Any]:
        """Declare private mutation metadata before entering the FSDP root.

        FSDP swaps module attributes between sharded masters and full forward
        views. The runtime must inspect this outside forward, when parameters
        are sharded. Identities detect local replacement; they never enter a
        shared key. Version reads do not copy or synchronize GPU contents.
        """
        parameters = {}
        for name, parameter in [*self._conditioning_parameters(), *self._conditioning_buffers()]:
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
        return {
            "frozen": bool(parameters) and not any(value["requires_grad"] for value in parameters.values()),
            "parameters": parameters,
            "semantic": self._conditioning_semantic(),
        }

    def _prepare_conditioning(self, data: dict[str, Any]) -> dict[str, Any]:
        from .pipeline import SenseNovaU1ImageUnit_PromptEmbedder

        return SenseNovaU1ImageUnit_PromptEmbedder.prepare_sft_inputs(self.pipe, data["prompt"], data["edit_image"])

    def conditioning_spec(
        self,
        data: dict[str, Any],
        shared: dict[str, Any],
        *,
        enabled: bool,
        prepared_prefix: dict[str, Any] | None = None,
    ) -> ConditioningSpec:
        """Describe effective CPU encoder inputs, independently of file names.

        Tokenizer mutations matter exactly when they change the encoder's token
        IDs, special-token selection or position indexes. Compute uses the same
        prepared tensors; it never tokenizes again after keying.
        """
        cacheable = self.conditioning_is_frozen()
        dependencies: dict[str, Any] = {}
        if enabled and cacheable:
            import hashlib
            import os

            from .processing import (
                IMAGE_PLACEHOLDER,
                IMAGENET_MEAN,
                IMAGENET_STD,
                IMG_CONTEXT_TOKEN,
                IMG_START_TOKEN,
                NON_THINK_PREFIX,
                SYSTEM_MESSAGE_FOR_GEN,
            )

            tokenizer = self.pipe.tokenizer
            prepared = self._prepare_conditioning(data) if prepared_prefix is None else prepared_prefix
            # Only freshly prepared CPU tensors may be inspected by keying.
            if prepared["input_ids"].device.type != "cpu" or prepared["indexes"].device.type != "cpu":
                raise ValueError("SenseNova conditioning keys require CPU encoder inputs")
            references = data["edit_image"]
            dependencies = {
                "prompt": data["prompt"],
                "references": [
                    {
                        "mode": image.mode,
                        "size": list(image.size),
                        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    }
                    for image in references
                ],
                "target_size": [shared["height"], shared["width"]],
                "dtype": str(self.pipe.torch_dtype),
                "encoder_semantic": self._conditioning_semantic(),
                "tokenizer": {
                    "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                    "input_ids": prepared["input_ids"][0].tolist(),
                    "special_token_ids": dict(prepared["special_token_ids"]),
                    "padding_side": getattr(tokenizer, "padding_side", None),
                    "truncation_side": getattr(tokenizer, "truncation_side", None),
                },
                "prefix_indexes": prepared["indexes"].tolist(),
                "reference_grid": prepared["grid_hw"].tolist() if prepared["grid_hw"] is not None else None,
                "template": [
                    SYSTEM_MESSAGE_FOR_GEN,
                    NON_THINK_PREFIX,
                    IMAGE_PLACEHOLDER,
                    IMG_CONTEXT_TOKEN,
                    IMG_START_TOKEN,
                ],
                "preprocessing": {
                    "version": 2,
                    "patch_size": self.pipe.dit.patch_size,
                    "downsample_ratio": self.pipe.dit.downsample_ratio,
                    "merge_size": self.pipe.dit.merge_size,
                    "min_pixels": 512 * 512,
                    "max_pixels": min(2048 * 2048, (4096 * 4096) // max(1, len(references))),
                    "mean": list(IMAGENET_MEAN),
                    "std": list(IMAGENET_STD),
                    "resize": "pillow-default",
                },
                "attention": os.environ.get("RELAX_ATTENTION_IMPLEMENTATION", "flash_attention_2"),
                "deterministic_attention": os.environ.get("RELAX_DETERMINISTIC_ATTENTION", "0"),
            }
        return ConditioningSpec("sensenova.prefix.v3", dependencies, cacheable)

    def conditioning_prefetch_spec(self, data: dict[str, Any]) -> ConditioningSpec:
        """Describe an already decoded sample without encoding or using RNG.

        The trainer calls this outside FSDP forward, on its main thread.
        Forward still prepares and keys its actual input independently; this
        hint never authorizes reuse by sample ID or by the prefetched data's
        identity.
        """
        height, width = self.pipe.check_resize_height_width(data["image"].height, data["image"].width)
        return self.conditioning_spec(data, {"height": height, "width": width}, enabled=True)

    @staticmethod
    def _export_conditioning(positive: dict[str, Any], *, detach: bool) -> dict[str, torch.Tensor]:
        features = {"indexes_image": positive["indexes_image"]}
        for index, layer in enumerate(positive["past_key_values"].layers):
            features[f"kv.{index}.key"] = layer.keys
            features[f"kv.{index}.value"] = layer.values
        return {name: tensor.detach() for name, tensor in features.items()} if detach else features

    def _restore_conditioning(self, features: dict[str, torch.Tensor], height: int, width: int) -> dict[str, Any]:
        from transformers.cache_utils import DynamicCache, DynamicLayer

        from .processing import PATCH_SIZE

        layers = self.pipe.dit.language_model.model.layers
        expected = {"indexes_image"} | {
            f"kv.{index}.{kind}" for index in range(len(layers)) for kind in ("key", "value")
        }
        if set(features) != expected:
            raise ValueError("Cached SenseNova conditioning fields differ from the model")
        indexes = features["indexes_image"]
        if indexes.dtype != torch.long or indexes.shape != (3, (height // PATCH_SIZE) * (width // PATCH_SIZE)):
            raise ValueError("Cached SenseNova image indexes have an invalid dtype or shape")
        cache = DynamicCache()
        prefix_length = None
        for index, layer in enumerate(layers):
            key, value = features[f"kv.{index}.key"], features[f"kv.{index}.value"]
            attention = layer.self_attn
            if (
                key.ndim != 4
                or key.shape != value.shape
                or key.shape[0] != 1
                or key.shape[1] != attention.num_key_value_heads
                or key.shape[-1] != attention.head_dim
                or key.dtype != self.pipe.torch_dtype
                or value.dtype != key.dtype
                or key.device != value.device
                or key.device != indexes.device
                or (prefix_length is not None and key.shape[2] != prefix_length)
            ):
                raise ValueError("Cached SenseNova prefix has an invalid dtype, device or shape")
            prefix_length = key.shape[2]
            # DynamicLayer.update (also used by ddp_cache_data construction)
            # concatenates with an empty tensor on first use. Bind the existing
            # immutable features to a fresh per-forward wrapper instead.
            cache_layer = DynamicLayer()
            cache_layer.keys, cache_layer.values = key, value
            cache_layer.dtype, cache_layer.device = key.dtype, key.device
            cache_layer.is_initialized = True
            cache.layers.append(cache_layer)
        return {"past_key_values": cache, "indexes_image": indexes}

    def forward(self, data: dict[str, Any]) -> torch.Tensor:
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
        # Default generation-only LoRA leaves prefix/vision frozen. Custom
        # targets or a later unfreeze must retain the conditioning graph.
        frozen_prefix = self.conditioning_is_frozen()
        with torch.no_grad() if self.training_mode == "lora" and frozen_prefix else nullcontext():
            if self.conditioning_cache is None:
                for unit in self.pipe.units:
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            else:
                inputs = self.pipe.unit_runner(self.pipe.units[0], self.pipe, *inputs)
                import time

                prepare_started = time.perf_counter()
                inputs[0]["prepared_prefix"] = self._prepare_conditioning(data)
                spec = self.conditioning_spec(
                    data,
                    inputs[0],
                    enabled=self.conditioning_cache.enabled,
                    prepared_prefix=inputs[0]["prepared_prefix"],
                )
                record_prepare = getattr(self.conditioning_cache, "record_prepare", None)
                if record_prepare is not None:
                    record_prepare(time.perf_counter() - prepare_started)

                def compute() -> dict[str, torch.Tensor]:
                    condition_inputs = (dict(inputs[0]), dict(inputs[1]), dict(inputs[2]))
                    # Numerical verification deliberately exercises the original
                    # tokenizer/device-index path as its independent reference.
                    # Performance runs never set force_online.
                    if getattr(self.conditioning_cache, "force_online", False):
                        condition_inputs[0].pop("prepared_prefix", None)
                    for unit in self.pipe.units[1:3]:
                        condition_inputs = self.pipe.unit_runner(unit, self.pipe, *condition_inputs)
                    record_transfer = getattr(self.conditioning_cache, "record_input_transfer", None)
                    if record_transfer is not None:
                        record_transfer(
                            sum(
                                value.numel() * value.element_size()
                                for name in ("edit_pixel_values", "edit_grid_hw")
                                if isinstance(value := condition_inputs[0].get(name), torch.Tensor)
                            )
                        )
                    return self._export_conditioning(condition_inputs[1], detach=spec.cacheable)

                restored = self.conditioning_cache.resolve(
                    spec,
                    compute,
                    lambda features: self._restore_conditioning(features, inputs[0]["height"], inputs[0]["width"]),
                )
                inputs[1].update(restored)
                inputs[2].update(restored)
                for unit in self.pipe.units[3:]:
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
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
            fsdp_branch_groups=config.get("fsdp_branch_groups", False),
        )

    def prepare_for_sharding(self, model: SenseNovaSFTModel) -> None:
        if model.fsdp_branch_groups and not model.conditioning_is_frozen():
            # Prefix K/V are intermediate branch outputs. A sibling group's
            # pre-backward hook on its final MLP output does not cover those
            # edges when the prefix is trainable. Keep the ordinary decoder
            # boundary for full/custom understanding training.
            raise ValueError("fsdp_branch_groups requires a frozen understanding branch; disable it to train prefix")
        # Match the reference BF16 adapter initialization before FP32 masters.
        model.pipe.to(dtype=torch.bfloat16)
        if model.training_mode == "lora" or model.fsdp_branch_groups:
            dit = model.pipe.dit
            # These modules are invoked through __call__, so independent FSDP
            # boundaries load vision/text weights only on online encoding.
            # The SFT generation path never calls the language-model head.
            # Apply this policy equally to online and cached LoRA training.
            for module in (dit.vision_model, dit.language_model.model.embed_tokens, dit.language_model.lm_head):
                module._fsdp_fine_grained = True
        if model.fsdp_branch_groups:
            groups = []
            for layer in model.pipe.dit.language_model.model.layers:
                attention = layer.self_attn
                for suffix in ("", "_mot_gen"):
                    # Every member is invoked on its branch, ending with MLP;
                    # grouped FSDP therefore reshards after the whole branch.
                    # Keep generation PEFT wrappers intact so their base and
                    # LoRA tensors have one owner. The understanding group is
                    # only used under no_grad, as checked before sharding.
                    groups.append(
                        (
                            getattr(layer, f"input_layernorm{suffix}"),
                            *(getattr(attention, f"{name}{suffix}") for name in ("q_proj", "k_proj", "v_proj")),
                            *(
                                getattr(attention, f"{name}{suffix}")
                                for name in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw")
                            ),
                            getattr(attention, f"o_proj{suffix}"),
                            getattr(layer, f"post_attention_layernorm{suffix}"),
                            getattr(layer, f"mlp{suffix}"),
                        )
                    )
            model._fsdp_module_groups = tuple(groups)

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
