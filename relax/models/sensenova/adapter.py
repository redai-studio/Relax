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

from relax.models.condition_cache import (
    ConditionFeatureCache,
    FrozenConditionSpec,
    all_ranks_hit,
    condition_cache_key,
)
from relax.utils.env import Envs


LORA_TARGETS = (
    "q_proj_mot_gen",
    "k_proj_mot_gen",
    "v_proj_mot_gen",
    "o_proj_mot_gen",
    "mlp_mot_gen.gate_proj",
    "mlp_mot_gen.up_proj",
    "mlp_mot_gen.down_proj",
)


def _condition_encoder_identity(model_path: str) -> str:
    """Bind a preverified asset manifest or hash the offline assets once.

    The manifest identity must be a content SHA validated by the launcher, not
    a model name/path. Without it we pay one cold-start sequential read. The
    cache is process-local and is never restored across model construction.
    """
    identity = Envs.RELAX_CONDITION_CACHE_IDENTITY
    if identity:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity.lower()):
            raise ValueError("RELAX_CONDITION_CACHE_IDENTITY must be a verified asset-manifest SHA256")
        return identity.lower()
    digest = hashlib.sha256()
    root = Path(model_path)
    files = sorted(p for p in root.iterdir() if p.is_file())
    if not any(p.name.startswith("model") and p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError("Frozen encoder identity requires the complete SenseNova assets")
    for path in files:
        digest.update(path.name.encode() + b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _condition_adapter_version() -> str:
    digest = hashlib.sha256()
    for name in ("adapter.py", "pipeline.py", "processing.py", "model.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


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

    def __init__(
        self,
        pipe: Any,
        training_mode: str = "lora",
        prefix_gradient_checkpointing: bool = True,
        condition_cache: ConditionFeatureCache | None = None,
        condition_identity: str | None = None,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.training_mode = training_mode
        self.prefix_gradient_checkpointing = prefix_gradient_checkpointing
        if condition_cache is not None and (training_mode != "lora" or not condition_identity):
            raise ValueError("Frozen prefix reuse requires LoRA mode and a content-bound encoder identity")
        self.condition_cache = condition_cache
        self.condition_identity = condition_identity
        self.condition_cache_group = None
        self._condition_version = _condition_adapter_version() if condition_cache is not None else ""
        self._prefix_versions = None
        self._condition_metrics = dict(
            prefix_encodes=0, local_hits=0, global_hits=0, global_misses=0, bypasses=0, invalidations=0
        )
        self._condition_encode_seconds = 0.0
        self.register_load_state_dict_post_hook(lambda module, incompatible: module.clear_condition_cache())
        if training_mode == "full":
            self._mark_fine_grained_modules()

    def _prefix_cache_is_frozen(self) -> bool:
        # Generation-specific MoT weights/adapters are not prefix dependencies.
        # Any adapter injected into understanding/vision must disable reuse.
        parameters = [
            (name, parameter)
            for name, parameter in self.pipe.dit.named_parameters()
            if "mot_gen" not in name and not name.startswith("fm_modules.")
        ]
        if not parameters or any(parameter.requires_grad for _, parameter in parameters):
            return False
        versions = tuple((name, parameter._version) for name, parameter in parameters)
        if self._prefix_versions is not None and versions != self._prefix_versions:
            self.clear_condition_cache()
            self._condition_metrics["invalidations"] += 1
        self._prefix_versions = versions
        # Future prefix dropout must not accidentally reuse a random feature.
        for name, module in self.pipe.dit.named_modules():
            if (
                "mot_gen" not in name
                and isinstance(module, torch.nn.modules.dropout._DropoutNd)
                and module.training
                and module.p
            ):
                return False
        return True

    def _prefix_key(self, data: dict[str, Any], unit: Any) -> str:
        images = data["edit_image"]
        if images is None:
            images = []
        elif not isinstance(images, (list, tuple)):
            images = [images]
        device_type = torch.device(self.pipe.device).type
        autocast = torch.is_autocast_enabled(device_type)
        # Hash the effective CPU pixels, after prepare_sample's crop/resize.
        # Never hash a filename or copy already-on-device pixels back to CPU.
        content = {
            "prompt": data["prompt"],
            "conditions": tuple((image.mode, image.size, image.tobytes()) for image in images),
            "target_size": (data["image"].height, data["image"].width),
            "think_mode": False,
            "attention": os.environ.get("RELAX_ATTENTION_IMPLEMENTATION", "flash_attention_2"),
            "deterministic_attention": os.environ.get("RELAX_DETERMINISTIC_ATTENTION", "0"),
            "torch_version": str(torch.__version__),
            "encoder_device": str(self.pipe.device),
            "autocast": autocast,
            "autocast_dtype": str(torch.get_autocast_dtype(device_type)) if autocast else None,
        }
        spec = FrozenConditionSpec(
            encoder_identity=self.condition_identity,
            preprocessing_version=self._condition_version,
            dtype=str(self.pipe.torch_dtype),
            feature_schema=unit.frozen_condition_schema,
            adapter_version=self._condition_version,
        )
        return condition_cache_key(spec, content)

    def clear_condition_cache(self) -> None:
        if self.condition_cache is not None:
            self.condition_cache.clear()

    def close_condition_cache(self) -> None:
        if self.condition_cache is not None:
            self.condition_cache.close()

    def condition_cache_stats(self) -> dict[str, Any]:
        return {
            "enabled": self.condition_cache is not None,
            **(self.condition_cache.stats() if self.condition_cache is not None else {}),
            **self._condition_metrics,
            "encode_seconds_host": self._condition_encode_seconds,
        }

    def _run_condition_unit(self, unit: Any, inputs: Any, data: dict[str, Any]) -> Any:
        cache = self.condition_cache
        reusable = self.training_mode == "lora" and self._prefix_cache_is_frozen()
        # Agreement applies to both eligibility and hit decisions. Independent
        # rank-local skips would mismatch the sharded decoder collectives.
        eligible = all_ranks_hit(reusable, self.condition_cache_group)
        if not eligible:
            self._condition_metrics["bypasses"] += 1
            cache.clear()
            started = time.monotonic()
            result = self.pipe.unit_runner(unit, self.pipe, *inputs)
            self._condition_metrics["prefix_encodes"] += 1
            self._condition_encode_seconds += time.monotonic() - started
            return result
        key = self._prefix_key(data, unit)
        local_hit = cache.contains(key)
        self._condition_metrics["local_hits"] += int(local_hit)
        if all_ranks_hit(local_hit, self.condition_cache_group):
            self._condition_metrics["global_hits"] += 1
            payload = cache.get(key, device=self.pipe.device)
            if payload is None:
                # A single-owner cache cannot change between probe/read.
                raise RuntimeError("Frozen condition disappeared after collective hit agreement")
            shared, positive, negative = inputs
            outputs = unit.unpack_frozen_conditions(payload)
            return shared, {**positive, **outputs}, {**negative, **outputs}
        self._condition_metrics["global_misses"] += 1
        started = time.monotonic()
        result = self.pipe.unit_runner(unit, self.pipe, *inputs)
        self._condition_metrics["prefix_encodes"] += 1
        self._condition_encode_seconds += time.monotonic() - started
        cache.put(key, unit.pack_frozen_conditions(result[1]))
        return result

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
        try:
            return self._forward_sample(data)
        except BaseException:
            self.close_condition_cache()
            raise

    def _forward_sample(self, data: dict[str, Any]) -> torch.Tensor:
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
                if self.condition_cache is not None and hasattr(unit, "frozen_condition_schema"):
                    inputs = self._run_condition_unit(unit, inputs, data)
                else:
                    started = time.monotonic()
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
                    if hasattr(unit, "frozen_condition_schema"):
                        self._condition_metrics["prefix_encodes"] += 1
                        self._condition_encode_seconds += time.monotonic() - started
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
        budget = Envs.RELAX_CONDITION_CACHE_BYTES
        if budget < 0:
            raise ValueError("RELAX_CONDITION_CACHE_BYTES cannot be negative")
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
        condition_cache = None
        identity = None
        if budget and mode == "lora":
            identity = _condition_encoder_identity(args.model_path)
            condition_cache = ConditionFeatureCache(budget, storage_device=Envs.RELAX_CONDITION_CACHE_DEVICE)
        return SenseNovaSFTModel(
            pipe,
            training_mode=mode,
            prefix_gradient_checkpointing=config.get("prefix_gradient_checkpointing", True),
            condition_cache=condition_cache,
            condition_identity=identity,
        )

    def prepare_for_sharding(self, model: SenseNovaSFTModel) -> None:
        # Match the reference BF16 adapter initialization before FP32 masters.
        model.pipe.to(dtype=torch.bfloat16)

    def on_device_ready(self, model: SenseNovaSFTModel, device: torch.device) -> None:
        # Never move the already sharded parameters through Module.to().
        model.pipe.device = device
        if model.condition_cache is not None:
            import torch.distributed as dist

            if dist.is_initialized() and dist.get_world_size(dist.group.WORLD) > 1:
                from relax.utils.distributed_utils import get_gloo_group

                # Native Ray actor initializes this CPU control group. A
                # standalone runner may instead set condition_cache_group.
                try:
                    model.condition_cache_group = get_gloo_group()
                except RuntimeError:
                    model.condition_cache_group = dist.group.WORLD

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
