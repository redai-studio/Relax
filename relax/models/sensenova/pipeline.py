# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Copyright (c) 2023 Zhongjie Duan.
# Adapted and modified by Relax from DiffSynth-Studio, revision f7b7db96.
# Source: https://github.com/modelscope/DiffSynth-Studio
# SPDX-License-Identifier: Apache-2.0
# See the repository root LICENSE for the full license text.
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from einops import reduce, repeat
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import Qwen2Tokenizer

from .model import SenseNovaU1DiT
from .processing import (
    IMAGE_PLACEHOLDER,
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    NON_THINK_PREFIX,
    PATCH_SIZE,
    SYSTEM_MESSAGE_FOR_GEN,
    THINK_PREFIX,
    build_conversation_prompt,
    build_image_token_block,
    build_thw_indexes,
    create_block_causal_mask,
    load_image_native,
    patchify,
    unpatchify,
)
from .scheduler import FlowMatchScheduler


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None,
    ) -> None:
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names


class PipelineUnitRunner:
    def __call__(
        self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict
    ) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega
            )
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


class BasePipeline(torch.nn.Module):
    def __init__(self, device: Any, torch_dtype: Any, height_division_factor: Any, width_division_factor: Any) -> None:
        super().__init__()
        self.device, self.torch_dtype = device, torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.unit_runner = PipelineUnitRunner()

    def to(self, *args: Any, **kwargs: Any) -> Any:
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        result = super().to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return result

    def freeze_except(self, names: Any) -> Any:
        self.eval()
        self.requires_grad_(False)
        for name in names:
            getattr(self, name).train().requires_grad_(True)

    def load_models_to_device(self, names: Any) -> Any:
        # Models are owned by Relax/FSDP; pipeline units never move parameters.
        return None

    def check_resize_height_width(self, height: Any, width: Any) -> Any:
        h, w = self.height_division_factor, self.width_division_factor
        return ((height + h - 1) // h * h, (width + w - 1) // w * w)

    def step(self, scheduler: Any, latents: Any, progress_id: Any, noise_pred: Any, **kwargs: Any) -> Any:
        return scheduler.step(noise_pred, scheduler.timesteps[progress_id], latents)

    def cfg_guided_model_fn(
        self, model_fn: Any, cfg_scale: Any, inputs_shared: Any, inputs_posi: Any, inputs_nega: Any, **others: Any
    ) -> Any:
        positive = model_fn(**inputs_posi, **inputs_shared, **others)
        if cfg_scale == 1.0:
            return positive
        negative = model_fn(**inputs_nega, **inputs_shared, **others)
        return negative + cfg_scale * (positive - negative)

    def preprocess_image(
        self,
        image: Any,
        torch_dtype: Any = None,
        device: Any = None,
        pattern: Any = "B C H W",
        min_value: Any = -1,
        max_value: Any = 1,
    ) -> Any:
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def vae_output_to_image(
        self, vae_output: Any, pattern: Any = "B C H W", min_value: Any = -1, max_value: Any = 1
    ) -> Any:
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def generate_noise(
        self,
        shape: Any,
        seed: Any = None,
        rand_device: Any = "cpu",
        rand_torch_dtype: Any = torch.float32,
        device: Any = None,
        torch_dtype: Any = None,
    ) -> Any:
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise


class SenseNovaU1ImagePipeline(BasePipeline):
    def __init__(self, device: Any = "cpu", torch_dtype: Any = torch.bfloat16) -> None:
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=PATCH_SIZE,
            width_division_factor=PATCH_SIZE,
        )
        self.scheduler = FlowMatchScheduler("SenseNova-U1")
        self.dit: SenseNovaU1DiT = None
        self.tokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            SenseNovaU1ImageUnit_ShapeChecker(),
            SenseNovaU1ImageUnit_EditImageEmbedder(),
            SenseNovaU1ImageUnit_PromptEmbedder(),
            SenseNovaU1ImageUnit_NoiseInitializer(),
            SenseNovaU1ImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_sensenova_u1_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(model_path: Any, device: Any = "cpu", torch_dtype: Any = torch.bfloat16) -> Any:
        pipe = SenseNovaU1ImagePipeline(device=device, torch_dtype=torch_dtype)
        shards = sorted(Path(model_path).glob("model*.safetensors"))
        if not shards:
            raise FileNotFoundError(model_path)
        with init_empty_weights(include_buffers=False):
            pipe.dit = SenseNovaU1DiT()
        state = {}
        for shard in shards:
            values = load_file(str(shard), device="cpu")
            if state.keys() & values.keys():
                raise ValueError("Duplicate keys across model shards")
            state.update(values)
        pipe.dit.load_state_dict(state, strict=True, assign=True)
        del state
        pipe.dit.to(device=device, dtype=torch_dtype).eval()
        pipe.tokenizer = Qwen2Tokenizer.from_pretrained(model_path, local_files_only=True)
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        cfg_scale: float = 4.0,
        # Shape
        height: int = 2048,
        width: int = 2048,
        # Randomness
        seed: int = None,
        rand_device: str = "cuda",
        # Steps
        num_inference_steps: int = 50,
        # Scheduler
        shift: float = 3.0,
        # Reasoning
        think_mode: bool = False,
        # Image editing
        edit_image: Union[Image.Image, list[Image.Image]] = None,
        # Progress bar
        progress_bar_cmd: Any = tqdm,
    ) -> Any:
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=shift)

        # Parameters
        inputs_posi = {"prompt": prompt, "is_positive": True}
        inputs_nega = {"prompt": "", "is_positive": False}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "think_mode": think_mode,
            "edit_image": edit_image,
        }

        # Units
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.to(dtype=torch.float32, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep,
                progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        image = self.vae_output_to_image(inputs_shared["latents"])
        self.load_models_to_device([])
        return image


class SenseNovaU1ImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self) -> None:
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, height: Any, width: Any) -> Any:
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class SenseNovaU1ImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self) -> None:
        super().__init__(
            input_params=("edit_image",),
            output_params=("edit_pixel_values", "edit_grid_hw"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, edit_image: Any) -> Any:
        if edit_image is None:
            return {"edit_pixel_values": None, "edit_grid_hw": None}
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]

        max_pixels = min(2048 * 2048, (4096 * 4096) // len(edit_image))
        pixel_values, grid_hw = [], []
        for image in edit_image:
            cur_pixel_values, cur_grid_hw = load_image_native(
                image,
                pipe.dit.patch_size,
                pipe.dit.downsample_ratio,
                min_pixels=512 * 512,
                max_pixels=max_pixels,
                upscale=False,
            )
            pixel_values.append(cur_pixel_values.to(device=pipe.device, dtype=pipe.torch_dtype))
            grid_hw.append(cur_grid_hw.to(pipe.device))
        return {"edit_pixel_values": torch.cat(pixel_values), "edit_grid_hw": torch.cat(grid_hw)}


class SenseNovaU1ImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self) -> None:
        super().__init__(
            seperate_cfg=True,
            input_params=(
                "height",
                "width",
                "edit_pixel_values",
                "edit_grid_hw",
                "think_mode",
                "use_gradient_checkpointing",
                "use_gradient_checkpointing_offload",
            ),
            input_params_posi={"prompt": "prompt", "is_positive": "is_positive"},
            input_params_nega={"prompt": "prompt", "is_positive": "is_positive"},
            output_params=("past_key_values", "indexes_image"),
            onload_model_names=("dit",),
        )

    @staticmethod
    def insert_image_placeholders(prompt: Any, num_images: Any) -> Any:
        missing = num_images - prompt.count(IMAGE_PLACEHOLDER)
        if missing <= 0:
            return prompt
        if missing == num_images and num_images > 1:
            return "".join(f"Image-{i + 1}:{IMAGE_PLACEHOLDER}\n" for i in range(num_images)) + prompt
        return f"{IMAGE_PLACEHOLDER}\n" * missing + prompt

    @staticmethod
    def expand_image_placeholders(query: Any, grid_hw: Any, downsample_ratio: Any) -> Any:
        for i in range(grid_hw.shape[0]):
            num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * downsample_ratio**2)
            query = query.replace(IMAGE_PLACEHOLDER, build_image_token_block(num_patch_token), 1)
        return query

    @staticmethod
    def build_prefix_inputs(pipe: Any, query: Any, pixel_values: Any, grid_hw: Any) -> Any:
        input_ids = pipe.tokenizer(query, return_tensors="pt")["input_ids"].to(pipe.device)
        indexes = build_thw_indexes(
            input_ids[0],
            pipe.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN),
            pipe.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
            grid_hw=grid_hw,
            merge_size=pipe.dit.merge_size,
        )
        attention_mask = create_block_causal_mask(indexes[0])
        if pixel_values is None:
            return input_ids, None, indexes, attention_mask

        inputs_embeds = pipe.dit.embed_tokens(input_ids)
        batch_size, num_tokens, channels = inputs_embeds.shape
        vision_embeds = pipe.dit.extract_und_feature(pixel_values, grid_hw)
        selected = input_ids.reshape(-1) == pipe.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        inputs_embeds = inputs_embeds.reshape(-1, channels)
        inputs_embeds[selected] = vision_embeds.reshape(-1, channels).to(inputs_embeds.device)
        return input_ids, inputs_embeds.reshape(batch_size, num_tokens, channels), indexes, attention_mask

    @staticmethod
    def build_image_indexes(token_h: Any, token_w: Any, text_len: Any, device: Any) -> Any:
        t_image = torch.full((token_h * token_w,), text_len, dtype=torch.long, device=device)
        idx = torch.arange(token_h * token_w, device=device, dtype=torch.long)
        h_image = idx // token_w
        w_image = idx % token_w
        return torch.stack([t_image, h_image, w_image], dim=0)

    def process(
        self,
        pipe: SenseNovaU1ImagePipeline,
        prompt: Any,
        is_positive: Any,
        height: Any,
        width: Any,
        edit_pixel_values: Any,
        edit_grid_hw: Any,
        think_mode: Any,
        use_gradient_checkpointing: Any,
        use_gradient_checkpointing_offload: Any,
    ) -> Any:
        pipe.load_models_to_device(self.onload_model_names)
        num_images = 0 if edit_grid_hw is None else edit_grid_hw.shape[0]
        reasoning = bool(think_mode) and is_positive and not pipe.scheduler.training

        if is_positive:
            query = build_conversation_prompt(
                self.insert_image_placeholders(prompt, num_images),
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text=THINK_PREFIX if reasoning else NON_THINK_PREFIX + IMG_START_TOKEN,
            )
        else:
            query = build_conversation_prompt(
                IMAGE_PLACEHOLDER * num_images,
                system_message="",
                append_text=IMG_START_TOKEN,
            )
        if num_images > 0:
            query = self.expand_image_placeholders(query, edit_grid_hw, pipe.dit.downsample_ratio)

        input_ids, inputs_embeds, indexes, attention_mask = self.build_prefix_inputs(
            pipe,
            query,
            edit_pixel_values,
            edit_grid_hw,
        )
        if reasoning:
            past_key_values, text_len, token_ids = pipe.dit.generate_think(
                input_ids=None if inputs_embeds is not None else input_ids,
                inputs_embeds=inputs_embeds,
                indexes=indexes,
                attention_mask=attention_mask,
                eos_token_id=pipe.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                think_end_token_id=pipe.tokenizer.convert_tokens_to_ids("</think>"),
                append_ids=pipe.tokenizer(
                    "\n\n" + IMG_START_TOKEN,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"].to(pipe.device),
            )
            # `token_ids` holds the reasoning the model just wrote. Nothing downstream needs
            # it, so it is left undecoded; to read it, decode it here:
            # pipe.tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            past_key_values, _ = pipe.dit.encode_prefix(
                input_ids=None if inputs_embeds is not None else input_ids,
                inputs_embeds=inputs_embeds,
                indexes=indexes,
                attention_mask=attention_mask,
                use_gradient_checkpointing=bool(use_gradient_checkpointing),
                use_gradient_checkpointing_offload=bool(use_gradient_checkpointing_offload),
            )
            text_len = indexes[0].max()

        token_h, token_w = height // PATCH_SIZE, width // PATCH_SIZE
        indexes_image = self.build_image_indexes(token_h, token_w, text_len + 1, pipe.device)
        return {"past_key_values": past_key_values, "indexes_image": indexes_image}


class SenseNovaU1ImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self) -> None:
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise", "noise_scale"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, height: Any, width: Any, seed: Any, rand_device: Any) -> Any:
        noise_scale = pipe.dit.compute_noise_scale(height, width)
        noise = pipe.generate_noise(
            (1, 3, height, width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise_scale * noise, "noise_scale": noise_scale}


class SenseNovaU1ImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self) -> None:
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
        )

    def process(self, pipe: SenseNovaU1ImagePipeline, input_image: Any, noise: Any) -> Any:
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        input_latents = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"latents": noise, "input_latents": input_latents}


def model_fn_sensenova_u1_image(
    dit: SenseNovaU1DiT,
    latents: Any = None,
    timestep: Any = None,
    past_key_values: Any = None,
    indexes_image: Any = None,
    noise_scale: Any = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs: Any,
) -> Any:
    sigma = (timestep / 1000.0).flatten()[0]
    t = 1.0 - sigma

    grid_hw = torch.tensor(
        [[latents.shape[2] // dit.patch_size, latents.shape[3] // dit.patch_size]] * latents.shape[0],
        device=latents.device,
    )

    z = patchify(latents, PATCH_SIZE)
    num_tokens = z.shape[1]

    image_input = patchify(latents, dit.patch_size, channel_first=True)
    image_embeds = dit.extract_gen_feature(image_input.flatten(0, 1), grid_hw).view(latents.shape[0], num_tokens, -1)
    image_embeds = image_embeds + dit.embed_timestep(t, noise_scale, num_tokens, batch_size=latents.shape[0])

    x_pred = dit(
        image_embeds=image_embeds,
        indexes_image=indexes_image,
        past_key_values=past_key_values,
        image_size=(latents.shape[3], latents.shape[2]),
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    x_pred = unpatchify(x_pred, PATCH_SIZE, latents.shape[2], latents.shape[3])
    return (latents - x_pred) / sigma
