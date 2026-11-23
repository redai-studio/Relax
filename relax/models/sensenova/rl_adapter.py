# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SenseNova-U1.5 image-edit adapter for native generative RL.

LoRA and full generation-branch tuning use Flow-Factory's explicit target
surface. The understanding path stays frozen, so the condition prefix can be
recomputed once per replay micro-batch without retaining its graph. This is
not unrestricted whole-model fine-tuning.

The rollout engine returns canonical prefix inputs, not the 42-layer KV cache.
This keeps trajectory transport bounded at high resolution while still letting
the trainer reproduce the exact conditioning path with its own model weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

from relax.models.generative import ArtifactTrack, resolve_sde_indices

from .adapter import LORA_TARGETS


__all__ = ["SenseNovaU1ImageEditAdapter"]

RL_LORA_TARGETS = LORA_TARGETS + (
    "fm_head.0",
    "fm_head.2",
    "fm_head.conv1",
    "fm_head.conv2",
    "timestep_embedder.mlp.0",
    "timestep_embedder.mlp.2",
    "noise_scale_embedder.mlp.0",
    "noise_scale_embedder.mlp.2",
)


class SenseNovaU1ImageEditAdapter:
    """SenseNova-U1.5 I2I adapter for the common FSDP generative actor."""

    family = "sensenova_u1_5"
    supported_tasks: Tuple[str, ...] = ("i2i",)
    # Matches Flow-Factory's SenseNova default target surface. U1.0/U1.5 only
    # adapt entries that exist in the loaded checkpoint.
    lora_target_modules: Tuple[str, ...] = RL_LORA_TARGETS
    lora_task_type = "FEATURE_EXTRACTION"
    full_trainable_targets = RL_LORA_TARGETS

    def configure_full_training(self, model: torch.nn.Module) -> List[str]:
        """Match Flow-Factory's full-mode substring selection, not whole-model
        FT."""
        model.requires_grad_(False)
        selected = []
        for name, parameter in model.named_parameters():
            if any(target in name for target in self.full_trainable_targets):
                parameter.requires_grad_(True)
                selected.append(name)
        if not selected:
            raise ValueError("SenseNova full training found no generation parameters.")
        self.validate_full_trainable_boundary(model)
        return selected

    def validate_full_trainable_boundary(self, model: torch.nn.Module) -> None:
        from relax.backends.fsdp.weight_update import strip_transport_wrappers

        trainable = 0
        for name, parameter in model.named_parameters():
            expected = any(target in strip_transport_wrappers(name) for target in self.full_trainable_targets)
            if parameter.requires_grad != expected:
                raise ValueError(f"SenseNova full generation boundary disagrees at {name}.")
            trainable += int(expected)
        if not trainable:
            raise ValueError("SenseNova full training requires generation parameters.")

    def resolve_lora_target_modules(self, model: torch.nn.Module) -> List[str]:
        """Select the U1.0/U1.5 target aliases that exist in this checkpoint.

        The two released heads use either sequential ``fm_head.0/.2`` names or
        explicit ``fm_head.conv1/.conv2`` names.  The generic LoRA injector is
        deliberately strict and rejects any unmatched suffix, so resolve that
        versioned alias set against the materialized model before injection.
        """
        module_names = tuple(name for name, _ in model.named_modules())
        resolved = [
            target for target in self.lora_target_modules if any(name.endswith(target) for name in module_names)
        ]
        missing_core = [target for target in LORA_TARGETS if target not in resolved]
        if missing_core:
            raise ValueError(f"SenseNova checkpoint is missing required generation LoRA targets: {missing_core}.")
        if not any(target.startswith("fm_head.") for target in resolved):
            raise ValueError("SenseNova checkpoint has no supported fm_head LoRA target names.")
        return resolved

    def load_train_model(self, config: Any) -> torch.nn.Module:
        from .pipeline import SenseNovaU1ImagePipeline

        model_path = str(config.model_path)
        if not list(Path(model_path).glob("model*.safetensors")):
            raise FileNotFoundError(f"No model*.safetensors in {model_path}")
        dtype = torch.bfloat16 if getattr(config, "fsdp_param_dtype", "bf16") == "bf16" else torch.float32
        pipe = SenseNovaU1ImagePipeline.from_pretrained(model_path=model_path, torch_dtype=dtype, device="cpu")
        model = pipe.dit
        model.configure_rl_precision(dtype=dtype)
        if str(getattr(config, "fsdp_trainable_mode", "lora")) == "full":
            self.configure_full_training(model)
        # FSDP2 discovers block shard/checkpoint boundaries through this HF-style
        # attribute. SenseNova's upstream `_repeated_blocks` is not inspected by
        # Relax's generic wrapper.
        model._no_split_modules = ("SenseNovaU1DecoderLayer",)
        return model

    def build_rollout_request(self, sample: Any, sampling: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        from .rollout import build_sensenova_rollout_request

        guidance_scale = float(sampling.get("guidance_scale", 1.0))
        if guidance_scale != 1.0:
            raise ValueError(
                "SenseNova RL replay currently supports one positive-conditioned forward; set guidance_scale=1.0."
            )
        if sampling.get("think_mode", False):
            raise ValueError("SenseNova RL replay currently requires think_mode=False.")
        condition_paths = self._condition_image_paths(sample)
        rollout_id = int(sampling.get("_rollout_id", 0))
        sample_id = str(sampling.get("_sample_id", seed))
        group_id = f"rollout:{rollout_id}:group:{getattr(sample, 'group_index', 0)}"
        if sampling.get("_evaluation", False):
            group_id = f"eval:{sampling['_eval_dataset_id']}:group:{getattr(sample, 'group_index', 0)}"
        candidate_id = f"{group_id}:{sample_id}"
        # Unlike Qwen, this adapter does not supply driver-owned x_T. Derive
        # noise from identity even when the common driver's fallback seed is
        # constant. Training includes the round; held-out evaluation keeps a
        # stable dataset/candidate identity across rounds.
        key = f"{sampling.get('_base_seed', seed)}:{candidate_id}".encode()
        candidate_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)
        steps = int(sampling.get("num_inference_steps", 50))
        algorithm = str(sampling.get("algorithm", "grpo"))
        height, width = int(sampling.get("height", 1024)), int(sampling.get("width", 1024))
        fixed_pixels = sampling.get("fixed_input_image_pixels", 0)
        max_side = sampling.get("max_input_image_side_length", 0)
        if sampling.get("output_size_from_condition", False):
            from PIL import Image

            from .geometry import fixed_pixel_size

            with Image.open(condition_paths[0]) as image:
                width, height = image.size
            if fixed_pixels:
                width, height = fixed_pixel_size(width, height, fixed_pixels, max_side)
            if height % 32 or width % 32:
                raise ValueError("Primary reference output geometry must be 32-aligned")
        request = build_sensenova_rollout_request(
            sample.prompt,
            condition_paths,
            seed=candidate_seed,
            initial_seed=candidate_seed,
            candidate_id=candidate_id,
            group_id=group_id,
            member_id=int(sampling.get("_member_id", 0)),
            algorithm=algorithm,
            storage_dtype=str(sampling.get("storage_dtype", "bfloat16")),
            dynamics=str(sampling.get("sde_type", "ode" if algorithm == "art" else "cps")),
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=1.0,
            timestep_shift=float(sampling.get("shift", 3.0)),
            t_eps=float(sampling.get("t_eps", 0.02)),
            noise_level=0.0 if algorithm == "art" else float(sampling["eta"]),
            sde_step_indices=[] if algorithm == "art" else resolve_sde_indices(sampling),
            return_step_indices=(sampling.get("art") or {}).get("train_step_indices") if algorithm == "art" else None,
        )
        request["_task"] = "i2i"
        if fixed_pixels:
            request["extra_sampling_params"]["fixed_input_image_pixels"] = fixed_pixels
            request["extra_sampling_params"]["max_input_image_side_length"] = max_side
        return request

    @staticmethod
    def _condition_image_paths(sample: Any) -> List[str]:
        for container in (getattr(sample, "multimodal_inputs", None), getattr(sample, "metadata", None)):
            if not container:
                continue
            for key in ("images", "image", "edit_image"):
                value = container.get(key)
                if not value:
                    continue
                paths = value if isinstance(value, (list, tuple)) else [value]
                return [str(path) for path in paths]
        raise ValueError(
            "SenseNova i2i requires at least one condition image in Sample.multimodal_inputs or metadata "
            "under 'images', 'image' or 'edit_image'."
        )

    def validate_rollout_response(self, response: Mapping[str, Any]) -> None:
        for key in (
            "trajectory_latents",
            "timesteps",
            "sde_indices",
            "height",
            "width",
            "seed",
            "policy_version",
            "weight_manifest_sha256",
            "condition_fingerprint",
            "scheduler_fingerprint",
            "sigmas",
            "state_indices",
            "transition_indices",
        ):
            if response.get(key) is None:
                raise ValueError(f"SenseNova rollout response missing required field {key!r}.")
        capture = (response.get("rollout_metadata") or {}).get("capture")
        if capture == "ode_states":
            for key in ("selected_indices", "old_velocities"):
                if response.get(key) is None:
                    raise ValueError(f"SenseNova ART response missing required field {key!r}.")
        elif capture != "terminal":
            if response.get("rollout_log_probs") is None:
                raise ValueError("SenseNova GRPO rollout response is missing rollout_log_probs.")
        env = response.get("denoising_env") or {}
        condition = env.get("pos_cond_kwargs") or env
        required = (
            "prefix_inputs_embeds",
            "prefix_indexes",
            "prefix_attention_mask",
            "indexes_image",
            "noise_scale",
        )
        if capture == "ode_states":
            required += ("t_eps",)
        missing = [name for name in required if condition.get(name) is None]
        if missing:
            raise ValueError(
                "SenseNova rollout denoising_env is missing canonical replay conditions "
                f"{missing}; export prefix inputs rather than the full prefix KV cache."
            )
        valid_mask = response.get("valid_mask", True)
        if isinstance(valid_mask, torch.Tensor) and valid_mask.numel() != 1:
            raise ValueError("A per-candidate SenseNova rollout response must carry a scalar valid_mask.")

    def pack_trajectory(self, response: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        art_capture = (response.get("rollout_metadata") or {}).get("capture") == "ode_states"
        if art_capture and any(
            not isinstance(response.get(key), torch.Tensor)
            or response[key].dtype != torch.float32
            or response[key].ndim != 1
            for key in ("timesteps", "sigmas")
        ):
            raise ValueError("ART requires original FP32 timesteps and sigmas")
        trajectory = _as_tensor(response["trajectory_latents"])
        timesteps = _as_tensor(response["timesteps"]).float().reshape(-1)
        sde_indices = _as_tensor(response["sde_indices"]).long().reshape(-1)
        sigmas = _as_tensor(response["sigmas"]).float().reshape(-1)
        if sigmas.numel() != timesteps.numel() + 1 or sigmas[-1] != 0:
            raise ValueError("SenseNova replay requires the exact sigma grid including terminal zero.")
        if not torch.allclose(sigmas[:-1] * 1000.0, timesteps):
            raise ValueError("SenseNova timesteps disagree with the actual sigma grid.")
        if art_capture and (
            not torch.isfinite(sigmas).all() or sigmas[0] != 1 or not torch.all(sigmas[:-1] > sigmas[1:])
        ):
            raise ValueError("ART requires the actual strictly descending sigma grid from 1 to 0")
        state_indices = [int(index) for index in _as_tensor(response["state_indices"]).tolist()]
        transition_indices = [int(index) for index in _as_tensor(response["transition_indices"]).tolist()]
        if len(state_indices) != trajectory.shape[0] or len(set(state_indices)) != len(state_indices):
            raise ValueError("SenseNova state_indices do not identify the stored states uniquely.")
        state_slots = {index: slot for slot, index in enumerate(state_indices)}
        score_slots = {index: slot for slot, index in enumerate(transition_indices)}
        indices = [int(index) for index in sde_indices.tolist()]
        invalid = [index for index in indices if index not in state_slots or index + 1 not in state_slots]
        if invalid:
            raise ValueError(f"SenseNova replay boundaries are missing for transitions {invalid}.")
        env = response.get("denoising_env") or {}
        condition = env.get("pos_cond_kwargs") or env
        if art_capture and condition.get("t_eps") is None:
            raise ValueError("ART requires actual replay t_eps")
        packed: Dict[str, torch.Tensor] = {
            "sigmas": sigmas,
            "sde_indices": sde_indices,
            "cond_prefix_inputs_embeds": _candidate_tensor(condition["prefix_inputs_embeds"], expected_ndim=2),
            "cond_prefix_indexes": _candidate_tensor(condition["prefix_indexes"], expected_ndim=2).long(),
            "cond_prefix_attention_mask": _candidate_tensor(condition["prefix_attention_mask"], expected_ndim=4),
            "cond_indexes_image": _candidate_tensor(condition["indexes_image"], expected_ndim=2).long(),
            "cond_noise_scale": _candidate_tensor(condition["noise_scale"]).float().reshape(()),
            "cond_t_eps": _as_tensor(condition.get("t_eps", 0.02)).float().reshape(()),
        }
        if art_capture:
            from .rollout import _validate_art_ode_cache

            selected = _validate_art_ode_cache(
                response["rollout_metadata"],
                trajectory,
                response.get("old_velocities"),
                num_steps=int(timesteps.numel()),
                state_indices=state_indices,
                expected_indices=response.get("selected_indices"),
            )
            if indices or transition_indices or response.get("rollout_log_probs") is not None:
                raise ValueError("ART ODE capture must not contain stochastic transitions or scores")
            selection = torch.tensor(selected, dtype=torch.long)
            packed.update(
                art_states=trajectory[:-1],
                art_old_velocities=response["old_velocities"],
                art_step_indices=selection,
                art_timesteps=timesteps[selection],
                art_sigmas=sigmas[selection],
                art_ode_dts=sigmas[selection + 1] - sigmas[selection],
            )
        else:
            if not indices:
                raise ValueError("GRPO capture requires at least one stochastic transition.")
            scores = _as_tensor(response["rollout_log_probs"]).float().reshape(-1)
            if scores.numel() != len(transition_indices) or any(index not in score_slots for index in indices):
                raise ValueError("SenseNova behavior scores do not cover the trained transition indices.")
            packed.update(
                image_x_t=torch.stack([trajectory[state_slots[index]] for index in indices]),
                image_x_next=torch.stack([trajectory[state_slots[index + 1]] for index in indices]),
                rollout_old_logp=torch.stack([scores[score_slots[index]] for index in indices]),
            )
        if timesteps.numel() in state_slots:
            packed["terminal_state"] = trajectory[state_slots[int(timesteps.numel())]]
        if response.get("transition_noise") is not None:
            packed["transition_noise"] = _as_tensor(response["transition_noise"])
        if response.get("model_outputs") is not None:
            packed["rollout_model_outputs"] = _as_tensor(response["model_outputs"])
        if response.get("step_valid_mask") is not None:
            packed["step_valid_mask"] = _as_tensor(response["step_valid_mask"]).bool()
        return packed

    def prepare_replay_batch(self, model: torch.nn.Module, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recompute the frozen prefix once and reuse it for every replay
        step."""
        if batch.get("_sensenova_prefix_owner") is model and "_sensenova_prefix_cache" in batch:
            return batch
        dit = _base_model(model)
        embeds = batch["cond_prefix_inputs_embeds"]
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        indexes = _shared_condition(batch["cond_prefix_indexes"], expected_ndim=2, name="prefix_indexes")
        attention_mask = _shared_condition(
            batch["cond_prefix_attention_mask"], expected_ndim=4, name="prefix_attention_mask"
        )
        with torch.no_grad():
            prefix_cache, _ = dit(
                prefix_inputs_embeds=embeds,
                prefix_indexes=indexes,
                prefix_attention_mask=attention_mask,
            )
        prepared = dict(batch)
        prepared["_sensenova_prefix_cache"] = prefix_cache
        prepared["_sensenova_prefix_owner"] = model
        return prepared

    def predict_training_velocity(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        noised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a fresh forward-process state, without rollout step
        indices."""
        if "_sensenova_prefix_cache" not in batch:
            raise ValueError("SenseNova training batch must be prepared before velocity evaluation.")
        timestep = batch.get("_replay_timestep")
        if timestep is None:
            raise ValueError("ART replay requires the original scheduler timestep.")
        return _base_model(model)(
            replay_latents=noised,
            # ART preserves the original sampled coordinate. Reconstructing
            # it from projection sigma introduces an extra rounding boundary.
            replay_timestep=timestep,
            past_key_values=batch["_sensenova_prefix_cache"],
            indexes_image=_shared_condition(batch["cond_indexes_image"], expected_ndim=2, name="indexes_image"),
            replay_noise_scale=batch["cond_noise_scale"].reshape(-1),
            replay_t_eps=batch.get("cond_t_eps", 0.02),
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False,
        )

    def replay_transition(self, model: torch.nn.Module, batch: Mapping[str, Any], step_index: int) -> torch.Tensor:
        if "_sensenova_prefix_cache" not in batch:
            raise ValueError("SenseNova replay batch was not prepared; call prepare_replay_batch once first.")
        indices = [int(index) for index in batch["sde_indices"].tolist()]
        try:
            slot = indices.index(int(step_index))
        except ValueError as exc:
            raise ValueError(f"SenseNova replay step {step_index} is not in sde_indices {indices}.") from exc
        latents = batch["image_x_t"][:, slot]
        indexes_image = _shared_condition(batch["cond_indexes_image"], expected_ndim=2, name="indexes_image")
        noise_scale = batch["cond_noise_scale"]
        if noise_scale.ndim > 1:
            noise_scale = noise_scale.reshape(noise_scale.shape[0], -1)[:, 0]
        return _base_model(model)(
            replay_latents=latents,
            replay_timestep=batch["sigmas"][int(step_index)] * 1000.0,
            past_key_values=batch["_sensenova_prefix_cache"],
            indexes_image=indexes_image,
            replay_noise_scale=noise_scale,
            replay_t_eps=batch.get("cond_t_eps", 0.02),
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False,
        )

    def artifact_tracks(self, response: Mapping[str, Any]) -> List[ArtifactTrack]:
        output = response.get("output") or {}
        return [
            ArtifactTrack(
                track="image",
                uri=str(output.get("uri", "")),
                mime=str(output.get("mime", "image/png")),
                sha256=str(output.get("sha256", "")),
                meta={key: output[key] for key in ("height", "width") if key in output},
            )
        ]

    def weight_name_map(self, name: str) -> str:
        for prefix in ("base_model.model.", "dit."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _candidate_tensor(value: Any, *, expected_ndim: int = 0) -> torch.Tensor:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    tensor = _as_tensor(value)
    if tensor.ndim == expected_ndim + 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != expected_ndim:
        raise ValueError(f"SenseNova canonical condition must have rank {expected_ndim}, got {tensor.ndim}.")
    return tensor


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def _shared_condition(value: torch.Tensor, *, expected_ndim: int, name: str) -> torch.Tensor:
    if value.ndim == expected_ndim:
        return value
    if value.ndim != expected_ndim + 1:
        raise ValueError(f"SenseNova {name} must have rank {expected_ndim} or {expected_ndim + 1}, got {value.ndim}.")
    first = value[0]
    if value.shape[0] > 1 and not torch.equal(value, first.unsqueeze(0).expand_as(value)):
        raise ValueError(f"SenseNova {name} differs across candidates in one replay micro-batch.")
    return first
