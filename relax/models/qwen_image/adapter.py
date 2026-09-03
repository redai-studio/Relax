# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qwen-Image T2I generative model adapter.

Implements the Qwen-Image RL alignment geometry against the Relax
:class:`GenerativeModelAdapter` protocol (design doc 4.2):

* 2x2 latent patch geometry: the packed DiT sequence is ``(lh/2) * (lw/2)``
  tokens of ``C*4`` channels over the latent grid ``latent_grid(h, w)``.
* raw-sigma timestep (no x1000), single positive-conditioned forward — CFG
  rollout is rejected at request build (see :meth:`build_rollout_request`).
* weight name map = strip the ``transformer.`` prefix (diffusers-native, no
  rename).

Pure geometry / request / packing / name-map logic is CPU-testable; the model
forward (``load_train_model`` / ``replay_transition``) uses lazy diffusers
imports and runs on GPU in production.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Tuple

import torch

from relax.models.generative import ArtifactTrack, resolve_sde_indices
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

__all__ = ["QwenImageAdapter", "qwen_image_sigmas", "latent_grid"]

VAE_SCALE_FACTOR = 8
LATENT_CHANNELS = 16
PATCH_SIZE = 2

QWEN_BASE_SHIFT = 0.5
QWEN_MAX_SHIFT = 0.9
QWEN_BASE_IMAGE_SEQ_LEN = 256
QWEN_MAX_IMAGE_SEQ_LEN = 8192
QWEN_SHIFT_TERMINAL = 0.02


def latent_grid(height: int, width: int) -> Tuple[int, int]:
    """Latent grid ``(latent_h, latent_w) = (2*(H//16), 2*(W//16))`` — always even."""
    return 2 * (height // (VAE_SCALE_FACTOR * 2)), 2 * (width // (VAE_SCALE_FACTOR * 2))


@lru_cache(maxsize=64)
def qwen_image_sigmas(num_inference_steps: int, height: int, width: int) -> Tuple[float, ...]:
    """Qwen-Image dynamic FlowMatch sigma schedule, terminal zero included."""
    steps = int(num_inference_steps)
    if steps <= 0:
        raise ValueError(f"qwen_image_sigmas requires num_inference_steps > 0; got {steps}.")

    latent_h = int(height) // VAE_SCALE_FACTOR
    latent_w = int(width) // VAE_SCALE_FACTOR
    image_seq_len = (latent_h // PATCH_SIZE) * (latent_w // PATCH_SIZE)
    mu = _calculate_dynamic_mu(
        image_seq_len,
        base_seq_len=QWEN_BASE_IMAGE_SEQ_LEN,
        max_seq_len=QWEN_MAX_IMAGE_SEQ_LEN,
        base_shift=QWEN_BASE_SHIFT,
        max_shift=QWEN_MAX_SHIFT,
    )

    exp_mu = math.exp(mu)
    if steps == 1:
        base_sigmas = [1.0]
    else:
        end = 1.0 / steps
        stride = (1.0 - end) / (steps - 1)
        base_sigmas = [1.0 - (i * stride) for i in range(steps)]

    shifted = [exp_mu / (exp_mu + ((1.0 / sigma) - 1.0)) for sigma in base_sigmas]
    tail = shifted[-1]
    if tail < 1.0:
        scale_factor = (1.0 - tail) / (1.0 - QWEN_SHIFT_TERMINAL)
        shifted = [1.0 - ((1.0 - sigma) / scale_factor) for sigma in shifted]
    return tuple(float(sigma) for sigma in shifted) + (0.0,)


def _calculate_dynamic_mu(
    image_seq_len: int,
    *,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (float(max_shift) - float(base_shift)) / (int(max_seq_len) - int(base_seq_len))
    b = float(base_shift) - m * int(base_seq_len)
    return int(image_seq_len) * m + b


class QwenImageAdapter:
    """Adapter for Qwen-Image (T2I)."""

    family = "qwen_image"
    supported_tasks: Tuple[str, ...] = ("t2i",)

    # Default LoRA targets (--fsdp-trainable-mode lora): the eight attention
    # projections of ``QwenImageTransformerBlock.attn`` -- image stream
    # (``to_*``) plus text stream (``add_*_proj`` / ``to_add_out``). SGLang's own
    # Qwen-Image DiT names these identically and leaves QKV unfused outside
    # SVDQuant, so an adapter trained here loads there without a rename.
    # ``attn.to_out.0`` indexes the Linear inside diffusers' ``to_out``
    # ModuleList; bare ``attn.to_out`` would match the ModuleList itself, which
    # PEFT cannot wrap.
    lora_target_modules: Tuple[str, ...] = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    )
    lora_task_type: str = "FEATURE_EXTRACTION"

    # -- model construction (lazy diffusers) ---------------------------------

    def load_train_model(self, config) -> torch.nn.Module:
        """Load the diffusers DiT; every parameter is trainable by design.

        The VAE and the text encoder are never part of this module — they live
        frozen inside the rollout engine — so there is nothing left to freeze
        here (a LoRA run re-freezes the base weights itself via PEFT).
        """
        from diffusers import QwenImageTransformer2DModel

        dtype = torch.bfloat16 if getattr(config, "fsdp_param_dtype", "bf16") == "bf16" else torch.float32
        load_kwargs: Dict[str, Any] = {"subfolder": "transformer", "torch_dtype": dtype}
        revision = getattr(config, "model_revision", None)
        if revision:
            load_kwargs["revision"] = str(revision)
        transformer = QwenImageTransformer2DModel.from_pretrained(config.model_path, **load_kwargs)
        return transformer

    # -- rollout request / response ------------------------------------------

    def build_rollout_request(self, sample, sampling: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        """Build a SGLang ``RolloutRequest`` dict (see SGLang io_struct).

        One request == one candidate. Asking the server for a whole group in one
        shot (``num_outputs_per_prompt = n``) would collapse a group's n
        text-encoder passes and n batch-1 DiT forwards into one batched
        generation, and the plumbing for it is all there -- ``normalize_output_seeds``
        derives ``[seed, seed+1, ...]``, ``_prepare_grouped_latents`` keeps the
        per-candidate noise bit-identical, ``_build_response`` slices the
        trajectory per candidate -- but the Qwen-Image denoising path does NOT
        repeat the text conditioning to the expanded batch: ``schedule_batch.py``
        multiplies the latent batch by ``num_outputs_per_prompt`` while
        ``encoder_hidden_states`` stays B=1, and the AdaLN modulation then dies in
        ``fuse_scale_shift_kernel`` with ``expand([1, 27, 3072])`` vs ``[2, 1, 3072]``.
        Measured live, not inferred. Until that is fixed upstream, keep n=1 and get
        the parallelism from spreading candidates across engines instead
        (``_dispatch_groups``).

        Classifier-free guidance is rejected: :meth:`replay_transition` runs ONE
        positive-conditioned forward, so a CFG rollout would optimize a different
        policy than the one that generated the sample.
        """
        sde_indices = resolve_sde_indices(sampling)
        height = int(sampling.get("height", 384))
        width = int(sampling.get("width", 384))
        guidance_scale = float(sampling.get("guidance_scale", 1.0))
        if guidance_scale != 1.0:
            raise ValueError(
                f"guidance_scale={guidance_scale} is not supported: the actor replays a single "
                "positive-conditioned forward, so a CFG rollout would train against a different "
                "policy than it sampled from. Two-forward CFG replay is not implemented; set "
                "sampling_config.guidance_scale to 1.0."
            )
        req: Dict[str, Any] = {
            "prompt": sample.prompt,
            "seed": int(seed),
            "height": height,
            "width": width,
            "num_inference_steps": int(sampling.get("num_inference_steps", 12)),
            "num_outputs_per_prompt": 1,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": guidance_scale,
            # RL rollout flags (SGLang upstreamed rollout API)
            "rollout": True,
            "rollout_sde_type": str(sampling.get("sde_type", "sde")),
            # No default: preflight requires `eta`, and a default here would be a
            # second source of truth for the noise level the actor replays against.
            "rollout_noise_level": float(sampling["eta"]),
            "rollout_sde_step_indices": sde_indices or None,
            "rollout_return_dit_trajectory": True,
            "rollout_return_denoising_env": True,  # frozen conditions for replay
            "_task": "t2i",  # engine echoes this back into the mapped response
        }
        neg = (sample.metadata or {}).get("negative_prompt")
        if neg:
            req["negative_prompt"] = neg
        extra = dict(req.get("extra_sampling_params") or {})
        if bool(sampling.get("driver_sigmas", True)):
            extra["sigmas"] = list(qwen_image_sigmas(req["num_inference_steps"], height, width)[:-1])
        if bool(sampling.get("driver_xt", True)):
            latent_shape = sampling.get("init_noise_latent_shape")
            if latent_shape is None:
                latent_shape = [LATENT_CHANNELS, *latent_grid(height, width)]
            sample_index = getattr(sample, "index", None)
            sample_id = str(
                sampling.get("_sample_id") or f"sample:{sample_index if sample_index is not None else seed}"
            )
            rollout_id = int(sampling.get("_rollout_id", 0))
            extra.update(
                {
                    "initial_noise_group_ids": [f"r{rollout_id}:{sample_id}"],
                    "initial_noise_latent_shape": [int(v) for v in latent_shape],
                    "initial_noise_seed": int(sampling.get("_base_seed", seed)),
                    "denoise_seeds": [sample_id],
                }
            )
        if extra:
            req["extra_sampling_params"] = extra
        return req

    def validate_rollout_response(self, response: Mapping[str, Any]) -> None:
        # ``height`` / ``width`` are echoed from the request by the engine's
        # response mapper: they are the only source of the latent grid, and the
        # packed sequence length alone cannot recover a non-square one.
        for key in ("trajectory_latents", "timesteps", "sde_indices", "height", "width"):
            if response.get(key) is None:
                raise ValueError(f"Qwen rollout response missing required field {key!r}.")

    # -- trajectory packing ---------------------------------------------------

    def pack_trajectory(self, response: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Pack one candidate's (engine-mapped) response into sidecar tensors.

        Consumes the dict produced by ``SGLangNativeGenerationEngine._map_response``:
        ``trajectory_latents`` ``[T+1, ...]``, ``timesteps`` ``[T]``,
        ``sde_indices`` (echoed from the request), ``height`` / ``width``
        (echoed from the request), ``rollout_log_probs`` (the engine-emitted
        π_old anchor) and ``denoising_env`` (frozen conditions).
        For flow-matching, ``sigma = timestep / 1000`` — the FlowGRPO replay math
        works in sigma space.

        The rollout contract uses packed ``[T+1,S,64]`` latents and stores
        conditioning tensors under ``denoising_env``.
        """
        traj = _as_tensor(response["trajectory_latents"])  # [T+1, ...]
        timesteps = _as_tensor(response["timesteps"]).float()
        sigmas = timesteps / 1000.0  # flow-matching sigma space
        sde_indices = _as_long(response["sde_indices"])
        idx = sde_indices.tolist()
        transition_count = min(int(traj.shape[0]) - 1, int(sigmas.numel()))
        invalid = [i for i in idx if not 0 <= int(i) < transition_count]
        if invalid:
            raise ValueError(
                f"pack_trajectory: sde_indices {invalid} are outside valid transitions [0, {transition_count})."
            )

        x_t = torch.stack([traj[i] for i in idx], dim=0)  # [S, seq, 64] (packed)
        x_next = torch.stack([traj[i + 1] for i in idx], dim=0)
        packed: Dict[str, torch.Tensor] = {
            "image_x_t": x_t,
            "image_x_next": x_next,
            "sigmas": sigmas,
            "sde_indices": sde_indices,
            # Latent grid of THIS candidate. The packed sequence length is
            # gh*gw, which does not determine (gh, gw) for a non-square image,
            # and height/width are independent sampling knobs — so the grid has
            # to travel with the trajectory or replay would have to guess it.
            "image_grid": torch.tensor(latent_grid(int(response["height"]), int(response["width"])), dtype=torch.long),
        }
        # Conditions live in denoising_env.pos_cond_kwargs (confirmed live schema):
        # encoder_hidden_states [L, D], encoder_hidden_states_mask, txt_seq_lens.
        # The negative branch is deliberately NOT packed: replay is a single
        # positive-conditioned forward and build_rollout_request rejects CFG.
        env = response.get("denoising_env") or {}
        pos = env.get("pos_cond_kwargs") or {}

        def _first(v):
            return v[0] if isinstance(v, list) else v

        if pos.get("encoder_hidden_states") is not None:
            packed["cond_encoder_hidden_states"] = _as_tensor(_first(pos["encoder_hidden_states"]))
        if pos.get("encoder_hidden_states_mask") is not None:
            packed["cond_encoder_mask"] = _as_tensor(_first(pos["encoder_hidden_states_mask"]))
        if pos.get("txt_seq_lens") is not None:
            packed["cond_txt_seq_lens"] = _as_long(pos["txt_seq_lens"])
        # π_old anchor: per-step log-probs the engine emitted during sampling.
        if response.get("rollout_log_probs") is not None:
            packed["rollout_old_logp"] = _as_tensor(response["rollout_log_probs"]).float()
        return packed

    # -- replay ---------------------------------------------------------------

    def replay_transition(self, model, batch: Mapping[str, Any], step_index: int) -> torch.Tensor:
        """Recompute the packed velocity prediction at ``step_index``.

        Trajectory latents are stored **packed** ``[B, S, 64]`` (SGLang's DiT
        works in packed space), so no pack/unpack is needed — the diffusers
        ``QwenImageTransformer2DModel`` consumes packed hidden states directly
        and returns packed velocity. Conditions come from the stored
        denoising_env. Returns ``noise_pred[B, S, 64]`` aligned with
        x_t/x_next.
        """
        transition_count = int(batch["sigmas"].numel())
        if not 0 <= int(step_index) < transition_count:
            raise ValueError(
                f"replay_transition: step_index {step_index} is outside valid transitions [0, {transition_count})."
            )
        try:
            slot = batch["sde_indices"].tolist().index(int(step_index))
        except ValueError as exc:
            raise ValueError(f"replay_transition: step_index {step_index} is not present in sde_indices.") from exc
        x_t = batch["image_x_t"][:, slot]  # [B, S, 64] packed
        b, seq, _pc = x_t.shape
        # Patch grid from the stored latent grid: S = (lh/2) * (lw/2). One 2-element
        # host sync per replayed step, the same cost as the txt_seq_lens read below.
        img_shapes = [[(1, *self._patch_grid(batch["image_grid"], seq))]] * b

        sigma = batch["sigmas"][step_index]
        timestep = sigma.to(x_t.dtype).expand(b)
        embeds = batch["cond_encoder_hidden_states"]
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0).expand(b, -1, -1)
        mask = batch.get("cond_encoder_mask")
        txt_seq_lens = batch["cond_txt_seq_lens"].tolist() if "cond_txt_seq_lens" in batch else [embeds.shape[1]] * b

        out = model(
            hidden_states=x_t.to(embeds.dtype),
            timestep=timestep,
            encoder_hidden_states=embeds,
            encoder_hidden_states_mask=mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        return out

    @staticmethod
    def _patch_grid(image_grid: torch.Tensor, seq: int) -> Tuple[int, int]:
        """``(gh, gw)`` of the packed DiT sequence from the stored latent grid.

        ``image_grid`` is packed per candidate, so it arrives as ``[B, 2]``
        after the group stack; every candidate of a group shares one geometry.
        The ``gh * gw == seq`` check is what turns a geometry mix-up into a
        crash instead of silently wrong RoPE positions.
        """
        grid = image_grid[0] if image_grid.dim() == 2 else image_grid
        lh, lw = int(grid[0]), int(grid[1])
        gh, gw = lh // PATCH_SIZE, lw // PATCH_SIZE
        if gh * gw != seq:
            raise ValueError(f"replay_transition: latent grid {(lh, lw)} implies {gh * gw} tokens, but x_t has {seq}.")
        return gh, gw

    # -- artifacts + weight map ----------------------------------------------

    def artifact_tracks(self, response: Mapping[str, Any]) -> List[ArtifactTrack]:
        out = response.get("output", {})
        return [
            ArtifactTrack(
                track="image",
                uri=out.get("uri", ""),
                mime=out.get("mime", "image/png"),
                sha256=out.get("sha256", ""),
                meta={k: out[k] for k in ("height", "width") if k in out},
            )
        ]

    def weight_name_map(self, name: str) -> str:
        return name[len("transformer.") :] if name.startswith("transformer.") else name


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _as_long(x) -> torch.Tensor:
    return _as_tensor(x).long()
