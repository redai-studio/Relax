# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART global window planning and cached ODE replay, without Ray/FSDP imports.

Group aggregation and selected-state capture must precede this API; never feed
NFT re-noised states or re-evaluate old velocities after the first update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from relax.models.diffusion_art import diffusion_art_loss


def art_update_windows(
    group_ids: Sequence[int],
    *,
    world_size: int = 1,
    seed: int = 42,
    rollout_id: int = 0,
    order: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Plan [two updates][rank][global candidate row] on CPU.

    Each prompt instance contributes K/2 candidates to each update. Each
    candidate carries ALL selected training times. Rows are then balanced
    across ranks, NOT projected onto their former candidate-strided owner. The
    caller must route caches using these global row identities. No row is
    replicated or dropped. Unequal rank budgets fail rather than pad.
    """
    groups = list(group_ids)
    if not groups or any(type(group) is not int or group < 0 for group in groups):
        raise ValueError("group_ids must be nonempty nonnegative integer conditioning-instance IDs.")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer.")
    if type(seed) is not int or type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("seed and nonnegative rollout_id must be integers.")
    if len(groups) % (2 * world_size):
        raise ValueError("Each ART update must have an equal nonempty candidate budget on every rank.")
    if order is None:
        generator = torch.Generator(device="cpu").manual_seed((seed + 1000003 * rollout_id) % 2**63)
        rows = torch.randperm(len(groups), generator=generator).tolist()
    else:
        rows = list(order)
    if any(type(row) is not int for row in rows) or sorted(rows) != list(range(len(groups))):
        raise ValueError("order must be a permutation of all global candidate rows.")
    members: Dict[int, list[int]] = {}
    for row in rows:
        members.setdefault(groups[row], []).append(row)
    if any(len(group) % 2 for group in members.values()):
        raise ValueError("Every ART prompt group must have even K.")
    assignments = {}
    for group in members.values():
        for position, row in enumerate(group):
            assignments[row] = int(position >= len(group) // 2)
    windows = [tuple(row for row in rows if assignments[row] == update) for update in range(2)]
    return tuple(tuple(window[rank::world_size] for rank in range(world_size)) for window in windows)


@dataclass(frozen=True)
class ARTTrainingStep:
    state: torch.Tensor
    old_velocity: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    ode_velocity_coefficient: torch.Tensor


def art_steps_from_batch(batch: Mapping[str, torch.Tensor]) -> Tuple[ARTTrainingStep, ...]:
    """View a validated packed cache as time-major replay steps, without casts.

    Candidate tensors use [B, S, ...]; original scheduler coordinates are
    shared [S], even when B happens to equal S. Protocol validation and row
    routing happen before this helper; it never re-noises or reconstructs
    velocity from stored state differences.
    """
    states, old = batch["art_states"], batch["art_old_velocities"]
    if states.ndim < 3 or states.numel() == 0 or old.shape != states.shape or old.dtype != torch.float32:
        raise ValueError("ART requires nonempty [B,S,...] states and aligned original FP32 old velocities.")
    count, selected = states.shape[:2]
    coordinates = [batch[key] for key in ("art_timesteps", "art_sigmas", "art_ode_dts")]
    if any(value.shape != (selected,) or value.dtype != torch.float32 for value in coordinates):
        raise ValueError("ART scheduler coordinates must retain shared FP32 shape [selected steps].")
    times, sigmas, dts = coordinates
    return tuple(
        ARTTrainingStep(
            state=states[:, slot],
            old_velocity=old[:, slot],
            timestep=times[slot].expand(count),
            sigma=sigmas[slot].expand(count),
            ode_velocity_coefficient=dts[slot].expand(count),
        )
        for slot in range(selected)
    )


def art_backward(
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: Sequence[ARTTrainingStep],
    *,
    guidance_direction: torch.Tensor,
    guidance_magnitude: torch.Tensor,
    global_item_count: int,
    data_parallel_world_size: int = 1,
    sample_valid_mask: Optional[torch.Tensor] = None,
    corrective_mask_enabled: bool = True,
) -> Dict[str, torch.Tensor]:
    """Backward cached states, leaving optimizer boundaries to the caller.

    ``global_item_count`` includes every candidate/time item in the complete
    update window, not just this microbatch or active rows. The world-size
    factor compensates for data-parallel gradient averaging. Keep identical
    forward/backward counts on all ranks, even for all-masked local rows. The
    caller may skip a globally signal-free rollout, but must not skip an
    optimizer step merely because a later window becomes fully masked.
    """
    if not steps:
        raise ValueError("ART requires at least one cached training step.")
    if type(global_item_count) is not int or global_item_count < 1:
        raise ValueError("global_item_count must be a positive integer.")
    if type(data_parallel_world_size) is not int or data_parallel_world_size < 1:
        raise ValueError("data_parallel_world_size must be a positive integer.")
    shape = guidance_direction.shape
    if len(shape) < 2 or shape[0] == 0 or guidance_magnitude.shape != (shape[0],):
        raise ValueError("Guidance must have candidate/feature dimensions and one magnitude per candidate.")
    if global_item_count < len(steps) * shape[0] * data_parallel_world_size:
        raise ValueError("global_item_count is smaller than the equal-rank microbatch item budget.")
    if sample_valid_mask is not None and (
        sample_valid_mask.dtype != torch.bool or sample_valid_mask.shape != (shape[0],)
    ):
        raise ValueError("sample_valid_mask must have boolean shape [candidates].")
    # Reject malformed cache shapes before any partial backward changes grads.
    for step in steps:
        if step.state.shape != shape or step.old_velocity.shape != shape:
            raise ValueError("Cached states, old velocities and guidance must have identical shapes.")
        for value in (step.timestep, step.sigma, step.ode_velocity_coefficient):
            if value.shape != (shape[0],):
                raise ValueError("Cached scheduler coordinates must have shape [candidates].")
    sums, kept, masked = [], [], []
    for step in steps:
        velocity = predict(step.state.detach(), step.timestep.detach())
        terms = diffusion_art_loss(
            velocity,
            step.old_velocity,
            step.sigma,
            guidance_direction,
            guidance_magnitude,
            ode_velocity_coefficient=step.ode_velocity_coefficient,
            corrective_mask_enabled=corrective_mask_enabled,
        )
        loss = terms["per_item_loss"]
        valid = torch.ones_like(terms["keep_mask"]) if sample_valid_mask is None else sample_valid_mask.to(loss.device)
        loss = loss * valid
        total = loss.sum()
        (total * (data_parallel_world_size / global_item_count)).backward()
        sums.append(total.detach())
        kept.append((terms["keep_mask"] & valid).detach().sum())
        masked.append((terms["masked"] & valid).detach().sum())
    return {
        "loss_sum": torch.stack(sums).sum(),
        "kept_items": torch.stack(kept).sum(),
        "masked_items": torch.stack(masked).sum(),
    }


def art_replay_update(
    model: torch.nn.Module,
    adapter: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    global_item_count: int,
    data_parallel_world_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Replay one routed microbatch; caller owns both optimizer boundaries.

    Prepare the frozen understanding prefix only once. Pass the ORIGINAL
    selected timestep and sigma independently: reconstructing either from the
    other introduces a rounding boundary. No old-policy forward, fresh noise,
    NFT EMA, log-prob or per-valid-row renormalization is involved.
    """
    steps = art_steps_from_batch(batch)
    minimum_items = batch["art_states"].shape[0] * len(steps) * data_parallel_world_size
    if type(global_item_count) is not int or global_item_count < minimum_items:
        raise ValueError("ART global_item_count must include every original candidate/time in the window.")
    prepared = adapter.prepare_replay_batch(model, batch)
    results = []
    for step in steps:

        def predict(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return adapter.predict_training_velocity(
                model, dict(prepared, _replay_timestep=timestep), state, step.sigma
            )

        results.append(
            art_backward(
                predict,
                [step],
                guidance_direction=batch["art_guidance_direction"],
                guidance_magnitude=batch["art_guidance_magnitude"],
                global_item_count=global_item_count,
                data_parallel_world_size=data_parallel_world_size,
                sample_valid_mask=batch["valid_masks"].reshape(-1).bool(),
            )
        )
    return {key: torch.stack([result[key] for result in results]).sum() for key in results[0]}
