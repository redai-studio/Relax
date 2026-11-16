# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART cache hydration and bounded candidate routing, separate from generation.

The caller supplies shared group loading/sharding so NFT and GRPO retain their
existing data path. No Ray calls, model forwards or optimizer updates live
here.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Mapping

import torch


def hydrate_art_micro_batches(
    args: Any,
    rollout_id: int,
    rows: Any,
    by_group: Dict[int, Dict[int, int]],
    dp_rank: int,
    dp_world: int,
    *,
    load_group: Callable[[int, List[int]], Dict[str, torch.Tensor]],
    shard_group: Callable[[Dict[str, torch.Tensor], int, List[int]], Dict[str, torch.Tensor]],
) -> List[Dict[str, Any]]:
    """Complete-group covariance, then globally balanced two-window routing.

    First profile replays one candidate at a time: every rank has exactly the
    same number of prefix/selected-time forwards, regardless of how many of its
    rows share a condition. No legacy full-group replication fallback. Group
    caches are released after aggregation. When the replay budget permits,
    retain owned B1 copies for this rank and consume each once during replay.
    Otherwise retain small descriptors and load candidates lazily as before.
    Ray owns acquisition caches until retention in either case.
    """
    from relax.backends.fsdp.art import art_update_windows
    from relax.engine.rollout.art_cache import art_cache_limits, check_art_cache_budget, tensor_payload_bytes
    from relax.models.diffusion_art import art_scalar_covariance

    if not 0 <= dp_rank < dp_world or int(getattr(args, "micro_batch_size", 1)) != 1:
        raise ValueError("ART requires a valid DP rank and micro_batch_size=1.")
    if "art_rewards" not in rows or "valid_masks" not in rows:
        raise ValueError("ART requires dedicated raw art_rewards and valid_masks columns.")
    skip_acquisition = False
    if "skip_optimizer_step" in rows:
        skips = rows["skip_optimizer_step"].bool()
        if bool(skips.any()) != bool(skips.all()):
            raise ValueError("ART requires one acquisition-wide skip flag, not partially skipped rows.")
        skip_acquisition = bool(skips.all())
    if (not skip_acquisition and not bool(rows["valid_masks"].bool().all())) or not bool(
        torch.isfinite(rows["art_rewards"]).all()
    ):
        raise ValueError("ART requires complete finite valid rewards, not partially filtered groups.")
    size = int(args.n_samples_per_prompt)
    if size < 2 or size % 2:
        raise ValueError("ART requires even K >= 2.")
    if any(set(slots) != set(range(size)) for slots in by_group.values()):
        raise ValueError("ART requires every configured candidate slot exactly once in each group.")
    identities = [(group, slot) for group in sorted(by_group) for slot in range(size)]
    if len(identities) != int(args.global_batch_size):
        raise ValueError("ART requires the complete acquisition batch before window planning.")
    sizes, guidance_sizes = [], []
    for group in sorted(by_group):
        positions = list(by_group[group].values())
        for key, values in (("art_cache_bytes", sizes), ("art_guidance_bytes", guidance_sizes)):
            if key not in rows:
                raise ValueError(f"ART requires publisher byte descriptors: {key}.")
            reported = rows[key][positions].reshape(-1)
            if not bool((reported == reported[0]).all()):
                raise ValueError(f"ART group {group} has mixed {key} descriptors.")
            values.append(int(reported[0]))
    check_art_cache_budget(args.sampling_config, sizes, guidance_sizes)
    plans = art_update_windows(
        [group for group, _slot in identities],
        world_size=dp_world,
        seed=getattr(args, "seed", 42),
        rollout_id=rollout_id,
    )
    destinations = {
        identities[row]: (update, position)
        for update, window in enumerate(plans)
        for position, row in enumerate(window[dp_rank])
    }
    # Bound each retained candidate by its complete group, including shared
    # conditions. Reserve the original covariance working set too. This is a
    # conservative tensor-payload bound, not an RSS or allocator guarantee.
    retained_bound = sum(
        sizes[index] * sum((group, slot) in destinations for slot in range(size))
        for index, group in enumerate(sorted(by_group))
    )
    retain_candidates = (
        retained_bound + sum(guidance_sizes) + 8 * max(sizes) <= art_cache_limits(args.sampling_config)[1]
    )
    batches = {}
    has_signal = False
    coordinates = None
    config = args.sampling_config.get("art", {})
    for group in sorted(by_group):
        positions = [by_group[group][slot] for slot in range(size)]
        batch = load_group(group, positions)
        if tensor_payload_bytes(batch) != int(rows["art_cache_bytes"][positions[0]]):
            raise ValueError("ART cache size disagrees with the publisher byte descriptor.")
        if batch["terminal_state"].shape[0] != size or batch["art_states"].shape[0] != size:
            raise ValueError("ART sidecar candidate count disagrees with complete K rows.")
        _validate_art_group_cache(batch, size)
        if (
            "valid_masks" not in batch
            or batch["valid_masks"].shape != (size,)
            or not bool(batch["valid_masks"].bool().all())
        ):
            raise ValueError("ART requires valid complete-group sidecar provenance.")
        current = tuple(batch[key] for key in ("art_step_indices", "art_timesteps", "art_sigmas", "art_ode_dts"))
        if coordinates is None:
            coordinates = tuple(value.clone() for value in current)
        elif any(not torch.equal(left, right) for left, right in zip(coordinates, current)):
            raise ValueError("ART groups must share identical selected scheduler coordinates.")
        if current[0].tolist() != config["train_step_indices"]:
            raise ValueError("ART sidecar selected steps disagree with the training configuration.")
        # A failed judge invalidates the acquisition, not a numeric reward.
        # Build only zero-direction routing descriptors for the actor's
        # existing global no-op path; never use failed scores in covariance.
        rewards = torch.zeros(size) if skip_acquisition else rows["art_rewards"][positions].reshape(-1)
        covariance = art_scalar_covariance(
            batch["terminal_state"],
            rewards,
            torch.zeros(size, dtype=torch.long),
            1,
            guidance_gain=float(config.get("guidance_gain", 1.0)),
        )
        has_signal = has_signal or bool(covariance["group_active"].any())
        if covariance["group_directions"].numel() * 4 != int(rows["art_guidance_bytes"][positions[0]]):
            raise ValueError("ART guidance size disagrees with the publisher byte descriptor.")
        owned_conditions = {}
        for slot in range(size):
            destination = destinations.get((group, slot))
            if destination is None:
                continue
            # Keep the actor's provenance view small; the loader owns any B1
            # tensors so they can be released immediately after consumption.
            small = {
                key: value
                for key, value in batch.items()
                if key not in ("art_states", "art_old_velocities", "terminal_state") and not key.startswith("cond_")
            }
            # A small view can still pin an entire Plasma/mmap backing object.
            # Own these bytes instead of retaining group storage through it.
            local = {key: value.clone() for key, value in shard_group(small, size, [slot]).items()}
            load = partial(
                _load_art_candidate,
                load_group,
                shard_group,
                group,
                rows,
                positions,
                slot,
                local,
                covariance["group_directions"],
                covariance["group_guidance_magnitudes"],
            )
            if retain_candidates:
                candidate = shard_group(batch, size, [slot])
                # Conditions are CPU inputs read by replay. Own shared fields once
                # per group on this rank; candidate-major index_select tensors
                # remain distinct. No candidate retains the Ray/mmap backing.
                for key, value in candidate.items():
                    if value is not batch[key]:
                        continue
                    if key.startswith("cond_"):
                        if key not in owned_conditions:
                            owned_conditions[key] = value.clone()
                        candidate[key] = owned_conditions[key]
                    else:
                        candidate[key] = value.clone()
                local = {key: candidate[key] for key in small}
                candidate["art_guidance_direction"] = covariance["group_directions"]
                candidate["art_guidance_magnitude"] = covariance["group_guidance_magnitudes"]
                load = partial(_take_art_candidate, candidate)
            batches[destination] = {
                "batch": local,
                "art_load": load,
                "advantages": torch.zeros(1),
                "art_rewards": rows["art_rewards"][positions[slot] : positions[slot] + 1],
                "step_indices": config["train_step_indices"],
                "art_update_index": destination[0],
                "art_global_item_count": len(identities) // 2 * len(current[0]),
                "art_dp_world": dp_world,
                "group_index": group,
                "local_idx": [slot],
            }
        del batch, covariance, owned_conditions
    ordered = [batches[key] for key in sorted(batches)]
    for mb in ordered:
        mb["skip_optimizer_step"] = not has_signal
    return ordered


def _take_art_candidate(candidate: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Transfer an already validated owned candidate to its sole replay use."""
    if not candidate:
        raise RuntimeError("ART owned candidate was already consumed; restore before retry.")
    result = dict(candidate)
    candidate.clear()
    return result


def _validate_art_group_cache(batch: Mapping[str, torch.Tensor], size: int) -> None:
    from relax.backends.fsdp.art import art_steps_from_batch

    art_steps_from_batch(batch)
    states, old, terminal = batch["art_states"], batch["art_old_velocities"], batch["terminal_state"]
    if states.dtype not in (torch.float16, torch.bfloat16) or terminal.shape != (size, *states.shape[2:]):
        raise ValueError("ART cache requires original FP16/BF16 states and aligned complete endpoints.")
    if states.shape[0] != size or any(value.device.type != "cpu" for value in batch.values()):
        raise ValueError("ART cache must be complete and CPU-owned before replay routing.")
    for value in (states, old, terminal, batch["art_timesteps"], batch["art_sigmas"], batch["art_ode_dts"]):
        if not _finite_cpu_tensor(value):
            raise ValueError("ART cache contains non-finite replay data.")
    if not bool(((batch["art_sigmas"] > 0) & (batch["art_sigmas"] <= 1)).all()) or not bool(
        (batch["art_ode_dts"] < 0).all()
    ):
        raise ValueError("ART cache requires positive input sigmas and negative ODE coefficients.")


def _finite_cpu_tensor(value: torch.Tensor) -> bool:
    """Check finite CPU replay values without a full-size boolean temporary."""
    if value.device.type != "cpu":
        raise ValueError("ART finite validation requires CPU-owned tensors.")
    if value.layout != torch.strided or value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return bool(torch.isfinite(value).all())
    if not value.numel():
        return True
    low, high = torch.aminmax(value)
    return bool(torch.isfinite(low) & torch.isfinite(high))


def _load_art_candidate(
    load_group: Callable[[int, List[int]], Dict[str, torch.Tensor]],
    shard_group: Callable[[Dict[str, torch.Tensor], int, List[int]], Dict[str, torch.Tensor]],
    group: int,
    rows: Any,
    positions: List[int],
    slot: int,
    metadata: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    magnitude: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Lease one B1 copy; immutable Ray ownership remains with the producer."""
    from relax.engine.rollout.art_cache import tensor_payload_bytes

    batch = load_group(group, positions)
    if tensor_payload_bytes(batch) != int(rows["art_cache_bytes"][positions[0]]):
        raise ValueError("ART lazy cache size changed after planning.")
    _validate_art_group_cache(batch, len(positions))
    local = shard_group(batch, len(positions), [slot])
    for key, value in metadata.items():
        if not torch.equal(local[key], value):
            raise ValueError(f"ART lazy cache provenance/coordinates changed: {key}.")
    local["art_guidance_direction"] = direction
    local["art_guidance_magnitude"] = magnitude
    return local
