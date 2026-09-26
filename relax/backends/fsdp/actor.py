# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDP2 training actor for native generative RL (FlowGRPO over
diffusion/flow).

`FSDPTrainRayActor` implements the :class:`TrainRayActor` contract with an FSDP2
full-fine-tune backend. It is selected by ``--train-backend fsdp`` via
``_resolve_train_actor_class`` (design doc 7.1 / 8.3) and keeps the existing
Controller / Actor / Rollout orchestration unchanged.

The FlowGRPO optimizer update is factored into :meth:`flow_grpo_update`, a pure
``(model, batch, adapter) -> loss/metrics`` step with no Ray / TransferQueue
dependency, so the deterministic single-track and dual-track (T2AV) update paths
(the M1 exit criteria) are unit-testable on CPU. The Ray lifecycle methods wire
that core to the FSDP runtime, DCP checkpointing and the full-weight sync.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch

from relax.distributed.ray.train_actor import TrainRayActor
from relax.models import flow_grpo
from relax.models.generative import GenerativeModelAdapter
from relax.utils.logging_utils import get_logger
from relax.utils.utils import load_function


logger = get_logger(__name__)

__all__ = ["FSDPTrainRayActor", "flow_grpo_update", "WeightSyncError", "lr_at_step"]


def _adamw_kwargs(args) -> Dict[str, Any]:
    """AdamW hyperparameters from the shared optimizer flags.

    Without this the backend silently fell back to torch's defaults
    (``weight_decay=0.01``, ``betas=(0.9, 0.999)``) and ``--weight-decay`` /
    ``--adam-beta1`` / ``--adam-beta2`` / ``--adam-eps`` were no-ops.
    """
    kwargs: Dict[str, Any] = {"lr": float(args.lr)}
    weight_decay = getattr(args, "weight_decay", None)
    if weight_decay is not None:
        kwargs["weight_decay"] = float(weight_decay)
    beta1 = getattr(args, "adam_beta1", None)
    beta2 = getattr(args, "adam_beta2", None)
    if beta1 is not None and beta2 is not None:
        kwargs["betas"] = (float(beta1), float(beta2))
    eps = getattr(args, "adam_eps", None)
    if eps is not None:
        kwargs["eps"] = float(eps)
    return kwargs


def lr_at_step(args, step: int, base_lr: float) -> float:
    """Learning rate for optimizer ``step`` (0-based).

    Warmup is linear over ``--lr-warmup-iters`` optimizer steps; the post-
    warmup shape is chosen by ``--fsdp-lr-scheduler`` (default ``constant``,
    which preserves the validated full-FT behaviour) and decays toward ``--min-
    lr`` over ``--lr-decay-iters``. Units are OPTIMIZER STEPS, not rollouts —
    the generative actor takes several steps per rollout.
    """
    warmup = int(getattr(args, "lr_warmup_iters", 0) or 0)
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)

    style = getattr(args, "fsdp_lr_scheduler", "constant") or "constant"
    if style == "constant":
        return base_lr
    decay_iters = getattr(args, "lr_decay_iters", None)
    if not decay_iters:
        return base_lr
    min_lr = float(getattr(args, "min_lr", 0.0) or 0.0)
    span = max(1, int(decay_iters) - warmup)
    progress = min(1.0, max(0.0, float(step - warmup) / float(span)))
    if style == "linear":
        return min_lr + (base_lr - min_lr) * (1.0 - progress)
    if style == "cosine":
        import math

        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


class WeightSyncError(RuntimeError):
    """A colocate weight-sync transaction failed; the streamed version is
    INVALID.

    Raised by :meth:`FSDPTrainRayActor._run_weight_transaction` when any engine
    drops a bucket, the commit fails, or the post-commit checksum mismatches.
    :meth:`update_weights` catches it to trigger engine recovery + whole-round
    retry (fault tolerance) or to fail the job (design doc 10.3). No engine-side
    ``begin/abort`` staging is used — that is a planned SGLang-patch follow-up.
    """

    def __init__(self, version: int, message: str) -> None:
        super().__init__(f"weight sync v{version} failed: {message}")
        self.version = int(version)


def _collect_weight_sync_errors(local_error: Optional[str]) -> List[Tuple[int, str]]:
    """Collect a rank-local weight-sync failure over the global Gloo world."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return [(0, local_error)] if local_error is not None else []
    if dist.get_world_size(dist.group.WORLD) <= 1:
        return [(0, local_error)] if local_error is not None else []
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    errors: List[Optional[str]] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, local_error, group=group)
    return [(rank, error) for rank, error in enumerate(errors) if error is not None]


def _collect_rank_objects(value: Any) -> List[Any]:
    """All-gather a small control-plane value over the global Gloo world."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return [value]
    if dist.get_world_size(dist.group.WORLD) <= 1:
        return [value]
    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    values: List[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def _raise_weight_sync_errors(version: int, phase: str, errors: List[Tuple[int, str]]) -> None:
    if errors:
        detail = "; ".join(f"rank {rank}: {error}" for rank, error in errors)
        raise WeightSyncError(version, f"{phase} failed ({detail})")


# ---------------------------------------------------------------------------
# Pure FlowGRPO update core (CPU-testable, no Ray/TQ dependency)
# ---------------------------------------------------------------------------


def _replay_logp(
    model: torch.nn.Module,
    adapter: GenerativeModelAdapter,
    batch: Mapping[str, Any],
    step_indices: List[int],
    eta: float,
    sigma_max: float,
    sde_type: str = "sde",
) -> torch.Tensor:
    """Replay stored transitions and return PER-STEP log-probs, shaped ``[B,
    S]``.

    For each selected step the adapter recomputes the velocity prediction at
    the current weights; :func:`flow_grpo.replay_transition_logp` turns that
    into the Gaussian transition log-prob. One column per replayed step.

    The steps are deliberately NOT summed here. Summing makes the PPO ratio the
    PRODUCT of the per-step ratios, so a single ``clip_eps`` band has to contain
    the combined drift of all S steps (a ~sqrt(S)-S times tighter trust region
    per step), and one step drifting out clips the whole sample, zeroing the
    gradient of the other steps too. FlowGRPO clips each (sample, step)
    independently and mean-reduce afterwards; keeping the step axis lets
    :func:`grpo_clip_loss` do the same.
    """
    sigmas = batch["sigmas"]
    per_step: List[torch.Tensor] = []
    slots = _slot_map(batch)
    for step_index in step_indices:
        noise_pred = adapter.replay_transition(model, batch, step_index)
        slot = slots[int(step_index)]
        per_step.append(
            flow_grpo.replay_transition_logp(
                noise_pred,
                batch["image_x_t"][:, slot],
                batch["image_x_next"][:, slot],
                sigmas[step_index],
                sigmas[step_index + 1],
                eta=eta,
                sigma_max=sigma_max,
                reduce=True,
                sde_type=sde_type,
            )
        )
    return torch.stack(per_step, dim=1)


def _slot_map(batch: Mapping[str, Any]) -> Dict[int, int]:
    """``step_index -> dense storage slot``, built once per micro-batch.

    ``sde_indices`` is a CPU tensor; materializing it once per replay instead
    of once per (step, track) keeps the ``.tolist()`` host round-trip out of
    the inner loop.
    """
    return {int(step): slot for slot, step in enumerate(batch["sde_indices"].tolist())}


def flow_grpo_update(
    model: torch.nn.Module,
    adapter: GenerativeModelAdapter,
    batch: Mapping[str, Any],
    *,
    advantages: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    step_indices: List[int],
    eta: float = 1.0,
    sigma_max: float = 0.99,
    sde_type: str = "sde",
    clip_eps: float = 0.2,
    clip_eps_high: Optional[float] = None,
    loss_scale: float = 1.0,
) -> Dict[str, Any]:
    """One FlowGRPO micro-batch: replay -> clipped loss -> backward.

    ``old_logp`` is the frozen pi_old anchor, shaped ``[B, S]`` (one column per
    trained SDE step), so on the first on-policy step the ratio is exactly 1.
    Returns ``{loss, metrics, new_logp}``; the caller drives the optimizer step.

    The clipped objective is evaluated PER (sample, step) and mean-reduced over
    both axes -- the per-sample advantage is broadcast across steps. Collapsing
    the step axis first (summing the log-probs) would make the ratio the product
    of the per-step ratios and put all S steps inside one ``clip_eps`` band; see
    :func:`_replay_logp`.

    ``old_logp=None`` SELF-ANCHORS: the anchor becomes this call's own
    ``new_logp.detach()``. That is exact -- not an approximation -- on the first
    mini-epoch, because no optimizer step has fired yet, so the anchor the caller
    would otherwise capture in a separate ``no_grad`` replay is computed at
    exactly these weights. It buys back that whole second replay (one full
    transformer forward per SDE step per micro-batch, ~11% of the train phase at
    n=16), and the ratio == 1 / clip_fraction == 0 invariant is preserved
    bit-for-bit. Pass a real tensor for every later mini-epoch.
    """
    new_logp = _replay_logp(model, adapter, batch, step_indices, eta, sigma_max, sde_type=sde_type)

    if old_logp is None:
        old_logp = new_logp.detach()
    # One advantage per candidate, shared by all of its trained steps.
    adv = advantages.detach().reshape(-1, 1).expand_as(new_logp)
    loss_per_elem, metrics = flow_grpo.grpo_clip_loss(new_logp, old_logp, adv, clip_eps, clip_eps_high)
    loss = loss_per_elem.mean()
    (loss * loss_scale).backward()
    return {"loss": loss.detach(), "metrics": metrics, "new_logp": new_logp.detach()}


def _micro_batch_num_samples(mb: Mapping[str, Any]) -> int:
    advantages = mb.get("advantages")
    if advantages is None:
        raise ValueError("FSDP generative micro-batch missing 'advantages'.")
    return int(advantages.numel())


def _plan_micro_batch_updates(micro_batches: List[Mapping[str, Any]], num_updates: int) -> List[List[int]]:
    """Partition micro-batches into disjoint optimizer updates.

    CountPlanner-style: ``num_updates_per_batch`` splits the local
    rollout shard into N equal sample-count updates, and every sample
    contributes to exactly one optimizer step. Micro-batches must not straddle
    update boundaries; the native diffusion hydrator emits fixed-size chunks
    for the validated recipes, so this is normally a cheap consistency check.
    """
    if not micro_batches:
        return []
    n_updates = max(1, int(num_updates))
    if n_updates == 1:
        return [list(range(len(micro_batches)))]

    sizes = [_micro_batch_num_samples(mb) for mb in micro_batches]
    total = sum(sizes)
    if total <= 0:
        raise ValueError("FSDP generative train batch is empty.")
    if total % n_updates != 0:
        raise ValueError(
            f"num_updates_per_batch={n_updates} must evenly divide the local train sample count "
            f"({total}); got remainder {total % n_updates}."
        )

    target = total // n_updates
    updates: List[List[int]] = []
    current: List[int] = []
    current_count = 0
    for idx, size in enumerate(sizes):
        if size <= 0:
            raise ValueError(f"FSDP generative micro-batch {idx} has no samples.")
        if current_count + size > target:
            raise ValueError(
                f"micro-batch {idx} with {size} samples crosses a num_updates_per_batch boundary "
                f"(target update size {target}, current update already has {current_count}). "
                "Use a micro-batch size that divides each update evenly."
            )
        current.append(idx)
        current_count += size
        if current_count == target:
            updates.append(current)
            current = []
            current_count = 0

    if current or len(updates) != n_updates:
        raise ValueError(
            f"failed to build {n_updates} disjoint updates from {len(micro_batches)} micro-batches ({total} samples)."
        )
    return updates


# ---------------------------------------------------------------------------
# Ray training actor
# ---------------------------------------------------------------------------


class FSDPTrainRayActor(TrainRayActor):
    """Full-FT FSDP2 actor driving FlowGRPO over a generative model adapter."""

    def init(self, args, role, with_ref=False, with_opd_teacher=False) -> Optional[int]:
        super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        # Full-FT diffusion training peaks near the card limit; expandable segments
        # reclaims the "reserved but unallocated" fragmentation (the megatron actor
        # does the same) so the group-forward + AdamW-state allocation fits.
        from relax.utils.device import set_expandable_segments

        set_expandable_segments(True)

        # Metrics sink (mirrors megatron actor.py:147 / rollout.py:814): the actor
        # logs train/* and perf/* through ``tracking_utils``, which requires the
        # wandb / TensorBoard / ClearML / metrics-service adapter to be initialized
        # IN THIS process. Only rank 0 emits metrics, so only it inits. Without this
        # ``tracking_utils.log`` finds no adapter and every train metric is silently
        # dropped ("Metrics service adapter not initialized") — the visualization
        # backends then show only the RolloutManager's reward/rollout metrics, never
        # the training curves.
        if self._rank == 0:
            from relax.utils.tracking_utils import init_tracking

            init_tracking(args, primary=False)

        self.adapter: GenerativeModelAdapter = self._load_adapter(args)
        self.bundle = self._load_train_model_waved(args)
        self.model = getattr(self.bundle, args.fsdp_trainable_attr, self.bundle)

        # LoRA must be injected here: after the base weights are materialized
        # (``load_train_model`` returns the bundle already frozen down to the
        # policy module), but before fsdp2_wrap (once fully_shard has run every
        # parameter is a DTensor and PEFT's nn.Linear surgery no longer applies)
        # and before the optimizer, which snapshots the trainable parameter list.
        self._trainable_mode = getattr(args, "fsdp_trainable_mode", "full")
        self._lora_target_modules: List[str] = []
        self._lora_plan = None
        if self._trainable_mode == "lora":
            from relax.backends.fsdp.lora import inject_lora_adapter

            self._lora_target_modules = self._resolve_lora_target_modules(args)
            inject_lora_adapter(
                self.model,
                rank=int(args.lora_rank),
                alpha=int(args.lora_alpha),
                target_modules=self._lora_target_modules,
                dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
                task_type=self._lora_task_type(),
            )

        self._assert_trainable_boundary()

        from relax.backends.fsdp.runtime import fsdp2_wrap

        self.model = fsdp2_wrap(
            self.model,
            param_dtype=args.fsdp_param_dtype,
            reduce_dtype=args.fsdp_reduce_dtype,
            reshard_after_forward=args.fsdp_reshard_after_forward,
            cpu_offload=args.fsdp_cpu_offload,
            activation_checkpointing=getattr(args, "fsdp_activation_checkpointing", False),
            master_dtype=getattr(args, "fsdp_master_dtype", None),
        )
        if self._trainable_mode == "lora":
            from relax.backends.fsdp.lora import build_lora_sync_plan

            # Built once, after sharding: the plan holds parameter OBJECTS, whose
            # .data the optimizer and the offload/onload helpers mutate in place.
            self._lora_plan = build_lora_sync_plan(self.model)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], **_adamw_kwargs(args)
        )
        # Step-driven LR schedule (see :func:`lr_at_step`). Counts OPTIMIZER steps,
        # not rollouts, and round-trips through the checkpoint so a resumed run does
        # not restart warmup from zero.
        self._lr_steps = 0
        self._base_lr = float(args.lr)
        self.policy_version = 0
        # Observability counters surfaced as train/* series (a rolled-back or
        # retried weight sync must be correlatable with a reward regression).
        self._checkpoint_failures = 0
        self._weight_sync_failures = 0
        self._synced_versions = 0
        self._last_sync_tensors = 0
        self._last_sync_bytes = 0
        self._data_system_client = None  # lazy TransferQueue client for reading train partitions
        self._base_model_sha256 = self._hash_base_model(args)
        self._assert_sync_plan_aligned()
        # Publish the parallel config so TrainRayActor.set_rollout_manager (called
        # by the Controller after init) can forward it to the RolloutManager.
        self._get_parallel_config()

        # Torch / memory profiler (backend-agnostic; honours --use-pytorch-profiler,
        # --profile-target, --record-memory-history and the memory-snapshot flags).
        # Without this the whole profiling flag group parsed and did nothing.
        from relax.utils.profile_utils import TrainProfiler

        self.profiler = TrainProfiler(args)
        self.profiler.on_init_end()

        start_rollout_id = self._maybe_resume()

        # Vacate the GPU after init so the first rollout's engine onload has room.
        # The Controller calls update_weights() right after set_rollout_manager
        # (components/actor.py), which wakes the model back up; without this the
        # full-FT model + AdamW stay resident from init through that first weight
        # sync (which also onloads the engine transformer) → OOM on a tight card.
        # Mirrors the megatron actor (backends/megatron/actor.py init tail).
        if getattr(self.args, "offload_train", False):
            self.sleep()

        # Open the inter-step "train_wait" interval (mirrors the megatron actor's
        # with_defer(Timer().start("train_wait")) on init) so log_perf_data_raw can
        # emit perf/step_time (= train_wait + train) and perf/wait_time_ratio for
        # the diffusion path.
        from relax.utils.timer import Timer

        Timer().start("train_wait")
        return start_rollout_id

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _load_adapter(args) -> GenerativeModelAdapter:
        adapter_cls = load_function(args.model_adapter_path)
        return adapter_cls()

    @staticmethod
    def _device_group():
        """The explicit device (NCCL) world group for this actor's collectives.

        CLAUDE.md requires every collective to name its process group rather
        than rely on the implicit default. FSDP here is pure DP (dp_size ==
        world_size), so the device group IS the world group -- but naming it
        keeps the call sites honest and makes a future sub-group split a one-
        line change instead of an audit.
        """
        import torch.distributed as dist

        return dist.group.WORLD

    def _rank0_then_agree(self, work, *, what: str) -> None:
        """Run rank-0-only ``work``, then make its outcome unanimous.

        The naive shape -- ``if rank == 0: work()`` followed by
        ``dist.barrier()`` -- DEADLOCKS on failure: rank 0 unwinds past the
        barrier while every other rank blocks on it forever, so the job hangs
        instead of raising, and the caller's rollback never runs on any rank.
        (The rollback path's own comment assumes "every rank lands here", which
        is exactly what a rank-0-only raise breaks.)

        Broadcasting the outcome instead turns a rank-0 failure into the same
        exception on every rank. The broadcast doubles as the barrier, so this
        is not an extra collective. Runs on the gloo group: it carries a Python
        object and must work while the device group is busy or unhealthy.
        """
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group() if dist.is_initialized() and dist.get_world_size(dist.group.WORLD) > 1 else None
        outcome: List[Optional[str]] = [None]
        if group is None or dist.get_rank(group) == 0:
            try:
                work()
            except Exception as exc:  # re-raised below on every rank
                outcome[0] = f"{type(exc).__name__}: {exc}"
        if group is not None and dist.get_world_size(group) > 1:
            dist.broadcast_object_list(outcome, src=0, group=group)
        if outcome[0] is not None:
            raise RuntimeError(f"{what} failed on rank 0: {outcome[0]}")

    def _load_train_model_waved(self, args) -> Any:
        """Load the full base model in rank waves to bound the launch
        footprint.

        Colocate places one FSDP rank per GPU (world == num GPUs); each rank
        materializes the *full* base model on CPU before ``fsdp2_wrap`` shards it
        to its GPU. Doing all ``world`` loads at once means ``world`` full CPU
        copies plus ``world`` concurrent readers on the (often networked) model
        directory -- for a 20B bf16 model that is ~40GB/rank, enough to thrash
        host RAM and stall the launch. Gating the loads to ``fsdp_load_wave_size``
        ranks at a time (barrier between waves on the CPU/gloo group) caps the
        peak while keeping the load parallel. ``fsdp_load_wave_size <= 0`` (or a
        single-rank world) disables gating.
        """
        import torch.distributed as dist

        if dist.is_initialized():
            if dist.get_world_size(dist.group.WORLD) > 1:
                from relax.utils.distributed_utils import get_gloo_group

                group = get_gloo_group()
            else:
                group = dist.group.WORLD
            world = dist.get_world_size(group)
            rank = dist.get_rank(group)
        else:
            world = 1
            rank = 0
        wave_size = int(getattr(args, "fsdp_load_wave_size", 0) or 0)
        if wave_size <= 0 or world <= 1:
            return self.adapter.load_train_model(args)

        bundle = None
        for wave_start in range(0, world, wave_size):
            local_error = None
            if wave_start <= rank < wave_start + wave_size:
                try:
                    logger.info(f"[rank {rank}] loading base model (wave {wave_start}-{wave_start + wave_size - 1})")
                    bundle = self.adapter.load_train_model(args)
                except Exception as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
            errors = _collect_weight_sync_errors(local_error)
            if errors:
                detail = "; ".join(f"rank {failed_rank}: {error}" for failed_rank, error in errors)
                raise RuntimeError(f"Waved model load failed ({detail})")
        return bundle

    def _resolve_lora_target_modules(self, args) -> List[str]:
        """CLI list, else the model adapter's own default, else raise.

        The target names are diffusers module-name suffixes and are irreducibly
        per-family, so the adapter owns the default. It is read with
        ``getattr`` rather than declared on the :class:`GenerativeModelAdapter`
        Protocol so that adding it never invalidates an existing adapter -- the
        Protocol is ``@runtime_checkable`` and is ``isinstance``-asserted by
        the adapter unit tests.
        """
        explicit = getattr(args, "lora_target_modules", None)
        if explicit:
            return [str(t) for t in explicit]
        default = getattr(self.adapter, "lora_target_modules", None)
        if default:
            return [str(t) for t in default]
        raise ValueError(
            f"fsdp_trainable_mode='lora' needs target modules: pass --lora-target-modules, or give "
            f"{type(self.adapter).__name__} a 'lora_target_modules' attribute."
        )

    def _lora_task_type(self) -> str:
        return str(getattr(self.adapter, "lora_task_type", "FEATURE_EXTRACTION"))

    def _assert_trainable_boundary(self) -> None:
        """Enforce the configured trainable boundary at runtime (design doc
        13.1).

        Full FT: every parameter must be trainable. LoRA: the adapter must
        exist and the base must be frozen — a LoRA run that silently kept the
        base trainable would blow past its memory budget and, worse, would sync
        a base the engine's adapter was never trained against.
        """
        if self._trainable_mode == "lora":
            from relax.backends.fsdp.lora import is_lora_injected

            if not is_lora_injected(self.model):
                raise ValueError("fsdp_trainable_mode='lora' but no LoRA parameters were injected.")
            trainable_base = [n for n, p in self.model.named_parameters() if p.requires_grad and ".lora_" not in n]
            if trainable_base:
                raise ValueError(
                    f"fsdp_trainable_mode='lora' requires a frozen base; trainable non-adapter params: "
                    f"{trainable_base[:5]}..."
                )
            return
        frozen_trainable = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        if frozen_trainable:
            raise ValueError(
                f"train_backend=fsdp requires all transformer params trainable; frozen: {frozen_trainable[:5]}..."
            )

    def _hash_base_model(self, args) -> str:
        seed = f"{args.model_path}:{getattr(args, 'model_revision', None)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _named_trainable_params(self):
        """Parameters the optimizer owns.

        Under LoRA this is adapters only.
        """
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def _named_sync_params(self):
        """Parameters the weight sync streams.

        Deliberately distinct from :meth:`_named_trainable_params`: under LoRA
        merge-mode the engine still needs the *whole* transformer (with ``B @ A``
        folded in), which is a superset of what the optimizer owns. Full FT
        returns the identical list as before, so its behaviour is unchanged by
        construction rather than by inspection.
        """
        if self._lora_plan is None:
            return self._named_trainable_params()
        if self.args.weight_sync_mode == "adapter":
            from relax.backends.fsdp.lora import named_lora_params

            return named_lora_params(self.model)
        return self._lora_plan.named_base_params

    def _sync_tensor_source(self):
        """``tensor_source`` for the sync iterator; ``None`` outside merge-mode
        LoRA."""
        if self._lora_plan is None or self.args.weight_sync_mode == "adapter":
            return None
        return self._lora_plan.materialize

    # -- TrainRayActor contract ----------------------------------------------

    def _get_parallel_config(self) -> Dict[str, Any]:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.train_parallel_config = {
            "train_backend": "fsdp",
            "dp_size": world_size,
            "tp_size": 1,
            "pp_size": 1,
        }
        return self.train_parallel_config

    def _assert_microbatch_count_aligned(self, count: int, rollout_id: int) -> None:
        """Fail fast if DP ranks derived different micro-batch counts.

        Every rank must run the identical (micro-batch x update) sequence or
        the FSDP all-gather / reduce-scatter collectives desync and hang. The
        counts are identical by construction (rank 0 broadcasts the numeric
        rows and each rank derives the same group list), but a future
        independent-read path (see the DP-sharding TODO in
        ``_read_train_partition``) could break that, so MAX/MIN-reduce the
        count and raise a clear error instead of deadlocking mid-step. (old
        slime FSDP did a MAX all-reduce for the same reason.)
        """
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size(self._device_group()) <= 1:
            return
        # Pack (count, -count) so a single MAX reduce yields both the global max
        # and (negated) the global min.
        packed = torch.tensor([count, -count], device=self.device, dtype=torch.long)
        dist.all_reduce(packed, op=dist.ReduceOp.MAX, group=self._device_group())
        hi = int(packed[0])
        lo = -int(packed[1])
        if hi != lo:
            raise RuntimeError(
                f"[train {rollout_id}] FSDP ranks disagree on micro-batch count "
                f"(min={lo}, max={hi}); the training collectives would deadlock. "
                "This indicates a divergent train-partition read across ranks."
            )

    def _prepare_micro_batch(self, mb):
        """Move one micro-batch's tensors to the compute device.

        ``sde_indices`` and ``image_grid`` are index/geometry metadata used
        only for Python-side lookups (never in compute), so they stay on CPU:
        keeping them on GPU would turn every ``.tolist()`` in ``_slot_map`` and
        the adapter's ``replay_transition`` into a GPU->CPU sync, which
        CLAUDE.md forbids in hot paths.
        """
        host_only = ("sde_indices", "image_grid")
        batch = {
            k: (v if k in host_only else (v.to(self.device) if isinstance(v, torch.Tensor) else v))
            for k, v in mb["batch"].items()
        }
        return batch, mb["advantages"].to(self.device)

    def _freeze_anchor(self, batch, mb, eta: float, sigma_max: float, sde_type: str) -> torch.Tensor:
        """Frozen pi_old log-prob for one micro-batch."""
        with torch.no_grad():
            old_logp = _replay_logp(
                self.model, self.adapter, batch, mb["step_indices"], eta, sigma_max, sde_type=sde_type
            )
        if self._rank == 0 and getattr(self.args, "fsdp_debug_fingerprint", False):
            self._log_logp_parity(batch, old_logp, eta, sigma_max)
        return old_logp

    def _log_logp_parity(self, batch: Mapping[str, Any], old_logp: torch.Tensor, eta: float, sigma_max: float) -> None:
        """Compare the replayed pi_old anchor against the engine's sampling
        log- prob.

        The actor derives ``old_logp`` by REPLAYING the stored trajectory at the
        current weights, which forces ``ratio == 1`` on the first mini-epoch by
        construction -- so an actor-vs-engine mismatch is invisible in the normal
        metrics. The engine also returns its true per-step sampling log-prob
        (``rollout_log_probs`` -> ``batch['rollout_old_logp']``, zero-padded to the
        full schedule with entries only at the trained ``sde_indices``), which is
        the parity oracle: after a weight sync the two should agree, and a gap
        means the PPO importance ratio is anchored to the wrong behaviour policy.

        REDUCTION: SGLang's ``collect_rollout_log_probs`` divides the accumulated
        trajectory log-prob sum by its element count. The engine value and
        :func:`flow_grpo.replay_transition_logp` are therefore both per-element
        means and can be compared directly.

        Diagnostic only -- never raises.
        """
        try:
            engine = batch.get("rollout_old_logp")
            sigmas = batch.get("sigmas")
            idx = [int(i) for i in batch["sde_indices"].tolist()] if "sde_indices" in batch else []
            if sigmas is not None and idx:
                # Step 0 (sigma == 1) is where the derived coefficient is only
                # finite because of the sigma_max clamp; config validation
                # rejects training it, and this row is how you confirm the
                # schedule you actually got.
                rows = []
                for i in idx:
                    s, sn = sigmas[i].float(), sigmas[i + 1].float()
                    sdt = float(flow_grpo.flow_sde_transition_std_dev_t(s, eta, sigma_max))
                    var = float(flow_grpo.flow_sde_transition_std(s, sn, eta, sigma_max))
                    rows.append(f"step{i}(sigma={float(s):.4f}) std_dev_t={sdt:.5f} std_var={var:.5f}")
                logger.info("[logp parity/std] " + " | ".join(rows))
            if engine is None:
                logger.info("[logp parity] engine rollout_old_logp absent from the sidecar batch.")
                return
            engine = engine.detach().float()
            # The anchor is per (candidate, trained step), so compare against the
            # engine's per-step log-probs directly rather than summing them: a
            # summed comparison hides WHICH step disagrees, and the boundary step
            # is exactly the one that historically did.
            if engine.ndim > 1 and idx and engine.shape[-1] > max(idx):
                engine_sel = engine[..., idx]  # zero-padded schedule -> select trained steps
            else:
                engine_sel = engine
            actor = old_logp.detach().float()
            engine_sel = engine_sel.reshape(actor.shape) if engine_sel.numel() == actor.numel() else engine_sel
            if engine_sel.shape != actor.shape:
                logger.info(
                    f"[logp parity] shape mismatch engine={tuple(engine.shape)}->"
                    f"{tuple(engine_sel.shape)} actor={tuple(actor.shape)}; skipping comparison."
                )
                return
            engine_mean = engine_sel.to(actor.device)
            diff = actor - engine_mean
            log_r = diff.clamp(-20.0, 20.0)
            k3 = float((torch.expm1(log_r) - log_r).mean())
            per_step = ""
            if actor.ndim == 2 and actor.shape[1] == len(idx):
                per_step = (
                    " per-step k3=["
                    + ", ".join(
                        f"s{i}:{float((torch.expm1(log_r[:, c]) - log_r[:, c]).mean()):.4f}" for c, i in enumerate(idx)
                    )
                    + "]"
                )
            logger.info(
                f"[logp parity] n={actor.numel()} engine_mean={float(engine_mean.mean()):+.5f} "
                f"actor_mean={float(actor.mean()):+.5f} mean_diff={float(diff.mean()):+.5f} "
                f"ratio={float(torch.exp(log_r).mean()):.5f} k3={k3:.5f} (on-policy target ~0){per_step}"
            )
        except Exception as e:  # diagnostics must never break training
            logger.warning(f"[logp parity] skipped ({e}).")

    def _log_update(self, agg, rollout_id: int, update_index: int, start: int, skipped: bool) -> None:
        """Log one optimizer-update summary on rank 0 (means over micro-
        batches)."""
        if self._rank != 0:
            return

        def _tail_mean(key: str) -> float:
            vals = agg[key][start:]
            return sum(vals) / len(vals) if vals else 0.0

        logger.info(
            f"[rollout {rollout_id} u{update_index}] loss={_tail_mean('loss'):.5f} "
            f"ratio_mean={_tail_mean('ratio_mean'):.4f} ratio_std={_tail_mean('ratio_std'):.4f} "
            f"clip_frac={_tail_mean('clip_fraction'):.4f} "
            f"grad_norm={(agg['grad_norm'][-1] if agg['grad_norm'] else 0.0):.4f}"
            f"{' (skipped: degenerate)' if skipped else ''}"
        )

    def train(self, rollout_id: int, rollout_data_ref=None) -> None:
        """Consume this rollout's transitions and run FlowGRPO updates.

        Fetches the numeric TQ rows, hydrates each group's trajectory sidecar,
        then splits them into ``num_updates_per_batch`` disjoint optimizer
        updates, matching CountPlanner-style semantics. Single-update runs can
        self-anchor π_old on their own forward (see :func:`flow_grpo_update`);
        multi-update runs freeze every micro-batch anchor before any optimizer
        step moves the weights. The heavy diffusion replay is delegated to
        :func:`flow_grpo_update`.
        """
        from relax.utils.timer import Timer, timer

        # Bracket the whole step so log_perf_data_raw emits perf/step_time
        # (train_wait + train) and perf/wait_time_ratio, matching the megatron
        # actor's `with inverse_timer("train_wait"), timer("train")`. train_wait
        # was opened at init / reopened after the previous step.
        # `Timer.end` asserts the interval was started, so a step that raised
        # between start("train") and end("train") leaves train_wait closed and
        # makes the NEXT step assert right here -- masking the original error
        # behind a confusing timer failure. Only close it if it is actually open.
        if "train_wait" in Timer().start_time:
            Timer().end("train_wait")
        Timer().start("train")

        micro_batches = self._load_train_batches(rollout_id, rollout_data_ref)
        self._assert_microbatch_count_aligned(len(micro_batches), rollout_id)
        if not micro_batches:
            Timer().end("train")
            Timer().start("train_wait")
            raise RuntimeError(
                f"[rollout {rollout_id}] no training micro-batches were available; "
                "refusing to sync unchanged weights or advance policy_version."
            )
        # `eta` MUST come from the config, never from a local default: it is the
        # sampler's noise level, which defines the Gaussian this replay scores.
        # A default here that disagreed with the adapter's (they were 1.0 vs 0.7)
        # would silently anchor the PPO ratio to a distribution the sample was
        # never drawn from. Preflight requires the key, so there is no fallback.
        eta = float(self._sampling_cfg()["eta"])
        sigma_max = float(self._sampling_cfg().get("sigma_max", 0.99))
        sde_type = str(self._sampling_cfg().get("sde_type", "sde")).lower()

        # Onload the actor onto GPU (it is offloaded during rollout in colocate).
        # Model + optimizer together: the AdamW states are allocated ONCE (by the
        # first optimizer.step, right after the AC-reduced backward frees its
        # activations) and stay resident for the whole loop. This is the measured
        # peak either way (reserved ~59GB / 96GB), but resident avoids the per-step
        # 20GB alloc/free spike that fragmented the tightest rank mid-loop and
        # desynced the FSDP collectives (NCCL watchdog hang). Every rank now runs
        # identical optimizer ops at identical times.
        if getattr(self.args, "offload_train", False):
            wake_error = None
            try:
                self.wake_up(include_optimizer=True)
            except Exception as exc:
                wake_error = f"{type(exc).__name__}: {exc}"
            errors = _collect_weight_sync_errors(wake_error)
            if errors:
                Timer().end("train")
                Timer().start("train_wait")
                detail = "; ".join(f"rank {rank}: {error}" for rank, error in errors)
                raise RuntimeError(f"Train-model onload failed ({detail})")

        self.model.train()
        # Full-model abs-sum fingerprint is a debug-only diagnostic (a full-param
        # reduction + host sync); gate it so it does not run every step in prod.
        debug_fp = self._rank == 0 and getattr(self.args, "fsdp_debug_fingerprint", False)
        fp_before = self._trainable_fp() if debug_fp else 0.0
        agg: Dict[str, List[float]] = {
            k: []
            for k in (
                "loss",
                "ratio_mean",
                "ratio_std",
                "clip_fraction",
                "approx_kl",
                "grad_norm",
                "adv_mean",
                "adv_std",
            )
        }
        skips = 0
        # Throughput accounting: a "transition" is one replayed SDE step for one
        # candidate — the unit of work that actually dominates the diffusion train
        # step (each is a full transformer forward + backward).
        self._trained_transitions = 0
        self._trained_samples = 0
        # Ratio tails (MAX-reduced once per step, see _record_update_metrics).
        self._ratio_max_local = float("-inf")
        self._ratio_min_local = float("inf")
        num_updates = max(1, int(getattr(self.args, "num_updates_per_batch", 1)))
        clip_grad = getattr(self.args, "clip_grad", 1.0)
        clip_eps = float(getattr(self.args, "eps_clip", 0.2))
        # Gradient accumulation: ONE optimizer step covers the whole global
        # batch. Every micro-batch (= one prompt group) contributes a
        # 1/N-scaled backward and the step fires after the last one, so
        # --global-batch-size really is the outer batch boundary;
        # --num-updates-per-batch then partitions those micro-batches into
        # disjoint equal-size optimizer updates.
        #
        # The π_old anchor is normally SELF-anchored inside the u == 0
        # flow_grpo_update (see its docstring): no optimizer step has fired at
        # that point, so its own new_logp is the anchor, and the separate no_grad
        # replay it replaces is pure waste (~11% of this phase at n=16). The
        # explicit anchor is kept only for --fsdp-debug-fingerprint, whose parity
        # diagnostic needs the anchor as a standalone tensor to compare against
        # the engine's sampling log-prob. The flag is uniform across ranks, so
        # both branches issue the same FSDP collective sequence on every rank —
        # gating this on `self._rank == 0` instead would hang the job.
        explicit_anchor = bool(getattr(self.args, "fsdp_debug_fingerprint", False))
        with timer("actor_train"):
            update_plans = _plan_micro_batch_updates(micro_batches, num_updates)
            old_logps: Dict[int, torch.Tensor] = {}

            # For disjoint multi-update mode, later updates run AFTER the
            # first optimizer step, so their π_old anchors must be frozen
            # before any weights move. With one update we can keep the cheap
            # self-anchor path. Debug parity also needs explicit anchors.
            prefreeze_anchors = explicit_anchor or len(update_plans) > 1
            if prefreeze_anchors:
                for i, mb in enumerate(micro_batches):
                    batch, _advantages = self._prepare_micro_batch(mb)
                    old_logps[i] = self._freeze_anchor(batch, mb, eta, sigma_max, sde_type)
                    del batch, _advantages

            for u, update_indices in enumerate(update_plans):
                start = len(agg["loss"])
                self.optimizer.zero_grad(set_to_none=True)
                update_total = sum(_micro_batch_num_samples(micro_batches[i]) for i in update_indices)
                # The degenerate flag is set for the whole round by the
                # reward post-process, so it is uniform across
                # micro-batches AND ranks; `all` keeps a partially
                # degenerate batch stepping.
                skip_step = bool(update_indices) and all(
                    bool(micro_batches[i].get("skip_optimizer_step", False)) for i in update_indices
                )
                for i in update_indices:
                    mb = micro_batches[i]
                    batch, advantages = self._prepare_micro_batch(mb)
                    result = flow_grpo_update(
                        self.model,
                        self.adapter,
                        batch,
                        advantages=advantages,
                        # Missing only in the single-update fast path:
                        # self-anchor inside the update (identical anchor,
                        # one fewer full replay). Multi-update anchors were
                        # frozen above before any optimizer step.
                        old_logp=old_logps.get(i),
                        step_indices=mb["step_indices"],
                        eta=eta,
                        sigma_max=sigma_max,
                        sde_type=sde_type,
                        clip_eps=clip_eps,
                        loss_scale=float(_micro_batch_num_samples(mb)) / float(update_total),
                    )
                    old_logps.setdefault(i, result["new_logp"])
                    self._record_update_metrics(agg, result, advantages, rollout_id, u)
                    self._trained_transitions += len(mb["step_indices"]) * int(advantages.numel())
                    self._trained_samples += int(advantages.numel())
                    # Free this group's GPU copy before the next one — only the
                    # (param-shaped) accumulated grads persist across the loop.
                    del batch, advantages
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                if skip_step:
                    self.optimizer.zero_grad(set_to_none=True)
                    skips += len(update_indices)
                else:
                    self._apply_lr()
                    self.optimizer.step()
                    self._lr_steps += 1
                agg["grad_norm"].append(float(grad_norm))
                self._log_update(agg, rollout_id, u, start, skip_step)

        # Did the optimizer actually move the trainable weights? (diagnostic: a
        # nonzero grad_norm with an unchanged fingerprint means the step isn't
        # persisting to the params the sync reads.) Debug-gated — see fp_before.
        if debug_fp:
            fp_after = self._trainable_fp()
            logger.info(
                f"[train {rollout_id}] trainable_param_fp {fp_before:.6f} -> {fp_after:.6f} "
                f"(delta={fp_after - fp_before:.3e}, skips={skips}/{len(micro_batches)})"
            )

        self._sync_save_and_release(rollout_id)

        # Held-out evaluation (mirrors the megatron actor's _run_step_evaluation):
        # rank 0 asks the RolloutManager to run its eval pass; the rollout side owns
        # the --eval-interval / --eval-prompt-data gating. Without this hook
        # --eval-interval never fires on the generative path.
        self._run_step_evaluation(rollout_id)

        # Close the train interval and reopen the wait interval before logging so
        # perf/train_time is recorded when log_perf_data_raw reads the timers.
        Timer().end("train")
        Timer().start("train_wait")

        # Advance the profiler's step window (no-op unless profiling is enabled).
        profiler = getattr(self, "profiler", None)
        if profiler is not None:
            profiler.step(rollout_id)

        # Emit train/* + perf/*_time metrics (after the sync so weight_sync_time is
        # captured). log_perf_data_raw reads + resets the Timer.
        self._log_train_metrics(rollout_id, agg, skips, len(micro_batches))

    def _run_step_evaluation(self, rollout_id: int) -> None:
        """Trigger the RolloutManager's held-out eval pass (rank 0, best-
        effort).

        Mirrors ``backends/megatron/actor.py::_run_step_evaluation`` for the RL
        path: the actor only pokes ``GET /evaluate``; the rollout side decides
        whether this step is an eval step (``--eval-interval`` /
        ``--eval-prompt-data``). Deliberately no HTTP timeout — an eval pass runs
        real generation and can legitimately take minutes. Failures are logged,
        never fatal.
        """
        if self._rank != 0 or getattr(self, "rollout_manager", None) is None:
            return
        if getattr(self.args, "eval_interval", None) is None:
            return
        try:
            import requests

            from relax.utils.utils import get_serve_url

            rollout_serve_url = get_serve_url("rollout")
            response = requests.get(f"{rollout_serve_url}/evaluate", params={"train_step": rollout_id})
            response.raise_for_status()
            if rollout_id + 1 >= int(getattr(self.args, "num_rollout", 0) or 0):
                self._wait_for_final_evaluation(rollout_serve_url)
        except Exception as e:
            logger.warning(f"[train {rollout_id}] post-train evaluation failed: {e}")

    def _wait_for_final_evaluation(self, rollout_serve_url: str) -> None:
        """Keep the final eval alive before Ray tears down Serve.

        Periodic evals run asynchronously while the next rollout is produced.
        On the final step there is no next rollout, so the job can otherwise
        exit immediately after scheduling `/evaluate` and lose the final
        metric.
        """
        import requests

        timeout_arg = getattr(self.args, "final_eval_wait_timeout", None)
        poll_arg = getattr(self.args, "final_eval_poll_interval", None)
        timeout_s = float(1800.0 if timeout_arg is None else timeout_arg)
        poll_s = float(10.0 if poll_arg is None else poll_arg)
        start = time.monotonic()
        while True:
            response = requests.get(f"{rollout_serve_url}/is_eval_done", timeout=10)
            response.raise_for_status()
            if response.json().get("done", True):
                return
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                logger.warning(f"Timed out waiting for final evaluation after {elapsed:.1f}s; proceeding.")
                return
            logger.info("Waiting for final evaluation to finish before completing training...")
            time.sleep(poll_s)

    def _record_update_metrics(self, agg, result, advantages, rollout_id, update_index) -> None:
        """All-reduce one micro-batch's scalar metrics across the DP world
        group and record them into ``agg``.

        With DP sharding each rank's loss / ratio are over ITS
        candidate subset, so average across the world group (the reduction
        megatron does on the data-parallel group) to get the true global-batch
        value. Advantage mean/std are computed from global sum/sumsq/count
        instead of by averaging local stds: at ``n_samples_per_prompt == dp``
        each rank owns one candidate per prompt, so the local std is 0 even
        though the global advantage distribution is not. All ranks run the
        identical ``(mini-epoch, micro-batch)`` sequence (lockstep), so this
        collective can never desync.
        ``ratio_std`` / ``clip_frac`` become nonzero from update index 1 onward
        — the signal that PPO clipping is engaging (with a single update they
        are 0 and ``loss`` is 0 by construction).

        ``grad_norm`` is recorded separately by the caller: it is a property of
        the accumulated gradient (one value per optimizer step), not of an
        individual micro-batch, and is already global (FSDP
        ``clip_grad_norm_`` reduces internally).

        Every scalar is kept ON DEVICE until the single ``.tolist()`` after the
        all-reduce. Converting each one with ``float()`` first would be nine
        separate GPU->CPU syncs per micro-batch in the train hot loop, which
        CLAUDE.md forbids; ``torch.stack`` keeps it to one.
        """
        import torch.distributed as dist

        m = result["metrics"]
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        advantages_f = advantages.to(torch.float32)
        adv_count = torch.tensor(float(advantages_f.numel()), device=self.device, dtype=torch.float32)
        stats = torch.stack(
            [
                result["loss"].to(torch.float32),
                m["ratio_mean"].to(torch.float32),
                m["ratio_std"].to(torch.float32),
                m["clip_fraction"].to(torch.float32),
                m["approx_kl"].to(torch.float32),
                advantages_f.sum() if advantages_f.numel() else zero,
                advantages_f.pow(2).sum() if advantages_f.numel() else zero,
                adv_count,
                # Ratio tails ride outside the SUM reduction below so the min/max stay on device too; they
            ]
        )
        ratio_max = m["ratio_max"]
        ratio_min = m["ratio_min"]
        world_size = 1
        if dist.is_initialized():
            group = self._device_group()
            world_size = dist.get_world_size(group=group)
            if world_size > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        if world_size > 1:
            stats[:5] /= float(world_size)
        total_adv_count = stats[7]
        adv_denom = total_adv_count.clamp(min=1.0)
        adv_mean = stats[5] / adv_denom
        adv_var = torch.where(total_adv_count > 0, stats[6] / adv_denom - adv_mean.pow(2), zero)
        adv_std = torch.where(total_adv_count > 1, torch.sqrt(torch.clamp(adv_var, min=0.0)), zero)
        metric_stats = torch.stack([stats[0], stats[1], stats[2], stats[3], stats[4], adv_mean, adv_std])
        loss_g, ratio_mean_g, ratio_std_g, clip_frac_g, approx_kl_g, adv_mean_g, adv_std_g = metric_stats.tolist()
        agg["loss"].append(loss_g)
        agg["ratio_mean"].append(ratio_mean_g)
        agg["ratio_std"].append(ratio_std_g)
        agg["clip_fraction"].append(clip_frac_g)
        agg["approx_kl"].append(approx_kl_g)
        agg["adv_mean"].append(adv_mean_g)
        agg["adv_std"].append(adv_std_g)
        # Ratio tails are the most direct PPO-clip diagnostic. Accumulate them
        # LOCALLY here (a min/max cannot be averaged) and MAX-reduce once per step
        # in _log_train_metrics rather than adding a collective per micro-batch.
        self._ratio_max_local = max(self._ratio_max_local, float(ratio_max))
        self._ratio_min_local = min(self._ratio_min_local, float(ratio_min))

    def _sync_save_and_release(self, rollout_id: int) -> None:
        # Per-step weight sync to the rollout engines (mirrors megatron actor's
        # train() -> update_weights): train() owns the sync so the next rollout
        # samples under the updated policy. For train-owned saves, defer the final
        # actor sleep until after checkpointing so DCP reads real FSDP2 parameter
        # storage.
        post_sync_offload_engine = False
        if getattr(self, "rollout_manager", None) is not None:
            post_sync_offload_engine = self.update_weights(offload_after_sync=False)

        # Checkpoint / resume: the sync-colocate Controller path
        # (components/actor.py:_execute_training -> async_train) does NOT call
        # _maybe_save_model; only the fully_async branch does. The train backend
        # owns the save decision, exactly like the Megatron actor. This must run
        # after update_weights commits policy_version, but before actor sleep:
        # Relax's FSDP2 sleep frees CUDA storage and keeps a private CPU backup
        # that DCP/get_state_dict cannot see.
        try:
            self._maybe_save(rollout_id)
        finally:
            self._finish_weight_sync_memory_state(post_sync_offload_engine)

    def _maybe_save(self, rollout_id: int) -> None:
        """Save a checkpoint when the interval / rotation / final-step fires.

        Mirrors ``backends/megatron/actor.py`` train()-owned save gating so the
        sync-colocate path (which never routes through
        ``components.Actor._maybe_save_model``) still honors ``--save-
        interval``, ``--rotate-ckpt`` and the final step.
        """
        if getattr(self.args, "save", None) is None:
            return
        is_final = (rollout_id + 1) == getattr(self.args, "num_rollout", 0)
        save_interval = getattr(self.args, "save_interval", None)
        if (
            getattr(self.args, "rotate_ckpt", False)
            or is_final
            or (save_interval is not None and (rollout_id + 1) % int(save_interval) == 0)
        ):
            from relax.utils.timer import timer

            try:
                # Timed so perf/checkpoint_time shows up on the dashboard — a DCP
                # save of a full-FT 20B is minutes long and otherwise silently
                # inflates perf/train_time on save steps.
                with timer("checkpoint"):
                    self.save_model(rollout_id, force_sync=is_final)
            except Exception as e:
                # A checkpoint failure (e.g. shared-FS disk-quota exceeded on a full
                # ~100GB DCP write) must NOT kill a long training run. DCP raises a
                # CheckpointException collectively across ALL ranks, so every rank
                # lands here symmetrically and stays in sync for the next collective.
                # Log loudly (checkpoints are failing → resume is unavailable),
                # count it for the dashboard, and continue training.
                self._checkpoint_failures += 1
                logger.error(
                    f"[train {rollout_id}] checkpoint save FAILED ({type(e).__name__}: {e}); "
                    + (
                        "final checkpoint is mandatory; failing the training task."
                        if is_final
                        else "continuing training WITHOUT a checkpoint (resume unavailable until fixed — "
                        "check --save disk quota / free space)."
                    )
                )
                if is_final:
                    raise

    @torch.no_grad()
    def _trainable_fp(self) -> float:
        """Cheap fingerprint of the trainable weights (sum |local shard|).

        Detects whether ``optimizer.step`` actually moved the params the weight
        sync reads — a nonzero grad with an unchanged fingerprint localizes the
        bug to the step/offload path rather than the reward/advantage math.
        """
        total = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                loc = getattr(p.data, "_local_tensor", p.data)
                total += float(loc.detach().float().abs().sum())
        return total

    def _log_train_metrics(
        self, rollout_id: int, agg: Dict[str, List[float]], skips: int, num_microbatches: int
    ) -> None:
        """Emit FlowGRPO train metrics + per-stage perf times (design doc
        11.2).

        Reuses the framework sinks: ``log_perf_data_raw`` turns the ``Timer``
        stage records (actor_train / weight_sync) into ``perf/*_time`` and logs
        them; the RL metrics go through ``tracking_utils.log`` at the same
        step.

        The MAX-reduce runs OUTSIDE the swallowing ``try``: metric logging is
        best-effort, a collective is not. If one rank raised on its way into
        the reduce, logged a warning and moved on while every other rank
        blocked inside it, the job would hang with nothing but a metrics
        warning to explain it.
        """
        import torch.distributed as dist

        # ONE collective for every quantity that must be reduced with MAX rather
        # than averaged: the ratio tails (a min/max is not averageable) and the
        # GPU-memory watermarks (the worst rank is what OOMs). Packed as (-min)
        # so a single MAX reduce yields both ends.
        worst = torch.tensor(
            [
                self._ratio_max_local if self._ratio_max_local > float("-inf") else 0.0,
                -self._ratio_min_local if self._ratio_min_local < float("inf") else 0.0,
                torch.cuda.memory_allocated(self.device) / 1e9,
                torch.cuda.memory_reserved(self.device) / 1e9,
                torch.cuda.max_memory_allocated(self.device) / 1e9,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        device_group = self._device_group()
        if dist.is_initialized() and dist.get_world_size(device_group) > 1:
            dist.all_reduce(worst, op=dist.ReduceOp.MAX, group=device_group)
        ratio_max_g, neg_ratio_min_g, mem_alloc_g, mem_reserved_g, mem_peak_g = worst.tolist()
        # Reset the peak so the next step measures its own watermark.
        torch.cuda.reset_peak_memory_stats(self.device)

        try:
            from relax.utils import tracking_utils
            from relax.utils.metrics.metric_utils import compute_rollout_step
            from relax.utils.timer import Timer
            from relax.utils.training.train_metric_utils import log_perf_data_raw

            # log_perf_data_raw reads Timer().seq_lens (a megatron-set attr); the
            # diffusion path has no token counts, so ensure it exists (empty).
            if not hasattr(Timer(), "seq_lens"):
                Timer().seq_lens = []

            # Snapshot the stage timings BEFORE log_perf_data_raw, which resets the
            # Timer — the throughput metrics below need actor_train's duration.
            train_time = float(Timer().log_dict().get("actor_train", 0.0) or 0.0)

            is_primary = self._rank == 0
            log_perf_data_raw(
                rollout_id=rollout_id,
                args=self.args,
                is_primary_rank=is_primary,
                flops_counter=None,
                world_size=dist.get_world_size(device_group),
            )
            if not is_primary:
                return

            def _mean(xs: List[float]) -> float:
                return sum(xs) / len(xs) if xs else 0.0

            metrics = {
                "train/loss": _mean(agg["loss"]),
                "train/ratio_mean": _mean(agg["ratio_mean"]),
                "train/ratio_std": _mean(agg["ratio_std"]),
                "train/ratio_min": -neg_ratio_min_g,
                "train/ratio_max": ratio_max_g,
                "train/clip_fraction": _mean(agg["clip_fraction"]),
                "train/approx_kl": _mean(agg["approx_kl"]),
                "train/grad_norm": _mean(agg["grad_norm"]),
                "train/advantage_mean": _mean(agg["adv_mean"]),
                "train/advantage_std": _mean(agg["adv_std"]),
                "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                "train/num_microbatches": float(num_microbatches),
                "train/optimizer_skips": float(skips),
                "train/optimizer_steps": float(self._lr_steps),
                # Weight-sync provenance: without a version series a reward
                # regression cannot be correlated with a rolled-back/retried sync.
                "train/policy_version": float(self.policy_version),
                "train/weight_sync_tensors": float(self._last_sync_tensors),
                "train/weight_sync_gb": float(self._last_sync_bytes) / 1e9,
                "train/weight_sync_failures": float(self._weight_sync_failures),
                "train/checkpoint_failures": float(self._checkpoint_failures),
                # GPU watermarks (worst rank). These gate the colocate full-FT run
                # and were previously only free text in the actor log.
                "perf/mem_allocated_gb": mem_alloc_g,
                "perf/mem_reserved_gb": mem_reserved_g,
                "perf/mem_peak_gb": mem_peak_g,
                "rollout/step": compute_rollout_step(self.args, rollout_id),
            }
            # Throughput. A FLOPs/MFU estimate would need a DiT-specific analytic
            # model (the shared FlopsCounter dispatches on an LLM hf_config), so the
            # generative path reports the quantities that actually bound it: replayed
            # transitions and candidates per second over the measured train time.
            # NOTE: this reads the snapshot taken before log_perf_data_raw reset the
            # Timer, not Timer() itself.
            if train_time > 0:
                metrics["perf/train_transitions_per_s"] = float(self._trained_transitions) / train_time
                metrics["perf/train_samples_per_s"] = float(self._trained_samples) / train_time
            logger.info(
                f"[train {rollout_id}] "
                + " ".join(f"{k.split('/', 1)[-1]}={v:.4f}" for k, v in metrics.items() if k != "rollout/step")
            )
            tracking_utils.log(self.args, metrics, step_key="rollout/step")
        except Exception as e:  # metrics must never break training
            logger.warning(f"[rollout {rollout_id}] train metric logging skipped ({e}).")
        finally:
            # The metrics-service adapter BUFFERS on log(); the backends (ClearML /
            # TensorBoard / wandb) only receive the step when report_step is called
            # via flush_metrics (mirrors megatron actor.py:868). Without this the
            # ClearML task is created but shows no scalars.
            #
            # This MUST run even if the block above raised: log_perf_data_raw
            # already logged perf/*_time for this step, and the server buffer is
            # per-step — a later flush at step N+1 would never rescue step N, so
            # those metrics would be orphaned forever.
            try:
                from relax.utils import tracking_utils as _tracking
                from relax.utils.metrics.metric_utils import compute_rollout_step as _step

                if self._rank == 0:
                    _tracking.flush_metrics(self.args, int(_step(self.args, rollout_id)))
            except Exception as e:  # flushing must never break training either
                logger.warning(f"[rollout {rollout_id}] metric flush skipped ({e}).")

    def update_weights(self, *, offload_after_sync: bool = True) -> bool:
        """Stream the full transformer to every rollout engine (design doc
        10.3)."""
        if getattr(self.args, "debug_train_only", False):
            return False

        def _mem(tag: str) -> None:
            a = torch.cuda.memory_allocated(self.device) / 1e9
            r = torch.cuda.memory_reserved(self.device) / 1e9
            logger.info(f"[wsync mem] {tag}: allocated={a:.1f}GB reserved={r:.1f}GB")

        # Per-phase wall clock. `perf/weight_sync_time` is ~83 s/rollout at 40.9 GB
        # / 1933 tensors, which is NOT the IPC payload — CUDA IPC passes handles,
        # the bytes never cross the wire. The candidates are the CPU<->GPU offload
        # traffic (actor ~40 GB each way, engine ~15 GB each way) and the bucket
        # loop (one 8-way DTensor full_tensor() all-gather per parameter because
        # FSDP shards over ALL ranks — unlike Megatron, whose DP dimension is
        # already unsharded — plus one blocking ray.get + barrier per bucket).
        # Attribute it instead of guessing.
        import time as _time

        _phase: Dict[str, float] = {}
        _t0 = _time.time()

        def _lap(tag: str) -> None:
            now = _time.time()
            _phase[tag] = now - _lap.mark  # type: ignore[attr-defined]
            _lap.mark = now  # type: ignore[attr-defined]

        _lap.mark = _t0  # type: ignore[attr-defined]

        _mem("update_weights entry")
        # The IPC weight transaction reads the actor's params, so it must be on GPU
        # (it may have been offloaded after the previous step's rollout). Params
        # only — the optimizer is not needed for the sync and is already on CPU
        # after the train loop's bracketed step.
        if getattr(self.args, "offload_train", False):
            self.wake_up(include_optimizer=False)
            # Free any grads the just-finished optimizer.step materialized (the
            # optimizer states are already CPU-resident): the weight sync only
            # needs PARAMS on GPU (full_tensor gather → IPC). Keeping grads
            # resident makes the actor + the engine transformer onload below
            # co-exceed the card.
            self.optimizer.zero_grad(set_to_none=True)
            self._move_optimizer_state(torch.device("cpu"))
            # Return the freed blocks to the driver so the engine transformer onload
            # below has room (PyTorch caches freed memory; nvidia-smi otherwise still
            # shows it held by this process, colliding with the engine on GPU).
            self.clear_memory()
            _mem("after wake_up + optimizer offload + empty_cache")
        _lap("actor_onload")
        # Colocate memory time-share: the engine was offloaded to CPU after rollout
        # (offload_rollout). The full-FT actor + full engine do not co-fit, but the
        # actor + the engine's TRANSFORMER (the IPC dest) do — so onload only the
        # weight modules for the sync.
        offload_engine = (
            getattr(self.args, "offload_rollout", False) and getattr(self, "rollout_manager", None) is not None
        )
        if offload_engine:
            import ray
            from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

            onload_error = None
            try:
                ray.get(self.rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
            except Exception as exc:
                onload_error = f"{type(exc).__name__}: {exc}"
            _raise_weight_sync_errors(
                self.policy_version + 1,
                "rollout-engine weights onload",
                _collect_weight_sync_errors(onload_error),
            )
        _lap("engine_weights_onload")
        self.policy_version += 1
        manifest = self._build_manifest()
        self._last_sync_tensors = int(manifest.tensor_count)
        self._last_sync_bytes = int(manifest.total_bytes)
        iterator = self._build_weight_iterator()
        from relax.utils.timer import timer

        with timer("weight_sync"):
            try:
                self._run_weight_transaction(manifest, iterator)
            except WeightSyncError as exc:
                # The streamed version is INVALID on some/all engines. Roll back the
                # version bump so a whole-round retry rebuilds the manifest with the
                # SAME version number — otherwise the actor's policy_version and the
                # engines' committed version drift apart across a recovery (every
                # rank bumped and every rank lands here, so the rollback is uniform).
                self.policy_version -= 1
                self._weight_sync_failures += 1
                # With fault tolerance, rebuild the engines from the last committed
                # disk checkpoint (RolloutManager.recover_rollout_engines) and let
                # the Controller retry the whole round; otherwise fail the job.
                # Re-raise on every rank so no rank proceeds to the post-sync
                # offload/onload as if the sync succeeded.
                logger.error(f"[wsync] {exc}; aborting this weight version.")
                # Recovery may replace the engine actors. Invalidate on EVERY
                # rank even when recovery itself fails, so a retry cannot reuse
                # stale handles or skip the collective topology rebuild.
                try:
                    if getattr(self.args, "use_fault_tolerance", False):

                        def _recover() -> None:
                            import ray

                            ray.get(self.rollout_manager.recover_rollout_engines.remote())

                        self._rank0_then_agree(_recover, what="rollout-engine recovery")
                finally:
                    self._invalidate_ipc_topology()
                raise
        _mem("after weight transaction")
        _lap("bucket_loop")
        if offload_after_sync:
            self._finish_weight_sync_memory_state(offload_engine, phase_lap=_lap)
        if self._rank == 0:
            total = _time.time() - _t0
            parts = " ".join(f"{k}={v:.1f}s" for k, v in _phase.items())
            # Split bucket_loop into where it actually goes. gather = the
            # per-parameter DTensor all-gathers, serialize = CUDA-IPC handle
            # packing, transport = gather_object + engine RPC + barrier.
            gather, serialize, transport = getattr(self, "_last_sync_breakdown", (0.0, 0.0, 0.0))
            inner = f" [gather={gather:.1f}s serialize={serialize:.1f}s transport={transport:.1f}s]"
            logger.info(
                f"[wsync time] total={total:.1f}s {parts}{inner} "
                f"({self._last_sync_tensors} tensors, {self._last_sync_bytes / 1e9:.1f}GB)"
            )
        return offload_engine

    def _finish_weight_sync_memory_state(
        self, offload_engine: bool, phase_lap: Optional[Callable[[str], None]] = None
    ) -> None:
        """Vacate actor GPU memory and restore rollout engines after weight
        sync."""
        if getattr(self.args, "offload_train", False):
            self.sleep()
        if phase_lap is not None:
            phase_lap("actor_offload")
        if offload_engine:
            import ray

            onload_error = None
            try:
                ray.get(self.rollout_manager.onload.remote())
            except Exception as exc:
                onload_error = f"{type(exc).__name__}: {exc}"
            _raise_weight_sync_errors(
                self.policy_version,
                "rollout-engine full onload",
                _collect_weight_sync_errors(onload_error),
            )
        if phase_lap is not None:
            phase_lap("engine_full_onload")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:

        from relax.backends.fsdp import checkpoint as ckpt

        trainer_state = {
            "rollout_id": int(rollout_id),
            "policy_version": int(self.policy_version),
            "task": self.args.generation_task,
            "model_family": self.adapter.family,
            "adapter_path": self.args.model_adapter_path,
            "sampling_fingerprint": self.sampling_fingerprint(),
            # Resume-critical scheduler state (design doc 12): the LR schedule is
            # step-driven, so the step count must round-trip or a resumed run
            # restarts warmup from zero.
            "lr_scheduler_steps": int(self._lr_steps),
        }
        ckpt.save_checkpoint(
            self.args.save,
            self.args.generation_task,
            rollout_id,
            self.model,
            self.optimizer,
            trainer_state=trainer_state,
            weight_sync_manifest=self._build_manifest().to_dict(),
            adapter_contract=self._adapter_contract(),
            adapter_only=(self._trainable_mode == "lora"),
            is_rank0=(int(os.environ.get("RANK", 0)) == 0),
        )

        # Rotate only AFTER the new snapshot is committed. Failed/in-progress
        # directories never consume the retention quota, and a failed save can
        # no longer delete the last valid recovery point first.
        task_root = os.path.join(self.args.save, self.args.generation_task)

        def _rotate() -> None:
            from relax.utils.rotate_ckpt import rotate_ckpt

            rotate_ckpt(
                self.args,
                global_step=rollout_id,
                save_dir=task_root,
                committed_only=True,
            )

        self._rank0_then_agree(_rotate, what=f"checkpoint rotation at rollout {rollout_id}")

    def sleep(self, tags=None) -> None:
        """Offload the actor (model + optimizer) to CPU to free GPU for
        rollout.

        Sync colocate: the diffusion engine and this actor share the same GPUs,
        so during rollout the actor must vacate GPU memory (design doc 10;
        mirrors the megatron actor's per-step offload). No-op unless
        ``--offload-train``.
        """
        if not getattr(self.args, "offload_train", False):
            return
        from relax.backends.fsdp.runtime import offload_module_to_cpu

        offload_module_to_cpu(self.model)
        self._move_optimizer_state(torch.device("cpu"))
        self.clear_memory()

    def wake_up(self, tags=None, include_optimizer: bool = True) -> None:
        """Onload the actor back to GPU for train / weight sync.

        ``include_optimizer=False`` onloads only the model params (used by
        ``train()``, which keeps the ~20GB AdamW states on CPU during the
        memory heavy FlowGRPO replay and streams them to GPU only around the
        optimizer step) — the weight-sync path never needs the optimizer on GPU
        either.
        """
        if not getattr(self.args, "offload_train", False):
            return
        from relax.backends.fsdp.runtime import onload_module_to_device

        onload_module_to_device(self.model, self.device)
        if include_optimizer:
            self._move_optimizer_state(self.device)

    @torch.no_grad()
    def _apply_lr(self) -> None:
        """Set the current step's learning rate on every optimizer param
        group."""
        lr = lr_at_step(self.args, self._lr_steps, self._base_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @torch.no_grad()
    def _move_optimizer_state(self, device: torch.device) -> None:
        """Move AdamW state tensors (exp_avg / exp_avg_sq) to ``device``.

        These dominate the actor's GPU footprint (~2x params in fp32), so they
        must move with the model or the offload frees too little to fit the
        engine.
        """
        for state in self.optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor) and val.device != device:
                    state[key] = val.to(device, non_blocking=True)

    def update_weights_fully_async(self, rollout_id: int, rollout_only=False, actor_fwd_only=False) -> None:
        # First release only exercises the synchronous colocate path; the async
        # branch reuses the same manifest + iterator (design doc 10.4).
        raise NotImplementedError("fully_async weight transport is not enabled in the first release.")

    # -- internals ------------------------------------------------------------

    def _sampling_cfg(self) -> Mapping[str, Any]:
        return getattr(self.args, "sampling_config", None) or {}

    def sampling_fingerprint(self) -> str:
        import json

        payload = json.dumps(
            {"sampling": self._sampling_cfg(), "task": self.args.generation_task},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_manifest(self):
        from relax.backends.fsdp.weight_update import build_full_weight_manifest

        return build_full_weight_manifest(
            self._named_sync_params(),
            model_family=self.adapter.family,
            task=self.args.generation_task,
            policy_version=self.policy_version,
            base_model_sha256=self._base_model_sha256,
            wire_dtype=self.args.weight_sync_wire_dtype,
            bucket_size_bytes=int(self.args.weight_sync_bucket_size_mb) * 1024 * 1024,
            name_map=self.adapter.weight_name_map,
        )

    def _build_weight_iterator(self):
        from relax.backends.fsdp.weight_update import FullWeightChunkIterator

        return FullWeightChunkIterator(
            self._named_sync_params(),
            wire_dtype=self.args.weight_sync_wire_dtype,
            bucket_size_bytes=int(self.args.weight_sync_bucket_size_mb) * 1024 * 1024,
            name_map=self.adapter.weight_name_map,
            # Under --fsdp-cpu-offload the shards live on CPU; gather over NCCL by
            # moving each shard to this rank's GPU before full_tensor().
            gather_device=self.device,
            # Merge-mode LoRA folds B@A into each base weight as it is materialized.
            tensor_source=self._sync_tensor_source(),
        )

    def _assert_sync_plan_aligned(self) -> None:
        """Fail loudly if ranks disagree on WHAT gets synced.

        Every rank must enter the same sequence of DTensor ``full_tensor()``
        collectives. If LoRA were injected differently on different ranks the
        gathers would mismatch and NCCL would simply stop making progress — a
        watchdog timeout an hour into the run with no Python traceback.
        Comparing the manifest's ordered name-shape hash up front turns that
        into an error.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        if dist.get_world_size(dist.group.WORLD) <= 1:
            return
        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
        manifest = self._build_manifest()
        local = (manifest.tensor_count, manifest.ordered_name_shape_hash)
        gathered: List[Any] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local, group=group)
        mismatched = {i: got for i, got in enumerate(gathered) if got != gathered[0]}
        if mismatched:
            raise ValueError(
                f"weight-sync plan differs across ranks (rank0={gathered[0]}, diverging={mismatched}); "
                "every rank must inject the same LoRA target modules."
            )

    def _invalidate_ipc_topology(self) -> None:
        """Drop the cached CUDA-IPC topology so the next sync rebuilds it.

        The cache holds Ray actor HANDLES; after
        ``recover_rollout_engines`` those handles are dead. Must be called on
        every rank, because the rebuild in
        :meth:`_build_ipc_gather_topology` is collective.
        """
        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._ipc_tp = None
        self._ipc_block_offset = None

    def _build_ipc_gather_topology(self, engines, gpu_counts) -> None:
        """Partition FSDP ranks into per-engine Gloo gather groups (colocate
        IPC).

        Mirrors megatron ``UpdateWeightFromTensor.connect_rollout_engines``: an
        engine spans ``gpu_counts[e]`` (= rollout TP) physical GPUs starting at its
        ``base_gpu_id``; the TP FSDP ranks co-located on those GPUs form one Gloo
        gather group whose src is the rank on the engine's base GPU. Each rank
        caches its own ``(group, src, engine, tp, block_offset)``.

        Everything is keyed by CUDA device UUID, not list position / raw index —
        the placement group reorders GPU ids, so both mislead. ``dist.new_group``
        is collective, so ALL ranks build every engine's group in the same order.
        Built once and reused across weight syncs; invalidated by
        :meth:`_invalidate_ipc_topology` when the engines are replaced.
        """
        import ray
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        if getattr(self, "_ipc_gather_group", None) is not None:
            return

        device_group = self._device_group()
        world = dist.get_world_size(device_group)
        my_rank = dist.get_rank(device_group)
        my_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        all_uuids: List[Optional[str]] = [None] * world
        dist.all_gather_object(all_uuids, my_uuid, group=get_gloo_group())
        uuid_to_rank = {u: r for r, u in enumerate(all_uuids) if u is not None}

        base_gpu_ids = None
        local_error = None
        try:
            base_gpu_ids = ray.get([e.get_base_gpu_id.remote() for e in engines])
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_weight_sync_errors(
            self.policy_version,
            "engine topology RPC",
            _collect_weight_sync_errors(local_error),
        )
        assert base_gpu_ids is not None

        block_plans: List[Tuple[Any, int, List[int]]] = []
        local_error = None
        try:
            for e, (engine, base) in enumerate(zip(engines, base_gpu_ids)):
                if base is None:
                    raise WeightSyncError(self.policy_version, f"engine {e} has no base_gpu_id")
                tp = int(gpu_counts[e])
                block_ranks: List[int] = []
                for offset in range(tp):
                    gpu = int(base) + offset
                    u = str(torch.cuda.get_device_properties(gpu).uuid)
                    r = uuid_to_rank.get(u)
                    if r is None:
                        raise WeightSyncError(
                            self.policy_version,
                            f"no FSDP rank co-located on gpu {gpu} (engine {e})",
                        )
                    block_ranks.append(r)
                block_plans.append((engine, tp, block_ranks))
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_weight_sync_errors(
            self.policy_version,
            "engine topology validation",
            _collect_weight_sync_errors(local_error),
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._ipc_tp = None
        self._ipc_block_offset = None
        for engine, tp, block_ranks in block_plans:
            # new_group must be called by every rank (collective), even non-members.
            group = dist.new_group(ranks=block_ranks, backend="gloo")
            if my_rank in block_ranks:
                self._ipc_gather_group = group
                self._ipc_gather_src = block_ranks[0]  # rank on the engine's base GPU
                self._ipc_engine = engine
                self._ipc_tp = tp
                self._ipc_block_offset = block_ranks.index(my_rank)
        mapping_error = None if self._ipc_engine is not None else f"rank {my_rank} not mapped to any engine block"
        _raise_weight_sync_errors(
            self.policy_version,
            "engine topology mapping",
            _collect_weight_sync_errors(mapping_error),
        )
        logger.info(
            f"[wsync] rank={my_rank} device={self.device} → engine src={self._ipc_gather_src} "
            f"tp={self._ipc_tp} block_offset={self._ipc_block_offset}"
        )

    def _run_weight_transaction(self, manifest, iterator) -> None:
        """Stream the full transformer to the co-located engine(s) via CUDA
        IPC.

        Mirrors the text ``UpdateWeightFromTensor`` colocate mapping generalized to
        rollout TP>1: the RolloutManager creates ``world / tp`` diffusion engines,
        each spanning ``tp`` physical GPUs. The ``tp`` FSDP ranks co-located on an
        engine's GPUs form a Gloo gather group; each all-gathers its full tensors
        (the iterator calls DTensor ``full_tensor()`` — identical data on every
        rank, resident on its own GPU), serializes them as CUDA-IPC handles, and
        the group's src ``gather_object``-collects the ``tp`` blobs *in engine-worker
        offset order* and hands the list to the engine (worker ``i`` opens blob
        ``i`` = the tensor on physical GPU base+i and shards internally). A barrier
        per bucket keeps every rank's source tensors alive until the engine has
        opened them. ``tp==1`` degenerates to one blob per single-GPU engine.

        Any engine drop / commit failure / checksum mismatch raises
        :class:`WeightSyncError` so the caller can abort the version. Disk update is
        used only for resume, never here.
        """
        import base64

        import ray
        import torch.distributed as dist
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        version = manifest.policy_version
        engine_state = None
        local_error = None
        try:
            engine_state = ray.get(self.rollout_manager.get_rollout_engines_and_lock.remote())
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_weight_sync_errors(
            version,
            "rollout-manager engine lookup",
            _collect_weight_sync_errors(local_error),
        )
        assert engine_state is not None
        engines, _lock, *_rest = engine_state
        if not engines:
            raise WeightSyncError(version, "no rollout engines available")
        if self.args.weight_sync_mode == "adapter":
            self._run_adapter_transaction(manifest, engines)
            return
        # _rest = [num_new_engines, engine_gpu_counts, engine_gpu_offsets].
        gpu_counts = _rest[1] if len(_rest) >= 2 else [1] * len(engines)
        world = dist.get_world_size(self._device_group())
        total_engine_gpus = sum(int(c) for c in gpu_counts)
        if total_engine_gpus != world:
            raise WeightSyncError(
                version,
                f"engine GPUs {total_engine_gpus} (counts={list(gpu_counts)}) != FSDP world {world}; "
                "rollout_num_gpus_per_engine must divide actor world size.",
            )

        self._build_ipc_gather_topology(engines, gpu_counts)
        group = self._ipc_gather_group
        src = self._ipc_gather_src
        engine = self._ipc_engine
        tp = self._ipc_tp
        offset = self._ipc_block_offset
        my_rank = dist.get_rank(self._device_group())
        assert group is not None and src is not None and engine is not None and tp is not None and offset is not None

        try:
            sent_count = 0
            sent_bytes = 0
            # Depth-1 pipeline: fire bucket k's RPC without waiting, let the
            # iterator start bucket k+1's gather, and only then drain k.
            # `pending` keeps bucket k's source tensors (and their IPC blob) alive
            # until the engine has actually opened the handles -- that lifetime
            # requirement is why the global drain agreement exists, and it is
            # what makes this a hold-a-reference change rather than fire-and-forget.
            #
            # MEASURED: this alone did NOT move bucket_loop (70.4s vs 70.0/73.9/69.6s
            # before). The transport was not the bottleneck. Hence the breakdown
            # below -- `gather` is the 1933 per-parameter DTensor full_tensor()
            # all-gathers the iterator drives, and at ~36ms each they are ~400x
            # their own bandwidth cost, i.e. collective LATENCY, not payload. Keep
            # this instrumentation until that is fixed (coalescing the per-tensor
            # collectives), so the next attempt is measured rather than guessed.
            import time as _time

            t_gather = t_serialize = t_transport = 0.0
            pending: Optional[Tuple[Any, Any, Any]] = None

            def _drain(p) -> Tuple[float, List[Tuple[int, str]]]:
                if p is None:
                    return 0.0, _collect_weight_sync_errors(None)
                t0 = _time.perf_counter()
                ref, _named, _blob = p
                local_error = None
                if ref is not None:
                    try:
                        ray.get(ref)
                    except Exception as exc:
                        local_error = f"{type(exc).__name__}: {exc}"
                # Global agreement replaces the old block-local barrier. It keeps
                # every block's source tensors alive until all engine RPCs finish,
                # and makes a failure on any engine src visible to every FSDP rank.
                errors = _collect_weight_sync_errors(local_error)
                return _time.perf_counter() - t0, errors

            bucket_iter = iter(iterator)
            while True:
                t0 = _time.perf_counter()
                bucket = None
                iterator_error = None
                has_bucket = True
                try:
                    bucket = next(bucket_iter)
                except StopIteration:
                    has_bucket = False
                except Exception as exc:
                    iterator_error = f"{type(exc).__name__}: {exc}"
                iterator_states = _collect_rank_objects((has_bucket, iterator_error))
                t_gather += _time.perf_counter() - t0
                iterator_errors = [
                    (rank, error) for rank, (_has_bucket, error) in enumerate(iterator_states) if error is not None
                ]
                has_bucket_states = {state[0] for state in iterator_states}
                if iterator_errors or len(has_bucket_states) != 1:
                    drain_time, drain_errors = _drain(pending)
                    t_transport += drain_time
                    pending = None
                    _raise_weight_sync_errors(version, "pending engine RPC", drain_errors)
                    if not iterator_errors:
                        iterator_errors = [(-1, f"ranks disagreed on iterator exhaustion: {iterator_states!r}")]
                    _raise_weight_sync_errors(version, "weight iterator", iterator_errors)
                if not has_bucket:
                    break
                assert bucket is not None

                t0 = _time.perf_counter()
                # CUDA-IPC requires the source tensors on GPU. Under --fsdp-cpu-offload
                # the DTensor shards live on CPU and full_tensor() gathers to CPU, so
                # move each bucket to this rank's device before serializing; it is a
                # no-op when the gather is already on-GPU (non-offload full-FT). At
                # most two buckets are resident at once (this one and the in-flight
                # one), so a CPU-offloaded 20B still never rebuilds ~40GB on GPU.
                named = []
                blob = None
                local_error = None
                try:
                    named = [(name, t.to(self.device)) for name, t in bucket]
                    blob = base64.b64encode(MultiprocessingSerializer.serialize(named)).decode("ascii")
                except Exception as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                prepare_errors = _collect_weight_sync_errors(local_error)
                if prepare_errors:
                    drain_time, drain_errors = _drain(pending)
                    t_transport += drain_time
                    pending = None
                    _raise_weight_sync_errors(version, "pending engine RPC", drain_errors)
                    _raise_weight_sync_errors(version, "IPC tensor preparation/serialization", prepare_errors)
                # Count what we actually stream so we can validate it against the
                # manifest before commit (below): a truncated / short bucket stream
                # would otherwise silently commit a partial weight set.
                sent_count += len(named)
                sent_bytes += sum(int(t.numel()) * t.element_size() for _, t in named)
                t_serialize += _time.perf_counter() - t0

                t0 = _time.perf_counter()
                # Gather (block_offset, blob) so the src can order blobs by engine
                # worker index regardless of the Gloo group's rank ordering: worker
                # i needs the IPC handle for the tensor on physical GPU base+i.
                gathered: Optional[List[Any]] = [None] * tp if my_rank == src else None
                dist.gather_object((offset, blob), gathered, dst=src, group=group)
                ref = None
                launch_error = None
                if my_rank == src:
                    try:
                        ordered: List[Optional[str]] = [None] * tp
                        for off, b in gathered:  # type: ignore[union-attr]
                            ordered[off] = b
                        if any(b is None for b in ordered):
                            raise RuntimeError(f"incomplete IPC gather for engine (got {gathered})")
                        # target_modules pins the update to the trainable module: without
                        # it the server routes tensors by "<module>." prefix across ALL
                        # pipeline modules and drops the actor's module-local names.
                        ref = engine.update_weights_from_tensor.remote(
                            ordered, target_modules=[self.args.fsdp_trainable_attr]
                        )
                    except Exception as exc:
                        launch_error = f"{type(exc).__name__}: {exc}"
                current = (ref, named, blob)
                launch_errors = _collect_weight_sync_errors(launch_error)
                t_transport += _time.perf_counter() - t0
                drain_time, drain_errors = _drain(pending)
                t_transport += drain_time
                pending = None
                if launch_errors or drain_errors:
                    current_time, current_errors = _drain(current)
                    t_transport += current_time
                    _raise_weight_sync_errors(version, "engine RPC launch", launch_errors)
                    _raise_weight_sync_errors(version, "pending engine RPC", drain_errors + current_errors)
                pending = current
                del named, blob, gathered
            drain_time, drain_errors = _drain(pending)
            t_transport += drain_time
            pending = None
            _raise_weight_sync_errors(version, "final engine RPC", drain_errors)
            self._last_sync_breakdown = (t_gather, t_serialize, t_transport)

            # Validate the streamed snapshot against the manifest BEFORE committing —
            # every rank streams the identical full set, so every rank checks the
            # same counts and raises uniformly. A mismatch means a truncated /
            # mis-ordered / mis-named stream; fail the version rather than commit a
            # partial weight set (design doc 10.2).
            count_error = None
            if sent_count != manifest.tensor_count or sent_bytes != manifest.total_bytes:
                count_error = (
                    f"streamed {sent_count} tensors / {sent_bytes} B != manifest "
                    f"{manifest.tensor_count} tensors / {manifest.total_bytes} B"
                )
            _raise_weight_sync_errors(
                version,
                "streamed manifest validation",
                _collect_weight_sync_errors(count_error),
            )

            # Broadcasts rank 0's outcome and doubles as the barrier, so a
            # failed verify/commit raises everywhere instead of hanging the
            # other ranks on a barrier rank 0 already unwound past.
            self._rank0_then_agree(
                lambda: self._verify_and_commit_engines(engines, manifest),
                what=f"weight v{version} verify/commit",
            )
            self._synced_versions += 1
        except WeightSyncError:
            raise
        except Exception as exc:  # any transport / engine RPC failure invalidates the version
            raise WeightSyncError(version, str(exc)) from exc

    def _verify_and_commit_engines(self, engines, manifest) -> None:
        """Verify every engine replica before committing the policy version."""
        import ray

        version = manifest.policy_version
        try:
            checksums = ray.get(
                [engine.get_weights_checksum.remote([self.args.fsdp_trainable_attr]) for engine in engines]
            )
        except Exception as exc:
            raise WeightSyncError(version, f"checksum RPC failed: {exc}") from exc
        reference = None
        for index, checksum in enumerate(checksums):
            if not isinstance(checksum, dict) or checksum.get("success") is False:
                raise WeightSyncError(version, f"engine {index} returned an invalid checksum response: {checksum!r}")
            try:
                engine_count = int(checksum["tensor_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WeightSyncError(
                    version,
                    f"engine {index} checksum response has no valid tensor_count: {checksum!r}",
                ) from exc
            if engine_count != manifest.tensor_count:
                raise WeightSyncError(
                    version,
                    f"engine {index} has {engine_count} tensors != manifest {manifest.tensor_count}",
                )
            comparable = {
                key: value for key, value in checksum.items() if key not in ("success", "message", "tensor_count")
            }
            if not comparable:
                raise WeightSyncError(version, f"engine {index} checksum response contains no digest: {checksum!r}")
            invalid_digests = {
                key: value
                for key, value in comparable.items()
                if not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in value)
            }
            if invalid_digests:
                raise WeightSyncError(
                    version,
                    f"engine {index} checksum response contains invalid digests: {invalid_digests!r}",
                )
            if reference is None:
                reference = comparable
            elif comparable != reference:
                raise WeightSyncError(
                    version,
                    f"engine {index} checksum differs from engine 0: {comparable!r} != {reference!r}",
                )

        manifest_digest = manifest.sha256()
        try:
            commit_results = ray.get(
                [engine.commit_weight_version.remote(version, manifest_digest) for engine in engines]
            )
        except Exception as exc:
            raise WeightSyncError(version, f"commit RPC failed after verification: {exc}") from exc
        for index, result in enumerate(commit_results):
            if (
                not isinstance(result, dict)
                or int(result.get("active_version", -1)) != version
                or result.get("weight_manifest_sha256") != manifest_digest
            ):
                raise WeightSyncError(version, f"engine {index} failed to commit the version: {result!r}")
        logger.info(
            f"weight v{version} committed on {len(engines)} engine(s); "
            f"weight_manifest_sha256={manifest_digest} checksum={reference}"
        )

    def _run_adapter_transaction(self, manifest, engines) -> None:
        """Push only the LoRA adapter to every engine (``--lora-adapter-
        mode``).

        LoRA training never changes the base weights, so the engine keeps the
        base it loaded at launch and only the (tiny) adapter travels each step.
        That is not merely an optimization: SGLang's ``convert_to_lora_layers()``
        renames the DiT parameters to ``*.base_layer.weight``, and its weight
        loader skips names it does not recognize with a ``continue`` while still
        reporting success — so a full-weight sync into a LoRA-converted engine is
        a silent no-op. Never mixing the two transports avoids that entirely.

        The adapter is replicated across FSDP ranks (it is DP, not TP), so rank 0
        gathers and pushes on behalf of everyone. Serialization goes over the
        regular Ray object store rather than CUDA IPC: at r=64 the payload is a
        few hundred MB, and a host round-trip is far simpler than the per-engine
        IPC topology the full path needs.
        """
        import ray

        from relax.backends.fsdp.lora import LORA_ADAPTER_NAME, engine_adapter_state_dict
        from relax.backends.fsdp.weight_update import iter_full_named_tensors

        version = manifest.policy_version
        wire_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[self.args.weight_sync_wire_dtype]
        try:
            # Collective: every rank enters full_tensor() in the same sorted order.
            gathered = []
            gather_error = None
            try:
                gathered = list(
                    iter_full_named_tensors(
                        self._named_sync_params(),
                        wire_dtype=wire_dtype,
                        gather_device=self.device,
                    )
                )
            except Exception as exc:
                gather_error = f"{type(exc).__name__}: {exc}"
            _raise_weight_sync_errors(
                version,
                "adapter tensor gather",
                _collect_weight_sync_errors(gather_error),
            )

            def _push_adapter() -> None:
                if len(gathered) != manifest.tensor_count:
                    raise WeightSyncError(
                        version,
                        f"gathered {len(gathered)} adapter tensors != manifest {manifest.tensor_count}",
                    )
                payload = engine_adapter_state_dict(gathered, alpha=int(self.args.lora_alpha))
                a_stems = {name[: -len(".lora_A.weight")] for name in payload if name.endswith(".lora_A.weight")}
                b_stems = {name[: -len(".lora_B.weight")] for name in payload if name.endswith(".lora_B.weight")}
                alpha_stems = {name[: -len(".alpha")] for name in payload if name.endswith(".alpha")}
                if not a_stems or a_stems != b_stems or a_stems != alpha_stems:
                    raise WeightSyncError(
                        version,
                        "incomplete LoRA payload: every layer must have exactly lora_A, lora_B, and alpha "
                        f"(A={sorted(a_stems)}, B={sorted(b_stems)}, alpha={sorted(alpha_stems)})",
                    )
                expected_layers = len(a_stems)
                payload = {name: t.detach().cpu() for name, t in payload.items()}
                results = ray.get(
                    [
                        e.set_lora_from_tensors.remote(
                            payload,
                            lora_name=LORA_ADAPTER_NAME,
                            target=self.args.fsdp_trainable_attr,
                        )
                        for e in engines
                    ]
                )
                for engine_index, result in enumerate(results):
                    if (
                        not isinstance(result, dict)
                        or result.get("success") is not True
                        or int(result.get("adapted_layers", -1)) != expected_layers
                    ):
                        raise WeightSyncError(
                            version,
                            f"engine {engine_index} adapted an incomplete LoRA payload: "
                            f"expected {expected_layers} layers, got {result!r}",
                        )
                self._last_sync_tensors = len(payload)
                self._last_sync_bytes = sum(int(t.numel()) * t.element_size() for t in payload.values())
                logger.info(
                    f"adapter v{version} pushed: {self._last_sync_tensors} tensors "
                    f"({manifest.tensor_count} adapter + {len(payload) - manifest.tensor_count} alpha), "
                    f"{self._last_sync_bytes / 1e6:.1f}MB to {len(engines)} engine(s)"
                )
                digest = manifest.sha256()
                commit_results = ray.get([e.commit_weight_version.remote(version, digest) for e in engines])
                for engine_index, result in enumerate(commit_results):
                    if (
                        not isinstance(result, dict)
                        or int(result.get("active_version", -1)) != version
                        or result.get("weight_manifest_sha256") != digest
                    ):
                        raise WeightSyncError(
                            version,
                            f"engine {engine_index} failed to commit adapter version: {result!r}",
                        )

            # See :meth:`_rank0_then_agree`: a plain rank-0 push + barrier would
            # strand ranks 1..N-1 on the barrier if the push failed.
            self._rank0_then_agree(_push_adapter, what=f"adapter v{version} push")
        except WeightSyncError:
            raise
        except Exception as exc:
            raise WeightSyncError(version, str(exc)) from exc

    def _tq_client(self):
        """Lazily create this actor's TransferQueue client (mirrors megatron
        actor)."""
        if self._data_system_client is None:
            import transfer_queue as tq

            if getattr(self.args, "tq_config", None) is not None:
                tq.init(self.args.tq_config)
            self._data_system_client = tq.get_client()
        return self._data_system_client

    def _load_train_batches(self, rollout_id: int, rollout_data_ref) -> List[Dict[str, Any]]:
        """Materialize micro-batches from the numeric TQ rows + sidecars.

        The rollout driver writes numeric rows (advantages, group index,
        trajectory slot, policy version) to the TransferQueue partition
        ``train_{rollout_id}`` and the heavy trajectory tensors to per-group
        safetensors sidecars under ``artifact_root``. When no in-memory
        ``rollout_data_ref`` is supplied (the sync colocate path calls
        ``async_train(rollout_id)`` without one), consume those rows from the
        TransferQueue here — otherwise the update loop gets no data and
        silently no-ops. Rank 0 reads the FULL numeric rows and broadcasts them;
        each rank then DP-shards the candidates *within each group* in
        :func:`hydrate_micro_batches` (design doc 3.1), so the step count (and the
        FSDP collective sequence) stays identical across ranks while the heavy
        replay is split.
        """
        import torch.distributed as dist

        from relax.engine.rollout.native_generation import hydrate_micro_batches

        if dist.is_initialized():
            world_group = dist.group.WORLD
            if dist.get_world_size(world_group) > 1:
                from relax.utils.distributed_utils import get_gloo_group

                world_group = get_gloo_group()
            world = dist.get_world_size(world_group)
        else:
            world = 1
        rows = rollout_data_ref
        if rows is None:
            # Only rank 0 consumes the partition from the TransferQueue, then
            # broadcasts the tiny numeric rows to all FSDP ranks. A single consumer
            # avoids a multi-rank race on the partition's consumption cursor; each
            # rank then hydrates the same per-group object-store trajectory and
            # selects its own candidate subset (DP sharding below). Legacy rows
            # without an ObjectRef still fall back to shared-disk sidecars.
            read_error = None
            if self._rank == 0:
                try:
                    rows = self._read_train_partition(rollout_id)
                except Exception as exc:
                    read_error = f"{type(exc).__name__}: {exc}"
            payload = [rows, read_error]
            if dist.is_initialized() and world > 1:
                # Object collectives must go over gloo (the codebase convention and
                # the old FSDP backend): broadcast_object_list on the default NCCL
                # group pickles to a CUDA byte tensor, which is slower and more
                # fragile than a host-side gloo broadcast of these tiny numeric rows.
                from relax.utils.distributed_utils import get_gloo_group

                dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
            rows, read_error = payload
            if read_error is not None:
                raise RuntimeError(f"TransferQueue read failed on rank 0: {read_error}")
        if rows is None:
            return []
        micro_batches: List[Dict[str, Any]] = []
        hydrate_error = None
        try:
            micro_batches = hydrate_micro_batches(
                self.args,
                self.adapter,
                rollout_id,
                rows,
                dp_rank=self._rank,
                dp_world=world,
            )
        except Exception as exc:
            hydrate_error = f"{type(exc).__name__}: {exc}"
        errors = _collect_weight_sync_errors(hydrate_error)
        if errors:
            detail = "; ".join(f"rank {rank}: {error}" for rank, error in errors)
            raise RuntimeError(f"Training sidecar hydration failed ({detail})")
        return micro_batches

    def _read_train_partition(self, rollout_id: int):
        """Rank-0 read of this step's numeric rows from the TransferQueue.

        Polls until the rollout's async_put to ``train_{rollout_id}`` is
        visible (producer and actor are separate processes), then returns the
        row TensorDict, or ``None`` if it never becomes ready.
        """
        import time

        from relax.engine.sft.runtime import sft_partition_id, sft_task_name

        client = self._tq_client()
        partition_id = sft_partition_id(self.args, rollout_id)
        task_name = sft_task_name(self.args, component="backend")
        data_fields = [
            "group_indices",
            "trajectory_slots",
            "trajectory_refs",
            "advantages",
            "raw_reward",
            "total_lengths",
            "skip_optimizer_step",
        ]
        deadline = time.time() + float(getattr(self.args, "rollout_request_timeout", 3600.0))
        while time.time() < deadline:
            meta = client.get_meta(
                data_fields=data_fields,
                batch_size=int(self.args.global_batch_size),
                partition_id=partition_id,
                task_name=task_name,
                # TODO(agent): DP work-sharding. dp_rank=0 makes rank 0 read the FULL
                # batch and broadcast it, so every FSDP rank replays every candidate
                # (N-way redundant diffusion replay; grads still correct via FSDP
                # all-reduce). Shard by dp_rank here + in hydrate_micro_batches so
                # replay compute scales with card count. Non-urgent (throughput, not
                # correctness); schedule separately.
                sampling_config={"dp_rank": 0, "task_name": task_name},
            )
            if getattr(meta, "size", 0) > 0:
                return client.get_data(meta)
            time.sleep(0.5)
        logger.warning(f"[rollout {rollout_id}] train partition {partition_id!r} never became ready; skipping.")
        return None

    def _lora_contract(self) -> Optional[Dict[str, Any]]:
        """The LoRA block of ``adapter_contract.json``; ``None`` under full
        FT."""
        if self._trainable_mode != "lora":
            return None
        from relax.backends.fsdp.lora import lora_metadata_dict

        return lora_metadata_dict(
            rank=int(self.args.lora_rank),
            alpha=int(self.args.lora_alpha),
            dropout=float(getattr(self.args, "lora_dropout", 0.0) or 0.0),
            target_modules=self._lora_target_modules,
            task_type=self._lora_task_type(),
        )

    def _adapter_contract(self) -> Dict[str, Any]:
        return {
            "family": self.adapter.family,
            "tasks": list(self.adapter.supported_tasks),
            "trainable_mode": self._trainable_mode,
            "save_mode": "adapter" if self._trainable_mode == "lora" else "full",
            "base_model_sha256": self._base_model_sha256,
            "lora": self._lora_contract(),
        }

    def _assert_resumable(self, ckpt_dir: str) -> None:
        """Reject a checkpoint whose adapter contract disagrees with this run.

        A LoRA checkpoint stores only the adapter — the base always comes back
        from ``--model-path`` — so resuming against a different base, rank or
        target list produces a policy that was never trained. Rank is at least
        a shape error deep inside DCP; **alpha is completely silent**, it just
        rescales the adapter. Check all of it before the load.
        """
        import json

        path = os.path.join(ckpt_dir, "adapter_contract.json")
        if not os.path.exists(path):
            return  # pre-contract checkpoint; nothing to compare against
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f)

        problems: List[str] = []
        want_mode = recorded.get("trainable_mode", "full")
        if want_mode != self._trainable_mode:
            problems.append(f"trainable_mode: checkpoint={want_mode!r} != current={self._trainable_mode!r}")
        recorded_sha = recorded.get("base_model_sha256")
        if recorded_sha and recorded_sha != self._base_model_sha256:
            problems.append("base_model_sha256 differs (the adapter was trained against another base model)")
        if self._trainable_mode == "lora":
            from relax.backends.fsdp.lora import contract_mismatches

            problems.extend(contract_mismatches(recorded.get("lora") or {}, self._lora_contract() or {}))
        if problems:
            raise ValueError(
                f"Refusing to resume {ckpt_dir}: adapter contract mismatch:\n  - " + "\n  - ".join(problems)
            )

    def _maybe_resume(self) -> int:
        from relax.backends.fsdp import checkpoint as ckpt

        if not getattr(self.args, "load", None):
            return 0
        latest = ckpt.find_latest_committed(self.args.load, self.args.generation_task)
        if latest is None:
            return 0
        self._assert_resumable(os.path.join(self.args.load, self.args.generation_task, ckpt.iter_dir_name(latest)))
        state = ckpt.load_checkpoint(
            self.args.load,
            self.args.generation_task,
            latest,
            self.model,
            self.optimizer,
            adapter_only=(self._trainable_mode == "lora"),
        )
        self.policy_version = int(state.get("policy_version", 0))
        # Resume the LR schedule where it left off (warmup must not restart).
        self._lr_steps = int(state.get("lr_scheduler_steps", 0))
        logger.info(
            f"Resumed FSDP checkpoint iter={latest} policy_version={self.policy_version} lr_steps={self._lr_steps}"
        )
        return latest + 1
