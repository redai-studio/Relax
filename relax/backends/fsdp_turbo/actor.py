# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDPTurbo actor for colocate synchronous GRPO and SFT.

GRPO synchronizes HF weights to colocated SGLang engines; SFT consumes
TransferQueue batches and eval chunks. Both regimes use HF-native checkpoints.
"""

from __future__ import annotations

import math
import os
import re
import time
from argparse import Namespace
from collections import defaultdict
from typing import Any

import ray
import torch
import torch.distributed as dist
import transfer_queue as tq

from relax.algorithms.advantages import compute_advantages_and_returns
from relax.backends.fsdp_turbo.checkpoint import load_checkpoint, save_checkpoint
from relax.backends.fsdp_turbo.lr_scheduler import MegatronParityWarmupCosineScheduler
from relax.backends.fsdp_turbo.model_utils import (
    _is_qwen4_exp,
    build_microbatches,
    build_turbo_model,
    compute_rollout_log_probs,
    grpo_microbatch_loss,
    sft_microbatch_loss,
    sft_microbatch_nll_stats,
    sft_step_supervised_tokens,
)
from relax.backends.fsdp_turbo.turbo_config_utils import load_turbo_config, validate_turbo_config
from relax.backends.fsdp_turbo.weight_update import FSDPTurboUpdateWeight
from relax.distributed.ray.train_actor import TrainRayActor
from relax.engine.sft.runtime import (
    evaluation_step_for_rollout,
    is_sft_mode,
    sft_partition_id,
    sft_partition_ids,
    sft_task_name,
    should_run_sft_eval,
)
from relax.utils.async_utils import run
from relax.utils.data.stream_dataloader import (
    get_data_from_transfer_queue,
    post_process_rollout_data,
)
from relax.utils.device import make_current_torch_device
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.timer import Timer
from relax.utils.training.data_fields import build_data_fields
from relax.utils.utils import get_debug_data


logger = get_logger(__name__)


def _summarize_chrome_trace(path: str, top_k: int = 50, max_bytes: int = 512 * 1024 * 1024) -> str:
    """Summarize an exported chrome trace in process.

    Aggregate trace JSON directly because this torch_npu stack exposes no
    ``key_averages``/``events`` API.
    """
    import json
    from collections import Counter

    if os.path.getsize(path) > max_bytes:
        raise RuntimeError(f"trace is {os.path.getsize(path)} bytes (> {max_bytes}); analyze it offline instead")
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    cat_tot = Counter()
    agg = defaultdict(lambda: [0, 0, 0])
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or not e.get("dur"):
            continue
        cat = e.get("cat", "?")
        cat_tot[cat] += e["dur"]
        rec = agg[(cat, e.get("name", "?"))]
        rec[0] += e["dur"]
        rec[1] += 1
        rec[2] = max(rec[2], e["dur"])
    lines = ["-- total time by category (s) --"]
    lines.extend(f"  {tot / 1e6:12.2f}  {cat}" for cat, tot in cat_tot.most_common(8))
    lines.append(f"-- top {top_k} events by total time --")
    ranked = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top_k]
    lines.extend(
        f"  {tot / 1e6:10.2f}s {cnt:7d}x avg {tot / cnt / 1e3:9.3f}ms max {mx / 1e3:9.2f}ms [{cat}] {name[:90]}"
        for (cat, name), (tot, cnt, mx) in ranked
    )
    return "\n".join(lines)


_FETCH_SPLIT_MAX_RETRIES = 50


def _concat_rollout_batches(batches: list[dict]) -> dict:
    """Merge per-fetch dicts, concatenating per-sample lists."""
    merged: dict[str, Any] = {}
    for key, first_value in batches[0].items():
        values = [batch[key] for batch in batches]
        if isinstance(first_value, list):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values[-1]
    return merged


class FSDPTurboTrainRayActor(TrainRayActor):
    """Synchronous SFT/GRPO trainer over an FSDPTurbo-wrapped HF model."""

    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int:
        super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        if role != "actor":
            raise ValueError(f"fsdp_turbo backend only supports the 'actor' role, got {role!r}")
        if with_ref or with_opd_teacher:
            raise ValueError("fsdp_turbo backend does not support reference/teacher models (keep --kl-coef 0).")

        # SFT has no rollout service to install a manager.
        self.rollout_manager = None

        # Heavy optional dependency: imported inside the actor process only.
        from transformers import AutoConfig, AutoTokenizer

        turbo_config = load_turbo_config(args.turbo_config)
        self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # CP is SFT-only because GRPO logprob/advantage handling is not CP-aware;
        # qwen4_exp shards fused sequences natively in its consolidated modeling.
        validate_turbo_config(turbo_config, allow_cp=is_sft_mode(args), native_cp=_is_qwen4_exp(self.hf_config))

        tq.init(args.tq_config)
        self.data_system_client = tq.get_client()

        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        self.turbo = build_turbo_model(args, turbo_config, self.hf_config)
        self._has_trainable_visual = any(
            param.requires_grad and "visual" in name.split(".") for name, param in self.turbo.named_parameters()
        )

        # MTP (Qwen3.8/qwen4_exp multi-token prediction): scaling applied to
        # the MTP loss stashed by the patched forward; None disables it.
        mtp_depth = int(getattr(turbo_config.model, "mtp_num_layers", 0) or 0)
        self._mtp_loss_scaling = (
            float(getattr(turbo_config.model, "mtp_loss_scaling_factor", 0.0) or 0.0) / mtp_depth
            if mtp_depth > 0
            else None
        )
        # Enable HF activation checkpointing for training; no_grad eval and logprob forwards are unaffected.
        if hasattr(self.turbo.model, "gradient_checkpointing_enable"):
            self.turbo.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.turbo.train()
        self._init_optimizer(args, turbo_config)

        from fsdp_turbo.distributed.parallel_state import get_parallel_state

        self.parallel_state = get_parallel_state()
        self._dp_size = int(self.parallel_state.get_group_size("data"))
        self._dp_rank = int(self.parallel_state.get_rank("data"))
        self._fsdp_size = int(self.parallel_state.get_group_size("fsdp"))
        self._tp_size = int(self.parallel_state.get_group_size("tp"))
        if args.global_batch_size % self._dp_size:
            raise ValueError("global_batch_size must be divisible by the Turbo data group size")
        self._build_same_data_group()

        self.weight_updater = FSDPTurboUpdateWeight(args, self.turbo)

        from relax.utils.training.flops_counter import FlopsCounter

        self.flops_counter = FlopsCounter(self.hf_config)

        # Initialize the metrics adapter on the primary rank.
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            from relax.utils.tracking_utils import init_tracking

            init_tracking(args, primary=False)

        self.train_parallel_config = {
            "train_backend": "fsdp_turbo",
            "dp_size": self._dp_size,
            "tp_size": self._tp_size,
            "pp_size": 1,
        }

        resumed_rollout_id = 0
        if args.load is not None:
            resumed_rollout_id = load_checkpoint(
                args, self.turbo, self.optimizer, self.hf_config, self.tokenizer, self.lr_scheduler
            )

        logger.info(
            f"[fsdp_turbo] actor initialized: dp={self._dp_size} fsdp={self._fsdp_size} tp={self._tp_size} "
            f"role={self.role} resume_rollout_id={resumed_rollout_id}"
        )
        # Start inter-step train_wait; _train_actor brackets full-step train and compute-only actor_train.
        Timer().start("train_wait")
        return resumed_rollout_id

    def _init_optimizer(self, args: Namespace, turbo_config) -> None:
        """Initialize the optimizer and its learning-rate schedule."""
        from fsdp_turbo.quantization.cache import hook_optimizer_step
        from fsdp_turbo.training.factories import OptimizerFactory, SchedulerFactory

        optimizer_config = turbo_config.optimizer
        from .performance_config import optimizer_options

        self.optimizer = OptimizerFactory.create(
            optimizer_config.optimizer_type,
            self.turbo.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
            eps=optimizer_config.adam_epsilon,
            **optimizer_options(turbo_config),
        )
        hook_optimizer_step(self.turbo, self.optimizer)
        total_steps = (
            int(turbo_config.run.max_steps) if int(turbo_config.run.max_steps) > 0 else max(int(args.num_rollout), 1)
        )
        self.lr_scheduler = SchedulerFactory.create(
            optimizer_config.lr_scheduler_type,
            self.optimizer,
            T_max=total_steps,
            eta_min=optimizer_config.min_lr,
        )
        # The factory has no warmup, so match Megatron: linear warmup, then cosine over
        # total_steps - warmup_span. Keep warmup_span fractional and step after optimizer.step().
        warmup_span = float(getattr(optimizer_config, "warmup_steps", 0) or 0)
        if not hasattr(optimizer_config, "warmup_steps"):
            warmup_ratio = float(getattr(optimizer_config, "warmup_ratio", 0.0) or 0.0)
            if warmup_ratio > 0:
                warmup_span = warmup_ratio * total_steps
        if warmup_span > 0:
            if optimizer_config.lr_scheduler_type != "CosineAnnealingLR":
                raise ValueError(
                    "[fsdp_turbo] optimizer.warmup_ratio/warmup_steps is set but warmup is only "
                    f"implemented for CosineAnnealingLR, got '{optimizer_config.lr_scheduler_type}'"
                )
            base_lr = float(optimizer_config.lr)
            min_lr = float(optimizer_config.min_lr)
            decay_span = max(float(total_steps) - warmup_span, 1.0)

            def _warmup_cosine_lr(step_idx: int) -> float:
                if step_idx < warmup_span:
                    # megatron warmup: init_lr + (max_lr - init_lr) * step / warmup_steps.
                    return base_lr * step_idx / warmup_span
                progress = step_idx - warmup_span
                if progress >= decay_span:
                    # Clamp beyond the decay span instead of letting the cosine rise again.
                    return min_lr
                # megatron cosine: min_lr + (max_lr - min_lr) * (1 + cos(pi * progress / decay)) / 2.
                return min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress / decay_span)) / 2.0

            self.lr_scheduler = MegatronParityWarmupCosineScheduler(self.optimizer, _warmup_cosine_lr)
            logger.info(
                f"[fsdp_turbo] lr warmup enabled: {warmup_span:g} step(s) ramping from 0, then cosine decay "
                f"to {optimizer_config.min_lr} over {decay_span:g} steps (megatron WarmupDecayLR parity)"
            )

    def _build_same_data_group(self) -> None:
        """Build a Gloo group for ranks sharing one logical DP slot.

        FSDP shards participate in data parallelism. Fetch agreement only
        couples TP/CP partners sharing the same data coordinate.
        """
        # FSDP ranks consume different samples. Only ranks with the same
        # data coordinate (TP/CP partners) replay one input batch.
        coordinates = [None] * dist.get_world_size()
        dist.all_gather_object(coordinates, self._dp_rank, group=get_gloo_group())
        self._same_data_group = None
        for data_rank in range(self._dp_size):
            ranks = [rank for rank, coordinate in enumerate(coordinates) if coordinate == data_rank]
            group = dist.new_group(ranks=ranks, backend="gloo")
            if data_rank == self._dp_rank:
                self._same_data_group = group
                self._same_data_ranks = ranks

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------

    def all_consumed(
        self, task_name: str, rollout_id: int, partition_id: str | None = None, *, all_data: bool = False
    ) -> bool:
        """Unanimous drain check over the ranks sharing this DP slot."""
        if partition_id is None:
            partition_id = sft_partition_id(self.args, rollout_id)
        status = run(self.data_system_client.async_check_consumption_status(task_name, partition_id))
        flag = torch.tensor([1 if status else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=get_gloo_group() if all_data else self._same_data_group)
        return bool(flag.item())

    def _fetch_batch_with_agreement(
        self,
        task_name: str,
        partition_id: str,
        data_fields: list[str],
        batch_size: int,
        batch_index: int,
        *,
        all_data: bool = False,
    ) -> dict | None:
        """Fetch one TQ batch after DP-slot ranks agree on empty versus data.

        Ranks use identical ``dp_rank``/``batch_index`` sampling keys, so a re-
        fetch replays the same samples.
        """
        from .data_parallel import fetch_with_agreement

        sampling_config = {"dp_rank": self._dp_rank, "task_name": task_name}
        agreement_group = get_gloo_group() if all_data else self._same_data_group

        def fetch() -> dict | None:
            rollout_data, _batch_meta = get_data_from_transfer_queue(
                self.args,
                self.data_system_client,
                data_fields,
                batch_size,
                partition_id,
                task_name,
                sampling_config,
                batch_index,
                broadcast_pp=False,
                per_rank_fetch=True,
                synchronize_per_rank_fetch=False,
            )
            return rollout_data

        def readiness(ready: bool) -> tuple[int, int]:
            tally = torch.tensor([1, 0] if ready else [0, 1], dtype=torch.int32)
            dist.all_reduce(tally, op=dist.ReduceOp.SUM, group=agreement_group)
            return tuple(tally.tolist())

        return fetch_with_agreement(
            fetch,
            readiness,
            attempts=_FETCH_SPLIT_MAX_RETRIES,
            pause=time.sleep,
            context=f"partition={partition_id} batch_index={batch_index}",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, rollout_id: int) -> None:
        if self.args.debug_train_only:
            logger.info(f"[fsdp_turbo] debug train only: rollout_id={rollout_id}")
            rollout_data = get_debug_data(
                self.args, rollout_id, self.args.global_batch_size // self._dp_size, dp_rank=self._dp_rank
            )
            post_process_rollout_data(self.args, rollout_data)
            self._train_actor(rollout_id, rollout_data)
            return

        task_name = sft_task_name(self.args, component="backend")
        partition_id = sft_partition_id(self.args, rollout_id)
        data_fields = build_data_fields(self.args, consumer="actor")
        batch_size = max(self.args.global_batch_size // self._dp_size, 1)

        logger.info(f"[fsdp_turbo] fetching rollout_id={rollout_id} data (task={task_name}, fields={data_fields})")
        fetched: list[dict] = []
        batch_index = 0
        empty_poll_sleep_s = Envs.RELAX_EMPTY_POLL_SLEEP_MS / 1000.0
        while not self.all_consumed(task_name, rollout_id, partition_id):
            rollout_data = self._fetch_batch_with_agreement(
                task_name, partition_id, data_fields, batch_size, batch_index
            )
            if rollout_data is None:
                if empty_poll_sleep_s > 0:
                    time.sleep(empty_poll_sleep_s)
                continue
            fetched.append(rollout_data)
            batch_index += 1

        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, sum(len(batch["total_lengths"]) for batch in fetched), group=get_gloo_group())
        if any(count == 0 for count in counts) and any(count > 0 for count in counts):
            raise RuntimeError(f"Uneven Turbo data slots cannot enter FSDP training safely: {counts}")
        if not fetched:
            logger.warning(f"[fsdp_turbo] no training data for rollout_id={rollout_id}; skipping step")
            return
        rollout_data = _concat_rollout_batches(fetched) if len(fetched) > 1 else fetched[0]
        self._train_actor(rollout_id, rollout_data)

    def _aligned_microbatches(self, lengths: list[int], max_tokens: int | None) -> list[list[int]]:
        from .data_parallel import align_microbatches

        microbatches = build_microbatches(lengths, max_tokens)
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, (len(microbatches), len(lengths)), group=get_gloo_group())
        return align_microbatches(microbatches, counts)

    def _mean_data_count(self, local_count: float, device: torch.device) -> float:
        """Use a global count / data_size to offset FSDP gradient averaging."""
        count = torch.tensor(float(local_count), dtype=torch.float32, device=device)
        dist.all_reduce(count, group=self.parallel_state.get_group("data"))
        return float(count.item()) / self._dp_size

    def _validate_visual_microbatches(self, rollout_data: dict, microbatches: list[list[int]]) -> None:
        if not self._has_trainable_visual:
            return
        inputs = rollout_data.get("multimodal_train_inputs") or []
        schedule = [
            tuple(
                any(inputs[index] is not None and inputs[index].get(key) is not None for index in batch)
                if inputs
                else False
                for key in ("pixel_values", "pixel_values_videos")
            )
            for batch in microbatches
        ]
        group = get_gloo_group()
        schedules = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(schedules, schedule, group=group)
        if any(other != schedule for other in schedules):
            raise ValueError("fsdp_turbo requires matching visual forward participation across ranks per microbatch")

    def _train_actor(self, rollout_id: int, rollout_data: dict) -> None:
        # Close stale half-open intervals before starting full-step train so a prior exception cannot
        # leave timer state that masks the next error.
        if "train" in Timer().start_time:
            Timer().end("train")
        if "train_wait" in Timer().start_time:
            Timer().end("train_wait")
        Timer().start("train")

        # actor_train covers only fwd/bwd/optimizer, excluding checkpoint, weight sync, and eval;
        # close it before the perf logger snapshots and resets timers.
        if "actor_train" in Timer().start_time:
            Timer().end("actor_train")
        Timer().start("actor_train")

        device = make_current_torch_device()
        max_tokens = getattr(self.args, "max_tokens_per_gpu", None)
        microbatches = self._aligned_microbatches(rollout_data["total_lengths"], max_tokens)
        # Token counts for the perf logger (perf/actor_train_tokens, tok_per_s).
        Timer().seq_lens = [int(length) for length in rollout_data["total_lengths"]]

        mean_step_samples = self._mean_data_count(len(rollout_data["total_lengths"]), device)
        if not is_sft_mode(self.args) and round(mean_step_samples * self._dp_size) > self.args.global_batch_size:
            raise ValueError("fsdp_turbo RL supports only one global optimizer batch per rollout")
        if not is_sft_mode(self.args):
            has_images = self._mean_data_count(
                any(item is not None and bool(item) for item in (rollout_data.get("multimodal_train_inputs") or [])),
                device,
            )
            if has_images:
                raise ValueError("fsdp_turbo image GRPO is not supported")
        skip_optimizer_step = False
        if is_sft_mode(self.args):
            metrics, skip_optimizer_step = self._train_sft(
                rollout_id, rollout_data, microbatches, device, mean_step_samples
            )
        else:
            mean_tokens = self._mean_data_count(sum(float(mask.sum()) for mask in rollout_data["loss_masks"]), device)
            skip_optimizer_step = mean_tokens <= 0
            metrics = (
                {}
                if skip_optimizer_step
                else self._train_rl(rollout_id, rollout_data, microbatches, device, mean_step_samples)
            )

        if skip_optimizer_step:
            # Clear stale gradients and leave optimizer and LR state untouched.
            self.optimizer.zero_grad()
            metrics["grad_norm"] = 0.0
            metrics["optimizer_step_skipped"] = 1.0
        else:
            from fsdp_turbo.training.clip_grads import clip_grad_norm

            grad_norm = clip_grad_norm(
                self.turbo,
                max_norm=self.turbo.config.optimizer.clip_grad or None,
                norm_type=self.turbo.config.optimizer.clip_grad_norm_type,
                group=None,
            )
            grad_norm_value = float(grad_norm) if grad_norm is not None else 0.0
            finite = torch.tensor([int(math.isfinite(grad_norm_value))], dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=get_gloo_group())
            if bool(finite.item()):
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            else:
                logger.warning(f"[fsdp_turbo] rollout={rollout_id}: non-finite gradient; skipping optimizer and LR")
            self.optimizer.zero_grad()
            metrics["grad_norm"] = grad_norm_value
            metrics["optimizer_step_skipped"] = float(not bool(finite.item()))
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])

        # Exclude checkpoint, weight sync, and eval from actor_train.
        Timer().end("actor_train")

        # Token-mean SFT loss has already been reduced across unique data ranks.
        is_primary = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_primary:
            from relax.utils import tracking_utils

            log_dict = {f"train/{key}": value for key, value in metrics.items()}
            log_dict["train/global_batch_size"] = float(mean_step_samples * self._dp_size)
            for param_group_id, param_group in enumerate(self.optimizer.param_groups):
                log_dict[f"train/lr-pg_{param_group_id}"] = float(param_group["lr"])
            log_dict["train/step"] = rollout_id
            num_per_epoch = getattr(self.args, "num_rollout_per_epoch", None)
            if num_per_epoch:
                log_dict["train/cur_epoch"] = (rollout_id + 1) / num_per_epoch
            tracking_utils.log(self.args, log_dict, step_key="train/step")
            tracking_utils.flush_metrics(self.args, rollout_id)
            logger.info(f"step {rollout_id}: {log_dict}")

        is_train_done = (rollout_id + 1) == self.args.num_rollout
        if (
            self.args.save is not None
            and self.args.save_interval is not None
            and ((rollout_id + 1) % self.args.save_interval == 0 or is_train_done)
        ):
            self.save_model(rollout_id, force_sync=is_train_done)

        if self.rollout_manager is not None and not self.args.debug_train_only:
            self.update_weights()

        evaluation_step = evaluation_step_for_rollout(self.args, rollout_id)
        if is_sft_mode(self.args) and should_run_sft_eval(self.args, evaluation_step):
            self._run_sft_eval(rollout_id, evaluation_step)

        # Close full-step train before the perf logger resets timers, then reopen train_wait.
        Timer().end("train")
        Timer().start("train_wait")

        self._log_rollout_and_perf(rollout_id, rollout_data, is_primary)

    def _train_sft(
        self,
        rollout_id: int,
        rollout_data: dict,
        microbatches: list[list[int]],
        device: torch.device,
        mean_step_samples: float,
    ) -> tuple[dict[str, float], bool]:
        """Accumulate SFT gradients and metrics for one optimizer batch."""
        metrics: dict[str, float] = defaultdict(float)
        skip_optimizer_step = False
        # Masked next-token NLL; loss_masks mark supervised tokens
        # (response_lengths == total_lengths on the SFT data plane).
        num_local_samples = mean_step_samples
        # Optionally match Megatron's token-mean NLL instead of the sample-mean default.
        per_token_loss = bool(getattr(self.args, "calculate_per_token_loss", False))
        step_token_count = sft_step_supervised_tokens(rollout_data) if per_token_loss else None
        if per_token_loss:
            # Average complete step counts over the actual data group. Its size already
            # excludes CP replicas; dp * fsdp includes them and would shrink the denominator
            # by CP-size. The CP loss's backward compensation remains separate and unchanged.
            data_group = self.parallel_state.get_group("data")
            data_size = dist.get_world_size(group=data_group)
            if data_size < 1:
                raise RuntimeError(f"SFT token normalization requires membership in the data group, got {data_size}")
            if data_size > 1:
                token_t = torch.tensor(float(step_token_count), dtype=torch.float32, device=device)
                dist.all_reduce(token_t, op=dist.ReduceOp.SUM, group=data_group)
                step_token_count = float(token_t.item()) / data_size
            if (step_token_count or 0) <= 0:
                # A global zero-token step skips optimizer and LR updates;
                # sample-mean mode retains its zero-grad step behavior.
                logger.warning(
                    f"[fsdp_turbo] rollout={rollout_id}: zero supervised tokens in the whole "
                    "step; skipping fwd/bwd and optimizer + LR updates"
                )
                skip_optimizer_step = True
                metrics["loss"] = 0.0
        if not skip_optimizer_step:
            self._validate_visual_microbatches(rollout_data, microbatches)
        for microbatch in microbatches:
            if skip_optimizer_step:
                break
            loss, micro_metrics = sft_microbatch_loss(
                self.turbo,
                rollout_data,
                microbatch,
                device,
                num_local_samples,
                self._mtp_loss_scaling,
                per_token_loss=per_token_loss,
                step_token_count=step_token_count,
            )
            loss.backward()
            for key, value in micro_metrics.items():
                if key in ("nll_sum", "token_count", "mtp_numerator", "mtp_denominator"):
                    # Preserve raw sums for the exact step token-mean below.
                    metrics[key] += value
                else:
                    metrics[key] += value * len(microbatch) / num_local_samples
        if per_token_loss and not skip_optimizer_step:
            totals = torch.tensor(
                [metrics.pop("nll_sum", 0.0), metrics.pop("token_count", 0.0)],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(totals, group=self.parallel_state.get_group("data"))
            metrics["loss"] = float((totals[0] / totals[1].clamp(min=1)).item())
        elif not per_token_loss:
            loss_metric = torch.tensor(metrics["loss"], dtype=torch.float32, device=device)
            dist.all_reduce(loss_metric, group=self.parallel_state.get_group("data"))
            metrics["loss"] = float(loss_metric.item()) / self._dp_size
        if self._mtp_loss_scaling:
            mtp_stats = torch.tensor(
                [metrics.pop("mtp_numerator", 0.0), metrics.pop("mtp_denominator", 0.0)],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(mtp_stats, group=self.parallel_state.get_group("data"))
            metrics["mtp_loss"] = float((mtp_stats[0] / mtp_stats[1].clamp(min=1)).item())

        return metrics, skip_optimizer_step

    def _train_rl(
        self,
        rollout_id: int,
        rollout_data: dict,
        microbatches: list[list[int]],
        device: torch.device,
        mean_step_samples: float,
    ) -> dict[str, float]:
        """Accumulate GRPO gradients and metrics for one optimizer batch."""
        metrics: dict[str, float] = defaultdict(float)
        from relax.algorithms import get_algorithm

        token_global = get_algorithm(self.args.advantage_estimator).advantage_normalization == "token_global"
        mean_step_tokens = None
        if token_global or getattr(self.args, "calculate_per_token_loss", False):
            mean_step_tokens = self._mean_data_count(
                sum(float(mask.sum()) for mask in rollout_data["loss_masks"]), device
            )
        # Recompute old-policy log-probs unless rollout log-probs are trusted.
        if not self.args.use_rollout_logprobs:
            logger.info(
                f"[fsdp_turbo] rollout={rollout_id}: recomputing old log-probs ({len(microbatches)} microbatches)"
            )
            rollout_data["log_probs"] = compute_rollout_log_probs(
                self.turbo, rollout_data, microbatches, device, temperature=self.args.rollout_temperature
            )
            logger.info(f"[fsdp_turbo] rollout={rollout_id}: old log-probs done")

        # KL is zero because this backend requires kl_coef == 0.
        kl = [
            torch.zeros(int(length), dtype=torch.float32, device=device) for length in rollout_data["response_lengths"]
        ]
        advantages, returns = compute_advantages_and_returns(
            self.args,
            rewards=rollout_data["rewards"],
            kl=kl,
            loss_masks=rollout_data["loss_masks"],
            response_lengths=rollout_data["response_lengths"],
            total_lengths=rollout_data["total_lengths"],
        )
        rollout_data["advantages"] = advantages
        rollout_data["returns"] = returns

        # RELAX_PROFILE_MB=N profiles the first N GRPO microbatches on the primary rank.
        profile_mb_count = int(os.environ.get("RELAX_PROFILE_MB", "0") or 0)
        profile_is_primary = (not dist.is_initialized()) or dist.get_rank() == 0
        for mb_idx, microbatch in enumerate(microbatches):
            use_prof = profile_mb_count and mb_idx < profile_mb_count and profile_is_primary
            if use_prof:
                from torch_npu import profiler

                # Level0 avoids the default Level1's per-aten-op events and oversized traces.
                experimental_config = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level0)
                prof = profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                    experimental_config=experimental_config,
                )
                prof.__enter__()
            loss, micro_metrics = grpo_microbatch_loss(
                self.turbo,
                self.args,
                rollout_data,
                microbatch,
                device,
                step_sample_count=mean_step_samples,
                step_token_count=mean_step_tokens,
            )
            loss.backward()
            if use_prof:
                prof.__exit__(None, None, None)
                trace_path = f"/tmp/relax_npu_prof_rollout{rollout_id}_mb{mb_idx}.json"
                prof.export_chrome_trace(trace_path)
                try:
                    summary = _summarize_chrome_trace(trace_path)
                    logger.info(f"[fsdp_turbo] NPU kernel top summary (rollout={rollout_id} mb={mb_idx}):\n{summary}")
                except Exception as e:  # noqa: BLE001 - summary is best-effort
                    logger.warning(f"[fsdp_turbo] kernel summary unavailable: {e}")
                try:
                    import gzip
                    import shutil

                    with open(trace_path, "rb") as f_in, gzip.open(trace_path + ".gz", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(trace_path)
                    trace_path += ".gz"
                except Exception as e:  # noqa: BLE001 - compression is best-effort
                    logger.warning(f"[fsdp_turbo] trace gzip failed, kept raw json: {e}")
                logger.info(f"[fsdp_turbo] NPU op-level profile exported to {trace_path}")
            for key, value in micro_metrics.items():
                weight = 1.0 if key == "pg_loss" else len(microbatch) / mean_step_samples
                metrics[key] += value * weight
        metric_names = ("pg_loss", "ratio_clipped_frac", "entropy")
        reduced = torch.tensor([metrics[key] for key in metric_names], dtype=torch.float32, device=device)
        dist.all_reduce(reduced, group=self.parallel_state.get_group("data"))
        for key, value in zip(metric_names, reduced.tolist()):
            metrics[key] = value / self._dp_size

        return metrics

    def _log_rollout_and_perf(self, rollout_id: int, rollout_data: dict, is_primary: bool) -> None:
        """Log rollout statistics and performance timing.

        Gather unique data-rank sample lengths for global throughput. Every
        rank resets timers; only the primary rank emits logs.
        """
        from relax.utils import tracking_utils
        from relax.utils.metrics.metric_utils import compute_rollout_step
        from relax.utils.training.train_metric_utils import log_perf_data_raw

        total_lengths = [int(length) for length in rollout_data.get("total_lengths") or []]
        # Count unique samples once across TP/CP partners, then expose global
        # lengths to the existing throughput/FLOPs logger on every rank.
        lengths_by_rank = [None] * dist.get_world_size()
        unique_lengths = total_lengths if dist.get_rank() == self._same_data_ranks[0] else []
        dist.all_gather_object(lengths_by_rank, unique_lengths, group=get_gloo_group())
        Timer().seq_lens = [length for lengths in lengths_by_rank for length in lengths]
        total_lengths = Timer().seq_lens
        if total_lengths and is_primary:
            rollout_log = {
                "total_lengths/max": float(max(total_lengths)),
                "total_lengths/min": float(min(total_lengths)),
                "total_lengths/mean": sum(total_lengths) / len(total_lengths),
            }
            prefixed = {f"rollout/{key}": value for key, value in rollout_log.items()}
            prefixed["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            tracking_utils.log(self.args, prefixed, step_key="rollout/step")
            logger.info(f"rollout {rollout_id}: {rollout_log}")

        log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=is_primary,
            flops_counter=self.flops_counter,
            world_size=dist.get_world_size() if dist.is_initialized() else 1,
        )

    # ------------------------------------------------------------------
    # SFT eval (PPL over the producer's eval chunks)
    # ------------------------------------------------------------------

    def _run_sft_eval(self, rollout_id: int, evaluation_step: int) -> None:
        """Consume SFT eval chunks and log PPL.

        All-reduce sufficient statistics over Gloo, then divide out TP/CP
        duplicate fetches so supervised tokens are counted once.
        """

        from relax.engine.sft.eval.ppl import compute_ppl_metrics

        args = self.args
        task_name = "sft_eval"
        data_fields = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
        if getattr(args, "multimodal_keys", None) is not None:
            data_fields.append("multimodal_train_inputs")
        batch_size = max(args.global_batch_size // self._dp_size, 1)
        device = make_current_torch_device()

        # The producer waits for partition removal, not consumption; clear training partitions here
        # because the Actor component would otherwise clear them only after train() returns.
        if dist.get_rank() == 0:
            for partition_id in sft_partition_ids(args, rollout_id):
                run(self.data_system_client.async_clear_partition(partition_id=partition_id))
        dist.barrier(group=get_gloo_group())

        pattern = re.compile(rf"^sft_eval_{evaluation_step}_n(\d+)_\d+$")
        n_chunks = 0
        while n_chunks == 0:
            partitions = run(self.data_system_client.async_get_partition_list())
            if partitions:
                for partition in partitions:
                    match = pattern.match(partition)
                    if match:
                        n_chunks = int(match.group(1))
                        break
            if n_chunks == 0:
                time.sleep(1)

        max_tokens = getattr(args, "max_tokens_per_gpu", None)
        local_neg_log_prob = 0.0
        local_num_tokens = 0
        self.turbo.eval()
        try:
            for chunk_idx in range(n_chunks):
                partition_id = f"sft_eval_{evaluation_step}_n{n_chunks}_{chunk_idx}"
                # Wait for the chunk so a missing partition cannot end the loop early.
                while True:
                    partitions = run(self.data_system_client.async_get_partition_list())
                    if partitions and partition_id in partitions:
                        break
                    time.sleep(1)
                batch_index = 0
                while not self.all_consumed(task_name, evaluation_step, partition_id=partition_id, all_data=True):
                    eval_data = self._fetch_batch_with_agreement(
                        task_name, partition_id, data_fields, batch_size, batch_index, all_data=True
                    )
                    if eval_data is None:
                        time.sleep(0.25)
                        continue
                    batch_index += 1
                    microbatches = self._aligned_microbatches(eval_data["total_lengths"], max_tokens)
                    self._validate_visual_microbatches(eval_data, microbatches)
                    for microbatch in microbatches:
                        nll, count = sft_microbatch_nll_stats(self.turbo, eval_data, microbatch, device)
                        local_neg_log_prob += nll
                        local_num_tokens += count
                # TQ does not auto-clean consumed partitions; free this chunk for the producer.
                dist.barrier(group=get_gloo_group())
                if dist.get_rank() == 0:
                    run(self.data_system_client.async_clear_partition(partition_id=partition_id))
        finally:
            self.turbo.train()

        # gloo all_reduce requires CPU tensors (mirrors all_consumed above).
        stats = torch.tensor([local_neg_log_prob, float(local_num_tokens)], dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=get_gloo_group())
        # Only TP/CP ranks replay identical data; FSDP samples are distinct.
        stats /= len(self._same_data_ranks)
        metrics = compute_ppl_metrics(float(stats[0].item()), int(stats[1].item()))
        logger.info(f"[fsdp_turbo] SFT eval @ step={evaluation_step}: {metrics}")

    # ------------------------------------------------------------------
    # Weight sync / checkpointing / lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        if self.args.offload_rollout and dist.get_rank() == 0:
            # Onload rollout weights before copying buckets because offload drops their NPU pages.
            ray.get(self.rollout_manager.onload_weights.remote())
        dist.barrier(group=get_gloo_group())
        rollout_engines, rollout_engine_lock, num_new_engines, engine_gpu_counts, engine_gpu_offsets = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())
        self.weight_updater.update_weights()
        # Warm KV and graph regions for the next generate round.
        if self.args.offload_rollout and dist.get_rank() == 0:
            ray.get(self.rollout_manager.onload_kv.remote())

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only or self.role != "actor":
            return
        save_checkpoint(
            self.args, rollout_id, self.turbo, self.optimizer, self.hf_config, self.tokenizer, self.lr_scheduler
        )

    def sleep(self, tags=None) -> None:
        logger.warning(
            "[fsdp_turbo] sleep() is a no-op: sharded train weights stay resident. "
            "Budget --sglang-mem-fraction-static so engine + train sides fit together."
        )

    def wake_up(self, tags=None) -> None:
        # Resident sharded weights need no reload.
        logger.warning("[fsdp_turbo] wake_up() is a no-op: sharded train weights stay resident.")

    def _get_parallel_config(self) -> dict:
        return self.train_parallel_config

    # ------------------------------------------------------------------
    # Unsupported service entry points (colocate GRPO only)
    # ------------------------------------------------------------------

    def train_async(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async training")

    def train_hybrid(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support hybrid training")

    def train_critic(self, rollout_id: int, rollout_data: dict | None = None) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support critic training")

    def compute_ref_log_prob(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support reference log-probs (keep --kl-coef 0)")

    def compute_actor_log_prob(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support standalone actor log-probs")

    def update_weights_fully_async(
        self, rollout_id: int, rollout_only: bool = False, actor_fwd_only: bool = False
    ) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async weight updates")

    def recv_weight_fully_async(self, rollout_id: int) -> None:
        raise NotImplementedError("fsdp_turbo backend does not support fully-async weight updates")
