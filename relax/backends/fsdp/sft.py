# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Supervised image training delegated to by the common FSDP Ray actor.

Owns the SFT data cursor, scheduler and update. Model math and image processing
belong to the adapter; initialization, FSDP and DCP belong to the common actor.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from relax.backends.fsdp import checkpoint
from relax.engine.sft.image import ImageBatchSampler, ImageEditDataset, assign_image_batch
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
TASK = "i2i_sft"


def _report_sft_metrics(args: Any, values: dict[str, Any]) -> None:
    from relax.utils.tracking_utils import flush_metrics, log

    try:
        log(args, values, step_key="train/step")
    finally:
        # MetricsService buffers until this exact completed-step is reported.
        # Keep telemetry failures from turning a valid optimizer update into a
        # failed training step, consistent with the generative FSDP path.
        try:
            flush_metrics(args, values["train/step"])
        except Exception as exc:
            logger.warning("Image SFT metric flush skipped: %s", exc)


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def restore_rng(directory: Path, rank: int) -> None:
    """Strict restoration for trusted, locally produced training
    checkpoints."""
    state = torch.load(directory / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available():
        if state["cuda"] is None:
            raise ValueError("GPU checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])


def clip_sharded_gradients(parameters: list[torch.nn.Parameter], max_norm: float, group: Any) -> torch.Tensor:
    """Global L2 norm of disjoint FSDP2 shards, with no scalar host
    transfers."""
    grads = [p.grad.to_local() for p in parameters if p.grad is not None]
    squares = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    for grad in grads:
        squares.add_(grad.float().square().sum())
    dist.all_reduce(squares, group=group)
    norm = squares.sqrt()
    torch._assert_async(torch.isfinite(norm), "Non-finite gradient norm")
    if max_norm > 0 and grads:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(grads, scale)
    return norm


class FSDPSFTTrainer:
    def __init__(self, actor: Any) -> None:
        from relax.backends.fsdp.actor import _collect_rank_objects, _collect_weight_sync_errors

        self.actor, self.args = actor, actor.args
        self.group = actor._device_group()
        self.world, self.rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        if self.args.global_batch_size % self.world:
            raise ValueError("Image SFT global batch size must be divisible by the FSDP world size")
        config = self.args.i2i_sft_config
        if config.get("dp_tier_micro_balance", False) and actor.adapter.family != "sensenova":
            raise ValueError("DP tier micro balancing currently requires the SenseNova per-row tier policy")
        from relax.utils.profile_utils import TrainProfiler

        self.profiler = TrainProfiler(self.args)
        self.step = 0
        dataset_error = None
        try:
            self.dataset = ImageEditDataset(self.args, actor.adapter)
        except Exception as exc:
            dataset_error = f"{type(exc).__name__}: {exc}"
        errors = _collect_weight_sync_errors(dataset_error)
        if errors:
            raise ValueError(f"Image dataset initialization failed: {errors}")
        self.sampler = ImageBatchSampler(config["dataset_size"], self.args.global_batch_size, self.args.seed)
        self.parameters = [p for p in actor.model.parameters() if p.requires_grad]
        warmup = int(getattr(self.args, "lr_warmup_iters", 0) or 0)
        self.scheduler = (
            torch.optim.lr_scheduler.LinearLR(actor.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup)
            if warmup > 0
            else torch.optim.lr_scheduler.LambdaLR(actor.optimizer, lambda _: 1.0)
        )
        self.contract = {
            "schema_version": 1,
            "family": actor.adapter.family,
            "adapter_path": self.args.model_adapter_path,
            "model_path": os.path.realpath(self.args.model_path),
            "metadata_sha256": self.dataset.sha256,
            # CPU scheduling controls do not change samples or optimizer math.
            # Keep old checkpoints resumable when prefetch is toggled.
            "config": {k: v for k, v in config.items() if k not in ("image_prefetch_depth", "image_prefetch_workers")},
            "world_size": self.world,
            "global_batch_size": self.args.global_batch_size,
            "seed": self.args.seed,
            "sampler": (
                "global-padded-randperm+dp-tier-micro-rotate-v2"
                if config.get("dp_tier_rank_rotation", False)
                else "global-padded-randperm+dp-tier-micro-v1"
                if config.get("dp_tier_micro_balance", False)
                else "global-padded-randperm-v1"
            ),
            "trainable_mode": actor._trainable_mode,
            "lora": actor._lora_contract(),
            "optimizer": {k: actor.optimizer.defaults[k] for k in ("lr", "betas", "eps", "weight_decay", "foreach")},
            "warmup": warmup,
            "clip_grad": self.args.clip_grad,
            "param_dtype": self.args.fsdp_param_dtype,
            "reduce_dtype": self.args.fsdp_reduce_dtype,
            "master_dtype": self.args.fsdp_master_dtype,
            "torch_version": torch.__version__,
        }
        # JSON roundtrip canonicalizes optimizer tuples before disk comparison.
        import json

        self.contract = json.loads(json.dumps(self.contract))
        if any(contract != self.contract for contract in _collect_rank_objects(self.contract)):
            raise ValueError("Image SFT dataset/model/config differs between training ranks")
        actor.adapter.on_device_ready(actor.model, actor.device)
        if self.args.load:
            self._resume()
        else:
            if self.args.save and checkpoint.find_latest_committed(self.args.save, TASK) is not None:
                raise FileExistsError("Output already contains SFT checkpoints; use --load or a fresh --save")
            seed_training(self.args.seed + self.rank)
        self.prefetch = None
        depth = config.get("image_prefetch_depth", 0)
        workers = config.get("image_prefetch_workers", 2)
        if type(depth) is not int or depth < 0 or type(workers) is not int or workers <= 0:
            raise ValueError("Image prefetch depth/workers must be nonnegative/positive integers")
        if depth:
            if actor.adapter.family != "sensenova":
                raise ValueError("Threaded image prefetch currently requires the CPU-only SenseNova adapter")
            from relax.engine.sft.image_prefetch import ImagePrefetch

            self.prefetch = ImagePrefetch(
                self.dataset,
                self.sampler,
                self.rank,
                self.world,
                self.step,
                self.args.num_rollout,
                depth,
                workers,
            )
        self.profiler.on_init_end()

    def _resume(self) -> None:
        latest = checkpoint.find_latest_committed(self.args.load, TASK)
        if latest is None:
            raise FileNotFoundError(f"No committed image SFT checkpoint under {self.args.load}")
        if checkpoint.read_adapter_contract(self.args.load, TASK, latest) != self.contract:
            raise ValueError("Checkpoint model/data/optimizer contract differs; refusing SFT resume")
        state = checkpoint.load_checkpoint(
            self.args.load,
            TASK,
            latest,
            self.actor.model,
            self.actor.optimizer,
            adapter_only=self.actor._trainable_mode == "lora",
        )
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["completed_steps"])
        if self.step != latest:
            raise ValueError("Checkpoint iteration and completed SFT steps disagree")
        explicit = getattr(self.args, "_start_rollout_id_explicit", False)
        if explicit and self.args.start_rollout_id != self.step:
            raise ValueError("--start-rollout-id disagrees with the restored SFT cursor")
        restore_rng(Path(self.args.load) / TASK / checkpoint.iter_dir_name(latest), self.rank)

    def train(self, step: int) -> None:
        try:
            self._train_step(step)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.prefetch:
            self.prefetch.close()

    def _train_step(self, step: int) -> None:
        from relax.backends.fsdp.actor import _collect_weight_sync_errors

        if step != self.step:
            raise ValueError(f"Expected SFT step {self.step}, received {step}")
        holder: list[Any] = [None]

        def read_batch() -> None:
            rows = self.actor._read_train_partition(step)
            if rows is None:
                raise TimeoutError(f"No image SFT batch for step {step}")
            holder[0] = rows["sample_id"].tolist()
            if holder[0] != self.sampler.batch(step):
                raise ValueError("TransferQueue sample order differs from the deterministic SFT sampler")

        self.actor._rank0_then_agree(read_batch, what=f"read image SFT step {step}")
        from relax.utils.distributed_utils import get_gloo_group

        dist.broadcast_object_list(holder, src=0, group=get_gloo_group())
        ids = assign_image_batch(holder[0], self.args.i2i_sft_config, step=step, world=self.world)[
            self.rank :: self.world
        ]
        started = time.monotonic()
        optimizer = self.actor.optimizer
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.actor.device, dtype=torch.float32)
        data_wait = 0.0
        for micro, sample_id in enumerate(ids):
            data, error = None, None
            read_started = time.monotonic()
            try:
                data = self.prefetch.get(step, micro, sample_id) if self.prefetch else self.dataset[sample_id]
            except Exception as exc:
                error = f"sample {sample_id}: {type(exc).__name__}: {exc}"
            data_wait += time.monotonic() - read_started
            errors = _collect_weight_sync_errors(error)
            if errors:
                if self.prefetch:
                    self.prefetch.close()
                raise ValueError(f"Image decoding failed: {errors}")
            loss = self.actor.model(data)
            torch._assert_async(torch.isfinite(loss), "Non-finite SFT loss")
            (loss / len(ids)).backward()
            loss_sum.add_(loss.detach().float() / len(ids))
        norm = clip_sharded_gradients(self.parameters, self.args.clip_grad, self.group)
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        self.scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.step += 1
        if self.prefetch and self.step == self.args.num_rollout:
            self.prefetch.close()
        self.profiler.step(step)
        dist.all_reduce(loss_sum, group=self.group)
        peak_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(self.actor.device),
                torch.cuda.max_memory_reserved(self.actor.device),
                int(data_wait * 1e9),
            ],
            device=self.actor.device,
            dtype=torch.int64,
        )
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX, group=self.group)
        metrics = torch.stack([loss_sum / self.world, norm]).cpu().tolist()
        peak_allocated, peak_reserved, max_data_wait_ns = peak_memory.cpu().tolist()
        if self.rank == 0:
            values = {
                "train/loss": metrics[0],
                "train/grad_norm": metrics[1],
                "train/lr": lr,
                "perf/train_seconds": time.monotonic() - started,
                "perf/data_wait_seconds": max_data_wait_ns / 1e9,
                "perf/peak_allocated_bytes": peak_allocated,
                "perf/peak_reserved_bytes": peak_reserved,
            }
            values["train/step"] = self.step
            _report_sft_metrics(self.args, values)
            logger.info(
                "i2i SFT step=%s loss=%.8f grad_norm=%.8f lr=%g peak_allocated=%d peak_reserved=%d",
                self.step,
                *metrics,
                lr,
                peak_allocated,
                peak_reserved,
            )
        interval = self.args.save_interval
        if self.args.save and (self.step == self.args.num_rollout or (interval and self.step % interval == 0)):
            self.save()

    def save(self) -> None:
        try:
            self._save_checkpoint()
        except BaseException:
            self.close()
            raise

    def _save_checkpoint(self) -> None:
        checkpoint.save_checkpoint(
            self.args.save,
            TASK,
            self.step,
            self.actor.model,
            self.actor.optimizer,
            trainer_state={"completed_steps": self.step, "scheduler": self.scheduler.state_dict()},
            weight_sync_manifest={},
            adapter_contract=self.contract,
            adapter_only=self.actor._trainable_mode == "lora",
            is_rank0=self.rank == 0,
            require_rng_state=True,
        )
        from relax.utils.rotate_ckpt import rotate_ckpt

        self.actor._rank0_then_agree(
            lambda: rotate_ckpt(
                self.args, global_step=self.step, save_dir=os.path.join(self.args.save, TASK), committed_only=True
            ),
            what=f"SFT checkpoint rotation at step {self.step}",
        )
