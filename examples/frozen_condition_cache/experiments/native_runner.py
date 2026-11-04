"""Real adapter + native FSDPSFTTrainer acceptance run, without Ray deployment.

Run under torchrun. All dataset preparation, loss, backward accumulation, FSDP
reduction, clipping, AdamW, scheduler and metric synchronization are the
upstream FSDPSFTTrainer path. Only Ray/TQ sample-ID delivery and metric
transport are replaced locally; results explicitly exclude those orchestration
costs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.copy_audit import (
    LogicalCopyAudit,
    condition_copy_phase,
    suspend_logical_copy_audit,
)
from experiments.evidence import (
    atomic_json,
    capture_rng,
    cpu_clone,
    local_tensor,
    payload_digest,
    restore_rng,
)
from experiments.resources import cache_storage_contract, storage_snapshot


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("correctness", "performance"), required=True)
    result.add_argument("--trajectory", choices=("a1", "a2", "bcold", "bwarm"))
    for key in (
        "adapter",
        "model-path",
        "metadata",
        "data-root",
        "run-dir",
        "initial-state-dir",
    ):
        result.add_argument(f"--{key}", required=True)
    result.add_argument(
        "--cache-bytes",
        type=int,
        default=0,
        help="Per-rank strict capacity; A must be zero",
    )
    result.add_argument("--cache-device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--cache-identity", required=True)
    result.add_argument("--world", type=int, choices=(1, 2, 4), required=True)
    result.add_argument("--global-batch", type=int, choices=(8,), default=8)
    result.add_argument("--lora-rank", type=int, choices=(32,), default=32)
    result.add_argument("--seed", type=int, choices=(20261030,), default=20261030)
    result.add_argument("--max-pixels", type=int, default=1048576)
    result.add_argument("--attention-implementation", choices=("torch",), default="torch")
    result.add_argument(
        "--image-prefetch-depth",
        type=int,
        choices=(0, 2),
        default=None,
        help="Default SenseNova=2, other adapters=0; fixed equally for A/B",
    )
    result.add_argument(
        "--dataset-size",
        type=int,
        required=True,
        help="Must equal the pinned metadata row count",
    )
    result.add_argument("--warmup", type=int, choices=(16,), default=16)
    result.add_argument("--measure", type=int, choices=(64,), default=64)
    return result


class CacheAudit:
    """Numerical-only bounded metadata audit of cache put -> get exact
    bytes."""

    def __init__(self, cache: Any) -> None:
        self.cache, self.expected = cache, {}
        self.checked_hits = self.failed = self.writes = 0
        self.get, self.put = cache.get, cache.put

        def put(key: Any, payload: Any) -> bool:
            with suspend_logical_copy_audit():
                digest = payload_digest(payload)
            accepted = self.put(key, payload)
            if accepted:
                self.writes += 1
                self.expected[key] = digest
            present = set(cache.keys())
            self.expected = {k: v for k, v in self.expected.items() if k in present}
            return accepted

        def get(key: Any, device: Any = None) -> Any:
            value = self.get(key, device=device)
            if value is not None:
                if key not in self.expected:
                    raise AssertionError("Cache hit has no audited write in this trajectory")
                self.checked_hits += 1
                with suspend_logical_copy_audit():
                    digest = payload_digest(value)
                if digest != self.expected[key]:
                    self.failed += 1
                    raise AssertionError("Cache payload dtype/shape/bytes changed")
            return value

        cache.put, cache.get = put, get

    def result(self) -> dict[str, int]:
        return {
            "writes": self.writes,
            "checked_hits": self.checked_hits,
            "failed": self.failed,
            "bounded_audit_entries": len(self.expected),
        }


class ConditionPhaseAudit:
    """Bounded correctness-only host/CUDA event spans around existing calls.

    CUDA durations are current-stream diagnostic spans, including launch gaps
    and observer overhead, not isolated kernel timings or performance results.
    Reading them never adds a synchronization: execute() already synchronizes.
    """

    def __init__(self, *, cuda: bool = False, max_samples_per_phase: int = 32) -> None:
        if max_samples_per_phase < 0:
            raise ValueError("Phase sample limit must be nonnegative")
        self.cuda, self.limit = cuda, max_samples_per_phase
        self.region = "training"
        self.rows: dict[str, dict[str, Any]] = {}
        self.restores = []

    def wrap(self, owner: Any, name: str, phase: str, eligible: Any = None) -> None:
        original = getattr(owner, name)
        had_instance_value = name in vars(owner)

        def observed(*args: Any, **kwargs: Any) -> Any:
            if eligible is not None and not eligible(*args, **kwargs):
                return original(*args, **kwargs)
            key = self.region + "/" + phase
            row = self.rows.setdefault(
                key,
                dict(
                    calls=0,
                    sampled_calls=0,
                    failed_calls=0,
                    host_seconds=0.0,
                    events=[],
                    timing_errors=[],
                ),
            )
            row["calls"] += 1
            if row["sampled_calls"] >= self.limit:
                with condition_copy_phase(phase):
                    return original(*args, **kwargs)
            row["sampled_calls"] += 1
            start_event = end_event = None
            if self.cuda:
                try:
                    start_event, end_event = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start_event.record()
                except RuntimeError as error:
                    row["timing_errors"].append(repr(error))
                    start_event = end_event = None
            started = time.perf_counter()
            try:
                with condition_copy_phase(phase):
                    return original(*args, **kwargs)
            except BaseException:
                row["failed_calls"] += 1
                raise
            finally:
                row["host_seconds"] += time.perf_counter() - started
                if end_event is not None:
                    try:
                        end_event.record()
                        row["events"].append((start_event, end_event))
                    except RuntimeError as error:
                        row["timing_errors"].append(repr(error))

        setattr(owner, name, observed)
        self.restores.append((owner, name, original, had_instance_value))

    def install(self, model: Any, family: str, cache: Any) -> None:
        if family == "sensenova":
            self.wrap(
                model.pipe,
                "unit_runner",
                "frozen_encoder",
                lambda unit, *_args, **_kwargs: hasattr(unit, "frozen_condition_schema"),
            )
        elif family == "qwen_image_edit":
            self.wrap(model.pipe, "encode_prompt", "frozen_encoder")
        else:
            raise ValueError(f"Unregistered condition timing adapter: {family}")
        if cache is not None:
            self.wrap(cache, "get", "cache_read")
            self.wrap(cache, "put", "cache_write")

    def result(self) -> dict[str, Any]:
        phases = {}
        for key, row in self.rows.items():
            seconds = []
            errors = list(row["timing_errors"])
            for start, end in row["events"]:
                try:
                    # Never wait for an event or inject a new synchronize here.
                    if end.query():
                        seconds.append(start.elapsed_time(end) / 1000)
                    else:
                        errors.append("event incomplete at collection")
                except RuntimeError as error:
                    errors.append(repr(error))
            phases[key] = {k: row[k] for k in ("calls", "sampled_calls", "failed_calls", "host_seconds")} | {
                "cuda_sample_seconds": seconds,
                "cuda_seconds": sum(seconds),
                "timing_errors": errors,
            }
        return {
            "scope": "existing frozen encoder/cache calls in correctness trajectories only; prefill and training separated",
            "max_samples_per_phase_per_region": self.limit,
            "phases": phases,
            "host_timing_scope": "wrapper wall time; includes diagnostic CPU source hashing when active",
            "cuda_timing_scope": "current-stream event spans including launch gaps/observer overhead; not pure kernel time",
            "not_installed_in_performance": True,
            "added_model_forwards": 0,
            "extra_cuda_synchronizations": 0,
            "cuda_timing_requested": self.cuda,
            "sample_window_complete": all(row["calls"] == row["sampled_calls"] for row in self.rows.values()),
            "cuda_event_samples_complete": (
                all(
                    len(row["cuda_sample_seconds"]) == row["sampled_calls"] and not row["timing_errors"]
                    for row in phases.values()
                )
                if self.cuda
                else None
            ),
        }

    def close(self) -> None:
        for owner, name, original, had_instance_value in reversed(self.restores):
            if had_instance_value:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.restores.clear()


def resource_snapshot(device: torch.device, run_dir: Path | None = None) -> dict[str, Any]:
    memory = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _ = line.split()
                memory[key.rstrip(":")] = int(value) * 1024
    except FileNotFoundError:
        pass
    result = {
        **memory,
        "cpu_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_allocated": torch.cuda.memory_allocated(device),
        "cuda_reserved": torch.cuda.memory_reserved(device),
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved": torch.cuda.max_memory_reserved(device),
    }
    if run_dir is not None:
        result["storage"] = storage_snapshot(run_dir)
    return result


class LocalActor:
    """Only the actor methods actually called by the existing SFT trainer."""

    def __init__(
        self,
        args: Any,
        adapter: Any,
        model: Any,
        optimizer: Any,
        group: Any,
        control: Any,
        device: torch.device | None = None,
    ) -> None:
        self.args, self.adapter, self.model, self.optimizer = (
            args,
            adapter,
            model,
            optimizer,
        )
        self.device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        self.group, self.control = group, control
        self._trainable_mode = "lora"
        self.trainer = None

    def _device_group(self) -> Any:
        return self.group

    def _lora_contract(self) -> dict[str, Any]:
        from relax.backends.fsdp.lora import lora_metadata_dict

        return lora_metadata_dict(
            rank=32,
            alpha=32,
            dropout=0.0,
            target_modules=list(self.adapter.lora_target_modules),
            task_type=self.adapter.lora_task_type,
        )

    def _read_train_partition(self, step: int) -> dict[str, torch.Tensor]:
        return {"sample_id": torch.tensor(self.trainer.sampler.batch(step), dtype=torch.int64)}

    def _rank0_then_agree(self, action: Any, *, what: str) -> None:
        error = None
        if dist.get_rank(self.control) == 0:
            try:
                action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        dist.broadcast_object_list(errors, src=0, group=self.control)
        if errors[0]:
            raise RuntimeError(f"{what}: {errors[0]}")


def native_args(options: argparse.Namespace, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        global_batch_size=8,
        seed=options.seed,
        model_adapter_path=options.adapter,
        model_path=options.model_path,
        prompt_data=options.metadata,
        fsdp_trainable_mode="lora",
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        lr=1e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        lr_warmup_iters=0,
        clip_grad=1.0,
        fsdp_param_dtype="bf16",
        fsdp_reduce_dtype="fp32",
        fsdp_master_dtype="fp32",
        save=None,
        load=None,
        save_interval=0,
        num_rollout=1 if options.mode == "correctness" else options.warmup + options.measure,
        use_pytorch_profiler=False,
        record_memory_history=False,
        i2i_sft_config={
            "data_root": options.data_root,
            "dataset_size": size,
            "max_pixels": options.max_pixels,
            "conditioning_images": 1,
            "attention_implementation": getattr(options, "attention_implementation", "torch"),
            "deterministic_attention": True,
            "prefix_gradient_checkpointing": True,
            "image_prefetch_depth": getattr(options, "image_prefetch_depth", None) or 0,
            "image_prefetch_workers": 2,
        },
    )


def initial_state(model: Any, optimizer: Any, trainer: Any, contract: dict[str, Any]) -> dict[str, Any]:
    if optimizer.state:
        raise ValueError("Initial-state creation must precede the first optimizer update")
    return {
        "parameters": {n: cpu_clone(p) for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "rng": capture_rng(),
        "contract": contract,
    }


def restore_initial(
    state: dict[str, Any],
    model: Any,
    optimizer: Any,
    trainer: Any,
    contract: dict[str, Any],
) -> None:
    if state["contract"] != contract:
        raise ValueError("Initial state model/optimizer/world/precision contract differs")
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state["parameters"]):
        raise ValueError("Trainable parameter names differ from initial state")
    with torch.no_grad():
        for name, parameter in parameters.items():
            target = local_tensor(parameter)
            if target.dtype != torch.float32 or target.shape != state["parameters"][name].shape:
                raise ValueError(f"FP32 master or shard shape differs: {name}")
            target.copy_(state["parameters"][name].to(target.device))
    optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.step = 0
    restore_rng(state["rng"])


def execute(options: argparse.Namespace) -> None:
    from relax.backends.fsdp import checkpoint, sft
    from relax.backends.fsdp.lora import inject_lora_adapter
    from relax.backends.fsdp.runtime import fsdp2_wrap
    from relax.utils.distributed_utils import init_gloo_group

    if not torch.cuda.is_available():
        raise RuntimeError("Real adapter runner requires CUDA; CPU tests are in tests/experiments")
    world, rank, local_rank = (
        int(os.environ.get(k, default)) for k, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world != options.world or 8 % world:
        raise ValueError("torchrun WORLD_SIZE must match --world and divide global batch 8")
    torch.cuda.set_device(local_rank)
    # A seed does not control fused CUDA backward reduction ordering. Apply
    # the same strict policy to both models and to online/cached/performance
    # paths; unsupported nondeterministic operations must fail visibly.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    group, control = dist.group.WORLD, init_gloo_group()
    directory = Path(options.run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    os.environ["RELAX_CONDITION_CACHE_BYTES"] = str(options.cache_bytes)
    os.environ["RELAX_CONDITION_CACHE_DEVICE"] = options.cache_device
    os.environ["RELAX_CONDITION_CACHE_IDENTITY"] = options.cache_identity
    module, name = options.adapter.rsplit(".", 1)
    adapter = getattr(importlib.import_module(module), name)()
    if options.image_prefetch_depth is None:
        options.image_prefetch_depth = 2 if adapter.family == "sensenova" else 0
    rows = json.loads(Path(options.metadata).read_text())
    if len(rows) != options.dataset_size:
        raise ValueError("Metadata row count differs from --dataset-size")
    args = native_args(options, len(rows))
    model = None
    trainer = None
    condition_phases = None
    stage_start_resources = resource_snapshot(torch.device("cuda", local_rank), directory)
    try:
        sft.seed_training(options.seed)
        # Avoid simultaneous rank-local CPU copies of 20B-class model weights.
        for loading_rank in range(world):
            if rank == loading_rank:
                model = adapter.load_train_model(args)
                inject_lora_adapter(
                    model,
                    rank=32,
                    alpha=32,
                    target_modules=adapter.lora_target_modules,
                    dropout=0.0,
                    task_type=adapter.lora_task_type,
                )
                adapter.prepare_for_sharding(model)
                model.to(torch.device("cuda", local_rank))
            dist.barrier(group=group)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (world,))
        model = fsdp2_wrap(
            model,
            device_mesh=mesh,
            param_dtype="bf16",
            reduce_dtype="fp32",
            master_dtype="fp32",
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        actor = LocalActor(args, adapter, model, optimizer, group, control)
        trainer = sft.FSDPSFTTrainer(actor)
        actor.trainer = trainer
        contract = {
            "adapter": options.adapter,
            "base_identity": options.cache_identity,
            "world": world,
            "global_batch": 8,
            "lora_rank": 32,
            "seed": options.seed,
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "master_dtype": "fp32",
            "optimizer": {k: v for k, v in optimizer.defaults.items() if k != "params"},
            "lr_warmup_iters": 0,
            "attention_implementation": args.i2i_sft_config["attention_implementation"],
        }
        state_path = Path(options.initial_state_dir) / f"initial-rank{rank:04d}.pt"
        initial_reused = state_path.exists()
        if options.mode == "correctness" and options.trajectory == "a1":
            state_path.parent.mkdir(parents=True, exist_ok=True)
            if state_path.exists():
                # A bounded same-config retry reuses the first fixed state;
                # it must never silently select a new initialization.
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            else:
                state = initial_state(model, optimizer, trainer, contract)
                temporary = state_path.with_suffix(".partial")
                torch.save(state, temporary)
                temporary.replace(state_path)
        else:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        restore_initial(state, model, optimizer, trainer, contract)
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not parameters or any(local_tensor(p).dtype != torch.float32 for p in parameters.values()):
            raise AssertionError("All trainable optimizer masters must be FP32")
        model.train()
        cache = getattr(model, "condition_cache", None)
        if options.mode == "correctness":
            condition_phases = ConditionPhaseAudit(cuda=True)
            condition_phases.install(model, adapter.family, cache)
        audit = CacheAudit(cache) if cache is not None and options.mode == "correctness" else None
        if options.cache_bytes > 0 and cache is None:
            raise AssertionError("Requested cache is not exposed by the model")
        if options.mode == "correctness" and options.trajectory == "bwarm":
            condition_phases.region = "prefill"
            ids = trainer.sampler.batch(0)[rank::world]
            with torch.no_grad():
                for index in ids:
                    model(trainer.dataset[index])
            # Cache warm-up never advances training data/noise/optimizer state.
            restore_initial(state, model, optimizer, trainer, contract)
            condition_phases.region = "training"
        rng_before = payload_digest(capture_rng())
        cache_before_training = model.condition_cache_stats()
        metrics: list[dict[str, Any]] = []
        original_report = sft._report_sft_metrics
        sft._report_sft_metrics = lambda _args, values: metrics.append(dict(values))
        gradients = {}
        original_clip = sft.clip_sharded_gradients

        def capture_clip(params: Any, maximum: float, process_group: Any) -> torch.Tensor:
            # Evidence extraction is not a copy performed by training/cache.
            with suspend_logical_copy_audit():
                gradients.update({n: None if p.grad is None else cpu_clone(p.grad) for n, p in parameters.items()})
            return original_clip(params, maximum, process_group)

        if options.mode == "correctness":
            sft.clip_sharded_gradients = capture_clip
        snapshots = []
        logical_copy_audit = None
        warmup_seconds = 0.0
        torch.cuda.synchronize()
        dist.barrier(group=group)
        section_started = time.monotonic()
        try:
            for step in range(args.num_rollout):
                if options.mode == "performance" and step == options.warmup:
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    warmup_seconds = time.monotonic() - section_started
                    section_started = time.monotonic()
                update_started = time.monotonic()
                if options.mode == "correctness":
                    # One already-required full update; no extra forward.
                    with LogicalCopyAudit(collect_repeated_transfers=True) as copy_audit:
                        trainer.train(step)
                    logical_copy_audit = copy_audit.result()
                else:
                    trainer.train(step)
                cache_stats = model.condition_cache_stats()
                if cache_stats.get("bytes", 0) > options.cache_bytes:
                    raise AssertionError("Condition cache exceeded its registered per-rank byte budget")
                snapshots.append(
                    {
                        "completed_step": trainer.step,
                        "resources": resource_snapshot(actor.device),
                        "cache": cache_stats,
                        "wall_seconds": time.monotonic() - update_started,
                    }
                )
            torch.cuda.synchronize()
            dist.barrier(group=group)
            elapsed = time.monotonic() - section_started
        finally:
            sft._report_sft_metrics = original_report
            sft.clip_sharded_gradients = original_clip
        metrics_holder = [metrics if rank == 0 else None]
        dist.broadcast_object_list(metrics_holder, src=0, group=control)
        metrics = metrics_holder[0]
        timing = torch.tensor([elapsed, warmup_seconds], dtype=torch.float64, device=actor.device)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX, group=group)
        elapsed, warmup_seconds = timing.cpu().tolist()
        resource_summary = {
            "rank": rank,
            "peak_process_rss": max(row["resources"].get("VmRSS", 0) for row in snapshots),
            "peak_cuda_allocated": max(row["resources"]["cuda_peak_allocated"] for row in snapshots),
            "peak_cuda_reserved": max(row["resources"]["cuda_peak_reserved"] for row in snapshots),
            "peak_cache_bytes": max(row["cache"].get("bytes", 0) for row in snapshots),
            "final_cache": snapshots[-1]["cache"],
        }
        all_rank_resources = [None] * world
        if not all(
            torch.isfinite(torch.tensor(row["train/loss"])) and torch.isfinite(torch.tensor(row["train/grad_norm"]))
            for row in metrics
        ):
            raise AssertionError("Nonfinite native training loss/gradient norm")
        manifest = {
            "schema": 1,
            "status": "completed",
            "passed": True,
            "stagepass": True,
            "mode": options.mode,
            "rank": rank,
            "scope": "real native FSDPSFTTrainer; Ray/TQ and checkpoint I/O excluded from measured time",
            "contract": contract,
            "native_contract": trainer.contract,
            "numerical_execution_policy": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "torch_version": torch.__version__,
            },
            "initial_reused": initial_reused,
            "metadata_sha256": trainer.dataset.sha256,
            "cache_budget_per_rank": options.cache_bytes,
            "cache_device": options.cache_device,
            "condition_cache_storage": cache_storage_contract(cache is not None, options.cache_device),
            "cache_before_training": cache_before_training,
            "warmup_updates": options.warmup if options.mode == "performance" else 0,
            "measured_updates": options.measure if options.mode == "performance" else 1,
            "warmup_seconds": warmup_seconds,
            "measured_seconds": elapsed,
            "initialization_plus_total_seconds": time.monotonic() - started,
            "samples_per_second": 8 * options.measure / elapsed if options.mode == "performance" else None,
            "metrics": metrics,
            "all_rank_resource_summary": all_rank_resources,
            "resources_and_cache": snapshots,
            "sample_id_occurrences": [trainer.sampler.batch(step) for step in range(args.num_rollout)],
            "checkpoint": str(directory / "final" / "i2i_sft" / checkpoint.iter_dir_name(trainer.step)),
            "cache_audit": audit.result() if audit else {"writes": 0, "checked_hits": 0, "failed": 0},
            "logical_copy_audit": logical_copy_audit,
            "condition_phase_diagnostics": condition_phases.result() if condition_phases is not None else None,
        }
        if options.mode == "correctness":
            evidence = {
                "rank": rank,
                "loss": metrics[0]["train/loss"],
                "gradients": gradients,
                "delta": {n: cpu_clone(p) - state["parameters"][n] for n, p in parameters.items()},
                "initial_parameters_sha256": payload_digest(state["parameters"]),
                "optimizer_initial_sha256": payload_digest(state["optimizer"]),
                "rng_before_sha256": rng_before,
                "rng_after_sha256": payload_digest(capture_rng()),
                "sample_ids": trainer.sampler.batch(0),
                "world": world,
                "parameter_global_shapes": {n: list(p.shape) for n, p in parameters.items()},
                "cache_audit": manifest["cache_audit"],
            }
            torch.save(evidence, directory / f"trajectory-rank{rank:04d}.pt")
        # Native DCP, adapter-only model + complete Adam/scheduler/rank RNG.
        checkpoint.save_checkpoint(
            str(directory / "final"),
            "i2i_sft",
            trainer.step,
            model,
            optimizer,
            trainer_state={
                "completed_steps": trainer.step,
                "scheduler": trainer.scheduler.state_dict(),
            },
            weight_sync_manifest={},
            adapter_contract=trainer.contract,
            adapter_only=True,
            is_rank0=rank == 0,
            require_rng_state=True,
        )
        resource_summary["stage_boundary_resources"] = {
            "before_model_load": stage_start_resources,
            "after_checkpoint_before_result": resource_snapshot(actor.device, directory),
        }
        dist.all_gather_object(all_rank_resources, resource_summary, group=control)
        atomic_json(directory / f"result-rank{rank:04d}.json", manifest)
        if rank == 0:
            atomic_json(directory / "result.json", manifest)
    except BaseException as exc:
        atomic_json(
            directory / f"failure-rank{rank:04d}.json",
            {
                "status": "failed",
                "exception": repr(exc),
                "traceback": traceback.format_exc(),
                "stage_start_resources": stage_start_resources,
                "failure_storage": storage_snapshot(directory),
                "condition_phase_diagnostics": condition_phases.result() if condition_phases is not None else None,
            },
        )
        raise
    finally:
        if condition_phases is not None:
            condition_phases.close()
        if trainer is not None:
            trainer.close()
        if model is not None and hasattr(model, "close_condition_cache"):
            model.close_condition_cache()
            after = model.condition_cache_stats()
            atomic_json(
                directory / f"lifecycle-rank{rank:04d}.json",
                {
                    "after_close": after,
                    "released": after.get("bytes", 0) == 0 and after.get("entries", 0) == 0,
                },
            )
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    options = parser().parse_args()
    if options.mode == "correctness":
        if options.trajectory is None:
            raise ValueError("Correctness requires one fixed trajectory name")
        if (options.trajectory.startswith("a") and options.cache_bytes != 0) or (
            options.trajectory.startswith("b") and options.cache_bytes <= 0
        ):
            raise ValueError("A must be online; B must use a positive cache budget")
    if options.cache_bytes < 0 or options.warmup < 0 or options.measure <= 0:
        raise ValueError("Invalid bounded cache or step counts")
    execute(options)


if __name__ == "__main__":
    main()
