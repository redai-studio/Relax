# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproduce Relax's Megatron Bridge HF-to-Megatron weight load path with
torchrun.

This intentionally follows the train actor initialization path up to:
initialize_model_and_optimizer -> load_checkpoint -> bridge.load_hf_weights.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import relax.utils.training.eval_config
from relax.backends.megatron.initialize import init as init_megatron
from relax.backends.megatron.model import initialize_model_and_optimizer
from relax.utils import device as device_utils
from relax.utils.arguments import parse_args
from relax.utils.checkpoint_write_patch import patch_checkpoint_write
from relax.utils.device import device_module
from relax.utils.distributed_utils import get_gloo_group, init_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.memory_utils import clear_memory, print_memory
from relax.utils.reloadable_process_group import monkey_patch_torch_dist
from relax.utils.utils import process_args


logger = get_logger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "off", "")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_cuda_memory_snapshot(rank_dir: Path, name: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    _write_text(rank_dir / f"{name}_memory_summary.txt", torch.cuda.memory_summary())


def _export_profile_artifact(rank_dir: Path, name: str, export_fn) -> None:
    try:
        export_fn()
    except Exception as exc:  # noqa: BLE001
        _write_text(rank_dir / f"{name}.error.txt", repr(exc))


def _install_bridge_progress_profiler(args, role: str) -> None:
    """Monkey-patch ``MegatronModelBridge._with_progress_tracking`` so that the
    first weight-conversion loop profiles a fixed window of tasks and dumps
    artifacts as soon as the window finishes, instead of waiting for the entire
    script to exit."""
    if not _env_flag("RELAX_REPRO_PROFILE"):
        return

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    num_steps = int(os.environ.get("RELAX_REPRO_PROFILE_STEPS", "50"))
    warmup = int(os.environ.get("RELAX_REPRO_PROFILE_WARMUP", "5"))
    exit_after_dump = _env_flag("RELAX_REPRO_PROFILE_EXIT_AFTER_DUMP")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    profile_root = Path(
        os.environ.get(
            "RELAX_REPRO_PROFILE_DIR",
            f"log/megatron_bridge_profile/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
    )
    rank_dir = profile_root / f"rank_{args.rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    original = MegatronModelBridge._with_progress_tracking
    state = {"profiled": False}

    def patched(self, tasks, description, show_progress=True):
        if state["profiled"]:
            yield from original(self, tasks, description, show_progress)
            return
        state["profiled"] = True

        total = len(tasks)
        is_rank0 = args.rank == 0

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"

        if is_rank0:
            logger.info(
                f"[bridge-profile] '{description}' total={total}, warmup={warmup}, "
                f"active={num_steps} — artifacts dir: {profile_root}"
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _write_cuda_memory_snapshot(rank_dir, "before")

        it = iter(tasks)
        emitted = 0

        for _ in range(warmup):
            try:
                task = next(it)
            except StopIteration:
                break
            yield task
            emitted += 1

        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
        prof.start()
        step_start = time.perf_counter()
        try:
            for i in range(num_steps):
                try:
                    task = next(it)
                except StopIteration:
                    break
                yield task
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.perf_counter()
                if is_rank0:
                    logger.info(
                        f"[bridge-profile] step {i + 1}/{num_steps} "
                        f"(global {emitted + 1}/{total}) elapsed={(now - step_start) * 1000:.1f}ms"
                    )
                step_start = now
                emitted += 1
        finally:
            prof.stop()

        _write_cuda_memory_snapshot(rank_dir, "after")
        _export_profile_artifact(
            rank_dir,
            "trace_gzip",
            lambda: torch.profiler.tensorboard_trace_handler(
                str(rank_dir), worker_name=f"rank_{args.rank}", use_gzip=True
            )(prof),
        )
        _export_profile_artifact(
            rank_dir,
            "operator_table",
            lambda: _write_text(
                rank_dir / "operator_table.txt",
                prof.key_averages(group_by_stack_n=10).table(sort_by=sort_by, row_limit=-1),
            ),
        )
        _export_profile_artifact(
            rank_dir,
            "stacks",
            lambda: prof.export_stacks(str(rank_dir / "stacks.txt"), metric=sort_by),
        )
        _write_text(
            rank_dir / "metadata.txt",
            "\n".join(
                [
                    f"rank={args.rank}",
                    f"local_rank={local_rank}",
                    f"world_size={args.world_size}",
                    f"role={role}",
                    f"description={description}",
                    f"total_tasks={total}",
                    f"warmup_steps={warmup}",
                    f"active_steps={num_steps}",
                    f"tp={args.tensor_model_parallel_size}",
                    f"pp={args.pipeline_model_parallel_size}",
                    f"ep={args.expert_model_parallel_size}",
                    f"load={args.load}",
                    f"hf_checkpoint={args.hf_checkpoint}",
                ]
            )
            + "\n",
        )
        if is_rank0:
            logger.info(f"[bridge-profile] dumped {num_steps}-step profile to {profile_root}")

        if exit_after_dump:
            if is_rank0:
                logger.info("[bridge-profile] RELAX_REPRO_PROFILE_EXIT_AFTER_DUMP=1 — exiting now")
            os._exit(0)

        for task in it:
            yield task

    MegatronModelBridge._with_progress_tracking = patched


def _init_distributed(args) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_module.set_device(f"{device_utils.get_device_name()}:{local_rank}")

    dist.init_process_group(
        backend=args.distributed_backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    numa_local_rank = int(os.environ.get("RANK", args.rank)) % args.num_gpus_per_node
    device_utils.set_numa_affinity(numa_local_rank)


def main() -> None:
    args = parse_args()
    role = os.environ.get("RELAX_REPRO_ROLE")
    if role is None:
        role = "reference" if args.only_load_weight else "actor"

    if role in ("reference", "actor_fwd"):
        if args.ref_actor_config is None:
            args.ref_actor_config = {}
        process_args(args, role)

    torch.serialization.add_safe_globals([relax.utils.training.eval_config.EvalDatasetConfig])

    monkey_patch_torch_dist(args)
    patch_checkpoint_write()
    _init_distributed(args)

    if args.rank == 0:
        logger.info(
            "Starting Megatron Bridge load reproduction "
            f"(role={role}, world_size={args.world_size}, tp={args.tensor_model_parallel_size}, "
            f"pp={args.pipeline_model_parallel_size}, ep={args.expert_model_parallel_size}, "
            f"load={args.load}, hf_checkpoint={args.hf_checkpoint}, only_load_weight={args.only_load_weight})"
        )

    _install_bridge_progress_profiler(args, role)

    init_megatron(args)
    dist.barrier(group=get_gloo_group())

    print_memory("before initialize_model_and_optimizer")
    start = time.perf_counter()
    model, optimizer, opt_param_scheduler, iteration = initialize_model_and_optimizer(args, role)
    dist.barrier(group=get_gloo_group())
    elapsed = time.perf_counter() - start

    if args.rank == 0:
        logger.info(
            "Finished initialize_model_and_optimizer "
            f"(iteration={iteration}, elapsed_seconds={elapsed:.2f}, "
            f"model_chunks={len(model)}, optimizer_loaded={optimizer is not None}, "
            f"scheduler_loaded={opt_param_scheduler is not None})"
        )

    del model, optimizer, opt_param_scheduler
    clear_memory()
    dist.barrier(group=get_gloo_group())


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
