# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in rank-local training evidence, without synchronizing CUDA or ranks."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_current: ContextVar[dict[str, Any] | None] = ContextVar("training_perf_trace", default=None)


def active() -> bool:
    return _current.get() is not None


def profiler_metadata() -> dict[str, Any]:
    state = _current.get()
    if state is None:
        return {}
    return {"rollout_id_at_export": state["rollout_id"], "topology": state["topology"]}


def _memory() -> dict[str, int]:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "step_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "step_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _topology(args: Any) -> dict[str, Any]:
    import torch.distributed as dist
    from megatron.core import mpu

    groups = []
    for label, getter in (
        ("TP", mpu.get_tensor_model_parallel_group),
        ("DP", lambda: mpu.get_data_parallel_group(with_context_parallel=False)),
        ("DP_CP", lambda: mpu.get_data_parallel_group(with_context_parallel=True)),
        ("PP", mpu.get_pipeline_model_parallel_group),
        ("CP", mpu.get_context_parallel_group),
        ("MODEL", mpu.get_model_parallel_group),
    ):
        group = getter()
        if group is None:
            continue
        # Offload replaces the underlying NCCL group. Resolve it anew each time.
        group = getattr(group, "group", group)
        groups.append(
            {
                "label": label,
                "name": str(getattr(group, "group_name", "")),
                "description": str(getattr(group, "group_desc", "")),
                "ranks": dist.get_process_group_ranks(group),
            }
        )
    return {
        "rank": dist.get_rank(group=dist.group.WORLD),
        "dp": mpu.get_data_parallel_rank(with_context_parallel=False),
        "tp": mpu.get_tensor_model_parallel_rank(),
        "pp": mpu.get_pipeline_model_parallel_rank(),
        "cp": mpu.get_context_parallel_rank(),
        "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        "tp_size": mpu.get_tensor_model_parallel_world_size(),
        "pp_size": mpu.get_pipeline_model_parallel_world_size(),
        "cp_size": mpu.get_context_parallel_world_size(),
        "sequence_parallel": bool(args.sequence_parallel),
        "groups": groups,
    }


def trace_training_step(func: Callable) -> Callable:
    """Wrap one actor RPC; disabled path does not query CUDA or distributed
    state."""

    @wraps(func)
    def wrapped(self: Any, rollout_id: int, *args: Any, **kwargs: Any) -> Any:
        directory = Envs.RELAX_TRAIN_PERF_DIR
        if not directory:
            return func(self, rollout_id, *args, **kwargs)
        import torch

        state = {
            "schema": 1,
            "rollout_id": rollout_id,
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "role": getattr(self, "role", "actor"),
            "start_unix_s": time.time(),
            # Colocated workers may have destroyed model groups until wake_up.
            "topology": {"rank": self._rank},
            "_args": self.args,
            "configuration": {
                name: getattr(self.args, name, None)
                for name in (
                    "global_batch_size",
                    "rollout_batch_size",
                    "n_samples_per_prompt",
                    "max_tokens_per_gpu",
                    "log_probs_max_tokens_per_gpu",
                    "micro_batch_size",
                    "use_dynamic_batch_size",
                    "freeze_vision_model",
                    "recompute_granularity",
                )
            },
            "phases": [],
            "workloads": [],
        }
        # These are allocator counters, not CUDA synchronization. Reset only in
        # this explicitly enabled diagnostic mode; do not reset per phase.
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            state["gpu_uuid"] = str(getattr(props, "uuid", ""))
            torch.cuda.reset_peak_memory_stats()
        state["memory_start"] = _memory()
        token = _current.set(state)
        start = time.perf_counter()
        try:
            with phase("train_rpc"):
                result = func(self, rollout_id, *args, **kwargs)
            state["status"] = "ok"
            return result
        except BaseException:
            state["status"] = "error"
            raise
        finally:
            state["cpu_wall_s"] = time.perf_counter() - start
            state["memory_end"] = _memory()
            _current.reset(token)
            state.pop("_args")
            try:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                rank = state["topology"]["rank"]
                with (path / f"{state['role']}_rank{rank}_pid{os.getpid()}.jsonl").open("a") as handle:
                    handle.write(json.dumps(state) + "\n")
            except OSError:
                logger.exception("Cannot write rank-local training performance evidence")

    return wrapped


@contextmanager
def phase(name: str) -> Iterator[None]:
    """CPU wall range plus profiler annotation; GPU work remains
    asynchronous."""
    state = _current.get()
    if state is None:
        yield
        return
    import torch

    start = time.perf_counter()
    row = {"name": name, "start_unix_s": time.time(), "memory_start": _memory()}
    try:
        with torch.profiler.record_function(f"relax.phase/{name}"):
            yield
    finally:
        row["cpu_wall_s"] = time.perf_counter() - start
        row["memory_end"] = _memory()
        state["phases"].append(row)


def record_workload(rollout_data: dict, iterators: list, num_microbatches: list[int], budget: int | None) -> None:
    """Read scheduling metadata and tensor shapes only; never consume
    iterators."""
    state = _current.get()
    if state is None:
        return
    state["topology"] = _topology(state["_args"])
    lengths = rollout_data["total_lengths"]
    iterator = iterators[0]  # VPP copies the same local schedule.
    indices = iterator.micro_batch_indices
    if indices is None:
        size = iterator.micro_batch_size
        indices = [list(range(i, min(i + size, len(lengths)))) for i in range(0, len(lengths), size)]
    dummy = iterator.dummy_micro_batch_offsets
    packs = [sum(lengths[i] for i in pack) for pack in indices]
    images, patches = 0, 0
    for mm in rollout_data.get("multimodal_train_inputs", []) or []:
        if not mm:
            continue
        grid = mm.get("image_grid_thw")
        pixels = mm.get("pixel_values")
        if grid is not None:
            images += grid.shape[0]
        if pixels is not None:
            patches += pixels.shape[0]
    row = {
        "purpose": "train" if budget is None else "log_probs",
        "budget_override": budget,
        "rows": len(lengths),
        "logical_trajectories": len(set(rollout_data.get("sample_indices", range(len(lengths))))),
        "tokens": sum(lengths),
        "response_tokens": sum(rollout_data.get("response_lengths", [])),
        "image_refs": images,
        "vision_patch_refs": patches,
        "num_microbatches": list(num_microbatches),
        "dummy_microbatches": len(dummy),
        "microbatch_tokens": packs,
        "scheduled_tokens_including_dummy": sum(packs),
        "dummy_tokens": sum(packs[i] for i in dummy),
    }
    state["workloads"].append(row)
