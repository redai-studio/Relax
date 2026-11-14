# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in worker RPC for installing a checkpoint through SGLang's disk
loader."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from .deadline import time_budget
from .fence import WorkerFence
from .store import atomic_write, canonical
from .traffic import IOObservation


class WorkerBusyError(RuntimeError):
    """The scheduler still has generation work to retire."""


def register() -> None:
    # SGLang discovers this entry point in each scheduler process. Ordinary
    # engines keep their existing update path and do not import the extension.
    if os.environ.get("RELAX_WEIGHT_VERSION_PLUGIN") != "1":
        return
    from sglang.srt.managers.scheduler import Scheduler

    Scheduler.relax_weight_version = worker_rpc


def worker_rpc(scheduler, payload: str) -> None:
    """Every TP member validates a command before any member mutates
    weights."""
    import torch
    import torch.distributed as distributed
    from sglang.srt.distributed.parallel_state import get_moe_tensor_parallel_rank, get_moe_tensor_parallel_world_size

    request = json.loads(payload)
    directory = Path(request["receipts"])
    rank = scheduler.ps.pp_rank * scheduler.ps.tp_size + scheduler.ps.tp_rank
    receipt = directory / f"rank-{rank}.json"
    fence = getattr(scheduler, "_relax_weight_fence", None)
    if fence is None:
        fence = scheduler._relax_weight_fence = WorkerFence()
    response = {"rank": rank, "request_id": request["request_id"], "success": False}
    response["device"] = {"pid": os.getpid(), "host": os.uname().nodename}
    try:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        response["device"].update(visible_index=device, uuid=str(properties.uuid), name=properties.name)
    except (AttributeError, RuntimeError, AssertionError) as error:
        # Diagnostics must not mask the actual RPC outcome (including mocked
        # CPU workers and unavailable-device error receipts).
        response["device"]["observation_error"] = f"{type(error).__name__}: {error}"
    try:
        phase = request["phase"]
        command = {key: value for key, value in request.items() if key not in ("request_id", "receipts", "timeout")}
        error = None
        try:
            if phase == "begin":
                fence.begin(
                    request["workers"][str(rank)]["instance"],
                    request["epoch"],
                    request["installation_id"],
                    request["root"],
                )
            elif phase != "identify":
                if request["workers"][str(rank)] != fence.identity():
                    raise ValueError("Worker instance or installation changed before this command")
                if not scheduler.is_fully_idle():
                    raise WorkerBusyError("Drain generation requests before installing weights")
                if phase != "check":
                    fence.validate(request["workers"][str(rank)], request["command_id"], phase, command)
        except Exception as exception:
            error = (isinstance(exception, WorkerBusyError), f"rank={rank}: {type(exception).__name__}: {exception}")
        errors = [None] * distributed.get_world_size(scheduler.tp_group.cpu_group)
        distributed.all_gather_object(errors, error, group=scheduler.tp_group.cpu_group)
        if any(errors):
            failures = [item for item in errors if item]
            exception_type = WorkerBusyError if all(item[0] for item in failures) else RuntimeError
            raise exception_type("; ".join(item[1] for item in failures))

        def operation():
            started = time.perf_counter()
            baseline = None
            verification = None
            observation = None
            multimodal_cache = None
            if phase == "load":
                from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput

                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with IOObservation(
                    path=request["checkpoint"],
                    interfaces=tuple(filter(None, os.environ.get("RELAX_WEIGHT_METRIC_INTERFACES", "").split(","))),
                ) as observation:
                    result = scheduler.weight_updater.update_weights_from_disk(
                        UpdateWeightFromDiskReqInput(
                            model_path=request["checkpoint"],
                            load_format="safetensors",
                            weight_version=request["version"],
                        )
                    )
                if not result.success:
                    raise RuntimeError(result.message)
            elif phase == "dump":
                scheduler.weight_updater.save_sharded_model(
                    {
                        "path": request["dump"],
                        "pattern": f"model-rank-{rank}-part-{{part}}.safetensors",
                        "max_size": None,
                    }
                )
            elif phase == "verify":
                from sglang.srt.model_loader.loader import ShardedStateLoader

                from .sglang_loader import verify_qwen3_dense

                layouts = {int(key): value for key, value in request["layouts"].items()}
                model = scheduler.tp_worker.model_runner.model
                physical = ShardedStateLoader._filter_subtensors(model.state_dict())
                schema = {name: [list(value.shape), str(value.dtype)] for name, value in physical.items()}
                if schema != request["expected_physical_schema"][str(rank)]:
                    raise ValueError("Physical model inventory changed since installation preparation")
                inventory_seconds = time.perf_counter() - started
                verification_timings = {}
                verification_pipeline = {}
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with (
                    IOObservation(path=request["checkpoint"]) as observation,
                    time_budget(float(request.get("timeout", 120))),
                ):
                    verification = verify_qwen3_dense(
                        Path(request["checkpoint"]),
                        None,
                        tp_size=scheduler.ps.tp_size,
                        pp_size=scheduler.ps.pp_size,
                        layouts=layouts,
                        physical_tensors={(rank, name): value for name, value in physical.items()},
                        worker_rank=rank,
                        timings=verification_timings,
                        pipeline_metrics=verification_pipeline,
                    )
            elif phase == "flush":
                from sglang.srt.managers import mm_utils

                if not scheduler.flush_cache():
                    raise RuntimeError("Scheduler cache is still in use")
                # SGLang's scheduler flush resets KV/grammar caches, while
                # vision embeddings are retained separately by image hash.
                # Every installation invalidates features from old weights.
                cache = mm_utils.embedding_cache
                if cache is not None:
                    multimodal_cache = {"entries_cleared": len(cache), "bytes_cleared": cache.current_size}
                    cache.clear()
            else:
                raise ValueError(f"Unknown worker phase: {phase}")
            torch.cuda.synchronize()
            result = {"device_complete": True}
            result["phase_seconds"] = {"total": time.perf_counter() - started}
            if phase == "verify":
                result["phase_seconds"]["physical_inventory"] = inventory_seconds
                result["phase_seconds"].update(verification_timings)
                result["verification_pipeline"] = verification_pipeline
            if multimodal_cache is not None:
                result["multimodal_cache"] = multimodal_cache
            if observation is not None:
                result["target_io"] = observation.report
            if verification is not None:
                result["verification"] = verification
            if baseline is not None:
                result["verification_memory" if phase == "verify" else "loader_memory"] = {
                    "allocated_before_bytes": baseline,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "temporary_allocated_peak_bytes": max(0, torch.cuda.max_memory_allocated() - baseline),
                }
            return result

        if phase in ("identify", "begin", "check"):
            from sglang.srt.model_loader.loader import ShardedStateLoader

            physical = ShardedStateLoader._filter_subtensors(scheduler.tp_worker.model_runner.model.state_dict())
            result = {
                "identity": fence.identity(),
                "physical_schema": {name: [list(value.shape), str(value.dtype)] for name, value in physical.items()},
                "layout": {
                    "tp_rank": scheduler.ps.tp_rank,
                    "tp_size": scheduler.ps.tp_size,
                    "pp_rank": scheduler.ps.pp_rank,
                    "pp_size": scheduler.ps.pp_size,
                    "start_layer": scheduler.tp_worker.model_runner.layer_info.start_layer,
                    "end_layer": scheduler.tp_worker.model_runner.layer_info.end_layer,
                    "moe_ep_rank": scheduler.ps.moe_ep_rank,
                    "moe_ep_size": scheduler.ps.moe_ep_size,
                    "moe_tp_rank": get_moe_tensor_parallel_rank(),
                    "moe_tp_size": get_moe_tensor_parallel_world_size(),
                },
                "parameter_dtypes": sorted(
                    {str(p.dtype) for p in scheduler.tp_worker.model_runner.model.parameters()}
                ),
            }
        else:
            result = fence.execute(
                request["workers"][str(rank)], request["command_id"], phase, operation, payload=command
            )
        response.update(success=True, **result)
    except Exception as exception:
        response["error"] = f"{type(exception).__name__}: {exception}"
        response["error_code"] = "WORKER_BUSY" if isinstance(exception, WorkerBusyError) else "WORKER_ERROR"
        response["identity"] = fence.identity()
    atomic_write(receipt, canonical(response))


@contextmanager
def version_plugin_environment():
    """Enable the version plugin only while spawning this engine's workers."""
    from importlib.metadata import entry_points

    registered = entry_points(group="sglang.srt.plugins")
    if not any(entry.name == "relax_weight_versions" for entry in registered):
        raise RuntimeError("Install this Relax checkout before starting the version consumer")
    previous = {key: os.environ.get(key) for key in ("RELAX_WEIGHT_VERSION_PLUGIN", "SGLANG_PLUGINS")}
    enabled = previous["SGLANG_PLUGINS"]
    if enabled and "relax_weight_versions" not in {name.strip() for name in enabled.split(",")}:
        os.environ["SGLANG_PLUGINS"] = enabled + ",relax_weight_versions"
    os.environ["RELAX_WEIGHT_VERSION_PLUGIN"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_engine(**options):
    """Start an engine whose spawned workers load the registered extension."""
    from sglang import Engine

    with version_plugin_environment():
        return Engine(**options)
