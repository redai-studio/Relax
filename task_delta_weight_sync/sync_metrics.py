# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit benchmark counters; never enabled by ordinary engine requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def parameter_sizes(model: Any) -> dict:
    """Read native parameter geometry and dtype sizes without tensor values."""
    groups = {}
    count = 0
    for name, parameter in model.named_parameters():
        count += 1
        key = ("vision" if "visual" in name else "other") + ":" + str(parameter.dtype)
        row = groups.setdefault(key, dict(parameters=0, elements=0, bytes=0))
        row["parameters"] += 1
        row["elements"] += parameter.numel()
        row["bytes"] += parameter.numel() * parameter.element_size()
    return dict(
        named_parameters=count,
        named_parameter_bytes=sum(row["bytes"] for row in groups.values()),
        parameter_groups=groups,
        parameter_size_scope="Native named parameters with duplicate-object removal; raw storage coverage independently audited",
    )


def record_http(counter: dict, endpoint: str, response: Any) -> None:
    """Count actual prepared JSON bodies, excluding HTTP headers and Ray
    RPC."""
    body = response.request.body
    if body is None and getattr(response.request, "method", None) == "GET":
        body = b""
    if not isinstance(body, (bytes, str)):
        raise ValueError("Synchronization metrics require an actual JSON request body")
    row = counter.setdefault(endpoint, dict(requests=0, request_json_bytes=0, decoded_response_body_bytes=0))
    row["requests"] += 1
    row["request_json_bytes"] += len(body.encode("utf-8") if isinstance(body, str) else body)
    row["decoded_response_body_bytes"] += len(response.content)


def optimizer_boundary(args: Any, entered_at: float, completed_at: float | None = None) -> dict | None:
    """Keep the observed host boundary distinct from GPU completion."""
    result = getattr(args, "_dws_optimizer_result", None)
    if result is None:
        return None  # Initialization has no preceding optimizer update.
    observed = result["observed_host_monotonic_s"]
    if entered_at < observed or (completed_at is not None and completed_at < entered_at):
        raise ValueError("Optimizer and synchronization boundaries are out of order")
    return dict(
        result,
        actor_pre_updater_s=entered_at - observed,
        host_result_to_completion_s=None if completed_at is None else completed_at - observed,
        completion_scope="Host result through the declared updater/probe boundary; includes observation setup if present; no claim of a separately timed optimizer GPU fence",
    )


def validate_stream_receipts(arm: dict, workers: list[dict]) -> dict:
    """Match actual published schemas with every successful TP installation."""
    root, version = arm["stream"], arm["version"]
    sources = arm["source_ranks"]
    expected = {f"{root}:expert:{owner}:tp:{tp}" for owner in range(8) for tp in range(2)} | {f"{root}:nonexpert"}
    streams = {}
    for owner, source in enumerate(sources):
        owned = {f"{root}:expert:{owner}:tp:{tp}" for tp in range(2)}
        if owner == 0:
            owned.add(f"{root}:nonexpert")
        if set(source["streams"]) != owned:
            raise ValueError("Published streams do not match their source owner")
        for stream, receipt in source["streams"].items():
            if stream in streams:
                raise ValueError("Duplicate published stream receipt")
            streams[stream] = receipt
    members = {worker["member"] for worker in workers}
    if (
        (len(sources), len(workers), len(members)) != (8, 8, 8)
        or sorted(worker["tp_rank"] for worker in workers) != [0] * 4 + [1] * 4
        or set(streams) != expected
    ):
        raise ValueError("Incomplete source stream or worker coverage")
    nonces = {receipt["group_update"] for receipt in streams.values()}
    if len(nonces) != 1:
        raise ValueError("Mixed published update identities")
    for stream, receipt in streams.items():
        selected = (
            members
            if stream.endswith(":nonexpert")
            else {worker["member"] for worker in workers if stream.endswith(f":tp:{worker['tp_rank']}")}
        )
        if (
            receipt["version"] != version
            or receipt["base"] != version - 1
            or set(receipt["members"]) != selected
            or not isinstance(receipt["schema"], str)
            or len(receipt["schema"]) != 64
        ):
            raise ValueError("Invalid published version, schema or member set")
    for worker in workers:
        stages = worker["last_tcp_receiver_stages"]
        schemas = {
            stream: receipt["schema"] for stream, receipt in streams.items() if worker["member"] in receipt["members"]
        }
        if (
            stages["stream"] != root
            or stages["version"] != version
            or stages["group_update"] not in nonces
            or stages["schemas"] != schemas
        ):
            raise ValueError("Receiver schema, stream or version differs from the published update")
    return dict(status="VERIFIED", stream=root, version=version, streams=len(streams), workers=len(workers))


def allocator_metrics(model: Any = None, *, reset: bool) -> dict:
    """Observe allocator values without copying any model parameters."""
    import os
    import resource

    import torch

    from relax.distributed.weight_sync.provenance import runtime_identity
    from task_delta_weight_sync.audit_cpu_budget import host_memory_budget

    if os.environ.get("DWS_COLLECT_METRICS") == "1" and (
        os.environ.get("NCCL_NET") != "Socket" or os.environ.get("NCCL_IB_DISABLE") != "1"
    ):
        raise ValueError("The formal comparison requires NCCL Socket with IB disabled on every process")
    device = (
        next(model.parameters()).device if model is not None else torch.device("cuda", torch.cuda.current_device())
    )
    if device.type != "cuda":
        raise ValueError("The GPU benchmark requires actual CUDA parameters")
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    status = Path("/proc/self/status").read_text().splitlines()
    rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
    compiler = sys.modules.get("torch._dynamo.utils")
    timers = getattr(compiler, "compilation_time_metrics", {})
    compiler_times = {name: dict(calls=len(values), seconds=sum(values)) for name, values in list(timers.items())}
    return dict(
        identity=runtime_identity(),
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        host_rss_bytes=rss,
        host_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        host_memory=host_memory_budget(),
        gpu_peak_scope="Since this arm's explicit reset; includes already resident allocations",
        host_peak_scope="Process lifetime high-water mark, including initialization",
        torch_compiler_cumulative=compiler_times,
        compiler_scope="Cumulative PyTorch dynamo_timed CPU compiler timers; nested keys overlap and must not be summed; preserve both snapshots and detect internal counter resets before differencing; other libraries' JIT is not covered",
        **(parameter_sizes(model) if model is not None else {}),
    )


def receiver_metrics(engines: Any, directory: Path, update_id: str, phase: str, *, start: bool) -> dict:
    """Collect eight worker records outside the completed synchronization
    timer."""
    import ray

    command = "measure_start" if start else "measure_end"
    control = dict(command=command, directory=str(directory), update_id=update_id, phase=phase, paused=False)
    replies = ray.get([engine.run_memory_audit.remote(control, 120) for engine in engines], timeout=130)
    workers = []
    for engine in engines:
        for tp in range(2):
            member = f"{engine._actor_id.hex()}:tp:{tp}"
            report = json.loads((directory / member / f"{update_id}-{phase}.json").read_text())
            if (
                report.get("success") is not True
                or report.get("member") != member
                or report.get("update_id") != update_id
            ):
                raise RuntimeError("Incomplete worker allocator observation")
            if phase.startswith("nccl-"):
                report.pop("last_tcp_receiver_stages", None)  # A native arm leaves the previous TCP state intact.
            workers.append(report)
    if len(workers) != 8 or len({row["identity"]["cuda"]["uuid"] for row in workers}) != 8:
        raise RuntimeError("Allocator observations require eight distinct actual GPUs")
    if len(replies) != 4 or any(reply.get("success") is not True for reply in replies):
        raise RuntimeError("Incomplete engine HTTP measurement")
    return dict(
        workers=workers,
        engines=replies,
        http_scope="Successful synchronization POST JSON bodies, layout GETs and decoded response bodies; includes pause/continue; excludes HTTP headers, Ray RPC and TCP frames",
        request_replication="One HTTP request per engine (four); eight TP workers execute those requests",
    )
