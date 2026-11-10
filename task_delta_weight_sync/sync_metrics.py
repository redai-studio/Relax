# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit benchmark counters; never enabled by ordinary engine requests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def record_http(counter: dict, endpoint: str, response: Any) -> None:
    """Count actual prepared JSON bodies, excluding HTTP headers and Ray
    RPC."""
    body = response.request.body
    if not isinstance(body, (bytes, str)):
        raise ValueError("Synchronization metrics require an actual JSON request body")
    row = counter.setdefault(endpoint, dict(requests=0, request_json_bytes=0, decoded_response_body_bytes=0))
    row["requests"] += 1
    row["request_json_bytes"] += len(body.encode("utf-8") if isinstance(body, str) else body)
    row["decoded_response_body_bytes"] += len(response.content)


def allocator_metrics(model: Any = None, *, reset: bool) -> dict:
    """Observe allocator values without copying any model parameters."""
    import os
    import resource

    import torch

    from relax.distributed.weight_sync.provenance import runtime_identity

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
    return dict(
        identity=runtime_identity(),
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        host_rss_bytes=rss,
        host_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        gpu_peak_scope="Since this arm's explicit reset; includes already resident allocations",
        host_peak_scope="Process lifetime high-water mark, including initialization",
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
            workers.append(report)
    if len(workers) != 8 or len({row["identity"]["cuda"]["uuid"] for row in workers}) != 8:
        raise RuntimeError("Allocator observations require eight distinct actual GPUs")
    if len(replies) != 4 or any(reply.get("success") is not True for reply in replies):
        raise RuntimeError("Incomplete engine HTTP measurement")
    return dict(
        workers=workers,
        engines=replies,
        http_scope="Successful _make_request JSON request bodies and decoded response bodies; excludes HTTP headers, direct pause/continue requests, Ray RPC and TCP frames",
        request_replication="One HTTP request per engine (four); eight TP workers execute those requests",
    )
