# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in sender collective-stream timing, collected after the common probes.

This is a profiling run. Events include collective enqueue gaps and receiver
arrival waits; they do not measure physical NIC transfer or receiver copies.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any


def install(args: Any) -> None:
    import torch
    import torch.distributed as dist

    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from task_delta_weight_sync import measure_sync

    if getattr(args, "memory_weight_sync_config", None) or getattr(args, "disk_weight_sync_config", None):
        raise ValueError("Sender collective timing requires the ordinary NCCL updater")
    module = importlib.import_module("relax.backends.megatron.weight_update.update_weight_from_distributed")
    cls = module.UpdateWeightFromDistributed
    if getattr(cls, "_dws_collective_profiled", False):
        return
    measure_sync.install(args)
    events = []

    def profiled_broadcast(group_name, group, weight_version, rollout_engines, converted_named_tensors):
        # Keep the ordinary function's metadata, collectives, waits and return
        # contract. Metadata dispatch precedes the first recorded event.
        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                group_name=group_name,
                weight_version=str(weight_version),
            )
            for engine in rollout_engines
        ]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        host_started = monotonic()
        handles = [
            module.dist.broadcast(param.data, 0, group=group, async_op=True) for _, param in converted_named_tensors
        ]
        for handle in handles:
            handle.wait()
        end.record()
        events.append(
            dict(
                start=start,
                end=end,
                host_enqueue_s=monotonic() - host_started,
                bytes=sum(param.numel() * param.element_size() for _, param in converted_named_tensors),
            )
        )
        return refs

    module.update_weights_from_distributed = profiled_broadcast
    original_update = cls.update_weights
    group = get_gloo_group()
    rank = dist.get_rank(group)
    root = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1] / "sync-measurements"

    def profiled_update(self, *call_args, **call_kwargs):
        events.clear()
        result = original_update(self, *call_args, **call_kwargs)
        error = None
        if rank == 0:
            try:
                started = monotonic()
                buckets = []
                for pair in events:
                    # The ordinary update and all eight generation probes have
                    # already completed. Collection stays outside their timers.
                    pair["end"].synchronize()
                    buckets.append(
                        dict(
                            sender_collective_stream_s=pair["start"].elapsed_time(pair["end"]) / 1000,
                            host_enqueue_s=pair["host_enqueue_s"],
                            bytes=pair["bytes"],
                        )
                    )
                path = root / f"{self._dws_version:06d}.json"
                metrics = json.loads(path.read_text())
                metrics.update(
                    collective_buckets=buckets,
                    sender_collective_stream_s=sum(b["sender_collective_stream_s"] for b in buckets),
                    event_collection_s=monotonic() - started,
                    event_scope="source stream; includes enqueue gaps/arrival waits; excludes receiver parameter copies",
                )
                atomic_write(path, json.dumps(metrics, indent=2).encode())
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"Collective timing collection failed: {errors}")
        return result

    cls.update_weights = profiled_update
    cls._dws_collective_profiled = True
