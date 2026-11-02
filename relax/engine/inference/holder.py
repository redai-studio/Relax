# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Caller identities and conservative, bounded liveness probes."""

from __future__ import annotations

import os
from dataclasses import dataclass

from relax.utils.env import Envs


@dataclass(frozen=True)
class HolderIdentity:
    actor_id: str = ""
    worker_id: str = ""
    node_id: str = ""
    pid: int = 0
    create_time: float = 0.0


def current_holder() -> HolderIdentity:
    import psutil
    import ray

    process = psutil.Process(os.getpid())
    if not ray.is_initialized():
        return HolderIdentity(pid=process.pid, create_time=process.create_time())
    context = ray.get_runtime_context()
    return HolderIdentity(
        actor_id=context.get_actor_id() or "",
        worker_id=context.get_worker_id() or "",
        node_id=context.get_node_id(),
        pid=process.pid,
        create_time=process.create_time(),
    )


def _process_alive(holder: HolderIdentity) -> bool:
    import psutil

    try:
        process = psutil.Process(holder.pid)
        return process.create_time() == holder.create_time and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def holder_alive(holder: HolderIdentity | None) -> bool | None:
    """Only a terminal receipt returns False; unreachable means unknown."""
    if holder is None:
        return None
    import ray

    try:
        if holder.actor_id:
            from ray._private.state import actors

            record = actors(holder.actor_id)
            state = record.get("State") if record else None
            return False if state == "DEAD" else True if state == "ALIVE" else None
        if not holder.node_id:
            return _process_alive(holder) if not ray.is_initialized() else None
        node = next((node for node in ray.nodes() if node["NodeID"] == holder.node_id), None)
        if node is None:
            return None
        if not node["Alive"]:
            return False
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        probe = ray.remote(num_cpus=0)(_process_alive).options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(holder.node_id, soft=False)
        )
        return ray.get(probe.remote(holder), timeout=Envs.RELAX_INFERENCE_HOLDER_CHECK_S)
    except Exception:
        return None
