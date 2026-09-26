# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Publish weight-transfer boundaries without leaving peer ranks in
collectives."""

from typing import Any

from relax.engine.inference import actor_identity


def notify_inference_weight_update(
    manager: Any,
    process_group: Any,
    completed: dict[str, Any] | None = None,
    *,
    engines: list[Any] | None = None,
    confirm_topology: bool = False,
    participating_urls: set[str] | None = None,
) -> dict[str, Any] | None:
    """All ranks participate; only group rank zero contacts the manager.

    Pass the returned token back only after a successful transfer. Engine
    handles can override the token's initial membership after recovery.
    """
    import ray
    import torch
    import torch.distributed as dist

    error = None
    token = None
    if confirm_topology:
        memberships = [None] * dist.get_world_size(process_group)
        dist.all_gather_object(memberships, participating_urls, group=process_group)
        senders = [set(urls) for urls in memberships if urls is not None]
        confirmed = set.intersection(*senders) if senders else set()
    if dist.get_rank(process_group) == 0:
        try:
            if completed is not None and engines is not None:
                completed = {**completed, "engines": [actor_identity(engine) for engine in engines]}
            if confirm_topology:
                unconfirmed = [
                    identity for identity in completed["engines"] if completed["urls"].get(identity) not in confirmed
                ]
                completed = {
                    **completed,
                    "unconfirmed_engines": unconfirmed,
                    "engines": [
                        identity for identity in completed["engines"] if completed["urls"].get(identity) in confirmed
                    ],
                }
            token = ray.get(manager.inference_weight_update.remote(completed), timeout=30)
        except Exception as exc:
            error = exc
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device="cpu")
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=process_group)
    if int(failed[0]):
        raise RuntimeError("Failed to publish inference weight update state") from error
    return token
