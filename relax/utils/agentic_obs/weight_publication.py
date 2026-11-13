# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded control-plane observation of successful rollout publications."""

from __future__ import annotations

import asyncio
import copy
import threading
import uuid
from concurrent.futures import Future
from typing import Any

from relax.utils.agentic_obs.config import agentic_obs_enabled
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_TIMEOUT = 5.0
_FRONTIERS: dict[str, dict[str, Any] | None] = {}


def weight_audit_enabled(args: Any) -> bool:
    """Custom router middleware may require numeric labels; leave it
    unknown."""
    return agentic_obs_enabled(args) and not (
        getattr(args, "use_slime_router", False) and getattr(args, "slime_router_middleware_paths", None)
    )


def publication_label(owner: Any) -> str:
    """Keep the training counter numeric and the observed backend label
    unique."""
    return getattr(owner, "_weight_audit_label", str(owner.weight_version))


def observe_install_result(owner: Any, result: Any) -> None:
    """Retain explicit receiver rejection without changing updater behavior."""
    if not weight_audit_enabled(getattr(owner, "args", None)) or not getattr(
        getattr(owner, "args", None), "use_agentic_rollout", False
    ):
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            observe_install_result(owner, item)
    elif isinstance(result, dict) and result.get("success") is False:
        owner._weight_audit_rejected = True


def _resolve_coordinator() -> Any:
    """Bound the entire Serve lookup, including its synchronous controller RPC.

    A daemon thread also avoids synchronously waiting on a Serve handle bound
    to the caller's running asyncio loop. A timed-out publisher never retries.
    """
    result: Future = Future()

    def resolve() -> None:
        try:
            from ray import serve

            from relax.agentic import AGENTIC_CHAT_API_SERVICE_NAME

            response = serve.get_app_handle(AGENTIC_CHAT_API_SERVICE_NAME).runtime_resources.remote()
            _, coordinator = response.result(timeout_s=_TIMEOUT)
            result.set_result(coordinator)
        except Exception as error:
            result.set_exception(error)

    threading.Thread(target=resolve, name="weight-audit-lookup", daemon=True).start()
    return result.result(timeout=_TIMEOUT)


def consumption_snapshot(args: Any) -> dict[str, Any] | None:
    """Read the local rank's publication evidence across argument copies."""
    return copy.deepcopy(_FRONTIERS.get(str(getattr(args, "agentic_obs_run_id", ""))))


def begin_publication(owner: Any) -> None:
    """Invalidate local evidence on all ranks; begin remotely on rank zero.

    Kept outside training hot paths. Failure disables subsequent certification
    for this publisher lifetime, without affecting weight installation.
    """
    args = getattr(owner, "args", None)
    if not weight_audit_enabled(args) or not getattr(args, "use_agentic_rollout", False):
        return
    owner._weight_audit_rejected = False
    _FRONTIERS[str(args.agentic_obs_run_id)] = None
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    rank = dist.get_rank(group)
    label = [None]
    if rank == 0:
        if not hasattr(owner, "_weight_audit_epoch"):
            owner._weight_audit_epoch = uuid.uuid4().hex
        label[0] = f"{owner._weight_audit_epoch}:{owner.weight_version}"
    # Even an unavailable coordinator cannot make a restarted publisher reuse
    # a previously certified backend label.
    dist.broadcast_object_list(label, src=dist.get_global_rank(group, 0), group=group)
    owner._weight_audit_label = label[0]
    if rank != 0:
        return
    if getattr(owner, "_weight_audit_failed", False):
        return
    try:
        import ray

        if not hasattr(owner, "_weight_audit_coordinator"):
            owner._weight_audit_coordinator = _resolve_coordinator()
        owner._weight_audit_attempt = (
            str(args.agentic_obs_run_id),
            owner._weight_audit_epoch,
            publication_label(owner),
        )
        accepted = ray.get(
            owner._weight_audit_coordinator.begin_weight_publication.remote(*owner._weight_audit_attempt),
            timeout=_TIMEOUT,
        )
        if accepted is not True:
            raise RuntimeError("publication begin was not acknowledged")
    except Exception as error:
        owner._weight_audit_failed = True
        logger.warning("weight audit disabled for publisher: %s", error)


def commit_publication(owner: Any) -> None:
    """Record a confirmed install before resume; share evidence with all
    ranks."""
    args = getattr(owner, "args", None)
    if not weight_audit_enabled(args) or not getattr(args, "use_agentic_rollout", False):
        return
    import ray
    import torch
    import torch.distributed as dist

    from relax.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    rejected = torch.tensor([int(getattr(owner, "_weight_audit_rejected", False))], dtype=torch.int32)
    dist.all_reduce(rejected, op=dist.ReduceOp.MAX, group=group)
    value = [None]
    if dist.get_rank(group) == 0 and not rejected.item() and not getattr(owner, "_weight_audit_failed", False):
        try:
            value[0] = ray.get(
                owner._weight_audit_coordinator.commit_weight_publication.remote(*owner._weight_audit_attempt),
                timeout=_TIMEOUT,
            )
            if value[0] is None:
                raise RuntimeError("publication commit was not acknowledged")
        except Exception as error:
            owner._weight_audit_failed = True
            logger.warning("weight audit publication unavailable: %s", error)
    dist.broadcast_object_list(value, src=dist.get_global_rank(group, 0), group=group)
    _FRONTIERS[str(args.agentic_obs_run_id)] = copy.deepcopy(value[0])


async def generation_snapshot(coordinator: Any) -> dict[str, Any] | None:
    """Read a frontier without allowing observer failure to fail a request."""
    if coordinator is None:
        return None
    try:
        return await asyncio.wait_for(coordinator.weight_publication_snapshot.remote(), timeout=_TIMEOUT)
    except Exception as error:
        logger.warning("weight audit snapshot unavailable: %s", error)
        return None
