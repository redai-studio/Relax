# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Admission and weight-update fences, independent of GPU libraries."""

import asyncio
from functools import wraps
from typing import Any, Callable
from uuid import uuid4


class AdmissionLedger:
    """One actor-loop ledger; drains yield so completions always make
    progress."""

    def __init__(self) -> None:
        self.fences: set[str] = set()
        self.permits: dict[str, Any] = {}
        self.changed = asyncio.Condition()
        self.model_fences: dict[str, set[str]] = {}

    def admit(self, model_id: str) -> str:
        if self.fences:
            raise RuntimeError("Inference admission is closed")
        token = uuid4().hex
        self.permits[token] = model_id
        return token

    async def complete(self, token: str) -> None:
        async with self.changed:
            self.permits.pop(token, None)
            self.changed.notify_all()

    async def drain(self, fence: str, timeout: float) -> None:
        # A failure or cancellation deliberately leaves the fence closed.
        self.fences.add(fence)
        async with self.changed:
            await asyncio.wait_for(self.changed.wait_for(lambda: not self.permits), timeout=timeout)


class AdmissionScope:
    """A model view of the task's one permit registry and condition."""

    def __init__(self, ledger: AdmissionLedger, scope: str) -> None:
        self.root = ledger
        self.scope = scope

    @property
    def fences(self) -> set[str]:
        return self.root.model_fences.setdefault(self.scope, set())

    @property
    def permits(self) -> dict[str, str]:
        return {token: model for token, (scope, model) in self.root.permits.items() if scope == self.scope}

    def admit(self, model_id: str) -> str:
        if self.fences or self.root.fences:
            raise RuntimeError("Inference admission is closed")
        token = uuid4().hex
        self.root.permits[token] = (self.scope, model_id)
        return token

    async def complete(self, token: str) -> None:
        await self.root.complete(token)

    async def drain(self, fence: str, timeout: float) -> None:
        self.fences.add(fence)
        async with self.root.changed:
            await asyncio.wait_for(self.root.changed.wait_for(lambda: not self.permits), timeout=timeout)


def committed_policy_weight_version(actor: Any) -> str:
    """Read the transport's committed version after its actual sync returns."""
    updater = getattr(actor, "weight_updater", None)
    dcs = getattr(actor, "checkpoint_engine_client", None)
    if getattr(actor, "_use_dcs_hybrid_weight_sync", False):
        version = getattr(dcs.backend, "weight_version", None) if dcs is not None else None
    elif updater is not None:
        version = getattr(updater, "weight_version", None)
    elif hasattr(actor, "policy_version"):
        version = actor.policy_version
    elif dcs is not None:
        version = getattr(dcs.backend, "weight_version", None)
    else:
        version = None
    if version is None:
        raise RuntimeError("Weight sync completed without a transport-confirmed policy version")
    return str(version)


def managed_weight_update(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Fence rank-zero backend updates only when using the managed rollout
    handle.

    A failed update keeps admission closed until the task is recovered. Legacy
    handles and their existing lifecycle are unaffected.
    """

    @wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        from relax.distributed.ray.inference_handle import ManagedRolloutHandle

        handle = getattr(self, "rollout_manager", None)
        manager = handle.inference_manager if isinstance(handle, ManagedRolloutHandle) else None
        if (
            manager is None
            or getattr(self.args, "debug_train_only", False)
            or getattr(self.args, "debug_rollout_only", False)
        ):
            return fn(self, *args, **kwargs)
        import ray
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group() if dist.is_initialized() else None

        def rank_zero(action: Callable[[], Any]) -> Any:
            outcome = [None, None]
            if self.args.rank == 0:
                try:
                    outcome[0] = action()
                except Exception as exc:
                    outcome[1] = type(exc).__name__
            if group is not None:
                dist.broadcast_object_list(outcome, src=0, group=group)
            if outcome[1] is not None:
                raise RuntimeError(f"Inference weight-update fence failed: {outcome[1]}")
            return outcome[0]

        source = "dcs" if getattr(self, "_use_dcs_hybrid_weight_sync", False) else "backend"
        token = rank_zero(lambda: ray.get(manager.begin_weight_update.remote(source)))
        local_error = None
        result = None
        try:
            result = fn(self, *args, **kwargs)
        except Exception as exc:
            local_error = exc
        if group is not None:
            failures = [None] * dist.get_world_size(group=group)
            failure = f"{type(local_error).__name__}: {local_error}" if local_error is not None else None
            dist.all_gather_object(failures, failure, group=group)
            if any(item is not None for item in failures):
                failed_ranks = [rank for rank, item in enumerate(failures) if item is not None]
                raise RuntimeError(
                    f"Policy weight synchronization failed on ranks {failed_ranks}: {failures}"
                ) from local_error
        elif local_error is not None:
            raise local_error
        rank_zero(lambda: ray.get(manager.end_weight_update.remote(token, committed_policy_weight_version(self))))
        return result

    return wrapped
