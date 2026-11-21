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
        self.permits: dict[str, str] = {}
        self.changed = asyncio.Condition()

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
        if manager is None:
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

        token = rank_zero(lambda: ray.get(manager.begin_weight_update.remote("backend")))
        result = fn(self, *args, **kwargs)
        rank_zero(lambda: ray.get(manager.end_weight_update.remote(token)))
        return result

    return wrapped
