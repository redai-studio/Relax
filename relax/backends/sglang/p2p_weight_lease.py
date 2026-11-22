# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import hashlib
import json
from typing import Any, Callable

from .weight_sync_guard import _WorkerWeightGuard


def control_fingerprint(loader: str, command: dict) -> str:
    return hashlib.sha256(json.dumps((loader, command), sort_keys=True).encode()).hexdigest()


def agree_control(model: Any, header: Any, error: str | None) -> None:
    import torch.distributed as dist

    guard = model._relax_native_weight_guard
    try:
        rows = [None] * dist.get_world_size(guard.manager.tp_cpu_group)
        dist.all_gather_object(rows, (header, error), group=guard.manager.tp_cpu_group)
        if any(peer != header or failure is not None for peer, failure in rows):
            raise RuntimeError(f"Native control TP agreement failed: {rows}")
    except BaseException:
        guard.lease.fail()
        raise


def _worker_control_finished(guard: _WorkerWeightGuard, control: tuple, error: str | None) -> None:
    """Meet loader peers even when native decoding/cache checks fail first."""
    header = control_fingerprint(*control)
    if not guard.control_entry_seen:
        try:
            agree_control(guard.models[0], header, error or "Native worker did not enter the control loader")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    agree_control(guard.models[0], (header, "native-worker-complete"), error)


def _p2p_native_request(
    manager: Any, guard: _WorkerWeightGuard, control: tuple, method: Callable, args: tuple, kwargs: dict
) -> Any:
    """Keep native worker failures ahead of the scheduler's outer barrier."""
    worker = manager.tp_worker
    guard.control_entry_seen = False
    guard.control_worker_succeeded = False
    worker_completed, installed = False, False

    def update(request):
        nonlocal worker_completed
        result, error = None, None
        try:
            result = original(request)
            if not isinstance(result, tuple) or len(result) != 2 or result[0] is not True:
                raise RuntimeError(f"Native control worker rejected the update: {result}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        worker_completed = True
        _worker_control_finished(guard, control, error)
        guard.control_worker_succeeded = True
        return result

    try:
        try:
            original = worker.update_weights_from_tensor
            had_override = "update_weights_from_tensor" in vars(worker)
            saved = vars(worker).get("update_weights_from_tensor")
            worker.update_weights_from_tensor = update
            installed = True
            return method(manager, *args, **kwargs)
        except Exception as exc:
            if not worker_completed:
                _worker_control_finished(guard, control, f"{type(exc).__name__}: {exc}")
            raise
    finally:
        if installed:
            if had_override:
                worker.update_weights_from_tensor = saved
            else:
                del worker.update_weights_from_tensor


def native_worker_dispatch_supported() -> bool:
    """Require the worker hook that closes errors before the scheduler
    barrier."""
    from sglang.srt.managers.tp_worker import BaseTpWorker

    return callable(getattr(BaseTpWorker, "update_weights_from_tensor", None)) and callable(
        getattr(BaseTpWorker, "_deserialize_own_rank", None)
    )
