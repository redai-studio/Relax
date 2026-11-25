# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import hashlib
import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable

from .weight_sync_guard import _WorkerWeightGuard


def control_fingerprint(loader: str, command: dict) -> str:
    return hashlib.sha256(json.dumps((loader, command), sort_keys=True).encode()).hexdigest()


class _ControlRejected(RuntimeError):
    """The TP collective completed, but at least one worker rejected the
    request."""


def agree_control(model: Any, header: Any, error: str | None) -> None:
    import torch.distributed as dist

    guard = model._relax_native_weight_guard
    try:
        rows = [None] * dist.get_world_size(guard.manager.tp_cpu_group)
        dist.all_gather_object(rows, (header, error), group=guard.manager.tp_cpu_group)
        if any(peer != header or failure is not None for peer, failure in rows):
            raise _ControlRejected(f"Native control TP agreement failed: {rows}")
    except BaseException:
        guard.lease.fail()
        raise


def _worker_control_finished(guard: _WorkerWeightGuard, control: tuple, error: str | None) -> None:
    """Meet loader peers even when native decoding/cache checks fail first."""
    header = control_fingerprint(*control)
    if not guard.control_entry_seen:
        try:
            agree_control(guard.models[0], header, error or "Native worker did not enter the control loader")
        except _ControlRejected as exc:
            error = f"{type(exc).__name__}: {exc}"
    agree_control(guard.models[0], (header, "native-worker-complete"), error)


class _RunnerWeightUpdater:
    def __init__(self, updater: Any, update: Callable) -> None:
        self._updater = updater
        self.update_weights_from_tensor = partial(update, updater.update_weights_from_tensor)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._updater, name)


@contextmanager
def _intercept_native_update(manager: Any, request: Any, update: Callable) -> Iterator[None]:
    worker = manager.tp_worker
    original = getattr(worker, "update_weights_from_tensor", None)
    if callable(original):
        had_override = "update_weights_from_tensor" in vars(worker)
        saved = vars(worker).get("update_weights_from_tensor")
        worker.update_weights_from_tensor = partial(update, original)
        try:
            yield
        finally:
            if had_override:
                worker.update_weights_from_tensor = saved
            else:
                del worker.update_weights_from_tensor
        return

    # The newer scheduler deserializes on the TP worker, then calls a frozen
    # WeightUpdater on each selected model runner.
    runners = tuple(manager._select_runners("all"))
    if len(runners) != 1 or runners[0][0] != "target":
        raise NotImplementedError("Colocate control requires exactly one target weight updater")
    selector = request.selector
    if selector is None and request.disable_draft_model is None and manager._session is not None:
        selector = manager._session.selector
    if selector not in (None, "target", "all"):
        raise ValueError("Colocate control did not select the target weight updater")

    runner = runners[0][1]
    updater = runner.weight_updater
    runner.weight_updater = _RunnerWeightUpdater(updater, update)
    try:
        yield
    finally:
        runner.weight_updater = updater


def _p2p_native_request(
    manager: Any, guard: _WorkerWeightGuard, control: tuple, method: Callable, args: tuple, kwargs: dict
) -> Any:
    """Keep native worker failures ahead of the scheduler's outer barrier."""
    guard.control_entry_seen = False
    guard.control_worker_succeeded = False
    worker_completed = False

    def update(original: Callable, *worker_args: Any, **worker_kwargs: Any) -> Any:
        nonlocal worker_completed
        result, error = None, None
        try:
            result = original(*worker_args, **worker_kwargs)
            if not isinstance(result, tuple) or len(result) != 2 or result[0] is not True:
                raise RuntimeError(f"Native control worker rejected the update: {result}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        worker_completed = True
        _worker_control_finished(guard, control, error)
        guard.control_worker_succeeded = True
        return result

    try:
        request = args[0] if args else kwargs.get("recv_req")
        with _intercept_native_update(manager, request, update):
            return method(manager, *args, **kwargs)
    except Exception as exc:
        if not worker_completed:
            _worker_control_finished(guard, control, f"{type(exc).__name__}: {exc}")
        raise


def native_worker_dispatch_supported() -> bool:
    """Require the worker hook that closes errors before the scheduler
    barrier."""
    from sglang.srt.managers.tp_worker import BaseTpWorker

    if callable(getattr(BaseTpWorker, "update_weights_from_tensor", None)) and callable(
        getattr(BaseTpWorker, "_deserialize_own_rank", None)
    ):
        return True
    try:
        from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager
        from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
    except ImportError:
        return False
    runner_update = getattr(WeightUpdater, "update_weights_from_tensor", None)
    return (
        callable(getattr(BaseTpWorker, "deserialize_own_rank", None))
        and callable(getattr(SchedulerWeightUpdaterManager, "_select_runners", None))
        and callable(runner_update)
        and tuple(inspect.signature(runner_update).parameters) == ("self", "named_tensors", "load_format")
    )
