# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Guard actual scheduler allocations during direct weight updates."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any


_NATIVE_WEIGHT_OPERATIONS = (
    "release_memory_occupation",
    "resume_memory_occupation",
    "begin_weight_update",
    "end_weight_update",
    "post_process_weights",
    "finalize_pending_bf16_weights",
    "update_weights_from_disk",
    "update_weights_from_distributed",
    "update_weights_from_tensor",
    "update_weights_from_ipc",
    "check_weights",
)
_OPTIONAL_WEIGHT_OPERATIONS = frozenset(
    {"begin_weight_update", "end_weight_update", "finalize_pending_bf16_weights", "check_weights"}
)


def _native_models(manager: Any) -> tuple:
    """Resolve every model through the actual manager's runner interface."""
    selector = getattr(manager, "_select_runners", None)
    if callable(selector):
        return tuple(runner.weight_updater.get_model() for _, runner in selector("all"))
    # Released managers before _select_runners keep the target runner directly
    # on the TP worker. Resolve the draft through that version's own helper;
    # an unrecognized draft layout must never bypass the allocation guard.
    models = [manager.tp_worker.model_runner.model]
    if manager.draft_worker is not None:
        from sglang.srt.managers.scheduler_components.weight_updater import _get_draft_model_runner

        runner = _get_draft_model_runner(manager.draft_worker)
        if runner is None:
            raise NotImplementedError("Unsupported SGLang draft runner for the native weight guard")
        models.append(runner.model)
    return tuple(models)


class _WorkerWeightGuard:
    def __init__(self, manager: Any) -> None:
        from relax.distributed.checkpoint_service.backends.shard_p2p.lifecycle import NativeWeightLease

        self.manager = manager
        self.lease = NativeWeightLease()
        self.models: tuple = ()
        self.operation_sequence = 0
        self.colocate_owner = None
        self.current_control = None

    def refresh(self) -> None:
        import torch

        if self.lease.failed or self.lease.active is not None or self.lease.pending:
            raise RuntimeError("Native weight allocations are leased or quarantined")
        models = _native_models(self.manager)
        if not models or any(not isinstance(model, torch.nn.Module) for model in models):
            raise RuntimeError("Native weight guard cannot resolve every actual model")
        if any(getattr(model, "_relax_native_weight_lease", self.lease) is not self.lease for model in models):
            raise RuntimeError("Native weights belong to another allocation lease")
        for model in models:
            model._relax_native_weight_lease = self.lease
            model._relax_native_weight_guard = self
        self.models = models


def native_weight_lease(model: Any) -> Any:
    """Resolve proof from the actual scheduler worker before native
    registration."""
    guard = getattr(model, "_relax_native_weight_guard", None)
    if (
        not isinstance(guard, _WorkerWeightGuard)
        or getattr(model, "_relax_native_weight_lease", None) is not guard.lease
        or not any(candidate is model for candidate in guard.models)
        or not any(candidate is model for candidate in _native_models(guard.manager))
    ):
        raise RuntimeError("Remote weights require a guarded actual scheduler model")
    return guard.lease


def commit_native_weight_lease(model: Any, identity: tuple[str, int, str], group: Any) -> None:
    """Only a complete TP agreement permits native offload after remote
    writes."""
    import torch.distributed as dist

    guard = getattr(model, "_relax_native_weight_guard", None)
    lease = guard.lease if isinstance(guard, _WorkerWeightGuard) else None
    # A misrouted argument must not move this worker's failure agreement to a
    # different group and strand the actual TP peers.
    agreement_group = guard.manager.tp_cpu_group if lease is not None else group
    header, error = None, None
    try:
        lease = native_weight_lease(model)
        if model._relax_native_weight_guard.manager.tp_cpu_group is not group:
            raise RuntimeError("Native weight completion used a different TP group")
        lease.ready(identity)
        header = identity[:2]
    except Exception as exc:
        error = str(exc)
    try:
        results = [None] * dist.get_world_size(agreement_group)
        dist.all_gather_object(results, (header, error), group=agreement_group)
        if any(peer != header or problem is not None for peer, problem in results):
            raise RuntimeError(f"Native weight completion was not agreed: {results}")
        lease.commit(identity)
    except BaseException:
        if lease is not None:
            lease.fail()
        raise


def _colocate_control(manager: Any, operation: str, request: Any) -> dict | None:
    """Decode only the checked NIXL probe and allocation-epoch requests."""
    if operation != "update_weights_from_tensor" or request is None:
        return None
    from .weight_loader import COLOCATE_NIXL_LOADER

    if getattr(request, "load_format", None) != COLOCATE_NIXL_LOADER:
        return None
    import json

    import torch.distributed as dist
    from sglang.srt.utils import MultiprocessingSerializer

    group = manager.tp_cpu_group
    values = request.serialized_named_tensors
    if len(values) != dist.get_world_size(group):
        raise ValueError("Colocate control must address every TP worker")
    weights = MultiprocessingSerializer.deserialize(values[dist.get_rank(group)])
    if len(weights) != 1 or weights[0][1].numel() != 0:
        raise ValueError("Colocate control requires one empty tensor")
    command = json.loads(weights[0][0])
    if (
        command.get("op") in ("probe_nixl", "bind_epoch", "retire")
        or str(command.get("op", "")).startswith("replica_")
        or command.get("op") == "apply"
        and "epoch_generation" in command
    ):
        return command
    return None


def _guard_native_weights(manager: Any, operation: str, request: Any = None) -> _WorkerWeightGuard:
    import torch.distributed as dist

    guard, error, control = None, None, None
    header = operation
    try:
        guard = getattr(manager.tp_worker, "_relax_native_weight_guard", None)
        if guard is None:
            guard = _WorkerWeightGuard(manager)
            manager.tp_worker._relax_native_weight_guard = guard
        if guard.manager is not manager:
            raise RuntimeError("Native weight manager incarnation changed")
        control = _colocate_control(manager, operation, request)
        if control is not None:
            header = (operation,) + tuple(control.get(key) for key in ("session", "engine", "op", "version"))
        if guard.colocate_owner is None:
            guard.refresh()
        else:
            from .shard_p2p_weight_loader import validate_epoch_continuation

            validate_epoch_continuation(guard, control)
        if operation in (
            "release_memory_occupation",
            "resume_memory_occupation",
            "update_weights_from_disk",
            "post_process_weights",
            "finalize_pending_bf16_weights",
        ):
            from .shard_p2p_weight_loader import retire_persistent_targets

            for model in guard.models:
                retire_persistent_targets(model)
                model.__dict__.pop("_relax_async_shard_layout", None)
    except Exception as exc:
        error = str(exc)
    try:
        outcomes = [None] * dist.get_world_size(manager.tp_cpu_group)
        dist.all_gather_object(outcomes, (header, error), group=manager.tp_cpu_group)
        if any(name != header or problem is not None for name, problem in outcomes):
            raise RuntimeError(f"Native weight lifetime guard rejected {operation}: {outcomes}")
    except BaseException:
        if guard is not None:
            guard.lease.fail()
        raise
    guard.current_control = control
    return guard


def install_native_weight_guard(manager_type: type | None = None) -> None:
    """Install before SGLang builds its worker dispatcher, never in Ray alone.

    The same guard covers all native weight mutations, including the owning
    tensor-loader RPC. That RPC may acquire and commit a lease inside its body;
    subsequent calls never bypass a still-active or failed lease by loader
    name.
    """
    if manager_type is None:
        from sglang.srt.managers.scheduler_components.weight_updater import SchedulerWeightUpdaterManager

        manager_type = SchedulerWeightUpdaterManager
    originals = {}
    for name in _NATIVE_WEIGHT_OPERATIONS:
        method = getattr(manager_type, name, None)
        if method is None and name in _OPTIONAL_WEIGHT_OPERATIONS:
            continue
        expected = ("self",) if name == "finalize_pending_bf16_weights" else ("self", "recv_req")
        if not callable(method) or tuple(inspect.signature(method).parameters) != expected:
            raise NotImplementedError(f"Unsupported SGLang native weight entry: {name}")
        originals[name] = method
    updates = {name for name in dir(manager_type) if name.startswith("update_weights_from_")}
    if not updates <= set(originals):
        raise NotImplementedError("Unsupported SGLang native weight mutation interface")
    if all(getattr(method, "_relax_native_weight_guard", False) for method in originals.values()):
        return
    if any(getattr(method, "_relax_native_weight_guard", False) for method in originals.values()):
        raise RuntimeError("SGLang native weight guards were partially installed")

    def wrapped(name: str, method: Callable) -> Callable:
        @wraps(method)
        def guarded(self, *args, **kwargs):
            guard = _guard_native_weights(self, name, args[0] if args else kwargs.get("recv_req"))
            try:
                guard.operation_sequence += 1
                for model in guard.models:
                    model.__dict__.pop("_relax_weight_sync_reply", None)
                control = guard.current_control
                if control is not None:
                    from .p2p_weight_lease import _p2p_native_request, agree_control

                    common = {key: control[key] for key in ("session", "engine", "op", "version")}
                    try:
                        result = _p2p_native_request(self, guard, ("colocate-shard", common), method, args, kwargs)
                    except Exception as exc:
                        if getattr(guard, "control_worker_succeeded", False):
                            agree_control(
                                guard.models[0],
                                ("colocate-shard", common, "native-complete"),
                                f"{type(exc).__name__}: {exc}",
                            )
                        raise
                else:
                    result = method(self, *args, **kwargs)
                # A loader returning before remote completion is not permission
                # for the next request to remap these native allocations.
                error = None
                try:
                    if (
                        (control is not None and getattr(result, "success", True) is False)
                        or guard.lease.failed
                        or (
                            (guard.lease.active is not None or guard.lease.pending)
                            and (
                                control is None
                                or control.get("op") not in ("bind_epoch", "replica_ready", "replica_gather")
                                or guard.colocate_owner is None
                            )
                        )
                    ):
                        raise RuntimeError("Native weight loader returned with an unfinished allocation lease")
                    if control is not None and control["op"] in ("replica_ready", "replica_gather"):
                        from .shard_p2p_weight_loader import validate_epoch_continuation

                        validate_epoch_continuation(guard, control)
                    if (
                        control is not None
                        and control["op"] in ("probe_nixl", "bind_epoch", "replica_bootstrap", "replica_ready")
                        and not any(getattr(model, "_relax_weight_sync_reply", None) for model in guard.models)
                    ):
                        raise RuntimeError("Colocate control returned without its worker reply")
                except Exception as exc:
                    error = str(exc)
                if control is not None:
                    agree_control(guard.models[0], ("colocate-shard", common, "native-complete"), error)
                if error is not None:
                    raise RuntimeError(error)
                replies = [model.__dict__.pop("_relax_weight_sync_reply", None) for model in guard.models]
                replies = [reply for reply in replies if reply is not None]
                if replies and result.success is True:
                    if name != "update_weights_from_tensor" or len(replies) != 1:
                        raise RuntimeError("Unexpected expert prepare response")
                    result.message = replies[0]
                return result
            except BaseException:
                guard.lease.fail()
                raise
            finally:
                guard.current_control = None
                for model in guard.models:
                    model.__dict__.pop("_relax_weight_sync_reply", None)

        guarded._relax_native_weight_guard = True
        return guarded

    for name, method in originals.items():
        setattr(manager_type, name, wrapped(name, method))
