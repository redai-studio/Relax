# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Own native P2P allocations across checked scheduler control requests."""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import (
    REMOTE_SHARD_LOADER,
    atomic_json,
)

from .weight_sync_guard import _WorkerWeightGuard


AUDIT_LOADER = "relax.backends.sglang.broadcast_weight_audit.control"
_LOADERS = frozenset((REMOTE_SHARD_LOADER, AUDIT_LOADER))


def decode_control(weights: list, loader: str, *, device: Any = None) -> dict:
    if (
        len(weights) != 1
        or (weights[0][1].device.type != "cpu" if device is None else weights[0][1].device != device)
        or weights[0][1].numel() != 0
        or weights[0][1].untyped_storage().nbytes() != 0
    ):
        raise ValueError("Native control requires one empty storage on the checked worker device")
    command = json.loads(weights[0][0])
    if not isinstance(command, dict):
        raise ValueError("Native control metadata must be an object")
    if type(command.get("version")) is not int or command["version"] < 0:
        raise ValueError("Native control requires a valid version")
    if type(command.get("engine")) not in (str, int) or str(command["engine"]) == "":
        raise ValueError("Native control requires an engine identity")
    command["engine"] = str(command["engine"])
    if not isinstance(command.get("directory"), str) or not Path(command["directory"]).is_absolute():
        raise ValueError("Native control requires an absolute directory")
    return command


def control_fingerprint(loader: str, command: dict) -> str:
    return hashlib.sha256(json.dumps((loader, command), sort_keys=True).encode()).hexdigest()


def scheduler_control(manager: Any, operation: str, request: Any) -> tuple[str, dict] | None:
    loader = getattr(request, "load_format", None)
    if operation != "update_weights_from_tensor" or loader not in _LOADERS:
        return None
    if getattr(request, "flush_cache", True) is not False:
        raise ValueError("Native P2P control requires the cache to be flushed before the transaction")
    import torch.distributed as dist
    from sglang.srt.utils import MultiprocessingSerializer

    group = manager.tp_cpu_group
    serialized = request.serialized_named_tensors
    if len(serialized) != dist.get_world_size(group):
        raise ValueError("Native control must address every actual TP worker")
    # A zero-storage tensor has no one-shot shared-storage descriptor. Validate
    # before the native worker deserializes the same request a second time.
    weights = MultiprocessingSerializer.deserialize(serialized[dist.get_rank(group)])
    return loader, decode_control(weights, loader)


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


class P2PWeightLease:
    """Bind one actual model, registration and update version to its lease."""

    def __init__(self, model: Any, command: dict, generation: str, params: dict, layouts: dict) -> None:
        from .weight_sync_guard import native_weight_lease

        self.model = model
        self.guard = model._relax_native_weight_guard
        self.lease = native_weight_lease(model)
        if self.guard.remote_owner is not None:
            raise RuntimeError("Native P2P transaction already has an owner")
        self.directory = str(Path(command["directory"]).resolve())
        self.engine, self.version = command["engine"], command["version"]
        self.session_mode = command["weight_update_session"]
        self.protocol = command["native_weight_lease_protocol"]
        self.params, self.layouts = params, layouts
        self.registration = self.state = None
        self.phase, self.receipt = "preparing", None
        self.token = (self.engine, generation)
        delivery = hashlib.sha256(
            json.dumps((self.directory, self.engine, generation, layouts), sort_keys=True).encode()
        ).hexdigest()
        self.identity = self.lease.begin(str(Path(self.directory).parent), self.version, delivery)
        self.lease.add(self.token, self.version)
        self.guard.remote_owner = self

    def bind(self, state: dict) -> None:
        self.state, self.registration = state, state["registration"]
        state["native_transaction"] = self
        self.phase = "prepared"

    def validate(self, control: tuple[str, dict] | None, *, after: bool = False) -> None:
        from .weight_sync_guard import _native_models, native_weight_lease

        if (
            self.guard.remote_owner is not self
            or native_weight_lease(self.model) is not self.lease
            or self.lease.failed
            or tuple(_native_models(self.guard.manager)) != self.guard.models
            or self.state is None
            or self.model._relax_p2p_state is not self.state
            or self.state.get("native_transaction") is not self
            or self.state["registration"] is not self.registration
            or self.state["version"] != self.version
            or self.state["directory"] != self.directory
            or not self.state["active"]
        ):
            raise RuntimeError("Native P2P allocation owner changed")
        self.state["native_reload"].validate()
        self.registration.validate(dict(self.model.named_parameters()))
        if self.phase != "committing" and (self.lease.active != self.identity or self.lease.pending != {self.token}):
            raise RuntimeError("P2P allocation lease changed")
        if control is None:
            raise RuntimeError("Native allocations are owned by a P2P transaction")
        loader, command = control
        if loader == AUDIT_LOADER:
            audit = getattr(self.model, "_relax_broadcast_audit", None)
            if (
                command.get("op") != "poison"
                or self.phase != "prepared"
                or audit is None
                or not audit["active"]
                or not audit["deferred"]
                or bool(audit["poison_complete"]) is not after
                or audit["version"] != self.version
                or command["version"] != self.version
                or command["engine"] != self.engine
                or audit["directory"] != command["directory"]
                or audit["p2p_directory"] != self.directory
                or command.get("p2p_directory") != self.directory
            ):
                raise RuntimeError("Audit does not own the prepared P2P allocation")
            return
        if loader != REMOTE_SHARD_LOADER or (
            command["directory"] != self.directory
            or command["engine"] != self.engine
            or command["version"] != self.version
            or command.get("weight_update_session") is not self.session_mode
            or command.get("native_weight_lease_protocol") != self.protocol
        ):
            raise RuntimeError("P2P continuation identity changed")
        op = command.get("op")
        expected = {"prepare": "prepared", "plan": "planned", "expert_allgather": "gathered", "commit": "committing"}
        if after:
            if expected.get(op) != self.phase:
                raise RuntimeError("P2P control returned without its phase completing")
        elif not (
            op == "plan"
            and self.phase == "prepared"
            or op == "expert_allgather"
            and self.phase == "planned"
            and self.state.get("collective_allgather")
            or op == "commit"
            and (
                self.phase == "planned"
                and not self.state.get("collective_allgather")
                or self.phase == "gathered"
                and self.state.get("collective_allgather")
            )
        ):
            raise RuntimeError("Unexpected P2P transaction phase")

    def transferred(self, receipt: dict) -> None:
        from .weight_sync_guard import commit_native_weight_lease

        error = None
        try:
            self.lease.complete(self.token)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        agree_control(self.model, (self.identity[:2], "transfer-complete"), error)
        commit_native_weight_lease(self.model, self.identity, self.guard.manager.tp_cpu_group)
        self.receipt, self.phase = receipt, "committing"

    def publish(self) -> None:
        if self.phase != "committing" or self.lease.active is not None or self.lease.pending:
            raise RuntimeError("P2P publication precedes allocation completion")
        atomic_json(Path(self.directory) / f"{self.state['key']}.verified.json", self.receipt)

    def finish(self) -> None:
        self.state.update(active=False, completed_version=self.version, reference={}, reference_buffers={})
        self.guard.remote_owner = None


def run_control(model: Any, weights: list, loader: str, action: Callable[[dict], Any]) -> Any:
    guard = model._relax_native_weight_guard
    guard.control_entry_seen = True
    command, header, error = None, None, None
    try:
        command = decode_control(weights, loader, device=next(iter(model.named_parameters()))[1].device)
        header = control_fingerprint(loader, command)
        control = (loader, command)
        owner = model._relax_native_weight_guard.remote_owner
        if owner is not None:
            owner.validate(control)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree_control(model, header, error)
    value, error = None, None
    try:
        value = action(command)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    agree_control(model, header, error)
    owner = model._relax_native_weight_guard.remote_owner
    if loader == REMOTE_SHARD_LOADER:
        if command["op"] == "commit":
            owner.transferred(value)
        elif command["op"] == "plan":
            owner.phase = "planned"
        elif command["op"] == "expert_allgather":
            owner.phase = "gathered"
    return value


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


def _guard_p2p_native_weights(manager: Any, operation: str, request: Any = None) -> _WorkerWeightGuard:
    import torch.distributed as dist

    guard, error, control = None, None, None
    operation_header = operation
    try:
        guard = getattr(manager.tp_worker, "_relax_native_weight_guard", None)
        if guard is None:
            guard = _WorkerWeightGuard(manager)
            guard.remote_owner, guard.current_control = None, None
            manager.tp_worker._relax_native_weight_guard = guard
        if guard.manager is not manager:
            raise RuntimeError("Native weight manager incarnation changed")
        control = scheduler_control(manager, operation, request)
        if control is not None:
            operation_header = operation, control_fingerprint(*control)
            if control[1]["op"] != "capability" and getattr(manager, "draft_worker", None) is not None:
                raise NotImplementedError("P2P control requires the checked target worker without a draft worker")
        if guard.remote_owner is not None:
            guard.remote_owner.validate(control)
        else:
            guard.refresh()
    except Exception as exc:
        error = str(exc)
    try:
        outcomes = [None] * dist.get_world_size(manager.tp_cpu_group)
        dist.all_gather_object(outcomes, (operation_header, error), group=manager.tp_cpu_group)
        if any(name != operation_header or problem is not None for name, problem in outcomes):
            raise RuntimeError(f"Native weight lifetime guard rejected {operation}: {outcomes}")
    except BaseException:
        if guard is not None:
            guard.lease.fail()
        raise
    guard.current_control = control
    return guard


def _finish_native_control(guard: _WorkerWeightGuard, control, result: Any, failure: str | None = None) -> None:
    owner, error = guard.remote_owner, failure
    try:
        if getattr(result, "success", True) is False:
            raise RuntimeError("Native control worker rejected the update")
        if owner is not None:
            owner.validate(control, after=True)
        elif guard.lease.failed or guard.lease.active is not None or guard.lease.pending:
            raise RuntimeError("Native control returned without a checked allocation owner")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    header = control_fingerprint(*control)
    agree_control(guard.models[0], header, error)
    if owner is not None and owner.phase == "committing":
        error = None
        try:
            owner.publish()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        agree_control(owner.model, (header, "publication"), error)
        owner.finish()


def wrap_remote_weight_operation(name: str, method: Callable) -> Callable:
    """Adapt the shared guard to a checked multi-RPC remote transaction."""

    @wraps(method)
    def guarded(self, *args, **kwargs):
        request = args[0] if args else kwargs.get("recv_req")
        guard = _guard_p2p_native_weights(self, name, request)
        control = guard.current_control
        guard.control_worker_succeeded = False
        try:
            guard.operation_sequence += 1
            try:
                result = (
                    _p2p_native_request(self, guard, control, method, args, kwargs)
                    if control is not None
                    else method(self, *args, **kwargs)
                )
            except Exception as exc:
                if control is not None and guard.control_worker_succeeded:
                    _finish_native_control(guard, control, None, f"{type(exc).__name__}: {exc}")
                raise
            if control is not None:
                _finish_native_control(guard, control, result)
            elif guard.lease.failed or guard.lease.active is not None or guard.lease.pending or guard.remote_owner:
                raise RuntimeError("Native operation returned with an unfinished remote allocation owner")
            return result
        except BaseException:
            guard.lease.fail()
            raise

    return guarded
