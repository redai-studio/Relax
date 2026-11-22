# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Versioned local expert updates with native seed parity and cold TP error
agreement."""

import base64
import hashlib
import json
import re
from collections.abc import Sequence
from math import prod
from typing import Any
from uuid import uuid4

import torch

from relax.backends.sglang.weight_loader import COLOCATE_NIXL_LOADER as COLOCATE_NIXL_LOADER
from relax.backends.sglang.weight_loader import EXPERT_IPC_LOADER as EXPERT_IPC_LOADER
from relax.backends.sglang.weight_loader import TP_IPC_LOADER as TP_IPC_LOADER
from relax.distributed.checkpoint_service.backends.shard_p2p.common import worker_identity
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import CompiledIPC, apply_ipc_plan, validate_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.layout import ExpertGeometry
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import bind_delivery, fingerprint
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


_NATIVE_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)$")
logger = get_logger(__name__)


def expert_shapes(geometry: dict) -> dict[str, tuple[int, int, int]]:
    """Describe actual MoE layers, including nonconsecutive layers in a hybrid
    model."""
    e, h, i, tp, count = (
        geometry[key] for key in ("num_experts", "hidden_size", "intermediate_size", "tp_size", "num_layers")
    )
    if any(type(value) is not int or value <= 0 for value in (e, h, i, tp, count)) or i % tp:
        raise ValueError("Invalid receiver expert geometry")
    layers = geometry.get("expert_layers", list(range(count)))
    if (
        not isinstance(layers, (list, tuple))
        or not layers
        or any(type(layer) is not int or not 0 <= layer < count for layer in layers)
        or list(layers) != sorted(set(layers))
    ):
        raise ValueError("Expert layers must be nonempty, unique, ordered model layer indices")
    layout = ExpertGeometry(e, h, i, tp)
    return {
        f"model.layers.{layer}.mlp.experts.{projection}": layout.native_shape(kind)
        for layer in layers
        for projection, kind in (("w13_weight", "gate_up"), ("w2_weight", "down"))
    }


def infer_expert_layers(writes: list[dict], geometry: dict) -> list[int]:
    """Infer source layer IDs; the receiver must still prove exact native
    coverage."""
    layers = sorted(
        {int(match[1]) for write in writes if (match := _NATIVE_EXPERT.fullmatch(write["param"])) is not None}
    )
    expert_shapes(dict(geometry, expert_layers=layers))
    return layers


def target_shapes(geometry: dict) -> dict:
    """Union native expert and ordinary targets in the same IPC plan."""
    shapes = expert_shapes(geometry) if geometry.get("expert_enabled", True) else {}
    ordinary = geometry.get("ordinary_targets", {})
    if not isinstance(ordinary, dict) or set(ordinary) & set(shapes):
        raise ValueError("Invalid or overlapping ordinary target inventory")
    for name, shape in ordinary.items():
        if not isinstance(name, str) or not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Invalid ordinary target shape")
    return dict(shapes, **{name: tuple(shape) for name, shape in ordinary.items()})


def _parameters(model: torch.nn.Module, geometry: dict) -> dict:
    shapes = target_shapes(geometry)
    return {
        name: value
        for name, value in model.named_parameters()
        if name in shapes or geometry.get("expert_enabled", True) and _NATIVE_EXPERT.fullmatch(name)
    }


def _remote_writes(state: dict) -> list[dict]:
    remote = set(state["delivery"]["routes"]["nixl"])
    if not remote:
        return []
    writes = [write for write in state["writes"] if write["id"] in remote]
    config = state.get("tp_replicas")
    if config is not None:
        from relax.distributed.checkpoint_service.backends.shard_p2p.replicas import partition_writes

        writes = partition_writes(
            writes, config["layouts"], config["parameters"], state["tp_rank"], len(config["members"])
        )
    return writes


def _validate_tp_replicas(model: torch.nn.Module, state: dict, parameters: dict, tp_rank: int) -> None:
    config = state.get("tp_replicas")
    if config is None:
        return
    from collections import Counter

    from relax.distributed.checkpoint_service.backends.shard_p2p.common import validate_worker_identity
    from relax.distributed.checkpoint_service.backends.shard_p2p.replicas import replica_write_signature

    from .replica_weight_sync import replica_parameters

    members, names = config["members"], config["parameters"]
    for member in members:
        validate_worker_identity(member)
    storages = Counter(
        (value.device, value.untyped_storage()._cdata)
        for _, value in (*model.named_parameters(remove_duplicate=False), *model.named_buffers(remove_duplicate=False))
    )
    if (
        state.get("persistent_targets")
        or not Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET
        or len(members) != state["geometry"]["tp_size"]
        or len(members) < 2
        or not 0 <= tp_rank < len(members)
        or len({(member["hostname"], member["boot_id"]) for member in members}) != 1
        or len({member["pid"] for member in members}) != len(members)
        or len({member["gpu_uuid"] for member in members}) != len(members)
        or members[tp_rank] != state["delivery"]["receiver"]
        or worker_identity(members[tp_rank]["generation"], torch.device("cuda", torch.cuda.current_device()))
        != members[tp_rank]
        or not names
        or names != sorted(config["layouts"])
        or names != sorted(config["signatures"])
        or not set(names) <= set(state["geometry"].get("ordinary_targets", {})) & set(replica_parameters(parameters))
    ):
        raise RuntimeError("TP replica configuration differs from actual native workers or parameters")
    remote = set(state["delivery"]["routes"]["nixl"])
    for name in names:
        value = parameters[name]
        writes = [write for write in state["writes"] if write["param"] == name]
        if (
            storages[value.device, value.untyped_storage()._cdata] != 1
            or config["layouts"][name] != dict(nbytes=value.numel() * 2)
            or value.numel() % len(members)
            or not writes
            or any(write["id"] not in remote for write in writes)
            or replica_write_signature(writes) != config["signatures"][name]
        ):
            raise RuntimeError("TP replica source, alias or equal output chunk changed")


def _pinned_remote_spans(parameters: dict[str, torch.Tensor], writes: list[dict], remote_ids: set[str]) -> dict:
    spans = {}
    for write in writes:
        if write["id"] not in remote_ids:
            continue
        name = write["param"]
        begin = write["dst_offset"]
        end = begin + (write["rows"] - 1) * write["dst_stride"] + write["row_bytes"]
        if name not in parameters or begin < 0 or end > parameters[name].numel() * parameters[name].element_size():
            raise RuntimeError("Colocate NIXL remote span exceeds its native parameter")
        previous = spans.get(name, (begin, end))
        spans[name] = (min(begin, previous[0]), max(end, previous[1]))
    if set(spans) != set(parameters):
        raise RuntimeError("Colocate NIXL remote span inventory changed")
    return spans


def _native_pinned_spans(parameters: dict[str, torch.Tensor], writes: list[dict], remote_ids: set[str]) -> dict | None:
    """Select spans with no local holes after full-plan coverage validation.

    The validated plan has disjoint destinations. Remote byte counts equal to
    their bounding span therefore prove every byte can be compared before
    overwrite, without destroying any local write's native reference.
    """
    sizes = {}
    for write in writes:
        if write["id"] in remote_ids:
            name = write["param"]
            sizes[name] = sizes.get(name, 0) + write["rows"] * write["row_bytes"]
    if not sizes:
        return None
    spans = _pinned_remote_spans({name: parameters[name] for name in sizes}, writes, remote_ids)
    return (
        spans
        if all(not (begin % 2 or end % 2) and sizes[name] == end - begin for name, (begin, end) in spans.items())
        else None
    )


def _prefetch_pinned_targets(state: dict, parameters: dict[str, torch.Tensor]) -> None:
    """Create one registration for the cold plan's remote spans only."""
    if state.get("persistent_targets") or not Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET or "nixl_agent" not in state:
        return
    from concurrent.futures import ThreadPoolExecutor

    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import PersistentPinnedTargets

    writes = _remote_writes(state)
    remote = {write["id"] for write in writes}
    if not writes:
        return
    if "pinned_future" in state or "pinned_targets" in state:
        raise RuntimeError("Pinned targets are already initialized for this cold plan")
    names = {write["param"] for write in writes}
    targets = {name: value for name, value in parameters.items() if name in names}
    spans = _pinned_remote_spans(targets, writes, remote)
    # The worker must not retain CUDA tensors after SGLang offloads its model.
    sizes = {name: value.numel() * value.element_size() for name, value in targets.items()}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2p-pinned-target")
    state["pinned_pool"] = pool
    state["pinned_future"] = pool.submit(
        PersistentPinnedTargets,
        state["nixl_agent"],
        sizes=sizes,
        spans=spans,
        writes=writes if state.get("target_encoding") is not None else None,
    )


def _finish_pinned_prefetch(state: dict) -> Any:
    """Drain the one cold-plan task before bind or safe session retirement."""
    future = state.get("pinned_future")
    if future is not None:
        try:
            pinned = future.result()
        finally:
            state["pinned_pool"].shutdown(wait=True)
        state["pinned_targets"] = pinned
        del state["pinned_future"], state["pinned_pool"]
    return state.get("pinned_targets")


def refresh_native_metadata(model: torch.nn.Module, geometry: dict) -> None:
    """Preserve a cold-qualified native lazy-cache invalidation after
    writes."""
    if geometry.get("native_refresh", False):
        model.load_weights([])


def validate_expert_targets(writes: list[dict], parameters: dict, geometry: dict) -> None:
    """Adapt native expert names and shapes to the shared storage contract."""
    validate_targets(writes, parameters, target_shapes(geometry), source_world_size=geometry["source_world_size"])


def _digests(parameters: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        name: hashlib.sha256(parameter.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, parameter in parameters.items()
    }


def _delivery_identity(command: dict) -> dict:
    return {
        **{key: command[key] for key in ("session", "engine", "tp_rank", "version", "nonce")},
        **({"target_encoding": command["target_encoding"]} if command.get("target_encoding") is not None else {}),
    }


def _prepare_delivery(model: torch.nn.Module, command: dict, tp_rank: int) -> dict:
    """Read-only cold negotiation against the native seed."""
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import validate_consistency_check

    validate_consistency_check(command, audit_key="audit_native")
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import PACKED_TARGET_ENCODING

    if command.get("target_encoding") not in (None, PACKED_TARGET_ENCODING) or (
        command.get("target_encoding") is not None and command.get("persistent_targets")
    ):
        raise ValueError("Unsupported remote target encoding")
    if (
        command.get("tp_rank") != tp_rank
        or type(command.get("version")) is not int
        or command["version"] < 0
        or type(command.get("engine")) is not int
        or command["engine"] < 0
        or any(not isinstance(command.get(key), str) or not command[key] for key in ("session", "nonce"))
    ):
        raise ValueError("Invalid expert delivery identity")
    parameters = _parameters(model, command["geometry"])
    validate_expert_targets(command["writes"], parameters, command["geometry"])
    devices = {value.device for value in parameters.values()}
    if len(devices) != 1:
        raise ValueError("Expert delivery requires one actual receiver device")
    binding = bind_delivery(
        command["writes"],
        command["geometry"],
        command["sources"],
        worker_identity(uuid4().hex, next(iter(devices))),
        layouts={name: dict(nbytes=prod(shape) * 2) for name, shape in target_shapes(command["geometry"]).items()},
    )
    if binding["logical_plan_hash"] != command["plan_hash"]:
        raise ValueError("Expert delivery logical checksum differs")
    reply = dict(identity=_delivery_identity(command), **binding)
    if command.get("persistent_targets") or (
        command.get("describe_tp_replicas") is True and Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET
    ):
        from .replica_weight_sync import replica_parameters

        reply["replica_parameters"] = replica_parameters(parameters)
    return reply


def validate_epoch_continuation(guard, command: dict | None) -> None:
    """Allow only this epoch's apply RPC through the native allocation
    guard."""
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import layout_of

    from .weight_sync_guard import _native_models

    epoch = guard.colocate_owner
    model = epoch["model"]
    state = getattr(model, "_relax_expert_ipc", None)
    if (
        command is None
        or command.get("op") not in ("apply", "replica_ready", "replica_gather")
        or tuple(_native_models(guard.manager)) != guard.models
        or len(guard.models) != 1
        or guard.models[0] is not model
        or state is None
        or state.get("epoch") is not epoch
        or guard.lease.failed
        or guard.lease.active != epoch["identity"]
        or guard.lease.pending != {epoch["token"]}
        or any(command.get(key) != state[key] for key in ("session", "engine"))
        or command.get("version") != epoch["version"]
        or command.get("delivery_hash") != state["delivery_hash"]
        or command.get("epoch_generation") != epoch["generation"]
    ):
        raise RuntimeError("Colocate NIXL allocation has an unrelated native request")
    registration = epoch["registration"]
    current = _parameters(model, state["geometry"])
    if any(current.get(name) is not value for name, value in epoch["parameters"].items()):
        raise RuntimeError("Colocate NIXL target parameter was replaced during registration")
    if (
        "gpu_layouts" in epoch
        and {name: layout_of(value) for name, value in current.items() if name in epoch["parameters"]}
        != epoch["gpu_layouts"]
    ):
        raise RuntimeError("Colocate NIXL target storage changed during the allocation epoch")
    if registration is not None:
        registration.validate({name: current[name] for name in epoch["parameters"]})


def _prefetch_persistent_targets(state: dict, parameters: dict) -> None:
    """Register actual resident targets while replica setup and fallback
    run."""
    if not state.get("persistent_targets") or "nixl_agent" not in state:
        return
    if "persistent_registration" in state:
        raise RuntimeError("Persistent registration is already owned by this session")
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import RegisteredMemory

    names = {write["param"] for write in _remote_writes(state)}
    if not names:
        return
    targets = {name: parameters[name] for name in names}
    registration = RegisteredMemory(state["nixl_agent"], targets)
    state.update(
        persistent_registration=registration,
        persistent_parameters=targets,
        persistent_storage_ids={name: value.untyped_storage()._cdata for name, value in targets.items()},
        persistent_registration_key=uuid4().hex,
    )
    registration.register(batched=True, asynchronous=True)


def _prepare_persistent_targets(model: torch.nn.Module, command: dict, binding: dict) -> None:
    """Start the initial resident registration after the exact plan agrees."""
    if (
        command.get("persistent_targets") is not True
        or getattr(model, "_relax_expert_ipc", None) is not None
        or not binding["delivery"]["routes"]["nixl"]
    ):
        return
    if getattr(model, "_relax_persistent_target_prefetch", None) is not None:
        raise RuntimeError("A prepared resident registration must be retired before replacement")
    state = dict(
        persistent_targets=True,
        nixl_agent=_probe_nixl_agent(model),
        delivery=binding["delivery"],
        writes=command["writes"],
    )
    # Keep live resources outside the JSON delivery identity, and own them
    # before registration can fail. No writes are possible before seed/bind.
    model._relax_persistent_target_prefetch = dict(binding=model._relax_expert_delivery, state=state)
    _prefetch_persistent_targets(state, _parameters(model, command["geometry"]))


def _adopt_persistent_targets(model: torch.nn.Module, state: dict, parameters: dict) -> bool:
    """Transfer the checked prepare's registration without waiting for it."""
    pending = getattr(model, "_relax_persistent_target_prefetch", None)
    if pending is None:
        return False
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import layout_of

    prepared = pending["state"]
    owners = prepared["persistent_parameters"]
    if (
        not state["persistent_targets"]
        or pending["binding"] is not model._relax_expert_delivery
        or state["nixl_agent"] is not prepared["nixl_agent"]
        or any(parameters.get(name) is not value for name, value in owners.items())
        or {name: layout_of(parameters[name]) for name in owners} != prepared["persistent_registration"].layouts
        or {name: value.untyped_storage()._cdata for name, value in owners.items()}
        != prepared["persistent_storage_ids"]
    ):
        raise RuntimeError("Prepared resident registration differs from the accepted native seed")
    for key in (
        "persistent_registration",
        "persistent_parameters",
        "persistent_storage_ids",
        "persistent_registration_key",
    ):
        state[key] = prepared[key]
    del model._relax_persistent_target_prefetch
    return True


def retire_persistent_targets(model: torch.nn.Module) -> None:
    """Retire an idle registration before native reload or allocation
    changes."""
    pending = getattr(model, "_relax_persistent_target_prefetch", None)
    if pending is not None:
        registration = pending["state"].get("persistent_registration")
        if registration is not None:
            registration.release()
        del model._relax_persistent_target_prefetch
    state = getattr(model, "_relax_expert_ipc", None)
    if state is None or not state.get("persistent_targets"):
        return
    if state.get("epoch") is not None or state["failed"]:
        raise RuntimeError("Cannot retire a pending or uncertain direct transfer")
    registration = state.get("persistent_registration")
    if registration is not None:
        registration.release()
        state.pop("persistent_registration")
    for key in (
        "persistent_parameters",
        "persistent_storage_ids",
        "persistent_registration_key",
        "persistent_agent_metadata",
    ):
        state.pop(key, None)


def refresh_persistent_targets(model: torch.nn.Module) -> None:
    """Keep registrations only when native post-processing preserved
    targets."""
    state = getattr(model, "_relax_expert_ipc", None)
    previous = state.get("persistent_registration") if state is not None else None
    if previous is None:
        return
    # Native post-processing may write in place without changing storage.
    # NIXL does not inherit the scheduler's CUDA stream ordering.
    if any(value.is_cuda for value in previous.parameters.values()):
        torch.cuda.synchronize()
    owners = state["persistent_parameters"]
    parameters = {name: value for name, value in model.named_parameters() if name in owners}
    if (
        {name: value.untyped_storage()._cdata for name, value in parameters.items()} != state["persistent_storage_ids"]
        or set(parameters) != set(owners)
        or any(owners[name] is not value for name, value in parameters.items())
    ):
        retire_persistent_targets(model)
    else:
        previous.validate(parameters)


def _bind_epoch(model: torch.nn.Module, command: dict, tp_rank: int) -> dict:
    """Register only remote target parameters in this fresh onload epoch."""
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import layout_of
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import RegisteredMemory

    from .weight_sync_guard import native_weight_lease

    state = getattr(model, "_relax_expert_ipc", None)
    if (
        state is None
        or state["failed"]
        or state.get("epoch") is not None
        or command["session"] != state["session"]
        or command["engine"] != state["engine"]
        or command["version"] != state["version"] + 1
        or command["delivery_hash"] != state.get("delivery_hash")
        or command.get("target_encoding") != state.get("target_encoding")
        or "delivery" not in state
    ):
        raise RuntimeError("Colocate NIXL epoch has no matching remote delivery")
    guard = model._relax_native_weight_guard
    if guard.colocate_owner is not None or len(guard.models) != 1:
        raise RuntimeError("Colocate NIXL requires one guarded native model")
    remote = set(state["delivery"]["routes"]["nixl"])
    names = {write["param"] for write in state["writes"] if write["id"] in remote}
    parameters = {name: value for name, value in _parameters(model, state["geometry"]).items() if name in names}
    if set(parameters) != names or any(
        not value.is_cuda or value.dtype != torch.bfloat16 for value in parameters.values()
    ):
        raise RuntimeError("Colocate NIXL target storage differs from the native seed")
    agent = state.get("nixl_agent") if parameters else None
    if parameters and agent is None:
        raise RuntimeError("Colocate NIXL session has no negotiated receiver agent")
    persistent = bool(state.get("persistent_targets"))
    use_pinned = agent is not None and not persistent and Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET
    _validate_tp_replicas(model, state, _parameters(model, state["geometry"]), tp_rank)
    remote_writes = _remote_writes(state)
    spans = (
        _pinned_remote_spans(parameters, remote_writes, {write["id"] for write in remote_writes}) if use_pinned else {}
    )
    registration = RegisteredMemory(agent, parameters) if agent is not None and not use_pinned else None
    if persistent and registration is not None:
        previous = state.get("persistent_registration")
        identities = {name: value.untyped_storage()._cdata for name, value in parameters.items()}
        owners = state.get("persistent_parameters", {})
        if previous is not None and (
            state["persistent_storage_ids"] != identities
            or set(owners) != set(parameters)
            or any(owners[name] is not value for name, value in parameters.items())
        ):
            retire_persistent_targets(model)
            previous = None
        if previous is None or "persistent_agent_metadata" not in state:
            if "replicas" in state:
                from .replica_weight_sync import replica_parameters

                if not set(state["replicas"]["config"]["parameters"]) <= set(replica_parameters(parameters)):
                    raise RuntimeError("Replica alias layout changed; native reseeding is required")
        if previous is not None:
            previous.validate(parameters)
            registration = previous
        else:
            registration.register(batched=True)
            state.update(
                persistent_registration=registration,
                persistent_parameters=parameters,
                persistent_storage_ids=identities,
                persistent_registration_key=uuid4().hex,
            )
        if "persistent_agent_metadata" not in state:
            state["persistent_agent_metadata"] = base64.b64encode(agent.get_agent_metadata()).decode()
    gpu_layouts = {name: layout_of(value) for name, value in parameters.items()}
    generation = uuid4().hex
    lease = native_weight_lease(model)
    identity = lease.begin(state["session"], command["version"], state["delivery_hash"])
    token = (state["engine"], tp_rank, generation)
    lease.add(token, command["version"])
    epoch = dict(
        model=model,
        identity=identity,
        token=token,
        version=command["version"],
        generation=generation,
        parameters=parameters,
        gpu_layouts=gpu_layouts,
        registration=registration,
        pinned=None,
        ready=False,
    )
    state["epoch"] = guard.colocate_owner = epoch
    if use_pinned:
        pinned = _finish_pinned_prefetch(state)
        if pinned is None:
            raise RuntimeError("Pinned targets were not prepared by the cold plan")
        pinned.validate(parameters, spans=spans)
        epoch["pinned"] = pinned
    if registration is not None and not persistent:
        registration.register(batched=True)
    pinned = epoch["pinned"]
    return dict(
        generation=generation,
        version=command["version"],
        tp_rank=tp_rank,
        gpu_id=torch.cuda.current_device(),
        target_memory_type="DRAM" if pinned is not None else "VRAM",
        target_device_id=0 if pinned is not None else torch.cuda.current_device(),
        persistent_registration_key=(
            pinned.registration_key if pinned is not None else state.get("persistent_registration_key")
        ),
        **(
            dict(target_encoding=state["target_encoding"], packed_regions_hash=pinned.regions_hash)
            if pinned is not None and pinned.regions is not None
            else {}
        ),
        agent_metadata=(
            state["persistent_agent_metadata"]
            if persistent and agent is not None
            else base64.b64encode(agent.get_agent_metadata()).decode()
            if agent is not None
            else None
        ),
        params=pinned.layouts(parameters) if pinned is not None else gpu_layouts,
    )


def _accept_delivery(model: torch.nn.Module, command: dict) -> dict | None:
    binding = command.get("delivery")
    if binding is None:
        if (
            getattr(model, "_relax_expert_delivery", None) is not None
            or getattr(model, "_relax_expert_ipc", {}).get("delivery_hash") is not None
        ):
            raise ValueError("Prepared expert delivery cannot be omitted from its seed")
        return None
    pending = getattr(model, "_relax_expert_delivery", None)
    if pending is None or pending["identity"] != _delivery_identity(command) or pending["binding"] != binding:
        raise ValueError("Expert delivery was not prepared by this receiver")
    return json.loads(json.dumps(binding))


def _copy_mode(command: dict) -> tuple[bool, bool]:
    batched, cached = command.get("batched", False), command.get("cache_plan", False)
    if type(batched) is not bool:
        raise ValueError("Expert IPC batching must be an explicit boolean")
    if type(cached) is not bool or cached and not batched:
        raise ValueError("Expert IPC plan caching requires explicit batched mode")
    if type(command.get("audit_native", False)) is not bool:
        raise ValueError("Expert native audit must be an explicit boolean")
    return cached, batched


def install_receiver_initialization() -> None:
    """Overlap transport setup with loading, before native KV pool sizing."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import wraps

    from sglang.srt.model_executor.model_runner import ModelRunner

    original = ModelRunner.initialize
    if getattr(original, "_relax_receiver_initialization", False):
        return

    @wraps(original)
    def initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.device != "cuda":
            return original(self, *args, **kwargs)
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import create_agent

        device = torch.cuda.current_device()

        def create() -> Any:
            torch.cuda.set_device(device)
            return create_agent()

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2p-receiver")
        future = pool.submit(create)
        pool.shutdown(wait=False)
        # Keep model startup errors unchanged. The future captures no runner,
        # model, weights or registration, even if initialization fails.
        result = original(self, *args, **kwargs)
        self.model._relax_nixl_probe_future = future
        prewarm = getattr(self.model, "_prewarm_mhc_kernels", None)
        if callable(prewarm):
            # This native hook has a TP barrier and must stay on every worker's
            # main thread. It only compiles kernels; native sync still loads weights.
            prewarm()
        return result

    initialize._relax_receiver_initialization = True
    ModelRunner.initialize = initialize


def _probe_nixl_agent(model: torch.nn.Module) -> Any:
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import create_agent

    agent = getattr(model, "_relax_nixl_probe_agent", None)
    if agent is None:
        future = getattr(model, "_relax_nixl_probe_future", None)
        # Keep failed initialization in the future: the original probe converts
        # this into unsupported capability, without retrying or leasing weights.
        agent = future.result() if future is not None else create_agent()
        if future is not None:
            del model._relax_nixl_probe_future
        model._relax_nixl_probe_agent = agent
    return agent


def _receiver_nixl_agent(model: torch.nn.Module) -> Any:
    _probe_nixl_agent(model)
    return model.__dict__.pop("_relax_nixl_probe_agent")


@torch.no_grad()
def apply_command(model: torch.nn.Module, command: dict, tp_rank: int) -> None:
    """Validate session and monotonic version before any source import."""
    if "tp" in command or "remote" in command:
        raise ValueError("This receiver supports local expert delivery only")
    if "relay" in command or "relay_plan_hash" in command:
        raise ValueError("Receiver relay is an archived experimental protocol")
    if (
        command.get("tp_rank") != tp_rank
        or type(command.get("version")) is not int
        or command["version"] < 0
        or not isinstance(command.get("session"), str)
        or not command["session"]
        or not isinstance(command.get("nonce"), str)
        or not command["nonce"]
    ):
        raise ValueError("Invalid expert IPC target or transfer identity")
    if command["op"] == "seed":
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import validate_consistency_check

        validate_consistency_check(command, audit_key="audit_native")
    state = getattr(model, "_relax_expert_ipc", None)
    geometry = command.get("geometry", {}) if command["op"] == "seed" else state["geometry"] if state else {}
    parameters = _parameters(model, geometry)
    if command["op"] == "seed":
        if state is not None and (
            state["failed"]
            or state.get("seed_pending", False)
            or state.get("reference") is not None
            or command["session"] != state["session"]
            or command["engine"] != state["engine"]
            or tp_rank != state["tp_rank"]
            or command.get("previous_version") != state["version"]
            or command["version"] != state["version"] + 1
            or command["nonce"] == state["nonce"]
        ):
            raise RuntimeError("An expert IPC session is already bound or cannot be safely reseeded")
        cached, batched = _copy_mode(command)
        if type(command.get("persistent_targets", False)) is not bool:
            raise ValueError("Persistent targets require an explicit boolean")
        audit_native = command.get("audit_native", False)
        # Own the seed description; later commands cannot mutate a compiled
        # plan or the geometry used for fresh native parameter qualification.
        description = json.loads(json.dumps(dict(geometry=command["geometry"], writes=command["writes"])))
        geometry, writes = description["geometry"], description["writes"]
        if (
            type(geometry.get("tp_size")) is not int
            or geometry["tp_size"] <= 0
            or type(geometry.get("source_world_size")) is not int
            or geometry["source_world_size"] <= 0
            or type(tp_rank) is not int
            or not 0 <= tp_rank < geometry["tp_size"]
        ):
            raise ValueError("Expert IPC requires valid source and receiver coordinates")
        validate_expert_targets(writes, parameters, geometry)
        plan_hash = fingerprint(dict(geometry=geometry, writes=writes))
        binding = _accept_delivery(model, command)
        if command["plan_hash"] != plan_hash or binding is not None and binding["logical_plan_hash"] != plan_hash:
            raise ValueError("Expert IPC seed plan checksum differs")
        compiled_plan = None
        if cached:
            compiled_plan = CompiledIPC(
                parameters,
                writes,
                geometry,
                target_shapes=target_shapes(geometry),
                owner_rotation=command["engine"] * geometry["tp_size"] + tp_rank,
                write_ids=frozenset(binding["delivery"]["routes"]["cuda_ipc"]) if binding is not None else None,
            )
            if compiled_plan.plan_hash != plan_hash:
                raise ValueError("Compiled expert IPC plan checksum differs")
        torch.cuda.synchronize()
        audit_parameters = dict(parameters)
        # The compiled copy compares every source bit with the still-intact
        # native destination before writing. Unbatched paths keep hash/poison.
        remote_ids = set(binding["delivery"]["routes"]["nixl"]) if binding else set()
        remote_writes = [write for write in writes if write["id"] in remote_ids]
        if command.get("tp_replicas") is not None:
            replica_state = dict(
                writes=writes,
                delivery=binding["delivery"],
                geometry=geometry,
                tp_rank=tp_rank,
                persistent_targets=command.get("persistent_targets", False),
                tp_replicas=command["tp_replicas"],
            )
            _validate_tp_replicas(model, replica_state, parameters, tp_rank)
            remote_writes = _remote_writes(replica_state)
        native_pinned_spans = (
            _native_pinned_spans(parameters, remote_writes, {write["id"] for write in remote_writes})
            if audit_native
            and compiled_plan is not None
            and remote_ids
            and not command.get("persistent_targets")
            and command.get("tp_replicas") is None
            and Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET
            else None
        )
        reference = (
            (
                "native"
                if compiled_plan is not None
                and command.get("tp_replicas") is None
                and (not remote_ids or native_pinned_spans is not None)
                else _digests(audit_parameters)
            )
            if audit_native
            else None
        )
        if state is not None:
            from .replica_weight_sync import retire_replica_group

            retire_replica_group(state)
            retire_persistent_targets(model)
            pinned = _finish_pinned_prefetch(state)
            if pinned is not None:
                pinned.registration.release()
        state = dict(
            session=command["session"],
            engine=command["engine"],
            tp_rank=tp_rank,
            version=command["version"] - 1,
            nonce=command["nonce"],
            geometry=geometry,
            writes=writes,
            plan_hash=plan_hash,
            reference=reference,
            native_pinned_spans=native_pinned_spans,
            seed_pending=True,
            failed=True,
            batched=batched,
            persistent_targets=command.get("persistent_targets", False),
            target_encoding=command.get("target_encoding"),
            **(
                {"tp_replicas": json.loads(json.dumps(command["tp_replicas"]))}
                if command.get("tp_replicas") is not None
                else {}
            ),
        )
        model._relax_expert_ipc = state
        if binding is not None and binding["delivery"]["routes"]["nixl"]:
            state["nixl_agent"] = _receiver_nixl_agent(model)
        if compiled_plan is not None:
            state["compiled_plan"] = compiled_plan
        if reference is not None and reference != "native":
            for parameter in audit_parameters.values():
                parameter.fill_(float("nan"))
        torch.cuda.synchronize()
        state["failed"] = False
        if binding is not None:
            state["delivery"] = binding["delivery"]
            state["delivery_hash"] = binding["plan_hash"]
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.plan import ReceiptValidator

            state["remote_receipt_validator"] = ReceiptValidator(dict(writes=_remote_writes(state)))
            _prefetch_pinned_targets(state, parameters)
            if not _adopt_persistent_targets(model, state, parameters):
                _prefetch_persistent_targets(state, parameters)
            del model._relax_expert_delivery
        return
    if command["op"] != "apply":
        raise ValueError("Unknown expert IPC operation")
    if (
        state is None
        or state["failed"]
        or command["session"] != state["session"]
        or command["engine"] != state["engine"]
        or tp_rank != state["tp_rank"]
        or command["version"] != state["version"] + 1
        or command["nonce"] == state["nonce"]
        or command["plan_hash"] != state["plan_hash"]
        or command.get("delivery_hash") != state.get("delivery_hash")
    ):
        raise RuntimeError("Stale, failed or misrouted expert IPC update")
    state["failed"] = True
    native_pinned_spans = state.get("native_pinned_spans") if state.get("reference") == "native" else None
    epoch = state.get("epoch")
    if epoch is not None:
        validate_epoch_continuation(model._relax_native_weight_guard, command)
        from relax.distributed.checkpoint_service.backends.shard_p2p.remote.rpc_receipts import index_receipts

        receipts = command.get("source_receipts")
        if not isinstance(receipts, list):
            raise RuntimeError("Colocate NIXL requires actual owner completion receipts")
        state["remote_receipt_validator"].validate(index_receipts(receipts), epoch["generation"], command["version"])
        from .replica_weight_sync import validate_replica_completion

        validate_replica_completion(state, receipts)
        if native_pinned_spans is not None:
            if epoch["pinned"] is None:
                raise RuntimeError("Native remote validation requires the qualified pinned target spans")
            epoch["pinned"].validate(epoch["parameters"], spans=native_pinned_spans)
            epoch["pinned"].copy_to(epoch["parameters"], verify_native=True)
        elif epoch["pinned"] is not None:
            epoch["pinned"].copy_to(epoch["parameters"])
    elif state.get("delivery", {}).get("routes", {}).get("nixl"):
        raise RuntimeError("Colocate NIXL update has no registered allocation epoch")
    bindings = {}
    for item in command["bindings"]:
        key = (item["source_rank"], item["source"])
        if key in bindings:
            raise ValueError("Duplicate expert IPC source binding")
        bindings[key] = item["payload"]
    compiled_plan = state.get("compiled_plan")
    if compiled_plan is not None:
        if compiled_plan.plan_hash != state["plan_hash"]:
            raise RuntimeError("Compiled expert IPC plan identity changed")
        # A remote-only plan has already checked every byte above. Empty
        # local plans must not invoke the IPC verifier's missing-tile guard.
        verify_local = state.get("reference") == "native" and (
            native_pinned_spans is None or bool(state["delivery"]["routes"]["cuda_ipc"])
        )
        compiled_plan.apply(parameters, bindings, verify_native=verify_local)
    else:
        apply_ipc_plan(
            parameters,
            state["writes"],
            bindings,
            state["geometry"],
            target_shapes=target_shapes(state["geometry"]),
            batched=state["batched"],
            owner_rotation=state["engine"] * geometry["tp_size"] + tp_rank,
            write_ids=frozenset(state["delivery"]["routes"]["cuda_ipc"]) if "delivery" in state else None,
        )
    if state.get("tp_replicas") is None:
        _finish_apply(model, command, tp_rank, parameters)


@torch.no_grad()
def _finish_apply(model: torch.nn.Module, command: dict, tp_rank: int, parameters: dict) -> None:
    """Audit and complete only after all output bytes are reconstructed."""
    state = model._relax_expert_ipc
    epoch, compiled_plan = state.get("epoch"), state.get("compiled_plan")
    native_pinned_spans = state.get("native_pinned_spans") if state.get("reference") == "native" else None
    cold_reference = state.get("reference")
    if cold_reference is not None and cold_reference != "native":
        actual = _digests(parameters)
        mismatched = sorted(
            name for name in actual.keys() | cold_reference.keys() if actual.get(name) != cold_reference.get(name)
        )
        if mismatched:
            raise RuntimeError(
                "Expert IPC differs from the native-loader reference: "
                f"engine={state['engine']} tp={tp_rank} "
                f"mismatched={len(mismatched)}/{len(cold_reference)} first={mismatched[:6]}"
            )
    if cold_reference is not None:
        del state["reference"]
        state.pop("native_pinned_spans", None)
        logger.info(
            "COLOCATE_EXPERT_IPC_AUDIT engine=%d tp=%d version=%d parameters=%d exact=True method=%s",
            state["engine"],
            tp_rank,
            command["version"],
            len(parameters),
            "native-pinned"
            if native_pinned_spans is not None
            else "native-ipc"
            if cold_reference == "native"
            else "sha256",
        )
        if compiled_plan is not None:
            logger.info(
                "COLOCATE_EXPERT_IPC_COMPILED engine=%d tp=%d version=%d rectangles=%d cpu_template_bytes=%d",
                state["engine"],
                tp_rank,
                command["version"],
                *compiled_plan.template_statistics,
            )
    refresh_native_metadata(model, state["geometry"])
    if epoch is not None:
        validate_epoch_continuation(model._relax_native_weight_guard, command)
        torch.cuda.synchronize()
        if epoch["registration"] is not None and not state.get("persistent_targets"):
            epoch["registration"].release()
        model._relax_native_weight_lease.complete(epoch["token"])
        epoch["ready"] = True
        return
    state.pop("reference", None)
    state.update(version=command["version"], nonce=command["nonce"], failed=False, seed_pending=False)


def _complete_epoch(model: torch.nn.Module, command: dict, group) -> None:
    from .weight_sync_guard import commit_native_weight_lease

    state = model._relax_expert_ipc
    epoch = state.get("epoch")
    if epoch is None or not epoch["ready"] or epoch["version"] != command["version"]:
        raise RuntimeError("Colocate NIXL completion has no fenced allocation epoch")
    commit_native_weight_lease(model, epoch["identity"], group)
    model._relax_native_weight_guard.colocate_owner = None
    state.pop("epoch")
    state.pop("reference", None)
    state.update(version=command["version"], nonce=command["nonce"], failed=False, seed_pending=False)


def _decode_command(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]], args) -> dict:
    if (
        type(args.tp_size) is not int
        or args.tp_size <= 0
        or args.pp_size != 1
        or args.enable_lora
        or args.speculative_algorithm is not None
    ):
        raise ValueError("Shard receiver requires managed TP/PP1 without LoRA or speculative decoding")
    if len(weights) != 1 or weights[0][1].numel() != 0:
        raise ValueError("Expert IPC control requires exactly one empty tensor")
    command = json.loads(weights[0][0])
    if "prepared_delivery_hash" in command:
        pending = getattr(model, "_relax_expert_delivery", None)
        if (
            command["op"] != "seed"
            or pending is None
            or any(key in command for key in ("geometry", "writes", "delivery"))
            or pending["identity"] != _delivery_identity(command)
            or pending["binding"]["logical_plan_hash"] != command["plan_hash"]
            or pending["binding"]["plan_hash"] != command["prepared_delivery_hash"]
        ):
            raise ValueError("Expert delivery reference was not prepared by this receiver")
        command.update(geometry=pending["geometry"], writes=pending["writes"], delivery=pending["binding"])
    geometry = (
        command.get("geometry", {})
        if command["op"] in ("describe_tp", "prepare", "seed", "probe_nixl", "retire")
        else getattr(model, "_relax_expert_ipc", {}).get("geometry", {})
    )
    if geometry.get("tp_size") != args.tp_size:
        raise ValueError("Expert IPC layout differs from the running receiver TP")
    if (
        command["op"] != "probe_nixl"
        and geometry.get("expert_enabled", True)
        and (args.dp_size != 1 or args.ep_size != 1 or args.quantization is not None or args.enable_dp_attention)
    ):
        raise ValueError("Direct experts require plain BF16 TP/DP1/EP1")
    return command


def control(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
    # A failed warm loader must not strand successful TP peers at SGLang's
    # outer completion barrier. Decode failures join the same agreement too.
    control_tp(model, weights)


def control_remote(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
    """Use the checked allocation epoch only for cross-node shard rounds."""
    control_tp(model, weights)


def control_tp(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel, get_server_args

    _control_tp(model, weights, get_server_args(), get_parallel().tp_rank, get_tp_group().cpu_group)


def _control_tp(
    model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]], args, tp_rank: int, group
) -> None:
    """Agree before mutation and after completion, including decoding
    failures."""
    import torch.distributed as dist

    guard = getattr(model, "_relax_native_weight_guard", None)
    if guard is not None and getattr(guard, "current_control", None) is not None:
        guard.control_entry_seen = True

    def agree(error, header=None):
        try:
            errors = [None] * dist.get_world_size(group)
            dist.all_gather_object(errors, (error, header), group=group)
            if any(item[0] is not None or item[1] != errors[0][1] for item in errors):
                raise RuntimeError(f"Colocate expert receiver failed: {errors}")
        except Exception:
            state = getattr(model, "_relax_expert_ipc", None)
            if state is not None:
                state["failed"] = True
            lease = getattr(model, "_relax_native_weight_lease", None)
            if lease is not None:
                lease.fail()
            raise

    error, header, command, delivery = None, None, None, None
    try:
        command = _decode_command(model, weights, args)
        if (
            command.get("tp_rank") != tp_rank
            or command.get("op")
            not in (
                "describe_tp",
                "prepare",
                "seed",
                "probe_nixl",
                "bind_epoch",
                "apply",
                "retire",
                "replica_bootstrap",
                "replica_configure",
                "replica_ready",
                "replica_gather",
            )
            or command.get("op") == "seed"
            and "delivery" not in command
            or "tp" in command
            or "remote" in command
        ):
            raise ValueError("Missing or misrouted cold expert command")
        header = tuple(command.get(key) for key in ("session", "engine", "op", "version", "previous_version"))
        header += ("delivery" in command,)
        if command["op"] in ("prepare", "seed"):
            from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import (
                validate_consistency_check,
            )

            validate_consistency_check(command, audit_key="audit_native")
            header += (command["consistency_check"], command.get("audit_native", False))
        if command["op"] == "seed":
            header += _copy_mode(command)
            header += (command.get("audit_native", False),)
            header += (fingerprint(command.get("tp_replicas")),)
        if command["op"] == "describe_tp":
            if command.get("description_hash") != fingerprint(command["description"]):
                raise ValueError("Ordinary TP description checksum differs")
        elif command["op"] == "prepare":
            delivery = _prepare_delivery(model, command, tp_rank)
        elif command["op"] == "seed":
            _accept_delivery(model, command)
        elif command["op"] == "retire":
            state = getattr(model, "_relax_expert_ipc", None)
            if state is not None and (
                not state.get("persistent_targets")
                or state.get("failed")
                or state.get("epoch") is not None
                or state.get("seed_pending")
                or state.get("reference") is not None
                or any(command.get(key) != state[key] for key in ("session", "engine", "version", "tp_rank"))
            ):
                raise RuntimeError("Cannot retire an incomplete or unrelated Async session")
            pending = getattr(model, "_relax_persistent_target_prefetch", None)
            if pending is not None and any(
                command.get(key) != pending["binding"]["identity"][key]
                for key in ("session", "engine", "version", "tp_rank")
            ):
                raise RuntimeError("Cannot retire an unrelated prepared resident registration")
    except Exception as exc:
        error = str(exc)
    agree(error, header)
    error = None
    try:
        if command["op"] == "describe_tp":
            from .tp_shard_layout import describe_tp_targets

            try:
                reply = describe_tp_targets(model, command["description"], tp_rank)
            except NotImplementedError as exc:
                logger.info("Ordinary TP native fallback: %s", exc)
                reply = dict(
                    description_hash=command["description_hash"],
                    tp_rank=tp_rank,
                    selected_hf=[],
                    components=[],
                    target_shapes={},
                    target_hf={},
                    writes=[],
                    native_refresh=False,
                )
            reply.update({key: command[key] for key in ("session", "engine", "version", "nonce")})
            if command.get("describe_source_locality") is True:
                reply["receiver"] = worker_identity(
                    command["session"], torch.device("cuda", torch.cuda.current_device())
                )
        elif command["op"] == "prepare":
            reply = dict(
                identity=delivery["identity"],
                receiver=delivery["delivery"]["receiver"],
                logical_plan_hash=delivery["logical_plan_hash"],
                plan_hash=delivery["plan_hash"],
            )
            if "replica_parameters" in delivery:
                reply["replica_parameters"] = delivery["replica_parameters"]
            model._relax_expert_delivery = dict(
                identity=reply["identity"],
                binding={key: delivery[key] for key in ("logical_plan_hash", "plan_hash", "delivery")},
                geometry=command["geometry"],
                writes=command["writes"],
            )
            _prepare_persistent_targets(model, command, delivery)
        elif command["op"] == "retire":
            state = getattr(model, "_relax_expert_ipc", None)
            if state is not None:
                from .replica_weight_sync import retire_replica_group

                retire_replica_group(state)
            retire_persistent_targets(model)
            if state is not None:
                if "nixl_agent" in state:
                    model._relax_nixl_probe_agent = state["nixl_agent"]
                del model._relax_expert_ipc
            model.__dict__.pop("_relax_expert_delivery", None)
        elif command["op"] == "probe_nixl":
            try:
                guard = getattr(model, "_relax_native_weight_guard", None)
                if guard is None or len(guard.models) != 1 or guard.models[0] is not model:
                    raise NotImplementedError("Colocate NIXL requires one guarded native model")
                state = getattr(model, "_relax_expert_ipc", None)
                reuse = bool(command.get("persistent_targets") and state and state.get("nixl_agent"))
                if not reuse:
                    _probe_nixl_agent(model)
            except Exception as exc:
                # Unsupported UCX is a capability result, not an uncertain
                # native write. Do not quarantine an otherwise usable model.
                reply = dict(supported=False, reason=str(exc))
            else:
                from .native_expert_layout import describe_native_fp8_experts
                from .native_flat_bucket import native_flat_bucket_supported

                reply = dict(supported=True, native_flat_bucket_v1=native_flat_bucket_supported(model))

                try:
                    native_experts = describe_native_fp8_experts(model)
                except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                    logger.info("Native FP8 expert layout is unavailable: %s", exc)
                    native_experts = None
                if native_experts is not None:
                    reply["native_fp8_experts"] = native_experts
                if command.get("persistent_targets"):
                    layout = getattr(model, "_relax_async_shard_layout", None)
                    if layout is None:
                        from relax.utils.misc import get_hf_config

                        config = get_hf_config(args.model_path)
                        config = getattr(config, "text_config", None) or config
                        keys = (
                            "pp_size",
                            "tp_size",
                            "ep_size",
                            "moe_dp_size",
                            "dwdp_size",
                            "enable_eplb",
                            "ep_num_redundant_experts",
                            "init_expert_location",
                            "ep_join_mode",
                            "elastic_ep_backend",
                            "speculative_algorithm",
                            "speculative_draft_model_path",
                            "moe_runner_backend",
                            "json_model_override_args",
                            "quantization",
                            "enable_lora",
                            "disable_shared_experts_fusion",
                            "enforce_shared_experts_fusion",
                            "kt_weight_path",
                            "custom_weight_loader",
                        )
                        layout = {key: getattr(args, key) for key in keys if hasattr(args, key)}
                        layout.update(
                            is_cuda=torch.version.cuda is not None and torch.version.hip is None,
                            dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
                            num_experts=getattr(config, "num_experts", None)
                            or getattr(config, "n_routed_experts", None),
                            hidden_size=getattr(config, "hidden_size", None),
                            moe_intermediate_size=getattr(config, "moe_intermediate_size", None),
                            quantization_config=getattr(config, "quantization_config", None),
                            native_flat_bucket_v1=reply["native_flat_bucket_v1"],
                        )
                        model._relax_async_shard_layout = layout
                    reply.update(layout=layout, session=state["session"] if state else None)
        elif command["op"] == "bind_epoch":
            reply = _bind_epoch(model, command, tp_rank)
        elif command["op"].startswith("replica_"):
            from .replica_weight_sync import control_replica

            state = model._relax_expert_ipc
            reply = control_replica(model, command, _parameters(model, state["geometry"]))
        else:
            apply_command(model, command, tp_rank)
    except Exception as exc:
        logger.exception("Shard IPC receiver command failed: %s", command["op"])
        error = str(exc)
    agree(error)
    state = getattr(model, "_relax_expert_ipc", None)
    if command["op"] == "apply" and state is not None and state.get("tp_replicas") is not None:
        error, contract, views, tp_group = None, None, [], None
        try:
            from sglang.srt.distributed import get_tp_group

            validate_epoch_continuation(model._relax_native_weight_guard, command)
            parameters = _parameters(model, state["geometry"])
            _validate_tp_replicas(model, state, parameters, tp_rank)
            tp_group = get_tp_group()
            config = state["tp_replicas"]
            if tp_group.world_size != len(config["members"]) or tp_group.rank_in_group != tp_rank:
                raise RuntimeError("Replica reconstruction differs from the native TP device group")
            for name in config["parameters"]:
                output = parameters[name].detach().view(-1)
                width = output.numel() // tp_group.world_size
                views.append((output, output.narrow(0, tp_rank * width, width)))
            contract = (fingerprint(config), tuple((value.numel(), str(value.dtype)) for value, _ in views))
        except Exception as exc:
            error = str(exc)
        # No rank enters NCCL before all partial DONE receipts and native views
        # agree. The lease remains active, with its token pending, through the
        # final GPU fence and full-output audit.
        agree(error, contract)
        error = None
        try:
            for output, local in views:
                tp_group.all_gather_into_tensor(output, local)
            torch.cuda.synchronize()
            _finish_apply(model, command, tp_rank, parameters)
        except Exception as exc:
            error = str(exc)
        agree(error)
    if command["op"] == "apply" and "epoch_generation" in command:
        _complete_epoch(model, command, group)
    if command["op"] in ("describe_tp", "prepare", "probe_nixl", "bind_epoch", "replica_bootstrap", "replica_ready"):
        replies = [None] * dist.get_world_size(group)
        dist.all_gather_object(replies, reply, group=group)
        # The scheduler guard forwards this one-shot reply through SGLang's
        # existing success/message response, after the native RPC completes.
        key = {
            "describe_tp": "shard_p2p_tp_targets",
            "prepare": "shard_p2p_receivers",
            "probe_nixl": "shard_p2p_nixl_capabilities",
            "bind_epoch": "shard_p2p_epochs",
            "replica_bootstrap": "shard_p2p_replica_bootstrap",
            "replica_ready": "shard_p2p_replica_ready",
        }[command["op"]]
        model._relax_weight_sync_reply = json.dumps({key: replies}, separators=(",", ":"))
