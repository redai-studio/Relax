# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Versioned local expert updates with native seed parity and cold TP error
agreement."""

import hashlib
import json
import re
from collections.abc import Sequence
from math import prod
from uuid import uuid4

import torch

from relax.backends.sglang.weight_loader import EXPERT_IPC_LOADER as EXPERT_IPC_LOADER
from relax.backends.sglang.weight_loader import TP_IPC_LOADER as TP_IPC_LOADER
from relax.distributed.checkpoint_service.backends.shard_p2p.common import worker_identity
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import CompiledIPC, apply_ipc_plan, validate_targets
from relax.distributed.checkpoint_service.backends.shard_p2p.layout import ExpertGeometry
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import bind_delivery, fingerprint
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
    return {key: command[key] for key in ("session", "engine", "tp_rank", "version", "nonce")}


def _prepare_delivery(model: torch.nn.Module, command: dict, tp_rank: int) -> dict:
    """Read-only cold negotiation against the native seed."""
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
    if binding["delivery"]["routes"]["nixl"]:
        raise ValueError("Local expert IPC requires every source on the receiver host")
    return dict(identity=_delivery_identity(command), **binding)


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
    if pending is None or pending != dict(identity=_delivery_identity(command), **binding):
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
        reference = ("native" if compiled_plan is not None else _digests(audit_parameters)) if audit_native else None
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
            seed_pending=True,
            failed=True,
            batched=batched,
        )
        model._relax_expert_ipc = state
        if compiled_plan is not None:
            state["compiled_plan"] = compiled_plan
        if audit_native and compiled_plan is None:
            for parameter in audit_parameters.values():
                parameter.fill_(float("nan"))
        torch.cuda.synchronize()
        state["failed"] = False
        if binding is not None:
            state["delivery"] = binding["delivery"]
            state["delivery_hash"] = binding["plan_hash"]
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
        compiled_plan.apply(parameters, bindings, verify_native=state.get("reference") == "native")
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
        logger.info(
            "COLOCATE_EXPERT_IPC_AUDIT engine=%d tp=%d version=%d parameters=%d exact=True",
            state["engine"],
            tp_rank,
            command["version"],
            len(parameters),
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
    state.pop("reference", None)
    state.update(version=command["version"], nonce=command["nonce"], failed=False, seed_pending=False)


def _decode_command(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]], args) -> dict:
    if (
        type(args.tp_size) is not int
        or args.tp_size <= 0
        or args.pp_size != 1
        or args.dp_size != 1
        or args.ep_size != 1
        or args.quantization is not None
        or args.enable_dp_attention
        or args.enable_lora
        or args.speculative_algorithm is not None
    ):
        raise ValueError("Expert IPC receiver requires plain BF16 TP/PP1/DP1/EP1")
    if len(weights) != 1 or weights[0][1].numel() != 0:
        raise ValueError("Expert IPC control requires exactly one empty tensor")
    command = json.loads(weights[0][0])
    geometry = (
        command.get("geometry", {})
        if command["op"] in ("describe_tp", "prepare", "seed")
        else getattr(model, "_relax_expert_ipc", {}).get("geometry", {})
    )
    if geometry.get("tp_size") != args.tp_size:
        raise ValueError("Expert IPC layout differs from the running receiver TP")
    return command


def control(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
    # A failed warm loader must not strand successful TP peers at SGLang's
    # outer completion barrier. Decode failures join the same agreement too.
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
            or command.get("op") not in ("describe_tp", "prepare", "seed", "apply")
            or command.get("op") == "seed"
            and "delivery" not in command
            or "tp" in command
            or "remote" in command
        ):
            raise ValueError("Missing or misrouted cold expert command")
        header = tuple(command.get(key) for key in ("session", "engine", "op", "version", "previous_version"))
        header += ("delivery" in command,)
        if command["op"] == "seed":
            header += _copy_mode(command)
            header += (command.get("audit_native", False),)
        if command["op"] == "describe_tp":
            if command.get("description_hash") != fingerprint(command["description"]):
                raise ValueError("Ordinary TP description checksum differs")
        elif command["op"] == "prepare":
            delivery = _prepare_delivery(model, command, tp_rank)
        elif command["op"] == "seed":
            _accept_delivery(model, command)
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
        elif command["op"] == "prepare":
            reply = dict(
                identity=delivery["identity"],
                receiver=delivery["delivery"]["receiver"],
                logical_plan_hash=delivery["logical_plan_hash"],
                plan_hash=delivery["plan_hash"],
            )
            model._relax_expert_delivery = delivery
        else:
            apply_command(model, command, tp_rank)
    except Exception as exc:
        logger.exception("Shard IPC receiver command failed: %s", command["op"])
        error = str(exc)
    agree(error)
    if command["op"] in ("describe_tp", "prepare"):
        replies = [None] * dist.get_world_size(group)
        dist.all_gather_object(replies, reply, group=group)
        # The scheduler guard forwards this one-shot reply through SGLang's
        # existing success/message response, after the native RPC completes.
        key = "shard_p2p_tp_targets" if command["op"] == "describe_tp" else "shard_p2p_receivers"
        model._relax_weight_sync_reply = json.dumps({key: replies}, separators=(",", ":"))
