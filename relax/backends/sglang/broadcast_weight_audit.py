# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Independent, opt-in receiver audit; never enabled in timing runs."""

import os
import socket
import uuid
from pathlib import Path

import torch

from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import atomic_json, digest


def _inventory(model: torch.nn.Module) -> dict:
    return {
        kind + name: dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=digest(tensor))
        for kind, tensors in (("parameter:", model.named_parameters()), ("buffer:", model.named_buffers()))
        for name, tensor in tensors
    }


def _storage(model: torch.nn.Module) -> dict:
    return {
        kind + name: (tensor.data_ptr(), tensor.storage_offset(), tensor.stride())
        for kind, tensors in (("parameter:", model.named_parameters()), ("buffer:", model.named_buffers()))
        for name, tensor in tensors
    }


def control(model: torch.nn.Module, weights: list[tuple[str, torch.Tensor]]) -> None:
    from .p2p_weight_lease import AUDIT_LOADER, run_control

    run_control(model, weights, AUDIT_LOADER, lambda command: _apply_control(model, command))


def _apply_control(model: torch.nn.Module, command: dict) -> None:
    from sglang.srt.runtime_context import get_parallel, get_server_args

    args, parallel = get_server_args(), get_parallel()
    if (
        type(args.tp_size) is not int
        or args.tp_size <= 0
        or not 0 <= parallel.tp_rank < args.tp_size
        or args.pp_size != 1
        or args.quantization is not None
    ):
        raise NotImplementedError("Broadcast audit requires unquantized TP/PP1")
    directory = Path(command["directory"])
    if not directory.is_dir() or directory.stat().st_mode & 0o077:
        raise ValueError("Broadcast audit requires a private directory")
    key = f"engine-{command['engine']}-tp-{parallel.tp_rank}"
    state = getattr(model, "_relax_broadcast_audit", None)
    identity = dict(
        hostname=socket.gethostname(),
        pid=os.getpid(),
        tp_rank=parallel.tp_rank,
        device=str(next(model.parameters()).device),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if command["op"] in ("reference", "reference_deferred"):
        if state is not None and state["active"]:
            raise RuntimeError("Unfinished broadcast audit")
        if state is not None and command["version"] <= state["version"]:
            raise RuntimeError("Stale broadcast audit version")
        reference = _inventory(model)
        poisoned = {
            name: tensor
            for name, tensor in model.named_parameters()
            if name.endswith((".w13_weight", ".w2_weight")) and tensor.is_floating_point()
        }
        prefixes = {name.rsplit(".", 1)[0] for name in poisoned}
        if not prefixes or set(poisoned) != {
            prefix + suffix for prefix in prefixes for suffix in (".w13_weight", ".w2_weight")
        }:
            raise ValueError(f"Incomplete native gated expert targets: {list(poisoned)}")
        if any(not torch.isfinite(tensor).all().item() for tensor in poisoned.values()):
            raise ValueError("Legacy expert reference contains nonfinite values")
        deferred = command["op"] == "reference_deferred"
        poisoned_buffers = {}
        if deferred:
            poisoned = {name: tensor for name, tensor in model.named_parameters() if tensor.is_floating_point()}
            poisoned_buffers = {
                name: tensor for name, tensor in model.named_buffers() if name.endswith(".gemma_weight")
            }
            if not poisoned:
                raise ValueError("P2P snapshot audit requires floating point parameters")
        state = dict(
            active=True,
            version=command["version"],
            generation=state["generation"] if state else str(uuid.uuid4()),
            reference=reference,
            poisoned=list(poisoned),
            poisoned_buffers=list(poisoned_buffers),
            deferred=deferred,
            poison_complete=not deferred,
            p2p_directory=command.get("p2p_directory") if deferred else None,
            directory=str(directory),
        )
        model._relax_broadcast_audit = state
        atomic_json(directory / f"{key}-reference.json", dict(**identity, **state))
        state["storage"] = _storage(model)
        if not deferred:
            with torch.no_grad():
                for tensor in poisoned.values():
                    tensor.fill_(float("nan"))
            torch.cuda.synchronize()
    elif command["op"] == "poison":
        p2p = getattr(model, "_relax_p2p_state", None)
        if (
            state is None
            or not state["active"]
            or not state["deferred"]
            or state["poison_complete"]
            or state["version"] != command["version"]
            or state["directory"] != str(directory)
            or p2p is None
            or not p2p["active"]
            or p2p["version"] != command["version"]
            or p2p["directory"] != state["p2p_directory"]
            or p2p["directory"] != command["p2p_directory"]
            or p2p["key"] != f"{command['engine']}-{parallel.tp_rank}"
            or _storage(model) != state["storage"]
        ):
            raise RuntimeError("Poison requires the prepared matching P2P snapshot")
        params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        with torch.no_grad():
            for name in state["poisoned"]:
                params[name].fill_(float("nan"))
            for name in state["poisoned_buffers"]:
                buffers[name].fill_(float("nan"))
        torch.cuda.synchronize()
        state["poison_complete"] = True
    elif command["op"] == "compare":
        if (
            state is None
            or not state["active"]
            or state["version"] != command["version"]
            or state["directory"] != str(directory)
            or not state["poison_complete"]
        ):
            raise RuntimeError("Stale broadcast audit reference")
        if state["deferred"]:
            p2p = getattr(model, "_relax_p2p_state", None)
            if (
                p2p is None
                or p2p["active"]
                or p2p["version"] != command["version"]
                or p2p["directory"] != state["p2p_directory"]
            ):
                raise RuntimeError("P2P comparison requires its matching completed commit")
        actual = _inventory(model)
        differences = sorted(
            name
            for name in actual.keys() | state["reference"].keys()
            if actual.get(name) != state["reference"].get(name)
        )
        storage_unchanged = _storage(model) == state["storage"]
        receipt = dict(
            **identity,
            generation=state["generation"],
            version=state["version"],
            poisoned=state["poisoned"],
            poisoned_buffers=state["poisoned_buffers"],
            actual=actual,
            differences=differences,
            storage_unchanged=storage_unchanged,
            passed=not differences and storage_unchanged,
        )
        atomic_json(directory / f"{key}-compare.json", receipt)
        if differences or not storage_unchanged:
            raise RuntimeError(
                f"Broadcast audit mismatches: {differences[:20]}, storage_unchanged={storage_unchanged}"
            )
        state["active"] = False
    else:
        raise ValueError(f"Unknown broadcast audit command: {command['op']}")
