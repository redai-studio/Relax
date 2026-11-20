# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Apply ADTS packages to the local Teacher weight shard."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

import torch

from .adts.bitpack import PackedU16Plan, read_package_header


@dataclass
class _State:
    base: str
    current: str
    plans: dict[tuple[str, str], PackedU16Plan]
    pointers: dict[str, int]
    recovery: dict[str, torch.Tensor]


def _get_target_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weights = dict(model.named_parameters())
    for name, tensor in model.named_buffers():
        # SGLang's GemmaRMSNorm caches weight + 1 in gemma_weight, so switch it too.
        if name.rsplit(".", 1)[-1] == "gemma_weight":
            weights[f"buffer:{name}"] = tensor
    return weights


def _hash_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


def _verify_base(weights: dict, recovery: dict, header: dict, base: str) -> None:
    records = header["tensors"]
    if weights.keys() != records.keys():
        raise ValueError(f"Runtime parameters differ from the ADTS base {base!r}")
    digest_key = "a_sha256" if base == header["provenance"]["source"] else "b_sha256"
    for name, record in records.items():
        tensor = weights[name]
        if list(tensor.shape) != record["shape"] or str(tensor.dtype) != record["dtype"]:
            raise ValueError(f"ADTS base layout mismatch: {name}")
        # The pinned recovery snapshot contains the exact bytes used by the first switch.
        if _hash_tensor(recovery[name]) != record[digest_key]:
            raise ValueError(f"ADTS base weight mismatch: {name}")


def _prepare(model: torch.nn.Module, root: Path, rank: int) -> _State:
    manifest = json.loads((root / "manifest.json").read_text())
    base = manifest["initial_teacher"]
    base_header = None
    weights = _get_target_tensors(model)
    plans = {}

    for filename, record in manifest["package_integrity"].items():
        if record.get("rank", 0) != rank:
            continue
        with (root / filename).open("rb") as stream:
            header = read_package_header(stream)
            source, target = header["provenance"]["source"], header["provenance"]["target"]
            if base_header is None and base in (source, target):
                base_header = header
            plan = PackedU16Plan(stream, weights, header)
            plans[source, target] = plans[target, source] = plan

    pointers = {name: tensor.data_ptr() for name, tensor in weights.items()}
    # Offload can discard buffers such as rotary caches as well as parameters.
    anchor = {**weights, **{f"buffer:{name}": t for name, t in model.named_buffers()}}
    recovery = {}
    for name, tensor in anchor.items():
        snapshot = torch.empty_like(tensor, device="cpu", pin_memory=True)
        recovery[name] = snapshot
        snapshot.copy_(tensor, non_blocking=True)

    torch.cuda.synchronize(next(model.parameters()).device)
    if base_header is None:
        raise ValueError(f"No ADTS package for base {base!r} on TP rank {rank}")
    _verify_base(weights, recovery, base_header, base)
    return _State(base, base, plans, pointers, recovery)


def _apply_delta(model: torch.nn.Module, state: _State, source: str, target: str) -> None:
    plan = state.plans[source, target]
    for name in plan.target_names:
        tensor = attrgetter(name.removeprefix("buffer:"))(model)
        if tensor.data_ptr() != state.pointers[name]:
            raise RuntimeError(f"Teacher write target moved: {name}")
    plan.apply(target)


def _restore_teacher(model: torch.nn.Module, state: _State, target: str) -> None:
    device = next(model.parameters()).device
    torch.cuda.synchronize(device)
    for name, snapshot in state.recovery.items():
        tensor = attrgetter(name.removeprefix("buffer:"))(model)
        tensor.copy_(snapshot, non_blocking=True)

    if target == state.base:
        torch.cuda.current_stream(device).synchronize()
    else:
        # The delta stream waits for the base copies on the caller stream.
        _apply_delta(model, state, state.base, target)


def _update_teacher(model: torch.nn.Module, named_tensors, rank: int) -> None:
    ((command, _),) = named_tensors
    _, operation, teacher = command.split(":")
    if operation == "prepare":
        model._relax_adts_teacher_state = _prepare(model, Path(os.environ["RELAX_ADTS_TEACHER_DIR"]), rank)
        return

    state = model._relax_adts_teacher_state
    if operation == "switch":
        if teacher != state.current:
            torch.cuda.synchronize(next(model.parameters()).device)
            _apply_delta(model, state, state.current, teacher)
            state.current = teacher
        return

    if operation == "restore":
        _restore_teacher(model, state, teacher)
        state.current = teacher
        return

    raise ValueError(f"Unknown Teacher operation: {operation}")


@torch.inference_mode()
def adts_teacher_loader(model: torch.nn.Module, named_tensors) -> None:
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group() if torch.distributed.is_initialized() else None
    if group is None or group.world_size == 1:
        return _update_teacher(model, named_tensors, 0)

    error = None
    try:
        _update_teacher(model, named_tensors, group.rank_in_group)
    except Exception as exc:
        error = f"rank {group.rank_in_group}: {exc}"

    # The endpoint reports one result for the TP group. Every shard must finish the update first.
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    if any(errors):
        raise RuntimeError(next(message for message in errors if message))
