# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One-update numerical verification for frozen SFT conditioning reuse."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class _TensorState:
    value: torch.Tensor
    device: torch.device
    mesh: Any = None
    placements: Any = None
    shape: Any = None
    stride: Any = None


class VerificationError(RuntimeError):
    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("Frozen-conditioning cache coverage, loss, gradient or FP32 update verification failed")
        self.report = report


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _snapshot_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        local = _local(value).detach()
        return _TensorState(
            local.cpu().clone(),
            local.device,
            getattr(value, "device_mesh", None),
            getattr(value, "placements", None),
            tuple(value.shape),
            tuple(value.stride()),
        )
    if isinstance(value, dict):
        return {key: _snapshot_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item) for item in value)
    return copy.deepcopy(value)


def _restore_tree(value: Any) -> Any:
    if isinstance(value, _TensorState):
        local = value.value.to(device=value.device, copy=True)
        if value.mesh is None:
            return local
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            local,
            device_mesh=value.mesh,
            placements=value.placements,
            run_check=False,
            shape=value.shape,
            stride=value.stride,
        )
    if isinstance(value, dict):
        return {key: _restore_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_tree(item) for item in value)
    return copy.deepcopy(value)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _world(group: Any) -> int:
    return dist.get_world_size(group) if group is not None else 1


def _reshard(model: torch.nn.Module) -> None:
    # Release any full-parameter copies before restoring the FP32 local shards.
    # The next root forward must all-gather from the restored masters.
    from torch.distributed.fsdp import FSDPModule

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def compare_sharded_tensors(
    reference: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    gloo_group: Any = None,
    relative_tolerance: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Compare each parameter over all of its disjoint FSDP shards.

    Inputs are already-local CPU snapshots. No full parameter is gathered.
    Norms use float64 accumulation; the quantities being compared retain their
    original FP32 values. A zero reference norm requires an exactly zero
    result.
    """
    if set(reference) != set(actual):
        raise ValueError("Verification parameter names differ")
    if not 0 <= relative_tolerance <= 0.01:
        raise ValueError("Verification relative tolerance must be between zero and one percent")
    names = sorted(reference)
    sums = torch.zeros(len(names), 3, dtype=torch.float64)
    maxima = torch.zeros(len(names), dtype=torch.float64)
    valid = torch.ones(len(names), dtype=torch.int64)
    for index, name in enumerate(names):
        ref, value = reference[name], actual[name]
        if ref.shape != value.shape or not bool(torch.isfinite(ref).all() and torch.isfinite(value).all()):
            valid[index] = 0
            continue
        ref, value = ref.double(), value.double()
        difference = value - ref
        sums[index] = torch.stack([ref.square().sum(), value.square().sum(), difference.square().sum()])
        if difference.numel():
            maxima[index] = difference.abs().max()
    if _world(gloo_group) > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=gloo_group)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=gloo_group)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=gloo_group)
    report = {}
    for index, name in enumerate(names):
        finite = bool(valid[index])
        ref_norm, actual_norm, error_norm = [math.sqrt(value) for value in sums[index].tolist()]
        finite = finite and all(math.isfinite(value) for value in (ref_norm, actual_norm, error_norm))
        passed = finite and (actual_norm == 0.0 if ref_norm == 0.0 else error_norm <= relative_tolerance * ref_norm)
        report[name] = {
            "finite": finite,
            "reference_l2": ref_norm if finite else None,
            "actual_l2": actual_norm if finite else None,
            "error_l2": error_norm if finite else None,
            "relative_l2_error": error_norm / ref_norm if finite and ref_norm else 0.0 if passed else None,
            "max_absolute_error": float(maxima[index]) if finite else None,
            "passed": passed,
        }
    return report


def _compare_loss(reference: float, actual: float, relative_tolerance: float) -> dict[str, Any]:
    finite = math.isfinite(reference) and math.isfinite(actual)
    error = abs(actual - reference) if finite else None
    return {
        "reference": reference if math.isfinite(reference) else None,
        "actual": actual if math.isfinite(actual) else None,
        "absolute_error": error,
        "finite": finite,
        "passed": finite and error <= relative_tolerance * abs(reference),
    }


def _cache_snapshot(statistics: Callable[[], dict[str, Any]] | None) -> dict[str, Any] | None:
    if statistics is None:
        return None
    try:
        values = statistics()
        counts = {name: values[name] for name in ("effective_hit", "encoding_calls")}
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts.values()):
            raise ValueError("Cache verification statistics must be nonnegative integer counters")
        return {**{name: int(value) for name, value in counts.items()}, "error": None}
    except Exception as exc:
        # Do not let a rank-local telemetry failure strand peers at the next
        # FSDP collective. The common coverage gate below rejects the result.
        return {"effective_hit": 0, "encoding_calls": 0, "error": f"{type(exc).__name__}: {exc}"}


def _cache_validation(
    before: dict[str, Any] | None,
    after_reference: dict[str, Any] | None,
    after_actual: dict[str, Any] | None,
    local_samples: int,
    gloo_group: Any,
) -> dict[str, Any]:
    if before is None:
        return {"statistics_provided": False, "branch_verified": False}
    reference_hits = after_reference["effective_hit"] - before["effective_hit"]
    reference_encodes = after_reference["encoding_calls"] - before["encoding_calls"]
    actual_hits = after_actual["effective_hit"] - after_reference["effective_hit"]
    actual_encodes = after_actual["encoding_calls"] - after_reference["encoding_calls"]
    errors = [snapshot["error"] for snapshot in (before, after_reference, after_actual)]
    local_verified = (
        not any(errors)
        and min(reference_hits, reference_encodes, actual_hits, actual_encodes) >= 0
        and actual_hits == local_samples
        and actual_encodes == 0
    )
    totals = torch.tensor(
        [reference_hits, reference_encodes, actual_hits, actual_encodes, local_samples], dtype=torch.int64
    )
    verified = torch.tensor(int(local_verified), dtype=torch.int64)
    if _world(gloo_group) > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=gloo_group)
        dist.all_reduce(verified, op=dist.ReduceOp.MIN, group=gloo_group)
    ref_hits_total, ref_encodes_total, actual_hits_total, actual_encodes_total, expected_hits_total = totals.tolist()
    return {
        "statistics_provided": True,
        "reference": {
            "local_effective_hits": reference_hits,
            "local_encoding_calls": reference_encodes,
            "global_effective_hits": ref_hits_total,
            "global_encoding_calls": ref_encodes_total,
        },
        "actual": {
            "local_effective_hits": actual_hits,
            "local_encoding_calls": actual_encodes,
            "global_effective_hits": actual_hits_total,
            "global_encoding_calls": actual_encodes_total,
        },
        "expected_local_hits": local_samples,
        "expected_global_hits": expected_hits_total,
        "local_branch_verified": local_verified,
        "branch_verified": bool(verified),
        "snapshot_errors": errors,
    }


def _write_report(report: dict[str, Any], directory: Path, rank: int, gloo_group: Any) -> None:
    error = [None]
    if rank == 0:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / "conditioning-verification.json.tmp"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(directory / "conditioning-verification.json")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if _world(gloo_group) > 1:
        dist.broadcast_object_list(error, src=0, group=gloo_group)
    if error[0] is not None:
        raise RuntimeError(f"Writing conditioning verification report failed: {error[0]}")


def verify_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    samples: Sequence[Any],
    forward: Callable[[Any], torch.Tensor],
    set_force_online: Callable[[bool], None],
    *,
    device: torch.device,
    group: Any,
    gloo_group: Any,
    loss_scale: float,
    output_dir: str | Path,
    rank: int = 0,
    clip_fn: Callable[[], Any] | None = None,
    relative_tolerance: float = 0.01,
    cache_statistics: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare online and warm-cache updates from exactly the same initial
    state.

    ``forward(sample)`` is called once per local microbatch and must not step
    the optimizer or scheduler. ``set_force_online(True)`` forces encoding but
    may populate the cache. Gradients are captured after full effective-batch
    accumulation and FSDP reduction, before ``clip_fn``. On success only the
    cached optimizer update remains applied; the caller advances its scheduler
    and training cursor once. On failure the original state is restored.

    ``named_parameters`` must be optimizer-owned FP32 masters. State and local
    shards are snapshotted to CPU, never an unsharded full model. All ranks must
    call this function together with identically ordered parameter names.

    For actual cache validation, supply a side-effect-free ``cache_statistics``
    callback returning cumulative ``effective_hit`` and ``encoding_calls``.
    Every rank must execute the cached branch for every sample with no online
    encoding in the second pass. Without this callback only numerical equality
    is checked, and the report explicitly leaves cache_branch_verified false.
    """
    device = torch.device(device)
    parameters = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if not parameters or not samples or loss_scale <= 0:
        raise ValueError("Verification requires trainable parameters, samples and positive loss_scale")
    if _world(group) > 1 and gloo_group is None:
        raise ValueError("Distributed verification requires an explicit Gloo group")
    if not 0 <= relative_tolerance <= 0.01:
        raise ValueError("Verification relative tolerance must be between zero and one percent")
    if any(_local(parameter).dtype != torch.float32 for _, parameter in parameters):
        raise ValueError("Parameter-update verification requires actual FP32 optimizer master parameters")
    names = [name for name, _ in parameters]
    if len(set(names)) != len(names):
        raise ValueError("Verification parameter names must be unique")
    if _world(gloo_group) > 1:
        schemas = [None] * _world(gloo_group)
        dist.all_gather_object(schemas, names, group=gloo_group)
        if any(item != names for item in schemas):
            raise ValueError("Verification parameter order differs between ranks")

    _reshard(model)
    initial = {name: _local(parameter).detach().cpu().clone() for name, parameter in parameters}
    optimizer_state = _snapshot_tree(optimizer.state_dict())
    rng = _rng_state(device)

    def restore() -> None:
        _reshard(model)
        with torch.no_grad():
            for name, parameter in parameters:
                _local(parameter).copy_(initial[name].to(_local(parameter).device))
        optimizer.load_state_dict(_restore_tree(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(rng, device)

    def run_pass() -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        for sample in samples:
            loss = forward(sample)
            if loss.ndim != 0:
                raise ValueError("Verification forward must return a scalar loss")
            (loss * loss_scale).backward()
            loss_sum.add_(loss.detach().float() * loss_scale)
        gradients = {
            name: _local(parameter.grad).detach().cpu().clone()
            if parameter.grad is not None
            else torch.zeros_like(initial[name])
            for name, parameter in parameters
        }
        norm = clip_fn() if clip_fn is not None else None
        norm = float(_local(norm).detach().cpu()) if isinstance(norm, torch.Tensor) else norm
        if norm is not None and not math.isfinite(norm):
            norm = None
        optimizer.step()
        updates = {name: _local(parameter).detach().cpu() - initial[name] for name, parameter in parameters}
        loss_cpu = loss_sum.double().cpu()
        if _world(gloo_group) > 1:
            dist.all_reduce(loss_cpu, op=dist.ReduceOp.SUM, group=gloo_group)
            loss_cpu /= _world(gloo_group)
        return {"loss": float(loss_cpu), "gradients": gradients, "updates": updates, "clip_norm": norm}

    try:
        cache_before = _cache_snapshot(cache_statistics)
        set_force_online(True)
        reference = run_pass()
        cache_after_reference = _cache_snapshot(cache_statistics)
        restore()
        set_force_online(False)
        actual = run_pass()
        cache_after_actual = _cache_snapshot(cache_statistics)
        cache_validation = _cache_validation(
            cache_before, cache_after_reference, cache_after_actual, len(samples), gloo_group
        )
        gradients = compare_sharded_tensors(
            reference["gradients"], actual["gradients"], gloo_group=gloo_group, relative_tolerance=relative_tolerance
        )
        updates = compare_sharded_tensors(
            reference["updates"], actual["updates"], gloo_group=gloo_group, relative_tolerance=relative_tolerance
        )
        loss = _compare_loss(reference["loss"], actual["loss"], relative_tolerance)
        report = {
            "schema_version": 1,
            "world_size": _world(group),
            "local_samples": len(samples),
            "relative_tolerance": relative_tolerance,
            "gradient_stage": "full effective batch, after FSDP reduction, before gradient clipping",
            "update_quantity": "FP32 theta_after - theta_before",
            "loss": loss,
            "reference_clip_norm": reference["clip_norm"],
            "actual_clip_norm": actual["clip_norm"],
            "cache_validation": cache_validation,
            "cache_branch_verified": cache_validation["branch_verified"],
            "gradients": gradients,
            "parameter_updates": updates,
            "passed": (
                loss["passed"]
                and all(row["passed"] for row in (*gradients.values(), *updates.values()))
                and (cache_statistics is None or cache_validation["branch_verified"])
            ),
        }
        _write_report(report, Path(output_dir), rank, gloo_group)
        if not report["passed"]:
            raise VerificationError(report)
        optimizer.zero_grad(set_to_none=True)
        return report
    except BaseException:
        restore()
        raise
    finally:
        set_force_online(False)
