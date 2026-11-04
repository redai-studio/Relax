"""CPU evidence helpers; thresholds are fixed by Task 17, not CLI options."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch


RELATIVE_LIMIT = 0.01


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return local_tensor(tensor).detach().cpu().clone()


def payload_digest(value: Any) -> str:
    """Hash structure, tensor dtype/shape and exact bytes (including signed
    zero)."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = cpu_clone(item).contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, bool, int, float, bytes)):
            digest.update(f"{type(item).__name__}:{item!r};".encode())
        elif hasattr(item, "tobytes") and hasattr(item, "dtype"):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        else:
            raise TypeError(f"Unsupported evidence payload type: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def capture_rng() -> dict[str, Any]:
    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_shards(reference: list[torch.Tensor | None], candidate: list[torch.Tensor | None]) -> dict[str, Any]:
    """One global parameter, assembled mathematically across disjoint FSDP
    shards.

    No per-shard relative threshold: a zero local shard must not incorrectly
    fail a globally nonzero reference. Conversely a globally zero reference
    requires exact zero difference, without an epsilon or user-selectable
    absolute slack.
    """
    if len(reference) != len(candidate) or not reference:
        return {"passed": False, "reason": "different_or_empty_shard_count"}
    squared_reference = squared_error = 0.0
    maximum_error = 0.0
    all_finite = True
    bitwise_equal = True
    for ref, actual in zip(reference, candidate):
        if ref is None or actual is None:
            if ref is not None or actual is not None:
                return {"passed": False, "reason": "gradient_presence_mismatch"}
            continue
        if ref.shape != actual.shape or ref.dtype != actual.dtype:
            return {"passed": False, "reason": "shape_or_dtype_mismatch"}
        finite = bool(torch.isfinite(ref).all()) and bool(torch.isfinite(actual).all())
        all_finite &= finite
        if not finite:
            continue
        error = actual.double() - ref.double()
        squared_reference += float(ref.double().square().sum())
        squared_error += float(error.square().sum())
        if error.numel():
            maximum_error = max(maximum_error, float(error.abs().max()))
        bitwise_equal &= payload_digest(ref) == payload_digest(actual)
    reference_norm = math.sqrt(squared_reference)
    error_norm = math.sqrt(squared_error)
    passed = all_finite and (
        error_norm == 0.0 if reference_norm == 0.0 else error_norm <= RELATIVE_LIMIT * reference_norm
    )
    return {
        "passed": passed,
        "all_finite": all_finite,
        "reference_l2": reference_norm,
        "error_l2": error_norm,
        "relative_l2": error_norm / reference_norm if reference_norm else None,
        "maximum_absolute_error": maximum_error,
        "bitwise_equal": bitwise_equal,
        "zero_reference_requires_exact_zero": reference_norm == 0.0,
    }


def compare_trajectories(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        return {"passed": False, "reason": "rank_count_mismatch"}
    failures: list[str] = []
    for rank, (ref, actual) in enumerate(zip(reference, candidate)):
        for field in (
            "initial_parameters_sha256",
            "optimizer_initial_sha256",
            "rng_before_sha256",
            "sample_ids",
            "parameter_global_shapes",
            "world",
        ):
            if ref[field] != actual[field]:
                failures.append(f"rank{rank}:{field}")
        if ref["rng_after_sha256"] != actual["rng_after_sha256"]:
            failures.append(f"rank{rank}:rng_after_sha256")
    loss = compare_shards(
        [torch.tensor(reference[0]["loss"], dtype=torch.float64)],
        [torch.tensor(candidate[0]["loss"], dtype=torch.float64)],
    )
    tables: dict[str, Any] = {}
    for kind in ("gradients", "delta"):
        names = set(reference[0][kind])
        if any(set(rank[kind]) != names for rank in reference + candidate):
            failures.append(f"{kind}:parameter_names")
            continue
        tables[kind] = {
            name: compare_shards([rank[kind][name] for rank in reference], [rank[kind][name] for rank in candidate])
            for name in sorted(names)
        }
        for name in names:
            shape = reference[0]["parameter_global_shapes"][name]
            expected_numel = math.prod(shape)
            for label, run in (("reference", reference), ("candidate", candidate)):
                present = [row[kind][name] for row in run if row[kind][name] is not None]
                if present and sum(t.numel() for t in present) != expected_numel:
                    failures.append(f"{kind}:{name}:{label}:incomplete_global_shards")
        failures.extend(f"{kind}:{name}" for name, row in tables[kind].items() if not row["passed"])
    if not loss["passed"]:
        failures.append("loss")
    return {"passed": not failures, "relative_limit": RELATIVE_LIMIT, "loss": loss, **tables, "failures": failures}


def numerical_components(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Name each loss/gradient/update quantity without mixing tensor scales."""
    components = {"loss": report["loss"]} if "loss" in report else {}
    for kind in ("gradients", "delta"):
        for name, values in report.get(kind, {}).items():
            components[f"{kind}:{name}"] = values
    return components


def assess_strict_acceptance(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Separate Task17 hard limits from unresolved baseline nondeterminism.

    The A1/A2 repeat is inspected first, component by component. An observed
    exact baseline requires an exact cached comparison for that same quantity.
    A nonexact baseline never creates a new absolute tolerance from its result:
    the existing 1%/zero/finite checks still apply, but official numerical
    acceptance remains unresolved. No GPU result can tune this policy.
    """
    reference = numerical_components(baseline)
    actual = numerical_components(candidate)
    same_components = bool(reference) and set(reference) == set(actual)
    nondeterministic = sorted(name for name, values in reference.items() if not values.get("bitwise_equal", False))
    mismatches = sorted(
        name
        for name, values in reference.items()
        if values.get("bitwise_equal", False) and not actual.get(name, {}).get("bitwise_equal", False)
    )
    hard_limits = bool(baseline.get("passed")) and bool(candidate.get("passed")) and same_components
    passed = hard_limits and not mismatches
    strict = passed and not nondeterministic
    status = "accepted" if strict else "needs_nondeterminism_review" if passed else "failed"
    return {
        "passed": passed,
        "hard_numerical_limits_passed": hard_limits,
        "strict_acceptance": strict,
        "acceptance_status": status,
        "needs_nondeterminism_review": passed and bool(nondeterministic),
        "component_sets_match": same_components,
        "baseline_nondeterministic_components": nondeterministic,
        "deterministic_mismatches": mismatches,
        "baseline_nondeterminism_errors": {name: reference[name] for name in nondeterministic},
    }
