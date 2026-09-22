# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bundle validation.

validate checks a bundle's format, integrity, safety and dependency closure
without executing any numerical replay. It is the gate that runs before every
replay so a stage can never start on an incomplete or tampered bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from relax.utils.replay.bundle import BundleReader, LoadedBundle
from relax.utils.replay.identity import ClosureError, expand_selection, sample_integrity_problems, validate_identity
from relax.utils.replay.schema import INDEX_KEYS, MANIFEST_KEYS, StageCapability


@dataclass
class ValidationResult:
    """Outcome of a bundle validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_bundle(path: str | Path) -> ValidationResult:
    """Validate format, integrity, safety and closure of a bundle."""
    result = ValidationResult(valid=True)
    _validate_unknown_fields(Path(path), result)
    try:
        bundle = BundleReader(path).load()
    except Exception as exc:  # noqa: BLE001 — surface every read failure as a validation error
        result.add_error(f"bundle read failed: {exc}")
        return result

    _validate_identity(bundle, result)
    _validate_sample_records(bundle, result)
    _validate_capabilities(bundle, result)
    _validate_numerics(bundle, result)
    _validate_closure(bundle, result)
    _validate_producer(bundle, result)
    return result


def _validate_unknown_fields(path: Path, result: ValidationResult) -> None:
    """Warn on metadata fields the V1 reader does not understand.

    A bundle may carry fields introduced by a newer-but-same-major schema; the
    reader must not silently drop them, so they surface as warnings.
    """
    for filename, allowed in (("manifest.json", MANIFEST_KEYS), ("index.json", INDEX_KEYS)):
        try:
            data = json.loads((path / filename).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # the reader's own load() will produce the authoritative error
        unknown = sorted(set(data) - set(allowed))
        if unknown:
            result.add_warning(f"{filename} has unknown field(s) {unknown}; the V1 reader will ignore them")


def _validate_producer(bundle: LoadedBundle, result: ValidationResult) -> None:
    """Flag missing build provenance so same-code-version replay is
    explicit."""
    if not bundle.manifest.producer.commit:
        result.add_warning("producer commit is unset; cannot verify the bundle was produced by the same code version")


def _validate_identity(bundle: LoadedBundle, result: ValidationResult) -> None:
    try:
        validate_identity(bundle.index)
    except ClosureError as exc:
        result.add_error(str(exc))


def _validate_sample_records(bundle: LoadedBundle, result: ValidationResult) -> None:
    if not bundle.index.samples:
        result.add_error("bundle index has no samples")
        return
    seen: set[str] = set()
    for record in bundle.index.samples:
        if record.sample_id in seen:
            result.add_error(f"duplicate sample_id {record.sample_id!r}")
        seen.add(record.sample_id)
        for _field, message in sample_integrity_problems(record):
            result.add_error(message)


def _validate_capabilities(bundle: LoadedBundle, result: ValidationResult) -> None:
    config = bundle.index.config
    for stage, contract in bundle.manifest.stage_contracts.items():
        if contract.capability == StageCapability.UNSUPPORTED:
            result.add_warning(f"stage {stage.value!r} is unsupported in this bundle")
    if config.advantage_estimator != "grpo":
        result.add_error(f"unsupported advantage_estimator {config.advantage_estimator!r} (V1 supports GRPO only)")
    context_parallel = bundle.index.identity.rank.get("cp", 1)
    if context_parallel != 1:
        result.add_error(f"unsupported context parallel cp={context_parallel} (V1 supports CP=1 only)")


def _validate_numerics(bundle: LoadedBundle, result: ValidationResult) -> None:
    for name, tensor in bundle.tensors.items():
        if not torch.isfinite(tensor).all():
            result.add_error(f"payload {name!r} contains NaN or Inf")
        if tensor.dtype not in (torch.float32, torch.float64):
            result.add_warning(f"payload {name!r} has dtype {tensor.dtype}; expect float for replay")


def _validate_closure(bundle: LoadedBundle, result: ValidationResult) -> None:
    try:
        closure = expand_selection(bundle.index)
        if len(closure.sample_ids) != len(bundle.index.samples):
            result.add_error("dependency closure does not cover all samples")
    except ClosureError as exc:
        result.add_error(str(exc))
