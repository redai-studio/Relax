# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DPO frozen-reference identity and byte-level integrity helpers."""

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REFERENCE_IDENTITY_FILENAME = "relax_dpo_reference.json"


@dataclass(frozen=True)
class DPOReferenceIdentity:
    """Identity persisted beside every standard-DPO checkpoint."""

    schema_version: int
    parameter_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DPOReferenceIdentity":
        return cls(
            schema_version=int(value["schema_version"]),
            parameter_sha256=str(value["parameter_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return value.view(torch.uint8).numpy().tobytes()


def canonical_tensor_sha256(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash names, dtype, shape and bytes in canonical name order."""
    normalized = sorted(((str(name), tensor) for name, tensor in named_tensors), key=lambda item: item[0])
    if not normalized:
        raise ValueError("canonical tensor digest requires at least one tensor")
    digest = hashlib.sha256()
    for name, tensor in normalized:
        _update_field(digest, name.encode())
        _update_field(digest, str(tensor.dtype).encode())
        _update_field(digest, json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        _update_field(digest, _tensor_bytes(tensor))
    return digest.hexdigest()


def reference_identity_path(checkpoint_root: str | os.PathLike[str], iteration: int) -> Path:
    root = Path(checkpoint_root)
    iteration_dir = root if root.name == f"iter_{iteration:07d}" else root / f"iter_{iteration:07d}"
    return iteration_dir / REFERENCE_IDENTITY_FILENAME


def write_reference_identity(path: Path, identity: DPOReferenceIdentity) -> None:
    """Atomically write a reference identity sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    payload = identity.to_dict()
    payload["schema_version"] = 2
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_reference_identity(path: Path) -> DPOReferenceIdentity:
    if not path.is_file():
        raise FileNotFoundError(f"DPO reference identity sidecar is missing: {path}")
    identity = DPOReferenceIdentity.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if identity.schema_version not in (1, 2):
        raise ValueError(f"unsupported DPO reference identity schema: {identity.schema_version}")
    return identity


__all__ = [
    "DPOReferenceIdentity",
    "REFERENCE_IDENTITY_FILENAME",
    "canonical_tensor_sha256",
    "read_reference_identity",
    "reference_identity_path",
    "write_reference_identity",
]
