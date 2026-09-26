# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DPO frozen-reference identity and byte-level integrity helpers."""

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


REFERENCE_IDENTITY_FILENAME = "relax_dpo_reference.json"
REFERENCE_LOADER_MODE = "hf_bridge_model_only_v1"
_GIT_COMMIT_SHA256_RE = re.compile(r"^[0-9a-f]{40}$")
_HF_ETAG_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_WEIGHT_INDEXES = (
    ("model.safetensors.index.json", ".safetensors"),
    ("pytorch_model.bin.index.json", ".bin"),
)
_SINGLE_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")


def _file_matches_hf_etag(path: Path, etag: str) -> bool:
    digest = hashlib.sha256() if len(etag) == 64 else hashlib.sha1()
    with path.open("rb") as file:
        if len(etag) == 40:
            size = os.fstat(file.fileno()).st_size
            digest.update(f"blob {size}\0".encode())
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == etag


def _validate_reference_weight_files(checkpoint: Path) -> None:
    found_weights = any((checkpoint / filename).is_file() for filename in _SINGLE_WEIGHT_NAMES)
    for index_name, shard_suffix in _WEIGHT_INDEXES:
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        found_weights = True
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"DPO reference weight index is invalid: {index_name!r}") from exc
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RuntimeError(f"DPO reference weight index has an empty weight_map: {index_name!r}")
        shard_names = set()
        for filename in weight_map.values():
            if not isinstance(filename, str) or not filename:
                raise RuntimeError(f"DPO reference weight index contains an invalid shard name: {index_name!r}")
            shard_names.add(filename)
        for filename in shard_names:
            relative_path = Path(filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"DPO reference weight index contains an unsafe shard path: {filename!r}")
            if relative_path.suffix != shard_suffix:
                raise RuntimeError(f"DPO reference weight index contains an invalid shard suffix: {filename!r}")
            if not (checkpoint / relative_path).is_file():
                raise RuntimeError(f"DPO reference weight index points to a missing shard: {filename!r}")
    if not found_weights:
        supported = ", ".join((*_SINGLE_WEIGHT_NAMES, *(name for name, _suffix in _WEIGHT_INDEXES)))
        raise RuntimeError(f"DPO reference has no supported model weights or index; expected one of: {supported}")


def _validate_local_download_metadata(checkpoint: Path, revision: str) -> None:
    metadata_root = checkpoint / ".cache" / "huggingface" / "download"
    snapshot_files = [
        path for path in checkpoint.rglob("*") if path.is_file() and checkpoint / ".cache" not in path.parents
    ]
    for path in snapshot_files:
        relative_path = path.relative_to(checkpoint)
        metadata_path = metadata_root.joinpath(*relative_path.parts).with_name(f"{relative_path.name}.metadata")
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
            metadata_revision, etag, timestamp_text = lines
            timestamp = float(timestamp_text)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"DPO reference file is missing valid Hugging Face local-dir metadata: {relative_path.as_posix()!r}"
            ) from exc
        if (
            metadata_revision != revision
            or _HF_ETAG_RE.fullmatch(etag) is None
            or not math.isfinite(timestamp)
            or path.stat().st_mtime - 1 > timestamp
        ):
            raise RuntimeError(
                "DPO reference file metadata does not match the pinned revision or file contents: "
                f"{relative_path.as_posix()!r}"
            )
        if not _file_matches_hf_etag(path, etag):
            raise RuntimeError(
                f"DPO reference file contents do not match its Hugging Face ETag: {relative_path.as_posix()!r}"
            )


def resolve_dpo_reference_checkpoint(repository: str, revision: str, local_checkpoint: str) -> str:
    """Resolve a pinned DPO reference in the configured local model
    directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("standard DPO requires huggingface_hub to resolve its frozen reference") from exc

    if _GIT_COMMIT_SHA256_RE.fullmatch(revision) is None:
        raise ValueError("standard DPO requires --dpo-reference-revision to be a full 40-character commit SHA")

    try:
        configured_checkpoint = Path(local_checkpoint).resolve(strict=True)
        checkpoint = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=str(configured_checkpoint),
                local_files_only=True,
            )
        ).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "standard DPO requires its pinned reference in --hf-checkpoint; prepare it with "
            f"`hf download {repository} --revision {revision} --local-dir {local_checkpoint}`"
        ) from exc
    if checkpoint != configured_checkpoint:
        raise RuntimeError(
            "DPO reference resolution returned a directory different from --hf-checkpoint: "
            f"configured={configured_checkpoint}, resolved={checkpoint}"
        )
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(
            "resolved DPO reference snapshot is missing config.json: "
            f"repository={repository!r}, revision={revision!r}, path={checkpoint}"
        )
    try:
        _validate_reference_weight_files(checkpoint)
        _validate_local_download_metadata(checkpoint, revision)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}; re-download with `hf download {repository} --revision {revision} --local-dir {local_checkpoint}`"
        ) from exc
    return str(checkpoint)


@dataclass(frozen=True)
class DPOReferenceIdentity:
    """Identity persisted beside every standard-DPO checkpoint."""

    schema_version: int
    repository: str
    revision: str
    loader_mode: str
    parameter_sha256: str
    probe_sha256: str | None
    probe_manifest: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DPOReferenceIdentity":
        return cls(
            schema_version=int(value["schema_version"]),
            repository=str(value["repository"]),
            revision=str(value["revision"]),
            loader_mode=str(value["loader_mode"]),
            parameter_sha256=str(value["parameter_sha256"]),
            probe_sha256=None if value.get("probe_sha256") is None else str(value["probe_sha256"]),
            probe_manifest=None if value.get("probe_manifest") is None else dict(value["probe_manifest"]),
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


def canonical_optimizer_sha256(optimizer: Any) -> str:
    """Hash optimizer master parameters and state without relying on object
    IDs."""
    digest = hashlib.sha256()

    def update(value: Any, path: str) -> None:
        _update_field(digest, path.encode())
        if isinstance(value, torch.Tensor):
            _update_field(digest, str(value.dtype).encode())
            _update_field(digest, json.dumps(list(value.shape), separators=(",", ":")).encode())
            _update_field(digest, _tensor_bytes(value))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                update(value[key], f"{path}/{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                update(item, f"{path}/{index}")
        else:
            _update_field(digest, repr(value).encode())

    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        optimizers = [getattr(item, "optimizer", item) for item in chained]
    else:
        optimizers = [getattr(optimizer, "optimizer", optimizer)]
    for optimizer_index, inner_optimizer in enumerate(optimizers):
        param_groups = getattr(inner_optimizer, "param_groups", None)
        state = getattr(inner_optimizer, "state", None)
        if param_groups is None or state is None:
            update(inner_optimizer.state_dict(), f"optimizer/{optimizer_index}/state_dict")
            continue
        for group_index, group in enumerate(param_groups):
            update(
                {key: value for key, value in group.items() if key != "params"},
                f"optimizer/{optimizer_index}/group/{group_index}/options",
            )
            for parameter_index, parameter in enumerate(group["params"]):
                path = f"optimizer/{optimizer_index}/group/{group_index}/parameter/{parameter_index}"
                update(parameter, f"{path}/master")
                update(state.get(parameter, {}), f"{path}/state")
    return digest.hexdigest()


def reference_probe_sha256(
    pair_ids: Sequence[int],
    branch_is_chosen: Sequence[bool],
    tokens: Sequence[Sequence[int] | torch.Tensor],
    loss_masks: Sequence[Sequence[int] | torch.Tensor],
    ref_log_probs: Sequence[Sequence[float] | torch.Tensor],
) -> str:
    """Hash the exact completion-only frozen-reference probe output."""
    size = len(pair_ids)
    if any(len(values) != size for values in (branch_is_chosen, tokens, loss_masks, ref_log_probs)):
        raise ValueError("reference probe fields must be branch aligned")
    digest = hashlib.sha256()
    for index in range(size):
        _update_field(digest, str(int(pair_ids[index])).encode())
        _update_field(digest, b"chosen" if bool(branch_is_chosen[index]) else b"rejected")
        token_tensor = torch.as_tensor(tokens[index], dtype=torch.int64)
        mask_tensor = torch.as_tensor(loss_masks[index], dtype=torch.bool)
        logp_tensor = torch.as_tensor(ref_log_probs[index], dtype=torch.float32)
        if logp_tensor.shape != mask_tensor.shape:
            raise ValueError("reference probe log-probability/mask shape mismatch")
        _update_field(digest, _tensor_bytes(token_tensor))
        _update_field(digest, _tensor_bytes(mask_tensor))
        masked_log_probs = logp_tensor * mask_tensor.to(device=logp_tensor.device, dtype=torch.float32)
        _update_field(digest, _tensor_bytes(masked_log_probs))
    return digest.hexdigest()


def reference_identity_path(checkpoint_root: str | os.PathLike[str], iteration: int) -> Path:
    root = Path(checkpoint_root)
    iteration_dir = root if root.name == f"iter_{iteration:07d}" else root / f"iter_{iteration:07d}"
    return iteration_dir / REFERENCE_IDENTITY_FILENAME


def write_reference_identity(path: Path, identity: DPOReferenceIdentity) -> None:
    """Atomically write a reference identity sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_reference_identity(path: Path) -> DPOReferenceIdentity:
    if not path.is_file():
        raise FileNotFoundError(f"DPO reference identity sidecar is missing: {path}")
    identity = DPOReferenceIdentity.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if identity.schema_version != 1:
        raise ValueError(f"unsupported DPO reference identity schema: {identity.schema_version}")
    return identity


__all__ = [
    "DPOReferenceIdentity",
    "REFERENCE_IDENTITY_FILENAME",
    "REFERENCE_LOADER_MODE",
    "canonical_optimizer_sha256",
    "canonical_tensor_sha256",
    "read_reference_identity",
    "reference_identity_path",
    "reference_probe_sha256",
    "resolve_dpo_reference_checkpoint",
    "write_reference_identity",
]
