# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Snapshot completed HF adapter exports without importing training
dependencies."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST_PATTERN = re.compile(r"[a-f0-9]{64}\Z")


class AdapterSealUnknownError(OSError):
    """A visible version could not be confirmed durable; retry the same
    ID/content."""


class AdapterArtifactCapacityError(RuntimeError):
    """The immutable store has insufficient space under its configured
    quota."""


class AdapterConflictError(ValueError):
    """An immutable version ID was reused with different content."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_model(directory: str | Path) -> str:
    """Hash a frozen local HF base model, including config/tokenizer/weight
    files."""
    root = Path(directory).resolve()
    suffixes = {".json", ".safetensors", ".bin", ".model", ".tiktoken", ".txt", ".py"}
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix in suffixes)
    if not (root / "config.json").is_file() or not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise ValueError("base-model directory must contain config.json and model weights")
    return hashlib.sha256(_canonical_json({path.name: _file_digest(path) for path in files})).hexdigest()


def _snapshot_files(directory: Path) -> tuple[str, ...]:
    # Sealing handles bytes; production ModelContract validation additionally
    # requires the trusted producer declaration.
    provenance = "producer_manifest.json"
    return (*_ADAPTER_FILES, provenance) if (directory / provenance).exists() else _ADAPTER_FILES


def _manifest(directory: Path, version_id: str, base_model_digest: str) -> dict[str, Any]:
    files = {
        name: {"sha256": _file_digest(directory / name), "size": (directory / name).stat().st_size}
        for name in _snapshot_files(directory)
    }
    source = {}
    if "producer_manifest.json" in files:
        source = json.loads((directory / "producer_manifest.json").read_bytes())
    return _build_manifest(files, source, version_id, base_model_digest)


def _build_manifest(files: dict, source: dict, version_id: str, base_model_digest: str) -> dict[str, Any]:
    # Provenance bytes are checked for integrity, but producer/step/time are not
    # content identity. The execution contract and fixed base are identity.
    if not isinstance(source, dict):
        raise ValueError("producer_manifest.json must be an object")
    manifest = {
        "format_version": 2,
        "version_id": version_id,
        "base_model_digest": base_model_digest,
        "contract": {key: source[key] for key in ("contract", "base_config_digest") if key in source},
        "files": files,
    }
    manifest["content_digest"] = _content_digest(manifest)
    return manifest


def _content_digest(manifest: dict[str, Any]) -> str:
    if not isinstance(manifest, dict) or manifest.get("format_version") != 2:
        raise ValueError("unsupported adapter snapshot manifest format")
    content = {
        "format_version": 2,
        "base_model_digest": manifest["base_model_digest"],
        "contract": manifest["contract"],
        "files": {name: manifest["files"][name] for name in _ADAPTER_FILES},
    }
    return hashlib.sha256(_canonical_json(content)).hexdigest()


@contextmanager
def _commit_lock(store: Path) -> Iterator[None]:
    # All writers use this lock, including same-process threads. This is a
    # filesystem contract: the configured shared filesystem must support flock.
    with (store / ".commit.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _store_bytes(store: Path) -> int:
    retained = sum(p.stat().st_size for p in (store / "versions").rglob("*") if p.is_file())
    for stage in (store / ".staging").iterdir():
        reservation = stage / ".reserved_bytes"
        if reservation.is_file():
            retained += int(reservation.read_text())
        elif stage.is_dir():
            # Existing producer export directories have their own writer. Count
            # visible bytes, but do not mistake them for our owned reservations.
            retained += sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    return retained


@contextmanager
def _reserved_stage(store: Path, size: int, max_bytes: int | None) -> Iterator[Path]:
    with _commit_lock(store):
        if max_bytes is not None and _store_bytes(store) + size > max_bytes:
            raise AdapterArtifactCapacityError("ARTIFACT_CAPACITY_EXCEEDED")
        temporary = tempfile.TemporaryDirectory(prefix="export-", dir=store / ".staging")
        stage = Path(temporary.name)
        try:
            (stage / ".reserved_bytes").write_text(str(size))
        except BaseException:
            temporary.cleanup()
            raise
    try:
        yield stage
    finally:
        # A crashed writer leaves its reservation charged; never infer that an
        # old directory is safe to remove from its age. Normal failures release
        # only this writer's directory, under the same accounting lock.
        with _commit_lock(store):
            temporary.cleanup()


@dataclass(frozen=True)
class AdapterSnapshot:
    """A sealed export; engines must verify its bytes before acknowledging
    load."""

    version_id: str
    digest: str
    base_model_digest: str
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not _VERSION_PATTERN.fullmatch(self.version_id):
            raise ValueError("version_id must be a nonempty, path-safe identifier of at most 128 characters")
        if any(
            not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value)
            for value in (self.digest, self.base_model_digest)
        ):
            raise ValueError("adapter and base-model digests must be lowercase SHA-256 hex strings")

    @property
    def lora_path(self) -> str:
        """Engine registration name, also used in every generate request."""
        return f"relax_policy@{self.version_id}"

    def verify(self) -> None:
        """Reject damaged or changed artifacts, including their manifest."""
        if self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("snapshot directory must be a regular directory")
        for name in (*_snapshot_files(self.path), "manifest.json"):
            path = self.path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"adapter snapshot requires a regular file: {name}")
        actual = json.loads((self.path / "manifest.json").read_bytes())
        if not isinstance(actual, dict):
            raise ValueError("snapshot manifest must be a JSON object")
        expected = _manifest(self.path, self.version_id, self.base_model_digest)
        if actual != expected or _content_digest(expected) != self.digest:
            raise ValueError(f"adapter snapshot checksum mismatch: {self.version_id}")

    def confirm_sealed(self) -> None:
        """Verify and finish a possibly interrupted directory commit.

        Visibility alone is not durability. Callers must use this before
        handing a descriptor to engines, including explicit publication of
        existing files.
        """
        self.verify()
        try:
            for name in (*_snapshot_files(self.path), "manifest.json"):
                with (self.path / name).open("rb") as handle:
                    os.fsync(handle.fileno())
            _sync_directory(self.path)
            _sync_directory(self.path.parent)
            _sync_directory(self.path.parent.parent)
        except OSError as error:
            raise AdapterSealUnknownError(f"SEAL_COMMIT_UNKNOWN: {self.version_id}") from error


def read_snapshot(directory: str | Path) -> AdapterSnapshot:
    """Read a sealed descriptor without trusting its path or claimed
    checksums."""
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("snapshot directory must be a regular directory")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("snapshot requires a regular manifest")
    manifest = json.loads(manifest_path.read_bytes())
    digest = _content_digest(manifest)
    snapshot = AdapterSnapshot(manifest["version_id"], digest, manifest["base_model_digest"], path)
    snapshot.verify()
    return snapshot


def snapshot_adapter(
    export_dir: str | Path,
    store_dir: str | Path,
    *,
    version_id: str,
    base_model_digest: str,
    max_bytes: int | None = None,
) -> AdapterSnapshot:
    """Seal a completed export from ``write_hf_peft_adapter``.

    The producer must finish its consistent training-step export before calling
    this function and leave the source unchanged until it returns. Copying a live
    training directory cannot establish a consistent training-step snapshot.

    Identity covers the base model, canonical config and exact safetensors bytes.
    A temporary sibling directory is renamed into place without replacing any
    existing version. Repeating the same ID/content returns the existing snapshot.
    """
    # Validate identifiers before using them as paths.
    AdapterSnapshot(version_id, "0" * 64, base_model_digest, Path(store_dir))
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer")
    source = Path(export_dir)
    store = Path(store_dir).resolve()
    created_parent_entries = []
    ancestor = store
    while not ancestor.exists():
        created_parent_entries.append(ancestor.parent)
        ancestor = ancestor.parent
    store.mkdir(parents=True, exist_ok=True)
    # mkdir(parents=True) may create more than one ancestor. Persist every new
    # directory entry, not just the eventual version's immediate parent.
    for parent in created_parent_entries:
        _sync_directory(parent)
    versions = store / "versions"
    staging = store / ".staging"
    for directory in (versions, staging):
        if directory.is_symlink():
            raise ValueError("store directories must not be symlinks")
        directory.mkdir(exist_ok=True)
    if (versions / "manifest.json").exists():
        raise ValueError("legacy version named 'versions' needs an explicit store migration")
    destination = versions / version_id
    files = {}
    for name in _snapshot_files(source):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"adapter export requires a regular file: {name}")
        files[name] = {"sha256": "0" * 64, "size": path.stat().st_size}
    config = json.loads((source / "adapter_config.json").read_bytes())
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    config_bytes = _canonical_json(config)
    files["adapter_config.json"] = {"sha256": hashlib.sha256(config_bytes).hexdigest(), "size": len(config_bytes)}
    provenance = (
        json.loads((source / "producer_manifest.json").read_bytes()) if "producer_manifest.json" in files else {}
    )
    planned = _build_manifest(files, provenance, version_id, base_model_digest)
    # Hash strings have fixed length: sizes and small JSON metadata suffice for
    # exact byte admission, before reading/copying the potentially large weights.
    reserved_bytes = sum(item["size"] for item in files.values()) + len(_canonical_json(planned))
    with _commit_lock(store):
        legacy = store / version_id
        if version_id != "versions" and (legacy.exists() or legacy.is_symlink()):
            raise ValueError("unsupported adapter store layout; use a new store and reseal completed exports")
    if destination.exists() or destination.is_symlink():
        # Already-sealed retries need no staging capacity or second weight copy.
        candidate = _manifest(source, version_id, base_model_digest)
        candidate["files"]["adapter_config.json"] = files["adapter_config.json"]
        with _commit_lock(store):
            existing = read_snapshot(destination)
            if existing.digest != _content_digest(candidate):
                raise AdapterConflictError(f"adapter version {version_id!r} already has different content")
            existing.confirm_sealed()
            return existing
    with _reserved_stage(store, reserved_bytes, max_bytes) as stage:
        (stage / "adapter_config.json").write_bytes(config_bytes)
        for name in files.keys() - {"adapter_config.json"}:
            # Bound writes by the reservation even if a producer violates the
            # completed-export contract and grows the file during copying.
            remaining = files[name]["size"]
            with (source / name).open("rb") as reader, (stage / name).open("wb") as writer:
                while remaining:
                    block = reader.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("adapter export changed during snapshot")
                    writer.write(block)
                    remaining -= len(block)
                if reader.read(1):
                    raise ValueError("adapter export changed during snapshot")
        manifest = _manifest(stage, version_id, base_model_digest)
        digest = _content_digest(manifest)
        manifest_bytes = _canonical_json(manifest)
        if sum(item["size"] for item in manifest["files"].values()) + len(manifest_bytes) > reserved_bytes:
            raise ValueError("adapter export changed during snapshot")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        snapshot = AdapterSnapshot(version_id, digest, base_model_digest, destination)
        for name in (*_snapshot_files(stage), "manifest.json"):
            (stage / name).chmod(0o444)
            with (stage / name).open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(stage)
        with _commit_lock(store):
            # Refuse old stores rather than returning a descriptor the managed
            # engines cannot load, or silently reusing a historical version ID.
            legacy = store / version_id
            if version_id != "versions" and (legacy.exists() or legacy.is_symlink()):
                raise ValueError("unsupported adapter store layout; use a new store and reseal completed exports")
            if destination.exists() or destination.is_symlink():
                existing = read_snapshot(destination)
                if existing.digest != digest:
                    raise AdapterConflictError(f"adapter version {version_id!r} already has different content")
                existing.confirm_sealed()
                return existing
            # Reservation -> retained bytes is one locked accounting transition.
            # Producer-owned staging may have grown since admission: recheck it.
            if max_bytes is not None and _store_bytes(store) > max_bytes:
                raise AdapterArtifactCapacityError("ARTIFACT_CAPACITY_EXCEEDED")
            (stage / ".reserved_bytes").unlink()
            os.rename(stage, destination)
            # From this point errors mean UNKNOWN, not rollback. TemporaryDirectory
            # only owns the now-missing staging path; the visible version survives.
            snapshot.confirm_sealed()
        return snapshot
