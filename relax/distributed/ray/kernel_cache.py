# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Portable node-local TorchInductor and Triton cache synchronization.

The shared directory is an append-only store of immutable delta archives.  A
detached agent on every GPU node restores those archives into a node-local,
writable cache before train actors start, periodically publishes new files, and
publishes one final delta when the owning training driver exits.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_AGENT_NAMESPACE = "relax_kernel_cache"
_AGENT_PREFIX = "relax-kernel-cache"
_STATE_FILE = ".relax-kernel-cache-state.json"
_PREPARED_FILE = ".relax-kernel-cache-prepared.json"
_STAGING_DIR = ".relax-kernel-cache-staging"
_CACHE_KINDS = ("inductor", "triton")
_COPY_BUFFER_SIZE = 4 * 1024 * 1024
_MAX_ARCHIVE_FILES = 250_000
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class KernelCacheConfig:
    shared_dir: str
    local_dir: str
    session_id: str
    cache_key: str
    build_fingerprint: str
    sync_interval_sec: int = 900
    lease_timeout_sec: int = 600
    startup_timeout_sec: int = 7200
    compression: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "shared_dir": self.shared_dir,
            "local_dir": self.local_dir,
            "session_id": self.session_id,
            "cache_key": self.cache_key,
            "build_fingerprint": self.build_fingerprint,
            "sync_interval_sec": self.sync_interval_sec,
            "lease_timeout_sec": self.lease_timeout_sec,
            "startup_timeout_sec": self.startup_timeout_sec,
            "compression": self.compression,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KernelCacheConfig":
        return cls(
            shared_dir=str(value["shared_dir"]),
            local_dir=str(value["local_dir"]),
            session_id=str(value["session_id"]),
            cache_key=str(value["cache_key"]),
            build_fingerprint=str(value["build_fingerprint"]),
            sync_interval_sec=int(value.get("sync_interval_sec", 900)),
            lease_timeout_sec=int(value.get("lease_timeout_sec", 600)),
            startup_timeout_sec=int(value.get("startup_timeout_sec", 7200)),
            compression=str(value.get("compression", "none")),
        )


@dataclass(frozen=True)
class SnapshotResult:
    published: bool
    changed_files: int
    archive_bytes: int
    manifest_path: str | None = None


def compute_build_fingerprint() -> str:
    """Hash the compiler ABI, accelerator architecture, and source
    revisions."""
    packages = {}
    for name in ("torch", "triton", "transformer-engine", "flash-linear-attention", "deep-ep"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        gpu = "unknown"
    project_root = Path(__file__).resolve().parents[3]
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        git_head = "unknown"
    source_tree = hashlib.sha256()
    for relative_root in ("relax/backends/megatron", "relax/distributed/ray", "relax/entrypoints", "relax/models"):
        source_root = project_root / relative_root
        for path in sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix in {".jinja", ".json", ".py", ".yaml", ".yml"}
        ):
            source_tree.update(path.relative_to(project_root).as_posix().encode("utf-8"))
            source_tree.update(_sha256_file(path).encode("ascii"))
    critical_sources = [Path(__file__)]
    megatron_root = os.environ.get("MEGATRON")
    if megatron_root:
        critical_sources.extend(
            [
                Path(megatron_root) / "megatron/core/jit.py",
                Path(megatron_root) / "megatron/core/ssm/gated_delta_net.py",
                Path(megatron_root) / "megatron/bridge/peft/utils.py",
            ]
        )
    source_hashes = {str(path): _sha256_file(path) for path in critical_sources if path.is_file()}
    value = {
        "schema": _SCHEMA_VERSION,
        "python": sys.version,
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "packages": packages,
        "gpu": gpu,
        "image": os.environ.get("RELAX_IMAGE_DIGEST", ""),
        "git_head": git_head,
        "source_tree_hash": source_tree.hexdigest(),
        "sources": source_hashes,
        "override": os.environ.get("RELAX_KERNEL_CACHE_BUILD_KEY", ""),
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_consistent_build_fingerprint(fingerprints: dict[str, str]) -> str:
    if not fingerprints:
        raise RuntimeError("Kernel cache requested but no alive GPU nodes were found")
    unique = set(fingerprints.values())
    if len(unique) != 1:
        details = ", ".join(f"{node_id}={value}" for node_id, value in sorted(fingerprints.items()))
        raise RuntimeError(f"Kernel cache build fingerprint differs across GPU nodes: {details}")
    return unique.pop()


def compute_gpu_cluster_build_fingerprint() -> str:
    """Compute one verified fingerprint from the Ray cluster's GPU nodes."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        ray.init(address="auto", configure_logging=False, log_to_driver=False)
    fingerprint_env = {
        name: os.environ[name]
        for name in ("MEGATRON", "PYTHONPATH", "RELAX_IMAGE_DIGEST", "RELAX_KERNEL_CACHE_BUILD_KEY")
        if name in os.environ
    }
    fingerprint_task = ray.remote(num_cpus=0)(compute_build_fingerprint)
    refs: dict[str, Any] = {}
    for node in ray.nodes():
        if not node.get("Alive", False) or float(node.get("Resources", {}).get("GPU", 0)) <= 0:
            continue
        node_id = str(node["NodeID"])
        refs[node_id] = fingerprint_task.options(
            runtime_env={"env_vars": fingerprint_env},
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote()
    values = ray.get(list(refs.values()))
    return _require_consistent_build_fingerprint({node_id: str(value) for node_id, value in zip(refs, values)})


def derive_local_cache_dir(shared_dir: str, cache_key: str = "default", build_fingerprint: str = "") -> str:
    """Return a stable absolute node-local directory for a shared cache."""
    resolved = str(Path(shared_dir).expanduser().resolve())
    identity = f"{resolved}\0{cache_key}\0{build_fingerprint}"
    cache_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"/tmp/relax-kernel-cache/{cache_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _is_cache_file(path: Path, local_dir: Path) -> bool:
    try:
        relative = path.relative_to(local_dir)
    except ValueError:
        return False
    if not relative.parts or relative.parts[0] not in _CACHE_KINDS:
        return False
    if not path.is_file() or path.is_symlink():
        return False
    lowered = path.name.lower()
    if lowered.endswith((".lock", ".tmp", ".partial")):
        return False
    if "lock" in relative.parts or _STAGING_DIR in relative.parts:
        return False
    return not any(part.startswith("tmp.pid_") for part in relative.parts)


def _iter_cache_files(local_dir: Path) -> Iterable[Path]:
    for kind in _CACHE_KINDS:
        root = local_dir / kind
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if _is_cache_file(path, local_dir):
                yield path


def _validate_member_name(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe kernel cache archive member: {name!r}")
    if not relative.parts or relative.parts[0] not in _CACHE_KINDS:
        raise ValueError(f"Unexpected kernel cache archive member: {name!r}")
    return relative


class LocalKernelCacheStore:
    """Restore and incrementally snapshot one node's compiler caches."""

    def __init__(self, config: KernelCacheConfig, node_id: str) -> None:
        self.config = config
        self.node_id = node_id
        self.shared_root = Path(config.shared_dir).expanduser().resolve()
        self.shared_dir = (
            self.shared_root
            / f"v{_SCHEMA_VERSION}"
            / "profiles"
            / self._safe_component(config.cache_key)
            / self._safe_component(config.build_fingerprint)
        )
        self.local_dir = Path(config.local_dir).expanduser().resolve()
        if self.local_dir == self.shared_root or self.shared_root in self.local_dir.parents:
            raise ValueError("Kernel cache local_dir must not be inside the shared cache directory")
        self.state_path = self.local_dir / _STATE_FILE
        self.prepared_path = self.local_dir / _PREPARED_FILE
        self._publisher_lock = threading.Lock()
        self._state = self._load_state()
        self._sequence = int(self._state.get("sequence", 0))

    @property
    def inductor_dir(self) -> Path:
        return self.local_dir / "inductor"

    @property
    def triton_dir(self) -> Path:
        return self.local_dir / "triton"

    def prepare(self) -> dict[str, Any]:
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.inductor_dir.mkdir(parents=True, exist_ok=True)
        self.triton_dir.mkdir(parents=True, exist_ok=True)

        # A previous driver may have been SIGKILLed while this node survived.
        # Publish those local additions before overlaying newer shared shards.
        salvage = self.snapshot("startup-salvage")
        restored = self.restore()
        self._rebuild_state()
        _atomic_json_dump(self.prepared_path, self._prepared_identity())
        return {
            "node_id": self.node_id,
            "restored_files": restored,
            "salvaged_files": salvage.changed_files,
            "local_dir": str(self.local_dir),
        }

    def verify_prepared(self) -> dict[str, Any]:
        try:
            with self.prepared_path.open(encoding="utf-8") as stream:
                prepared = json.load(stream)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Kernel cache was not prepared successfully on node {self.node_id}") from exc
        expected = self._prepared_identity()
        if any(prepared.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"Kernel cache prepared marker is incompatible on node {self.node_id}")
        return {"node_id": self.node_id, "local_dir": str(self.local_dir), "prepared": True}

    def restore(self) -> int:
        with self._publisher_lock, self._node_file_lock():
            return self._restore_locked()

    def _restore_locked(self) -> int:
        manifests = self._ready_manifests()
        restored = 0
        known_hashes: dict[str, str] = {}
        for manifest_path, manifest in manifests:
            try:
                archive_path = (self.shared_dir / str(manifest["archive"])).resolve()
                archive_path.relative_to(self.shared_dir)
            except (KeyError, ValueError):
                logger.warning("Skipping kernel cache manifest with unsafe archive path: %s", manifest_path)
                continue
            if not archive_path.is_file():
                logger.warning("Kernel cache archive is missing: %s", archive_path)
                continue
            if archive_path.stat().st_size > _MAX_ARCHIVE_BYTES:
                logger.warning("Skipping oversized kernel cache archive: %s", archive_path)
                continue
            expected_archive_hash = manifest.get("archive_sha256")
            if expected_archive_hash and _sha256_file(archive_path) != expected_archive_hash:
                logger.warning("Kernel cache archive checksum mismatch: %s", archive_path)
                continue
            expected_files = {entry["path"]: entry for entry in manifest.get("files", [])}
            extracted_bytes = sum(int(entry.get("size", 0)) for entry in expected_files.values())
            if (
                len(expected_files) > _MAX_ARCHIVE_FILES
                or int(manifest.get("archive_bytes", 0)) > _MAX_ARCHIVE_BYTES
                or extracted_bytes > _MAX_ARCHIVE_BYTES
            ):
                logger.warning("Skipping oversized kernel cache shard: %s", manifest_path)
                continue
            conflict = False
            for relative, entry in expected_files.items():
                try:
                    _validate_member_name(relative)
                except ValueError:
                    conflict = True
                    break
                expected_hash = str(entry.get("sha256", ""))
                if relative in known_hashes and known_hashes[relative] != expected_hash:
                    conflict = True
                    break
                destination = self.local_dir / Path(*PurePosixPath(relative).parts)
                if destination.is_file() and _sha256_file(destination) != expected_hash:
                    conflict = True
                    break
            if conflict:
                logger.warning("Skipping conflicting kernel cache shard as a unit: %s", manifest_path)
                continue
            mode = "r:gz" if manifest.get("compression") == "gzip" else "r:"
            try:
                self._validate_archive_members(archive_path, mode, expected_files)
                with tarfile.open(archive_path, mode) as archive:
                    for member in archive:
                        relative = _validate_member_name(member.name)
                        if member.isdir():
                            (self.local_dir / Path(*relative.parts)).mkdir(parents=True, exist_ok=True)
                            continue
                        if not member.isfile():
                            raise ValueError(f"Unsupported kernel cache archive member: {member.name!r}")
                        entry = expected_files.get(member.name)
                        if entry is None:
                            raise ValueError(f"Kernel cache member is absent from manifest: {member.name!r}")
                        if member.size != int(entry.get("size", -1)):
                            raise ValueError(f"Kernel cache member size mismatch: {member.name!r}")
                        expected_hash = str(entry["sha256"])
                        destination = self.local_dir / Path(*relative.parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.is_file() and _sha256_file(destination) == expected_hash:
                            known_hashes[member.name] = expected_hash
                            continue
                        if destination.exists():
                            raise ValueError(f"Kernel cache local path is not a regular file: {destination}")
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"Unable to read kernel cache member: {member.name!r}")
                        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
                        digest = hashlib.sha256()
                        try:
                            with temporary.open("wb") as output:
                                while chunk := extracted.read(_COPY_BUFFER_SIZE):
                                    output.write(chunk)
                                    digest.update(chunk)
                                output.flush()
                                os.fsync(output.fileno())
                            if digest.hexdigest() != expected_hash:
                                raise ValueError(f"Kernel cache member checksum mismatch: {member.name!r}")
                            os.chmod(temporary, int(entry.get("mode", 0o600)) & 0o777)
                            mtime_ns = int(entry.get("mtime_ns", time.time_ns()))
                            os.utime(temporary, ns=(mtime_ns, mtime_ns))
                            os.replace(temporary, destination)
                        finally:
                            temporary.unlink(missing_ok=True)
                        known_hashes[member.name] = expected_hash
                        restored += 1
            except (OSError, tarfile.TarError, ValueError) as exc:
                logger.warning("Skipping invalid kernel cache shard %s: %s", manifest_path, exc)
        return restored

    def snapshot(self, reason: str) -> SnapshotResult:
        with self._publisher_lock, self._node_file_lock():
            self._state = self._load_state()
            self._sequence = int(self._state.get("sequence", 0))
            return self._snapshot_locked(reason)

    def _snapshot_locked(self, reason: str) -> SnapshotResult:
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        incomplete_group_dirs = self._incomplete_triton_group_dirs()
        staging = (
            self.local_dir
            / _STAGING_DIR
            / f"{self._safe_component(self.config.session_id)}-{self._sequence}-{uuid.uuid4().hex}"
        )
        staging.mkdir(parents=True, exist_ok=False)
        changed: list[dict[str, Any]] = []
        state_files = self._state.setdefault("files", {})
        try:
            for source in _iter_cache_files(self.local_dir):
                if any(
                    group_dir == source.parent or group_dir in source.parents for group_dir in incomplete_group_dirs
                ):
                    continue
                relative = source.relative_to(self.local_dir).as_posix()
                before = source.stat()
                previous = state_files.get(relative)
                if (
                    previous
                    and int(previous.get("size", -1)) == before.st_size
                    and int(previous.get("mtime_ns", -1)) == before.st_mtime_ns
                ):
                    continue
                snapshot_path = staging / relative
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, snapshot_path)
                except OSError:
                    shutil.copy2(source, snapshot_path)
                after = source.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    snapshot_path.unlink(missing_ok=True)
                    continue
                digest = _sha256_file(snapshot_path)
                entry = {
                    "path": relative,
                    "size": snapshot_path.stat().st_size,
                    "mtime_ns": snapshot_path.stat().st_mtime_ns,
                    "mode": snapshot_path.stat().st_mode & 0o777,
                    "sha256": digest,
                }
                if previous and previous.get("sha256") == digest:
                    state_files[relative] = entry
                    snapshot_path.unlink(missing_ok=True)
                    continue
                changed.append(entry)

            if not changed:
                self._save_state()
                return SnapshotResult(False, 0, 0)
            if len(changed) > _MAX_ARCHIVE_FILES or sum(int(entry["size"]) for entry in changed) > _MAX_ARCHIVE_BYTES:
                raise ValueError("Kernel cache delta exceeds the publisher quota")

            shard_dir = (
                self.shared_dir
                / "incoming"
                / self._safe_component(self.config.session_id)
                / self._safe_component(self.node_id)
            )
            shard_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".tar.gz" if self.config.compression == "gzip" else ".tar"
            archive_name = f"{self._sequence:06d}{suffix}"
            archive_ready = shard_dir / f"{archive_name}.ready"
            archive_partial = shard_dir / f".{archive_name}.{uuid.uuid4().hex}.partial"
            tar_mode = "w:gz" if self.config.compression == "gzip" else "w:"
            tar_kwargs = {"compresslevel": 1} if self.config.compression == "gzip" else {}
            with tarfile.open(archive_partial, tar_mode, **tar_kwargs) as archive:
                archive.dereference = True
                for entry in changed:
                    archive.add(staging / entry["path"], arcname=entry["path"], recursive=False)
            with archive_partial.open("rb") as stream:
                os.fsync(stream.fileno())
            archive_hash = _sha256_file(archive_partial)
            os.replace(archive_partial, archive_ready)

            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "cache_key": self.config.cache_key,
                "build_fingerprint": self.config.build_fingerprint,
                "local_dir": str(self.local_dir),
                "session_id": self.config.session_id,
                "node_id": self.node_id,
                "sequence": self._sequence,
                "created_ns": time.time_ns(),
                "reason": reason,
                "compression": self.config.compression,
                "archive": archive_ready.relative_to(self.shared_dir).as_posix(),
                "archive_sha256": archive_hash,
                "archive_bytes": archive_ready.stat().st_size,
                "files": changed,
            }
            manifest_ready = shard_dir / f"{self._sequence:06d}.json.ready"
            _atomic_json_dump(manifest_ready, manifest)
            for entry in changed:
                state_files[entry["path"]] = entry
            self._sequence += 1
            self._state["sequence"] = self._sequence
            self._save_state()
            return SnapshotResult(
                True,
                len(changed),
                archive_ready.stat().st_size,
                str(manifest_ready),
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _ready_manifests(self) -> list[tuple[Path, dict[str, Any]]]:
        manifests: list[tuple[Path, dict[str, Any]]] = []
        incoming = self.shared_dir / "incoming"
        if not incoming.exists():
            return manifests
        for path in incoming.glob("*/*/*.json.ready"):
            try:
                with path.open(encoding="utf-8") as stream:
                    manifest = json.load(stream)
                if int(manifest.get("schema_version", -1)) != _SCHEMA_VERSION:
                    logger.warning("Skipping unsupported kernel cache manifest: %s", path)
                    continue
                if (
                    manifest.get("cache_key") != self.config.cache_key
                    or manifest.get("build_fingerprint") != self.config.build_fingerprint
                    or manifest.get("local_dir") != str(self.local_dir)
                ):
                    logger.warning("Skipping incompatible kernel cache manifest: %s", path)
                    continue
                manifests.append((path, manifest))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("Skipping unreadable kernel cache manifest %s: %s", path, exc)
        manifests.sort(
            key=lambda item: (
                int(item[1].get("created_ns", 0)),
                str(item[1].get("session_id", "")),
                str(item[1].get("node_id", "")),
                int(item[1].get("sequence", 0)),
            )
        )
        return manifests

    def _load_state(self) -> dict[str, Any]:
        try:
            with self.state_path.open(encoding="utf-8") as stream:
                state = json.load(stream)
            if (
                state.get("shared_dir") == str(self.shared_dir)
                and state.get("local_dir") == str(self.local_dir)
                and state.get("cache_key") == self.config.cache_key
                and state.get("build_fingerprint") == self.config.build_fingerprint
                and state.get("schema_version") == _SCHEMA_VERSION
            ):
                return state
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return {
            "schema_version": _SCHEMA_VERSION,
            "shared_dir": str(self.shared_dir),
            "local_dir": str(self.local_dir),
            "cache_key": self.config.cache_key,
            "build_fingerprint": self.config.build_fingerprint,
            "sequence": 0,
            "files": {},
        }

    def _rebuild_state(self) -> None:
        files: dict[str, dict[str, Any]] = {}
        incomplete_group_dirs = self._incomplete_triton_group_dirs()
        for path in _iter_cache_files(self.local_dir):
            if any(group_dir == path.parent or group_dir in path.parents for group_dir in incomplete_group_dirs):
                continue
            stat = path.stat()
            relative = path.relative_to(self.local_dir).as_posix()
            files[relative] = {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "mode": stat.st_mode & 0o777,
                "sha256": _sha256_file(path),
            }
        self._state["files"] = files
        self._save_state()

    def _save_state(self) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._state["schema_version"] = _SCHEMA_VERSION
        self._state["shared_dir"] = str(self.shared_dir)
        self._state["local_dir"] = str(self.local_dir)
        self._state["cache_key"] = self.config.cache_key
        self._state["build_fingerprint"] = self.config.build_fingerprint
        self._state["sequence"] = self._sequence
        _atomic_json_dump(self.state_path, self._state)

    def _prepared_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "shared_dir": str(self.shared_dir),
            "local_dir": str(self.local_dir),
            "cache_key": self.config.cache_key,
            "build_fingerprint": self.config.build_fingerprint,
        }

    def _incomplete_triton_group_dirs(self) -> set[Path]:
        incomplete = set()
        for group_path in self.triton_dir.rglob("__grp__*.json"):
            try:
                with group_path.open(encoding="utf-8") as stream:
                    group = json.load(stream)
                child_paths = group.get("child_paths", {})
                if not isinstance(child_paths, dict):
                    incomplete.add(group_path.parent)
                    continue
                for raw_path in child_paths.values():
                    child = Path(str(raw_path)).expanduser().resolve()
                    child.relative_to(self.triton_dir)
                    if not child.is_file():
                        incomplete.add(group_path.parent)
                        break
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                incomplete.add(group_path.parent)
        return incomplete

    @staticmethod
    def _validate_archive_members(
        archive_path: Path,
        mode: str,
        expected_files: dict[str, dict[str, Any]],
    ) -> None:
        seen = set()
        total_bytes = 0
        with tarfile.open(archive_path, mode) as archive:
            for member in archive:
                if len(seen) >= _MAX_ARCHIVE_FILES:
                    raise ValueError("Kernel cache archive contains too many members")
                relative = _validate_member_name(member.name)
                name = relative.as_posix()
                if name in seen:
                    raise ValueError(f"Duplicate kernel cache archive member: {name!r}")
                if not member.isfile():
                    raise ValueError(f"Unsupported kernel cache archive member: {name!r}")
                entry = expected_files.get(name)
                if entry is None or member.size != int(entry.get("size", -1)):
                    raise ValueError(f"Kernel cache archive member does not match manifest: {name!r}")
                total_bytes += member.size
                if total_bytes > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Kernel cache archive expands beyond the byte quota")
                seen.add(name)
        if seen != set(expected_files):
            raise ValueError("Kernel cache archive file set does not match manifest")

    @contextmanager
    def _node_file_lock(self):
        import fcntl

        self.local_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.local_dir / ".relax-kernel-cache-publisher.lock"
        with lock_path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _safe_component(value: str) -> str:
        safe = "".join(char if char.isalnum() or char in "-." else "_" for char in value).strip(".")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"{safe[:64] or 'value'}-{digest}"


class KernelCacheAgent:
    """Detached, node-pinned publisher that survives the training Ray Job."""

    def __init__(self, config: dict[str, Any], node_id: str) -> None:
        self.config = KernelCacheConfig.from_dict(config)
        self.node_id = node_id
        self.store = LocalKernelCacheStore(self.config, node_id)
        self._last_heartbeat = time.monotonic()
        self._last_snapshot = time.monotonic()
        self._stop = threading.Event()
        self._finalize_lock = threading.Lock()
        self._prepared = False
        self._finalized = False
        self._claimed = False
        self._monitor: threading.Thread | None = None
        try:
            os.nice(10)
        except OSError:
            pass

    def prepare(self) -> dict[str, Any]:
        actual_fingerprint = compute_build_fingerprint()
        if actual_fingerprint != self.config.build_fingerprint:
            raise RuntimeError(
                "Kernel cache build fingerprint differs across nodes: "
                f"expected={self.config.build_fingerprint} actual={actual_fingerprint} node={self.node_id}"
            )
        result = self.store.prepare()
        self._start_monitor()
        return result

    def attach(self) -> dict[str, Any]:
        actual_fingerprint = compute_build_fingerprint()
        if actual_fingerprint != self.config.build_fingerprint:
            raise RuntimeError(
                "Kernel cache build fingerprint differs across nodes: "
                f"expected={self.config.build_fingerprint} actual={actual_fingerprint} node={self.node_id}"
            )
        result = self.store.verify_prepared()
        self._start_monitor()
        return result

    def _start_monitor(self) -> None:
        self._prepared = True
        self._last_heartbeat = time.monotonic()
        self._last_snapshot = time.monotonic()
        self._monitor = threading.Thread(target=self._monitor_loop, name="kernel-cache-monitor", daemon=True)
        self._monitor.start()

    def claim(self) -> dict[str, Any]:
        with self._finalize_lock:
            if not self._prepared or self._finalized or self._stop.is_set():
                raise RuntimeError(f"Kernel cache agent is not active on node {self.node_id}")
            self._claimed = True
            self._last_heartbeat = time.monotonic()
            return {"node_id": self.node_id, "claimed": True}

    def heartbeat(self) -> bool:
        with self._finalize_lock:
            if not self._claimed or self._finalized or self._stop.is_set():
                return False
            self._last_heartbeat = time.monotonic()
            return True

    def request_finalize(self, reason: str) -> None:
        # This is only a SIGTERM/SIGINT fallback.  The driver performs the
        # authoritative final scan after train actors and Serve have stopped.
        self._stop.set()
        thread = threading.Thread(
            target=self._finalize_after_delay,
            args=(reason, 30),
            name="kernel-cache-finalize",
            daemon=True,
        )
        thread.start()

    def finalize(self, reason: str) -> dict[str, Any]:
        self._stop.set()
        result = self._finalize(reason)
        return {
            "node_id": self.node_id,
            "published": result.published,
            "changed_files": result.changed_files,
            "archive_bytes": result.archive_bytes,
            "manifest_path": result.manifest_path,
        }

    def status(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "prepared": self._prepared,
            "finalized": self._finalized,
            "claimed": self._claimed,
            "seconds_since_heartbeat": time.monotonic() - self._last_heartbeat,
        }

    def _monitor_loop(self) -> None:
        poll_interval = min(30, max(1, self.config.sync_interval_sec or 30))
        while not self._stop.wait(poll_interval):
            now = time.monotonic()
            timeout = self.config.lease_timeout_sec if self._claimed else self.config.startup_timeout_sec
            if now - self._last_heartbeat > timeout:
                self._finalize("heartbeat-timeout")
                return
            if self.config.sync_interval_sec > 0 and now - self._last_snapshot >= self.config.sync_interval_sec:
                try:
                    result = self.store.snapshot("periodic")
                    logger.info(
                        "Kernel cache periodic snapshot: node=%s files=%s bytes=%s",
                        self.node_id,
                        result.changed_files,
                        result.archive_bytes,
                    )
                except Exception as exc:  # noqa: BLE001 - cache publishing is best effort
                    logger.warning("Kernel cache periodic snapshot failed on %s: %s", self.node_id, exc)
                self._last_snapshot = time.monotonic()

    def _finalize(self, reason: str) -> SnapshotResult:
        with self._finalize_lock:
            try:
                result = self.store.snapshot(reason)
                logger.info(
                    "Kernel cache final snapshot: node=%s reason=%s files=%s bytes=%s",
                    self.node_id,
                    reason,
                    result.changed_files,
                    result.archive_bytes,
                )
                return result
            except Exception as exc:  # noqa: BLE001 - do not change the training exit status
                logger.warning("Kernel cache final snapshot failed on %s: %s", self.node_id, exc)
                return SnapshotResult(False, 0, 0)
            finally:
                self._finalized = True
                self._stop.set()
                # Keep the detached actor addressable long enough for the
                # driver's post-Serve final scan, even if the fallback fired.
                exit_timer = threading.Timer(600, os._exit, args=(0,))
                exit_timer.daemon = True
                exit_timer.start()

    def _finalize_after_delay(self, reason: str, delay_sec: int) -> None:
        # Give Ray time to terminate train actors so finalized cache files stop
        # changing. The detached agent survives the training Job's SIGKILL.
        time.sleep(delay_sec)
        self._finalize(reason)


def _agent_name(session_id: str, cache_key: str, node_id: str) -> str:
    safe_session = LocalKernelCacheStore._safe_component(session_id)
    return f"{_AGENT_PREFIX}-{cache_key}-{safe_session}-{node_id[:12]}"


def _agent_profile_key(shared_dir: str, cache_key: str, build_fingerprint: str) -> str:
    resolved = str(Path(shared_dir).expanduser().resolve())
    identity = f"{resolved}\0{cache_key}\0{build_fingerprint}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def _session_agents(session_id: str) -> list[Any]:
    import ray

    cache_key = _agent_profile_key(
        os.environ["RELAX_KERNEL_CACHE_DIR"],
        os.environ["RELAX_KERNEL_CACHE_KEY"],
        os.environ["RELAX_KERNEL_CACHE_BUILD_FINGERPRINT"],
    )
    prefix = f"{_AGENT_PREFIX}-{cache_key}-{LocalKernelCacheStore._safe_component(session_id)}-"
    actors = []
    for item in ray.util.list_named_actors(all_namespaces=True):
        if item["namespace"] != _AGENT_NAMESPACE or not item["name"].startswith(prefix):
            continue
        try:
            actors.append(ray.get_actor(item["name"], namespace=_AGENT_NAMESPACE))
        except ValueError:
            continue
    return actors


def prepare_kernel_cache_agents(config: KernelCacheConfig, *, attach_only: bool = False) -> list[dict[str, Any]]:
    """Create one detached cache agent on every alive GPU node."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        ray.init(address="auto", namespace=_AGENT_NAMESPACE, log_to_driver=False)
    cache_key = _agent_profile_key(config.shared_dir, config.cache_key, config.build_fingerprint)
    agent_type = ray.remote(num_cpus=0)(KernelCacheAgent)
    fingerprint_env = {
        name: os.environ[name]
        for name in ("MEGATRON", "PYTHONPATH", "RELAX_IMAGE_DIGEST", "RELAX_KERNEL_CACHE_BUILD_KEY")
        if name in os.environ
    }
    handles = []
    try:
        # A prior session using the same profile may have been SIGKILLed.  It
        # must be quiesced before a new agent restores into the same local root.
        prefix = f"{_AGENT_PREFIX}-{cache_key}-"
        stale = []
        for item in ray.util.list_named_actors(all_namespaces=True):
            if item["namespace"] == _AGENT_NAMESPACE and item["name"].startswith(prefix):
                try:
                    stale.append(ray.get_actor(item["name"], namespace=_AGENT_NAMESPACE))
                except ValueError:
                    pass
        if stale:
            refs = [actor.finalize.remote("superseded-session") for actor in stale]
            try:
                ray.get(refs, timeout=max(30, config.lease_timeout_sec))
            finally:
                for actor in stale:
                    ray.kill(actor, no_restart=True)

        for node in ray.nodes():
            if not node.get("Alive", False) or float(node.get("Resources", {}).get("GPU", 0)) <= 0:
                continue
            node_id = str(node["NodeID"])
            name = _agent_name(config.session_id, cache_key, node_id)
            handle = agent_type.options(
                name=name,
                lifetime="detached",
                runtime_env={"env_vars": fingerprint_env},
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(config.as_dict(), node_id)
            handles.append(handle)
        if not handles:
            raise RuntimeError("Kernel cache requested but no alive GPU nodes were found")
        refs = [handle.attach.remote() if attach_only else handle.prepare.remote() for handle in handles]
        return ray.get(refs, timeout=max(1, config.lease_timeout_sec))
    except Exception:
        for handle in handles:
            try:
                ray.kill(handle, no_restart=True)
            except Exception:  # noqa: BLE001 - cleanup after a failed prepare
                pass
        raise


def prepare_local_kernel_cache(config: KernelCacheConfig, node_id: str | None = None) -> dict[str, Any]:
    """Restore the cache on the current node before it joins a Ray cluster."""
    actual_fingerprint = compute_build_fingerprint()
    if actual_fingerprint != config.build_fingerprint:
        raise RuntimeError(
            "Kernel cache build fingerprint changed during local preparation: "
            f"expected={config.build_fingerprint} actual={actual_fingerprint}"
        )
    local_node_id = node_id or os.environ.get("POD_UID") or os.environ.get("HOST_IP") or platform.node()
    return LocalKernelCacheStore(config, local_node_id).prepare()


class KernelCacheHeartbeat:
    """Driver-side heartbeat and finalization client for detached agents."""

    def __init__(self, session_id: str, interval_sec: int = 30) -> None:
        self.session_id = session_id
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="kernel-cache-heartbeat", daemon=True)

    def start(self) -> None:
        import ray

        refs = [actor.claim.remote() for actor in _session_agents(self.session_id)]
        if not refs:
            raise RuntimeError("No kernel cache agents were found for this training session")
        claimed = {str(item["node_id"]) for item in ray.get(refs, timeout=60)}
        expected = {
            str(node["NodeID"])
            for node in ray.nodes()
            if node.get("Alive", False) and float(node.get("Resources", {}).get("GPU", 0)) > 0
        }
        if claimed != expected:
            raise RuntimeError(
                f"Kernel cache agent coverage mismatch: expected={sorted(expected)} claimed={sorted(claimed)}"
            )
        self._thread.start()

    def request_finalize(self, reason: str) -> None:
        for actor in _session_agents(self.session_id):
            actor.request_finalize.remote(reason)

    def finalize(self, reason: str, timeout_sec: int) -> list[dict[str, Any]]:
        import ray

        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1, self.interval_sec + 1))
        refs = [actor.finalize.remote(reason) for actor in _session_agents(self.session_id)]
        if not refs:
            return []
        try:
            return ray.get(refs, timeout=max(1, timeout_sec))
        except Exception as exc:  # noqa: BLE001 - do not change the training exit status
            logger.warning("Timed out finalizing kernel cache agents: %s", exc)
            return []

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                for actor in _session_agents(self.session_id):
                    actor.heartbeat.remote()
            except Exception as exc:  # noqa: BLE001 - heartbeat is best effort
                logger.warning("Kernel cache heartbeat failed: %s", exc)


def start_kernel_cache_heartbeat_from_env() -> KernelCacheHeartbeat | None:
    if not os.environ.get("RELAX_KERNEL_CACHE_DIR"):
        return None
    session_id = os.environ.get("RELAX_KERNEL_CACHE_SESSION_ID")
    if not session_id:
        logger.warning("RELAX_KERNEL_CACHE_DIR is set but RELAX_KERNEL_CACHE_SESSION_ID is missing")
        return None
    heartbeat = KernelCacheHeartbeat(
        session_id,
        interval_sec=int(os.environ.get("RELAX_KERNEL_CACHE_HEARTBEAT_INTERVAL_SEC", "30")),
    )
    heartbeat.start()
    return heartbeat
