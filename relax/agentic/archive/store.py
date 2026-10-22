# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Filesystem store with exclusive ownership for each open instance."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .schema import (
    ASSET_PREFIX,
    SCHEMA_VERSION,
    asset_digest,
    asset_ids,
    encode,
    materialize,
    path_fingerprint,
    quality,
    task_key,
    validate,
)


LOCAL_FILESYSTEMS = {"ext4", "ext3", "ext2", "xfs", "btrfs", "tmpfs", "overlay"}


def filesystem_type(path: Path) -> str:
    """Fail closed on shared/unknown filesystems; never infer flock
    semantics."""
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4]))
        if path == mount or mount in path.parents:
            matches.append((len(str(mount)), right.split()[0]))
    return max(matches)[1] if matches else "unknown"


class ArchiveStore:
    def __init__(
        self,
        root: str | Path,
        max_bytes: int | None = None,
        per_task_limit: int = 5,
        *,
        task_limits: dict[str, int] | None = None,
        allow_shared_filesystem: bool = False,
        lock_timeout: float = 0,
        new_asset_layout: str = "flat",
    ) -> None:
        self.root = Path(root).resolve()
        probe = self.root
        while not probe.exists():
            probe = probe.parent
        fs = filesystem_type(probe)
        if fs not in LOCAL_FILESYSTEMS and not allow_shared_filesystem:
            raise ValueError(f"Archive requires a local filesystem, found {fs}; use the central HTTP writer")
        if max_bytes is not None and (max_bytes < 1024 or per_task_limit < 1):
            raise ValueError("max_bytes must be >= 1024 and per_task_limit >= 1")
        if new_asset_layout not in ("flat", "task"):
            raise ValueError("unsupported asset layout")
        if task_limits is not None and (
            not isinstance(task_limits, dict)
            or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in task_limits.items())
        ):
            raise ValueError("task_limits must map task IDs to nonnegative integers")
        if not self.root.exists():
            if max_bytes is None:
                raise ValueError("archive does not exist")
            self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "archive.json"
        self._lock = (self.root / ".owner.lock").open("a+b")
        deadline = time.monotonic() + lock_timeout
        try:
            while True:
                try:
                    fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "archive is owned by another process; stop its server before export/recovery"
                        ) from None
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        except OSError:
            self._lock.close()
            raise
        except BaseException:
            self._lock.close()
            raise
        self.mutex = threading.RLock()
        try:
            if marker.exists():
                config = json.loads(marker.read_text())
                if config.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("unsupported archive schema")
                if max_bytes is not None and (max_bytes, per_task_limit) != (
                    config["max_bytes"],
                    config["per_task_limit"],
                ):
                    raise ValueError("existing archive budget/quota differ; reuse its original settings")
            else:
                if max_bytes is None:
                    raise ValueError("max_bytes is required for a new archive")
                if any(p.name != ".owner.lock" for p in self.root.iterdir()):
                    raise ValueError("archive_dir must be a new/empty dedicated directory")
                config = dict(schema_version=SCHEMA_VERSION, max_bytes=max_bytes, per_task_limit=per_task_limit)
                if new_asset_layout != "flat":
                    config["asset_layout"] = new_asset_layout
                if task_limits:
                    config["task_limits"] = task_limits
                if len(encode(config)) > max_bytes:
                    raise ValueError("archive configuration exceeds capacity")
                marker.write_bytes(encode(config))
            if task_limits is not None and task_limits != config.get("task_limits", {}):
                raise ValueError("existing archive task limits differ; reuse its original settings")
            self.max_bytes = config["max_bytes"]
            self.per_task_limit = config["per_task_limit"]
            self.task_limits = config.get("task_limits", {})
            self.asset_layout = config.get("asset_layout", "flat")
            if self.asset_layout not in ("flat", "task"):
                raise ValueError("unsupported asset layout")
            for name in ("staging", "trajectories", "assets"):
                (self.root / name).mkdir(exist_ok=True)
            # Reject unexpected paths/symlinks before any cleanup or accounting.
            for path in self.root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("symlinks are not supported inside the archive")
                if (
                    path.is_file()
                    and path.parent != self.root
                    and path.parent.name not in ("staging", "trajectories", "assets")
                    and self.root / "assets" not in path.parents
                ):
                    raise ValueError("unexpected archive layout")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._lock.close()

    def usage(self) -> int:
        # Logical file bytes, not filesystem allocation blocks; unique assets
        # have one filename. Includes staging, metadata and temporary writes.
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def _path(self, identity: str, folder: str = "staging") -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise ValueError("invalid trajectory ID")
        return self.root / folder / f"{identity}.jsonl"

    def _events(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()]

    def asset_path(self, key: str) -> Path:
        asset_digest(key)
        return self.root / "assets" / key

    def retained(self) -> list[dict]:
        return [materialize(self._events(p)) for p in sorted((self.root / "trajectories").glob("*.jsonl"))]

    def _gc(self, protected: set[str] | None = None) -> None:
        refs: set[str] = set(protected or ())
        for folder in ("staging", "trajectories"):
            for path in (self.root / folder).glob("*.jsonl"):
                try:
                    refs.update(asset_ids(self._events(path)))
                except (ValueError, OSError):
                    return  # A partial write is not evidence that assets are unreferenced.
        assets_root = self.root / "assets"
        for path in assets_root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(assets_root).as_posix()
            if path.name.startswith(".pending-"):
                path.unlink()
            elif key not in refs:
                try:
                    asset_digest(key)
                except ValueError:
                    continue
                path.unlink()
        for directory in sorted(assets_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()

    def recover(self) -> int:
        """Exclusive-owner recovery, never a timeout-based live-session
        cleaner."""
        with self.mutex:
            paths = list((self.root / "staging").glob("*.jsonl"))
            for path in paths:
                path.unlink()
            self._gc()
            return len(paths)

    def _delete(self, trajectory: dict, protected: set[str] | None = None) -> None:
        self._path(trajectory["trajectory_id"], "trajectories").unlink(missing_ok=True)
        self._gc(protected)

    def _reserve(self, size: int, candidate: dict) -> None:
        while self.usage() + size > self.max_bytes:
            kept = self.retained()
            counts: dict[tuple, int] = {}
            for t in kept:
                counts[task_key(t)] = counts.get(task_key(t), 0) + 1
            # Preserve the last representative of every task. Capacity pressure
            # may evict extras; if those do not suffice, reject the candidate.
            extras = [t for t in kept if counts[task_key(t)] > 1]
            if not extras:
                raise ValueError("capacity_exhausted: staging and retained data exceed budget")
            victim = max(extras, key=lambda t: (counts[task_key(t)], quality(t)))
            self._delete(victim, asset_ids(candidate))

    def _externalize(
        self, value: Any, assets: dict[str, bytes], *, prefix: str | None = None, index: list[int] | None = None
    ) -> Any:
        if index is None:
            index = [0]
        if isinstance(value, str) and value.startswith("data:"):
            header, content = value.split(",", 1)
            if ";base64" not in header or not header.startswith("data:image/"):
                raise ValueError("unsupported inline resource")
            raw = base64.b64decode(content, validate=True)
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(raw)) as image:
                extension = {"JPEG": "jpg", "TIFF": "tif"}.get(image.format, (image.format or "").lower())
                image.verify()
            identity = hashlib.sha256(raw).hexdigest()
            if prefix is not None:
                identity = f"{prefix}/{index[0]:06d}-{identity}.{extension}"
                asset_digest(identity)
                index[0] += 1
            assets[identity] = raw
            return ASSET_PREFIX + identity
        if isinstance(value, dict):
            # Remote/local file URLs cannot be assumed to be the bytes the model
            # saw. Require the producer to supply inline bytes at capture time.
            if value.get("type") in ("image_url", "image"):
                source = value.get("image_url", value.get("image"))
                source = source.get("url") if isinstance(source, dict) else source
                if not isinstance(source, str) or not source.startswith(("data:image/", ASSET_PREFIX)):
                    raise ValueError("image source is not captured inline")
            return {k: self._externalize(v, assets, prefix=prefix, index=index) for k, v in value.items()}
        if isinstance(value, list):
            return [self._externalize(v, assets, prefix=prefix, index=index) for v in value]
        return value

    def begin(self, metadata: dict) -> str:
        with self.mutex:
            identity = uuid.uuid4().hex
            data = dict(metadata, schema_version=SCHEMA_VERSION, trajectory_id=identity)
            try:
                self._append(identity, {"kind": "trajectory", "data": data}, create=True)
            except Exception:
                self.abort(identity)
                raise
            return identity

    def _append(self, identity: str, event: dict, *, create: bool = False) -> None:
        path = self._path(identity)
        if not create and not path.exists():
            raise ValueError("unknown or expired candidate")
        assets: dict[str, bytes] = {}
        prefix, image_index = None, [0]
        if self.asset_layout == "task":
            events = [] if create else self._events(path)
            metadata = event["data"] if create else events[0]["data"]
            task = quote(str(metadata["task_id"]), safe="")
            if task in ("", ".", "..") or len(task) > 180:
                task = "task-" + hashlib.sha256(str(metadata["task_id"]).encode()).hexdigest()
            base = f"{task}/{identity}"
            if event["kind"] == "observation":
                prefix = base + "/screenshots"
                image_index[0] = sum(k.startswith(prefix + "/") for k in asset_ids(events))
            else:
                category = "model_inputs" if event["kind"] == "model_call" else "events"
                call_index = sum(e["kind"] == "model_call" for e in events)
                sequence = call_index if category == "model_inputs" else len(events)
                prefix = f"{base}/{category}/{sequence:06d}"
        event = self._externalize(event, assets, prefix=prefix, index=image_index)
        line = encode(event) + b"\n"
        missing = {key: value for key, value in assets.items() if not self.asset_path(key).exists()}
        # A single serialized transaction reserves all new bytes BEFORE writing.
        self._reserve(len(line) + sum(map(len, missing.values())), event)
        for key, raw in missing.items():
            target = self.asset_path(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".pending-", delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        with path.open("xb" if create else "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, identity: str, event: dict) -> None:
        with self.mutex:
            try:
                self._append(identity, event)
            except Exception:
                self.abort(identity)
                raise

    def abort(self, identity: str) -> None:
        with self.mutex:
            self._path(identity).unlink(missing_ok=True)
            self._gc()

    def verify_assets(self, trajectory: dict) -> None:
        for key in asset_ids(trajectory):
            path = self.asset_path(key)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset_digest(key):
                raise ValueError("missing or corrupted asset")

    def finish(self, identity: str, outcome: dict) -> dict:
        with self.mutex:
            try:
                self._append(identity, {"kind": "outcome", "data": outcome})
                trajectory = materialize(self._events(self._path(identity)))
                validate(trajectory)
                self.verify_assets(trajectory)
                peers = [t for t in self.retained() if task_key(t) == task_key(trajectory)]
                fingerprint = path_fingerprint(trajectory)
                duplicates = [t for t in peers if path_fingerprint(t) == fingerprint]
                if duplicates and quality(trajectory)[:-1] >= min(quality(t)[:-1] for t in duplicates):
                    raise ValueError("duplicate_path")
                peers = [t for t in peers if t not in duplicates]
                limit = self.task_limits.get(trajectory["task_id"], self.per_task_limit)
                selected = sorted([*peers, trajectory], key=quality)[:limit]
                if trajectory not in selected:
                    raise ValueError("task_quota: lower_priority")
                self._append(identity, {"kind": "commit", "data": {"integrity": "complete"}})
                # No second trajectory copy: rename is the commit point.
                for victim in duplicates + [t for t in peers if t not in selected]:
                    self._delete(victim)
                self._path(identity).rename(self._path(identity, "trajectories"))
                return {"retained": True, "trajectory_id": identity}
            except Exception as exc:
                self.abort(identity)
                return {"retained": False, "reason": str(exc)}
