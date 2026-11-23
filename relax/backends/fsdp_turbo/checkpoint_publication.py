# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fence checkpoint publication from training groups retired by recovery."""

import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def checkpoint_saving_enabled(config) -> bool:
    return (
        getattr(config, "train_backend", None) == "fsdp_turbo"
        and config.save is not None
        and config.save_interval is not None
        and not config.debug_rollout_only
        and "actor" in config.resource
    )


def _validate_storage_mount(root: str) -> None:
    root_path = Path(root).resolve()
    mounts = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        mountpoint = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[1]))
        if root_path.is_relative_to(mountpoint):
            mounts.append((len(mountpoint.parts), fields[2], fields[3].split(",")))
    _, filesystem, options = max(reversed(mounts), key=lambda mount: mount[0])
    unsupported = {"nocto", "nolock", "local_lock=all", "local_lock=flock"}.intersection(options)
    if filesystem in {"nfs", "nfs4"} and unsupported:
        raise ValueError(
            f"Turbo checkpoint saving requires coherent NFS storage with remote flock: {root}: {unsupported}"
        )


@contextmanager
def _publication_lock(root: str, timeout: float):
    import fcntl

    _validate_storage_mount(root)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".checkpoint-publication.lock"), "a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for checkpoint publication: {root}") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _generation(root: str) -> str:
    path = Path(root) / ".checkpoint-generation"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def checkpoint_generation(root: str) -> str:
    with _publication_lock(root, 30.0):
        if not (Path(root) / ".checkpoint-generation").exists():
            _write_generation(root)
        return _generation(root)


def _write_generation(root: str) -> None:
    generation = uuid.uuid4().hex
    temporary = Path(root) / f".checkpoint-generation.{generation}"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(generation)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, Path(root) / ".checkpoint-generation")


def fence_checkpoint_publishers(root: str, timeout: float = 30.0) -> None:
    with _publication_lock(root, timeout):
        _write_generation(root)


@contextmanager
def checkpoint_publication(root: str, generation: str):
    with _publication_lock(root, 30.0):
        if generation != _generation(root):
            raise RuntimeError(
                f"Checkpoint publication refused at {root}: training group was retired by recovery "
                "or the save directory is not shared with the driver"
            )
        yield
