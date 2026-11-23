# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Fence checkpoint publication from training groups retired by recovery."""

import os
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


def _validate_storage_mount(fd: int, root: str) -> None:
    fdinfo = dict(
        line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{fd}").read_text(encoding="utf-8").splitlines()
    )
    mount_id = fdinfo["mnt_id"].strip()
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[0] != mount_id:
            continue
        separator = fields.index("-")
        filesystem = fields[separator + 1]
        options = set(fields[5].split(",")) | set(fields[separator + 3].split(","))
        unsupported = {"nocto", "nolock", "local_lock=all", "local_lock=flock"}.intersection(options)
        if filesystem in {"nfs", "nfs4"} and unsupported:
            raise ValueError(
                f"Turbo checkpoint saving requires coherent NFS storage with remote flock: {root}: {unsupported}"
            )
        return
    raise ValueError(f"Cannot identify the checkpoint publication mount: {root}")


@contextmanager
def _publication_lock(root: str, timeout: float):
    import fcntl

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".checkpoint-publication.lock"), "a") as lock:
        _validate_storage_mount(lock.fileno(), root)
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
