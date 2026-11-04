"""Stage-boundary disk/SHM accounting; no directory walks on the update
path."""

from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any


def filesystem_usage(path: Path) -> dict[str, Any]:
    """Filesystem totals include other processes and are never task
    attribution."""
    result: dict[str, Any] = {"path": str(path), "scope": "filesystem_global"}
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {**result, "available": False, "error": type(exc).__name__}
    return {
        **result,
        "available": True,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def directory_usage(path: Path) -> dict[str, Any]:
    """Count unique regular-file inodes under this directory, without following
    links.

    Logical bytes and allocated blocks are separate so sparse files are honest.
    Directory metadata, external targets and deleted-but-open files are
    excluded. A disappearing/inaccessible entry makes the sample incomplete,
    not zero.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "scope": "run_directory_only",
        "logical_file_bytes": 0,
        "allocated_file_bytes": 0,
        "unique_regular_files": 0,
        "symlinks_not_followed": 0,
        "errors": 0,
        "complete": True,
        "excludes": "directory metadata, external symlink targets, deleted-but-open files",
    }
    seen: set[tuple[int, int]] = set()
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(info.st_mode):
                            result["symlinks_not_followed"] += 1
                        elif stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) not in seen:
                            seen.add((info.st_dev, info.st_ino))
                            result["unique_regular_files"] += 1
                            result["logical_file_bytes"] += info.st_size
                            if hasattr(info, "st_blocks") and result["allocated_file_bytes"] is not None:
                                result["allocated_file_bytes"] += info.st_blocks * 512
                            else:
                                result["allocated_file_bytes"] = None
                    except OSError:
                        result["errors"] += 1
        except OSError:
            result["errors"] += 1
    result["complete"] = result["errors"] == 0
    return result


def storage_snapshot(run_dir: Path, shm_path: Path = Path("/dev/shm")) -> dict[str, Any]:
    return {
        "sampled_unix": time.time(),
        "sampling": "stage_boundary_only; no recursive walk during optimizer updates",
        "run_directory": directory_usage(run_dir),
        "run_filesystem": filesystem_usage(run_dir),
        "shared_memory_filesystem": filesystem_usage(shm_path),
    }


def cache_storage_contract(enabled: bool, device: str) -> dict[str, Any]:
    """Implementation declaration, distinct from measured global SHM/disk
    use."""
    return {
        "enabled": enabled,
        "implementation": "relax.models.condition_cache.ConditionFeatureCache",
        "storage": ("process_private_cpu_ram" if device == "cpu" else "process_private_cuda_tensors")
        if enabled
        else "disabled",
        "persistent_files_created_by_cache_backend": False,
        "shared_memory_segments_created_by_cache_backend": False,
        "basis": "implementation uses an in-process OrderedDict and torch.Tensor.to(copy=True); no file/SHM API",
        "scope": "condition-feature cache only; does not attribute DataLoader, DCP or global filesystem use",
    }
