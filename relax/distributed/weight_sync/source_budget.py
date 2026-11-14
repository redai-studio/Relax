# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Source capture admission using simultaneous tensor and scratch lifetimes."""

import os
import shutil
import socket
from pathlib import Path

from .store import stored_bytes
from .traffic import filesystem_info


def gpu_capacity(cuda, required: int) -> dict:
    free = cuda.mem_get_info()[0]
    before = free
    reclaimed = False
    if free < required and cuda.memory_reserved() > cuda.memory_allocated():
        cuda.empty_cache()
        free = cuda.mem_get_info()[0]
        reclaimed = True
    return {"available_bytes": free, "driver_free_before_bytes": before, "released_unused_cache": reclaimed}


def filesystem_key(path: Path) -> str:
    try:
        shared = filesystem_info(path)["shared_id"]
    except (OSError, ValueError, IndexError):
        shared = None
    return f"shared:{shared}" if shared else f"local:{socket.gethostname()}:{path.stat().st_dev}"


def memory_filesystem(path: Path) -> bool:
    try:
        return filesystem_info(path)["type"] in ("tmpfs", "ramfs")
    except (OSError, ValueError, IndexError):
        return False


def source_demand(
    local_bytes: int,
    logical_bytes: int,
    conversion_bytes: int,
    *,
    adaptive: bool,
    bridge: bool,
    gathered_source_bytes: int | None = None,
    encode_workers: int = 4,
) -> dict:
    gathered_source_bytes = logical_bytes if gathered_source_bytes is None else gathered_source_bytes
    if any(type(n) is not int or n < 0 for n in (local_bytes, logical_bytes, conversion_bytes, gathered_source_bytes)):
        raise ValueError("Source sizes must be nonnegative integers")
    # Frozen tensors survive conversion; complete gathers and host conversion
    # can coexist. Codec/patch processing uses bounded windows.
    if type(encode_workers) is not int or encode_workers < 1:
        raise ValueError("encode_workers must be a positive integer")
    # Each window includes raw/base, encoded candidates and zlib workspace.
    window = encode_workers * 64 * 1024**2
    ram = local_bytes + 4 * conversion_bytes + window
    parts = {"frozen_tensors": local_bytes, "conversion_buffers": 4 * conversion_bytes, "codec_windows": window}
    scratch = {"logical_export": logical_bytes, "publication_candidate": logical_bytes}
    if adaptive:
        # Every persisted replacement frame is smaller than its raw chunk.
        # Earlier frames retained before a dense fallback satisfy this bound too.
        scratch.update(local_shards=local_bytes, replacement_frames=local_bytes)
        if bridge:
            # Padded vocabulary and tied source tensors can make the gathered
            # Megatron representation larger than the exported HF checkpoint.
            scratch["gathered_source"] = gathered_source_bytes
    return {
        "ram_bytes": ram,
        "gpu_bytes": 3 * conversion_bytes,
        "scratch_additional_bytes": sum(scratch.values()),
        "ram_components": parts,
        "scratch_components": scratch,
    }


def reclaimable_file_cache(root: Path) -> int:
    """Estimate clean inactive file pages without counting tmpfs or writeback.

    The kernel can reclaim these pages when charging a new anonymous
    allocation. Active cache and slab are deliberately excluded from this
    estimate.
    """
    try:
        fields = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
        values = {key: int(fields[key]) for key in ("inactive_file", "file", "shmem", "file_dirty", "file_writeback")}
        if any(value < 0 for value in values.values()):
            return 0
        disk_cache = max(0, values["file"] - values["shmem"])
        return max(0, min(values["inactive_file"], disk_cache) - values["file_dirty"] - values["file_writeback"])
    except (OSError, KeyError, ValueError):
        return 0


def cgroup_available(root: Path, boundary: Path) -> int | None:
    values = []
    if root != boundary and boundary not in root.parents:
        return None
    while True:
        try:
            limit = (root / "memory.max").read_text().strip()
            if limit != "max":
                limit = int(limit)
                current = int((root / "memory.current").read_text())
                reclaimable = min(max(0, current), reclaimable_file_cache(root))
                values.append(max(0, min(limit, limit - current + reclaimable)))
        except (OSError, ValueError):
            pass
        if root == boundary:
            break
        root = root.parent
    return min(values) if values else None


def reclaimable_arc(meminfo: dict, proc: Path = Path("/proc"), sys: Path = Path("/sys")) -> int:
    """Read clean ARC data eligible for Linux reclaim, preserving its floor.

    Linux MemAvailable omits this external filesystem cache. Only resident
    MRU/MFU data is included; ghost entries, pinned data and metadata are not.
    The floor follows OpenZFS arc_evictable_memory in
    module/os/linux/zfs/arc_os.c. Missing counters leave the ordinary Linux
    estimate unchanged.
    """
    try:
        counters = {}
        for line in (proc / "spl/kstat/zfs/arcstats").read_text().splitlines():
            row = line.split()
            if len(row) == 3 and row[2].isdigit():
                counters[row[0]] = int(row[2])
        size, minimum, maximum, mru, mfu = (
            counters[key] for key in ("size", "c_min", "c_max", "mru_evictable_data", "mfu_evictable_data")
        )
        percent = int((sys / "module/zfs/parameters/zfs_arc_pc_percent").read_text())
        if not 0 <= percent <= 100 or not 0 <= minimum <= maximum or size > meminfo["MemTotal"]:
            return 0
        file_pages = sum(meminfo[key] for key in ("Cached", "Buffers", "SwapCached"))
        floor = max(minimum, min(maximum, file_pages * percent // 100))
        return max(0, min(mru + mfu, size - floor))
    except (OSError, KeyError, ValueError):
        return 0


def memory_capacity(proc: Path = Path("/proc"), sys: Path = Path("/sys")) -> dict:
    try:
        fields = {
            line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
            for line in (proc / "meminfo").read_text().splitlines()
        }
        arc = reclaimable_arc(fields, proc, sys)
        available = min(fields["MemTotal"], fields["MemAvailable"] + arc)
    except (OSError, KeyError, ValueError):
        return {"available_bytes": None}
    remaining = None
    try:
        line = next(line for line in (proc / "self/cgroup").read_text().splitlines() if line.startswith("0::"))
        boundary = (sys / "fs/cgroup").resolve()
        remaining = cgroup_available((boundary / line[3:].lstrip("/")).resolve(), boundary)
        if remaining is not None:
            available = min(available, remaining)
    except (OSError, ValueError, StopIteration):
        pass
    return {
        "available_bytes": available,
        "linux_available_bytes": fields["MemAvailable"],
        "reclaimable_arc_data_bytes": arc,
        "cgroup_available_bytes": remaining,
    }


def available_memory() -> int | None:
    return memory_capacity()["available_bytes"]


def source_preflight_record(
    scratch: Path,
    demand: dict,
    *,
    limits: dict | None = None,
    gpu_free: int | None = None,
    store=None,
    logical_bytes: int = 0,
    metadata_bytes: int = 64 * 1024,
    source_cache: Path | None = None,
) -> dict:
    limits = limits or {}
    allowed = {"ram_bytes", "gpu_bytes", "scratch_bytes"}
    if set(limits) - allowed or any(type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError("Source limits require positive ram_bytes, gpu_bytes or scratch_bytes")
    scratch = scratch.resolve()
    volumes = []
    if store is not None and logical_bytes:
        # FULL fallback may coexist with canonical raw content. Metadata and
        # temporary atomic replacements are included in the admission bound.
        if type(metadata_bytes) is not int or metadata_bytes < 0:
            raise ValueError("Metadata allowance must be a nonnegative integer")
        metadata = metadata_bytes
        for kind, path, needed, budget in (
            ("encoded_store", store.root, logical_bytes + metadata, store.disk_budget),
            ("canonical_store", store.canonical_directory, logical_bytes + metadata, store.canonical_budget),
        ):
            existing = path
            while not existing.exists():
                existing = existing.parent
            volumes.append(
                {
                    "kind": kind,
                    "path": str(path.resolve()),
                    "filesystem": existing.stat().st_dev,
                    "filesystem_key": filesystem_key(existing),
                    "additional_bytes": needed,
                    "available_bytes": shutil.disk_usage(existing).free,
                    "used_bytes": stored_bytes(path),
                    "budget": budget,
                    "memory_backed": memory_filesystem(existing),
                }
            )
        if store.canonical_budget is None:
            volumes[0]["additional_bytes"] += volumes[1]["additional_bytes"]
            volumes.pop()
    scratch_additional = demand["scratch_additional_bytes"]
    if source_cache is not None and source_cache.resolve() != scratch:
        # Exported tensors, shard bases and replacement frames live in the
        # source workspace. Only the immutable publication candidate is copied
        # into publication scratch. Reserve each future byte at its destination;
        # check_source_preflight combines paths that share a physical volume.
        source_cache = source_cache.resolve()
        scratch_additional = demand["scratch_components"]["publication_candidate"]
        volumes.append(
            {
                "kind": "source_cache",
                "path": str(source_cache),
                "filesystem": source_cache.stat().st_dev,
                "filesystem_key": filesystem_key(source_cache),
                "additional_bytes": demand["scratch_additional_bytes"] - scratch_additional,
                "available_bytes": shutil.disk_usage(source_cache).free,
                "used_bytes": stored_bytes(source_cache),
                "budget": None,
                "memory_backed": memory_filesystem(source_cache),
            }
        )
    memory = memory_capacity()
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "scratch": str(scratch),
        "filesystem": scratch.stat().st_dev,
        "filesystem_key": filesystem_key(scratch),
        "demand": demand,
        "limits": limits,
        "available_ram_bytes": memory["available_bytes"],
        "memory_capacity": memory,
        "available_gpu_bytes": gpu_free,
        "available_disk_bytes": shutil.disk_usage(scratch).free,
        "scratch_used_bytes": stored_bytes(scratch),
        "scratch_volume_additional_bytes": scratch_additional,
        "scratch_memory_bytes": scratch_additional if memory_filesystem(scratch) else 0,
        "publication_volumes": volumes,
        "metadata_allowance_bytes": metadata_bytes,
    }


def check_source_preflight(records: list[dict]) -> dict:
    """Check every rank before any rank clones weights or enters a gather.

    Shared scratch paths are counted once for existing files and once per rank
    for future writes. Host memory is aggregated across participating ranks.
    Explicit limits apply to the synchronization working set, not resident
    training models. Available memory separately includes resident allocations.
    Its instantaneous value is advisory: the conversion estimate includes
    buffers whose lifetimes may not overlap, and allocators can reclaim cache
    during the operation. Configured limits and filesystem capacity retain
    their existing semantics.
    """
    shortages = []
    warnings = []
    hosts = {}
    for record in records:
        hosts.setdefault(record["host"], []).append(record)
        wanted = record["demand"]["gpu_bytes"]
        for label, limit in (
            ("available_gpu", record["available_gpu_bytes"]),
            ("gpu_limit", record["limits"].get("gpu_bytes")),
        ):
            if limit is not None and wanted > limit:
                findings = warnings if label == "available_gpu" else shortages
                findings.append({"host": record["host"], "resource": label, "required": wanted, "available": limit})
    for host, members in hosts.items():
        wanted = sum(
            row["demand"]["ram_bytes"]
            + row.get("scratch_memory_bytes", 0)
            + sum(
                volume["additional_bytes"]
                for volume in row.get("publication_volumes", [])
                if volume.get("memory_backed")
            )
            for row in members
        )
        for label, key in (("available_ram", "available_ram_bytes"), ("ram_limit", None)):
            values = [row[key] if key else row["limits"].get("ram_bytes") for row in members]
            values = [value for value in values if value is not None]
            if values and wanted > min(values):
                findings = warnings if label == "available_ram" else shortages
                findings.append({"host": host, "resource": label, "required": wanted, "available": min(values)})
    additional = sum(row["demand"]["scratch_additional_bytes"] for row in records)
    locations = {}
    for row in records:
        filesystem = row.get("filesystem_key") or f"local:{row['host']}:{row['filesystem']}"
        key = (filesystem, Path(row.get("scratch", ".")))
        locations[key] = max(locations.get(key, 0), row["scratch_used_bytes"])
        for volume in row.get("publication_volumes", []):
            if volume["kind"] == "source_cache":
                key = (volume["filesystem_key"], Path(volume["path"]))
                locations[key] = max(locations.get(key, 0), volume["used_bytes"])
    retained = sum(
        used
        for (filesystem, path), used in locations.items()
        if not any(other_fs == filesystem and parent in path.parents for other_fs, parent in locations)
    )
    filesystems = {}
    for row in records:
        wanted = additional + retained
        limit = row["limits"].get("scratch_bytes")
        if limit is not None and wanted > limit:
            shortages.append(
                {"host": row["host"], "resource": "scratch_limit", "required": wanted, "available": limit}
            )
        volumes = row.get("publication_volumes", [])
        for volume in volumes:
            budget = volume["budget"]
            wanted = volume["used_bytes"] + volume["additional_bytes"]
            if budget is not None and wanted > budget:
                shortages.append(
                    {
                        "host": row["host"],
                        "resource": volume["kind"],
                        "required": wanted,
                        "available": budget,
                        "path": volume["path"],
                    }
                )
        allocations = [
            {
                "filesystem": row["filesystem"],
                "filesystem_key": row.get("filesystem_key"),
                "additional_bytes": row.get(
                    "scratch_volume_additional_bytes", row["demand"]["scratch_additional_bytes"]
                ),
                "available_bytes": row["available_disk_bytes"],
            },
            *volumes,
        ]
        for allocation in allocations:
            key = allocation.get("filesystem_key") or f"local:{row['host']}:{allocation['filesystem']}"
            group = filesystems.setdefault(key, {"required": 0, "free": [], "hosts": set()})
            group["required"] += allocation["additional_bytes"]
            group["free"].append(allocation["available_bytes"])
            group["hosts"].add(row["host"])
    for key, group in filesystems.items():
        if group["required"] > min(group["free"]):
            shortages.append(
                {
                    "hosts": sorted(group["hosts"]),
                    "resource": "combined_filesystem",
                    "required": group["required"],
                    "available": min(group["free"]),
                    "filesystem": key,
                }
            )
    result = {
        "status": "WAITING_ACTION" if shortages else "ADMITTED",
        "ranks": records,
        "scratch_workspace": {"retained_bytes": retained, "additional_bytes": additional},
        "shortages": shortages,
        "warnings": warnings,
        "actions": [
            "release unused source caches",
            "increase source limits or scratch capacity",
            "retry the same update",
        ]
        if shortages
        else [],
    }
    return result
