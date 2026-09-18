# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Hold host device locks across Actions steps; released by the post action."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import IO


def discover_nvidia(pool: str) -> tuple[dict[str, str], str]:
    """Resolve physical nvidia-smi indices to stable whole-GPU UUIDs."""
    if "CUDA_VISIBLE_DEVICES" in os.environ and not pool:
        raise ValueError("CUDA_VISIBLE_DEVICES is already set; specify devices using physical indices or full UUIDs")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,mig.mode.current", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    aliases: dict[str, str] = {}
    indices: dict[str, str] = {}
    mig_devices: set[str] = set()
    for line in result.stdout.splitlines():
        index, uuid, mig = (part.strip() for part in line.split(","))
        if not index.isdigit() or not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
            raise ValueError(f"Unexpected nvidia-smi device: {line!r}")
        aliases[index] = aliases[uuid] = uuid
        indices[uuid] = index
        if mig.lower() == "enabled":
            mig_devices.add(uuid)
    if pool:
        requested = [part.strip() for part in pool.split(",")]
        if any(device not in aliases for device in requested):
            raise ValueError("devices must contain known physical indices or full GPU UUIDs")
        devices = [aliases[device] for device in requested]
    else:
        devices = list(dict.fromkeys(aliases.values()))
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("Device pool must be nonempty and contain no duplicate physical devices")
    if mig_devices.intersection(devices):
        raise ValueError("MIG-enabled GPUs are not supported; choose a pool of whole GPUs")
    return {device: indices[device] for device in devices}, "CUDA_VISIBLE_DEVICES"


def format_devices(devices: list[str], indices: dict[str, str]) -> str:
    return ", ".join(f"{indices[device]} ({device})" for device in devices) or "none"


def lock_path(directory: Path, backend: str, device: str) -> Path:
    digest = hashlib.sha256(f"{backend}\0{device}".encode()).hexdigest()
    return directory / f"{digest}.lock"


def parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def cancelled(directory: Path, parent_pid: int) -> bool:
    if (directory / "release").exists():
        return True
    # Before handoff, a killed action must not leave a queued/acquired lease.
    # After handoff, do NOT expire the lease when a runner disappears: its
    # detached workloads may still be using the devices.
    return not (directory / "accepted").exists() and not parent_alive(parent_pid)


def acquire(
    directory: Path, config: dict, devices: dict[str, str], deadline: float
) -> tuple[list[str], list[IO[str]]]:
    lock_dir = Path(config["lock_dir"])
    lock_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    next_report = started
    while not cancelled(directory, config["parent_pid"]):
        selected: list[str] = []
        handles: list[IO[str]] = []
        try:
            for device in devices:
                handle = lock_path(lock_dir, config["backend"], device).open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                except BaseException:
                    handle.close()
                    raise
                handles.append(handle)
                selected.append(device)
                if len(selected) == config["count"]:
                    return selected, handles
        except BaseException:
            for handle in handles:
                handle.close()
            raise
        # A partial allocation must never block another request while waiting.
        for handle in handles:
            handle.close()
        now = time.monotonic()
        if now >= next_report:
            locked = [device for device in devices if device not in selected]
            sys.stdout.write(
                f"Waiting for {config['count']} {config['backend']} devices ({now - started:.0f}s elapsed): "
                f"{len(selected)}/{len(devices)} available\n"
                f"  Available: {format_devices(selected, devices)}\n"
                f"  Locked: {format_devices(locked, devices)}\n"
            )
            sys.stdout.flush()
            next_report = now + 30
        if now >= deadline:
            raise TimeoutError(f"Timed out waiting for {config['count']} {config['backend']} devices")
        time.sleep(min(random.uniform(0.1, 0.3), max(0, deadline - time.monotonic())))
    raise RuntimeError("Device acquisition cancelled")


def write_result(directory: Path, result: dict) -> None:
    temporary = directory / "result.tmp"
    temporary.write_text(json.dumps(result))
    temporary.replace(directory / "result.json")


def terminate(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main(directory: Path) -> None:
    handles: list[IO[str]] = []
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, terminate)
    try:
        config = json.loads((directory / "config.json").read_text())
        devices, visibility_env = discover_nvidia(config["devices"])
        if not 1 <= config["count"] <= len(devices):
            raise ValueError(f"count must be between 1 and the candidate pool size ({len(devices)})")
        deadline = time.monotonic() + config["timeout"]
        selected, handles = acquire(directory, config, devices, deadline)
        write_result(
            directory,
            {
                "devices": selected,
                "display_devices": format_devices(selected, devices),
                "visibility_env": visibility_env,
            },
        )
        while not cancelled(directory, config["parent_pid"]):
            time.sleep(0.1)
    except Exception as error:
        write_result(directory, {"error": str(error)})
    finally:
        for handle in handles:
            handle.close()
        with contextlib.suppress(FileNotFoundError):
            (directory / "released").touch()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
