# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Scoped communication instrumentation without changing global torch APIs.

Collective counters describe delivered tensor payloads (sender ownership), not
NCCL's internal algorithm or physical wire traffic. File writes have a separate
scope; Linux interface observations calibrate deployment overhead.
"""

import contextvars
import hashlib
import os
import pickle
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path

from .diagnostics import observe


_sink = contextvars.ContextVar("weight_sync_traffic", default=None)
_attempt = contextvars.ContextVar("weight_sync_attempt", default=1)


@contextmanager
def capture(metrics, *, attempt=1):
    token = _sink.set(metrics)
    attempt_token = _attempt.set(attempt)
    try:
        yield
    finally:
        _sink.reset(token)
        _attempt.reset(attempt_token)


def current():
    return _sink.get()


def emit(stage, size, link, scope="collective_payload"):
    sink = _sink.get()
    if sink is not None:
        sink.transferred(stage, size, link=link, scope=scope, attempt=_attempt.get())


def file_write(path, size):
    emit("W", size, str(path), "file_write")


class _MeasuredWork:
    def __init__(self, work, count, label, sink, attempt=1):
        self.work, self.count, self.label, self.sink = work, count, label, sink
        self.recorded = False
        self.attempt = attempt

    def wait(self, *args, **kwargs):
        try:
            result = self.work.wait(*args, **kwargs)
        except Exception:
            self.sink.unmeasured("G", f"failed collective may have sent a partial payload: {self.label}")
            raise
        if result is not False and not self.recorded:
            self.sink.transferred("G", self.count, link=self.label, scope="collective_payload", attempt=self.attempt)
            self.recorded = True
        return result


def _complete(work, count, label, asynchronous):
    sink = _sink.get()
    if sink is None:
        return work
    if asynchronous:
        return _MeasuredWork(work, count, label, sink, _attempt.get())
    emit("G", count, label)
    return work


def all_gather(tensor_list, tensor, group=None, async_op=False):
    import torch.distributed as dist

    work = dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
    if _sink.get() is None:
        return work
    ranks = dist.get_process_group_ranks(group) if group is not None else list(range(dist.get_world_size()))
    size = tensor.numel() * tensor.element_size() * (len(ranks) - 1)
    return _complete(work, size, f"all_gather:{dist.get_rank()}:{ranks}", async_op)


def broadcast(tensor, src, group=None, async_op=False):
    import torch.distributed as dist

    work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    if _sink.get() is None:
        return work
    ranks = dist.get_process_group_ranks(group) if group is not None else list(range(dist.get_world_size()))
    size = tensor.numel() * tensor.element_size() * (len(ranks) - 1) if dist.get_rank() == src else 0
    return _complete(work, size, f"broadcast:{src}:{ranks}", async_op)


def object_all_gather_bytes(objects):
    """Sender-owned padded pickle payload plus size exchange for a group."""
    if not objects:
        return 0
    size = max(len(pickle.dumps(value, protocol=pickle.DEFAULT_PROTOCOL)) for value in objects)
    return len(objects) * (len(objects) - 1) * (size + 8)


def all_gather_object(object_list, obj, group=None):
    import torch.distributed as dist

    result = dist.all_gather_object(object_list, obj, group=group)
    if _sink.get() is not None:
        # PyTorch pads object buffers to the maximum pickled length. Count
        # size exchange (int64) and padded payload once per sending rank.
        size = object_all_gather_bytes(object_list) // len(object_list)
        emit("C", size, f"object_all_gather:{dist.get_rank()}", "object_collective")
    return result


def broadcast_object_list(object_list, src, group=None):
    import torch.distributed as dist

    result = dist.broadcast_object_list(object_list, src=src, group=group)
    if _sink.get() is not None and dist.get_rank() == src:
        size = sum(len(pickle.dumps(value, protocol=pickle.DEFAULT_PROTOCOL)) + 8 for value in object_list)
        emit("C", size * (dist.get_world_size(group) - 1), f"object_broadcast:{src}", "object_collective")
    return result


def all_reduce(tensor, op=None, group=None):
    import torch.distributed as dist

    result = dist.all_reduce(tensor, op=dist.ReduceOp.SUM if op is None else op, group=group)
    if _sink.get() is not None:
        emit(
            "C",
            tensor.numel() * tensor.element_size() * (dist.get_world_size(group) - 1),
            f"reduction_operands:{dist.get_rank()}",
            "reduction_operands",
        )
    return result


def interface_snapshot(interfaces):
    """Explicit interfaces avoid counting both a bridge and its child NIC.

    A node-wide counter includes unrelated traffic. Keep that attribution
    visible instead of presenting the observation as a per-process result.
    """
    result = {}
    for name in interfaces:
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid network interface")
        root = Path("/sys/class/net") / name / "statistics"
        result[name] = {key: int((root / key).read_text()) for key in ("tx_bytes", "rx_bytes")}
    return result


def interface_delta(before, after):
    if set(before) != set(after):
        raise ValueError("Interface set changed during measurement")
    result = {name: {key: after[name][key] - before[name][key] for key in ("tx_bytes", "rx_bytes")} for name in before}
    if any(value < 0 for counters in result.values() for value in counters.values()):
        raise ValueError("Interface counter reset during measurement")
    return {"scope": "node_interface", "attribution": "includes_concurrent_traffic", "interfaces": result}


def filesystem_info(path):
    """Identify the actual mount; an NFS cache is still remote storage."""
    target = Path(path).resolve()
    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, _, tail = line.partition(" - ")
        columns, detail = fields.split(), tail.split()
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), columns[4]))
        if target == mount or mount in target.parents:
            candidates.append((len(mount.parts), mount, detail))
    if not candidates:
        raise ValueError("Storage mount is not identifiable")
    _, mount, detail = max(candidates, key=lambda item: item[0])
    kind = detail[0]
    local = kind in {"ext4", "ext3", "xfs", "btrfs", "tmpfs", "ramfs", "zfs"}
    remote = kind in {"nfs", "nfs4", "cifs", "ceph", "lustre", "gpfs", "fuse.sshfs"}
    return {
        "mount": str(mount),
        "type": kind,
        "placement": "local" if local else "remote" if remote else "unknown",
        "shared_id": hashlib.sha256(f"{kind}:{detail[1]}".encode()).hexdigest() if remote else None,
    }


def process_io():
    return {
        key: int(value) for key, value in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())
    }


def mount_operations(text, mount):
    """Read Linux NFS operation counters for one visible mount."""
    selected, counters = False, {}
    for line in text.splitlines():
        if line.startswith("device "):
            selected = f" mounted on {mount} with fstype " in line
        elif selected and ":" in line:
            name, _, values = line.strip().partition(":")
            fields = values.split()
            if name.isupper() and len(fields) >= 8 and all(value.isdigit() for value in fields):
                counters[name] = dict(
                    zip(
                        ("operations", "transmissions", "timeouts", "sent_bytes", "received_bytes"),
                        map(int, fields[:5]),
                        strict=True,
                    )
                )
    return counters


def mount_snapshot(mount):
    return mount_operations(Path("/proc/self/mountstats").read_text(), mount)


def mount_delta(before, after):
    if set(before) - set(after):
        raise ValueError("Filesystem operation counters changed during measurement")
    values = {
        name: {key: fields[key] - before.get(name, {}).get(key, 0) for key in fields} for name, fields in after.items()
    }
    if any(value < 0 for fields in values.values() for value in fields.values()):
        raise ValueError("Filesystem operation counters reset during measurement")
    return {"scope": "client_mount", "attribution": "includes_concurrent_traffic", "operations": values}


class IOObservation:
    """Retain process I/O and optional node counters around a concrete phase.

    mmap/cache and filesystem protocols make process I/O different from wire
    bytes. Both observations remain named; neither is silently substituted.
    """

    def __init__(self, *, path=None, interfaces=()):
        self.path, self.interfaces = path, interfaces
        self.report = {}

    def _before(self):
        before = process_io()
        network = interface_snapshot(self.interfaces) if self.interfaces else None
        storage = filesystem_info(self.path) if self.path is not None else None
        mount = mount_snapshot(storage["mount"]) if storage and storage["type"] in ("nfs", "nfs4") else None
        return before, network, storage, mount

    def __enter__(self):
        self.started = time.monotonic_ns()
        self.report.update(
            host=socket.gethostname(), pid=os.getpid(), start_unix_ns=time.time_ns(), counter_snapshots={}
        )
        self.errors = []
        self.before = self.network = self.mount_before = None
        result = observe(self._before, "I/O entry observation")
        if result.error:
            self.errors.append(result.error)
        else:
            self.before, self.network, storage, self.mount_before = result.value
            if storage is not None:
                self.report["storage"] = storage
        return self

    def _after(self):
        result = {"counter_snapshots": {}}
        if self.before is not None:
            after = process_io()
            result["counter_snapshots"]["process_io"] = {"before": self.before, "after": after}
            result["process_io_bytes"] = {
                key: after[key] - value
                for key, value in self.before.items()
                if key in ("rchar", "wchar", "read_bytes", "write_bytes", "cancelled_write_bytes")
            }
        if self.network is not None:
            after = interface_snapshot(self.interfaces)
            result["counter_snapshots"]["network"] = {"before": self.network, "after": after}
            result["network"] = interface_delta(self.network, after)
        if self.mount_before is not None:
            after = mount_snapshot(self.report["storage"]["mount"])
            result["counter_snapshots"]["filesystem_rpc"] = {"before": self.mount_before, "after": after}
            result["filesystem_rpc"] = mount_delta(self.mount_before, after)
        return result

    def __exit__(self, *args):
        if self.before is not None:
            result = observe(self._after, "I/O exit observation")
            if result.error:
                self.errors.append(result.error)
            else:
                self.report.update(result.value)
        self.report.update(
            start_monotonic_ns=self.started,
            end_monotonic_ns=time.monotonic_ns(),
            end_unix_ns=time.time_ns(),
            errors=self.errors,
        )
