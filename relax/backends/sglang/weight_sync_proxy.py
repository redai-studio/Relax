# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-owned local TP receivers.

CUDA payloads never enter the parent actor.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from typing import Any

from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup, remaining

from .delta_loader import LOADER as PROXY_MEMORY_LOADER


MAX_DOCUMENT_BYTES = 16 << 20


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


MAX_HTTP_BYTES = 65536


def _http_bytes(payload: dict) -> int:
    # requests' json= body uses these default separators/ASCII escaping.
    return len(json.dumps(payload, allow_nan=False).encode())


def _process_identity(pid: int) -> tuple[int, int]:
    import psutil

    return pid, int(psutil.Process(pid).create_time() * 1e6)


def _workers_exited(identities: list) -> bool:
    import psutil

    for pid, stamp in identities:
        try:
            process = psutil.Process(pid)
            if int(process.create_time() * 1e6) == stamp and process.status() != psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            pass
    return True


def _serialize(control: dict, tensor: Any = None, status: Any = None) -> str:
    import torch
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
        monkey_patch_torch_reductions()
    weights = [(json_bytes(control).decode(), _EmptyControlTensor() if tensor is None else tensor)]
    if status is not None:
        weights.append(("__delta_capability__", status))
    return MultiprocessingSerializer.serialize(weights, output_str=True)


class _EmptyControlTensor:
    """Use Torch's original CPU recipe even in a CUDA-patched parent
    process."""

    def __reduce__(self):
        import torch
        from torch.multiprocessing import reductions

        reducer = getattr(reductions, "_reduce_tensor_original", reductions.reduce_tensor)
        return reducer(torch.empty(0))


class _CudaWireSize:
    """A serialization-only upper bound; it never exports CUDA storage.

    The supported Torch CUDA reducer has three IPC handles and signed-64-bit
    sizes/offsets. Use oversized distinct handles and UUID text so pickle's
    memoization cannot underestimate a real one-dimensional uint8 slab. Never
    deserialize this object. GPU component tests compare the bound to the exact
    reducer in the deployment image.
    """

    def __reduce__(self):
        import torch
        from sglang.srt.utils.patch_torch import _rebuild_cuda_tensor_modified

        upper = (1 << 63) - 1
        return _rebuild_cuda_tensor_modified, (
            torch.Tensor,
            torch.Size([upper]),
            (upper,),
            upper,
            torch.storage.TypedStorage,
            torch.uint8,
            "GPU-" + "x" * 256,
            bytes(range(64)),
            upper,
            upper,
            False,
            b"r" * 512,
            upper,
            bytes(range(64, 128)),
            True,
        )


class _StagingArena:
    """Reuse fenced storage even while NCCL/IPC retain inactive Python
    references."""

    def __init__(self, capacity: int, device: Any) -> None:
        import torch

        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Positive staging capacity required")
        self.storage = torch.empty(capacity, dtype=torch.uint8, device=device)
        self.free = [(0, capacity)]
        self.active: dict[int, tuple[int, int]] = {}

    def reserve(self, key: int, size: int) -> Any:
        if key in self.active or type(size) is not int or size < 0:
            raise ValueError("Invalid or repeated arena reservation")
        if not size:
            self.active[key] = (0, 0)
            return self.storage[:0]
        charged = (size + 3) // 4 * 4
        for index, (offset, available) in enumerate(self.free):
            if size <= available:
                charged = min(charged, available)
                self.active[key] = (offset, charged)
                self.free[index : index + 1] = [(offset + charged, available - charged)] if charged < available else []
                return self.storage[offset : offset + size]
        raise ValueError("Retained round storage exceeds the staging arena")

    def release(self, key: int) -> None:
        offset, size = self.active.pop(key)
        if size:
            blocks = sorted([*self.free, (offset, size)])
            self.free = []
            for start, length in blocks:
                if self.free and self.free[-1][0] + self.free[-1][1] == start:
                    previous, count = self.free.pop()
                    self.free.append((previous, count + length))
                else:
                    self.free.append((start, length))


def _child(connection: Connection) -> None:
    """The allocating process serializes IPC and survives ambiguous
    consumption."""
    import torch

    group, buffers, workers, arena = None, {}, [], None
    try:
        config = connection.recv()
        workers = config["workers"]
        torch.cuda.set_device(config["device"])
        device = torch.device("cuda", config["device"])
        properties = torch.cuda.get_device_properties(device)
        identity = dict(pid=os.getpid(), device=device.index, uuid=str(properties.uuid), rank=config["rank"])
        free, total = torch.cuda.mem_get_info(device)
        connection.send(
            dict(
                success=True,
                **identity,
                free_bytes=free,
                total_bytes=total,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                staging_capacity_bytes=0,
            )
        )
        while True:
            command = connection.recv()
            op = command["op"]
            try:
                if op == "join":
                    if group is not None or buffers:
                        raise RuntimeError("A proxy transport can only join once before receiving")
                    if command["rank"] is not None:
                        group = GpuSyncGroup(
                            command["connection"], rank=command["rank"], device=device, timeout_s=config["timeout_s"]
                        )
                    result = dict(
                        success=True,
                        **identity,
                        transport="nccl" if group is not None else "tcp",
                        collective_rank=command["rank"],
                    )
                elif op == "receive":
                    key, batch = command["round"], command["batch"]
                    if key in buffers:
                        raise ValueError("A proxy round is already retained")
                    if arena is None:
                        free, _ = torch.cuda.mem_get_info(device)
                        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                        if command["budget"] > free + reusable:
                            raise ValueError("Proxy arena exceeds physically available memory")
                        arena = _StagingArena(command["budget"], device)
                    if arena.storage.numel() != command["budget"]:
                        raise ValueError("A live proxy arena cannot change capacity")
                    slab = arena.reserve(key, batch["bytes"])
                    if group is not None:
                        slab, works = group.receive(batch, slab)
                    else:
                        from relax.distributed.weight_sync.tcp_transport import receive

                        host = (
                            receive(
                                command["descriptor"],
                                str(config["rank"]),
                                timeout_s=config["timeout_s"],
                                max_bytes=command["budget"],
                            )
                            if command["descriptor"] is not None
                            else bytearray()
                        )
                        if host:
                            slab.copy_(torch.frombuffer(host, dtype=torch.uint8))
                        torch.cuda.synchronize(device)
                        works = []
                    buffers[key] = dict(slab=slab, works=works, exported=False, complete=False)
                    del slab, works
                    result = dict(success=True, posted=True)
                elif op == "ready":
                    item = buffers[command["round"]]
                    if group is not None:
                        group.fence(item["works"], deadline=command["deadline"])
                    item["complete"] = True
                    item["works"] = []
                    del item
                    result = dict(
                        success=True,
                        retained_payload_bytes=sum(v["slab"].numel() for v in buffers.values()),
                        staging_capacity_bytes=arena.storage.numel(),
                        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                        cuda_process_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        cuda_process_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    )
                elif op == "export":
                    item = buffers[command["round"]]
                    if not item["complete"]:
                        raise ValueError("Cannot export an incomplete NCCL receive")
                    # Mark before serialization: even an ambiguous send failure
                    # must not cause early producer destruction.
                    item["exported"] = True
                    item["status"] = torch.empty(command["count"], dtype=torch.uint8, device=device)
                    result = dict(
                        success=True, serialized=_serialize(command["control"], item["slab"], item["status"])
                    )
                    del item
                elif op == "status":
                    item = buffers[command["round"]]
                    result = dict(success=True, full_required=item["status"].cpu().tolist())
                    del item
                elif op == "release":
                    item = buffers[command["round"]]
                    if item["exported"] and command.get("consumed") is not True:
                        raise ValueError("Exported IPC slab lacks a worker consumption fence")
                    if not item["complete"]:
                        raise ValueError("Cannot release an in-flight NCCL receive")
                    del buffers[command["round"]]
                    del item
                    arena.release(command["round"])
                    torch.cuda.ipc_collect()
                    result = dict(success=True)
                elif op == "memory":
                    free, total = torch.cuda.mem_get_info(device)
                    result = dict(
                        success=True,
                        **identity,
                        free_bytes=free,
                        total_bytes=total,
                        allocated_bytes=torch.cuda.memory_allocated(device),
                        reserved_bytes=torch.cuda.memory_reserved(device),
                        staging_capacity_bytes=arena.storage.numel() if arena is not None else 0,
                        retained_payload_bytes=sum(v["slab"].numel() for v in buffers.values()),
                    )
                elif op == "close":
                    if buffers and not _workers_exited(workers):
                        raise RuntimeError("Retaining unresolved IPC/NCCL buffers until model worker exit")
                    connection.send(dict(success=True))
                    break
                else:
                    raise ValueError("Unknown proxy operation")
                connection.send(result)
            except BaseException as exc:
                connection.send(dict(success=False, error=f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as exc:
        try:
            connection.send(dict(success=False, error=f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        # Parent death/HTTP timeout is not a consumer fence. Keep every pending
        # producer alive until the original server tree actually exits.
        while buffers and not _workers_exited(workers):
            time.sleep(1)
        buffers.clear()
        if group is not None:
            group.close()
        connection.close()


class WeightSyncProxy:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.children: list[tuple[Any, Connection]] = []
        self.connection: dict | None = None
        self.identities: list[dict] = []
        self.version: str | None = None
        self.batch = 0
        self.metrics: dict = {}

    def _receive_replies(self, *, deadline: float) -> list:
        results, errors = [], []
        for _process, connection in self.children:
            if not connection.poll(remaining(deadline)):
                raise TimeoutError("Proxy subprocess did not complete before its deadline")
            result = connection.recv()
            if result.get("success") is not True:
                errors.append(result)
            results.append(result)
        if errors:
            raise RuntimeError(f"GPU proxy subprocess batch failed: {errors}")
        return results

    def _commands(self, commands: list[dict], *, deadline: float) -> list:
        if len(commands) != len(self.children):
            raise ValueError("Proxy commands must cover every TP process")
        for (_process, connection), command in zip(self.children, commands, strict=True):
            connection.send(command)
        return self._receive_replies(deadline=deadline)

    def prepare(self, ranks: list[int], *, timeout: float) -> list:
        if self.children:
            if [item["rank"] for item in self.identities] != ranks:
                raise ValueError("Proxy GPU delivery identities changed")
            return self._commands([dict(op="memory")] * len(ranks), deadline=time.monotonic() + timeout)
        args = getattr(self.engine, "_weight_sync_server_args", {})
        if not isinstance(args, dict) or any(
            type(args.get(key)) is not int for key in ("nnodes", "tp_size", "base_gpu_id", "gpu_id_step")
        ):
            raise ValueError("GPU proxy requires resolved integer GPU topology")
        if args["nnodes"] != 1 or args["tp_size"] != len(ranks) or self.children:
            raise ValueError("GPU proxy requires one fixed local process per actual TP GPU")
        if args["base_gpu_id"] < 0 or args["gpu_id_step"] <= 0:
            raise ValueError("GPU proxy requires a nonnegative base device and a positive device step")
        from multiprocessing import Pipe

        import psutil

        root = psutil.Process(self.engine.process.pid)
        workers = [_process_identity(p.pid) for p in [root, *root.children(recursive=True)]]
        if len(workers) <= 1:
            raise ValueError("Cannot verify the existing SGLang worker process tree")
        deadline = time.monotonic() + timeout
        for tp, rank in enumerate(ranks):
            parent, child = Pipe(duplex=True)
            fd = child.fileno()
            process = subprocess.Popen(
                [sys.executable, "-m", __name__, str(fd)],
                pass_fds=(fd,),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            child.close()
            self.children.append((process, parent))
            parent.send(
                dict(
                    rank=rank,
                    timeout_s=remaining(deadline),
                    workers=workers,
                    device=args["base_gpu_id"] + tp * args["gpu_id_step"],
                )
            )
        self.identities = self._receive_replies(deadline=deadline)
        if len({item["uuid"] for item in self.identities}) != len(ranks):
            raise ValueError("Proxy TP processes do not map to distinct physical GPUs")
        return self.identities

    def join(self, connection: dict, ranks: list[int | None], *, timeout: float) -> list:
        if len(ranks) != len(self.children) or not self.children or self.connection is not None:
            raise ValueError("Prepare fixed GPU owners before joining their transport")
        replies = self._commands(
            [dict(op="join", connection=connection, rank=rank) for rank in ranks], deadline=time.monotonic() + timeout
        )
        self.connection = connection
        return replies

    def receive(
        self, round_id: int, batches: list[dict], budget: int, *, timeout: float, descriptor: dict | None = None
    ) -> list:
        return self._commands(
            [dict(op="receive", round=round_id, batch=b, budget=budget, descriptor=descriptor) for b in batches],
            deadline=time.monotonic() + timeout,
        )

    def ready(self, round_id: int, *, timeout: float) -> list:
        deadline = time.monotonic() + timeout
        return self._commands(
            [dict(op="ready", round=round_id, deadline=deadline)] * len(self.children), deadline=deadline
        )

    def _request(self, endpoint: str, payload: dict, *, deadline: float) -> None:
        size = _http_bytes(payload)
        if size > MAX_HTTP_BYTES:
            raise ValueError("Serialized whole-engine proxy HTTP body exceeds 64KiB")
        started = time.monotonic()
        response = self.engine._make_request(endpoint, payload, timeout=remaining(deadline))
        if not isinstance(response, dict) or response.get("success") is not True:
            raise RuntimeError(f"SGLang {endpoint} failed during GPU proxy installation")
        self.metrics["http_calls"] += 1
        self.metrics["control_bytes"] += size
        self.metrics["http_s"] += time.monotonic() - started

    @staticmethod
    def _payload(serialized: list[str], *, flush: bool = False) -> dict:
        return dict(serialized_named_tensors=serialized, load_format=PROXY_MEMORY_LOADER, flush_cache=flush)

    def _controls(self, controls: list[dict], *, deadline: float, flush: bool = False) -> None:
        self._request(
            "update_weights_from_tensor",
            self._payload([_serialize(c) for c in controls], flush=flush),
            deadline=deadline,
        )

    def _documents(self, kind: str, documents: list, *, deadline: float) -> list[str]:
        encoded = [json_bytes(d).decode() for d in documents]
        if any(len(data.encode()) > MAX_DOCUMENT_BYTES for data in encoded):
            raise ValueError("Proxy document exceeds its bounded worker directory capacity")
        identities = [hashlib.sha256(data.encode()).hexdigest() for data in encoded]
        # Decide the complete fragment set before publishing its first fragment.
        width = max(1, MAX_HTTP_BYTES // (4 * len(encoded)))
        while True:
            count = max((len(data) + width - 1) // width for data in encoded)
            controls = [
                [
                    dict(
                        op="document",
                        kind=kind,
                        id=identity,
                        index=i,
                        count=count,
                        data=data[i * width : (i + 1) * width],
                    )
                    for data, identity in zip(encoded, identities, strict=True)
                ]
                for i in range(count)
            ]
            if all(_http_bytes(self._payload([_serialize(c) for c in part])) <= MAX_HTTP_BYTES for part in controls):
                break
            if width == 1:
                raise ValueError("One proxy control fragment cannot fit the whole-engine HTTP budget")
            width = max(1, width // 2)
        for part in controls:
            self._controls(part, deadline=deadline)
            self.metrics["schema_calls" if kind == "schema" else "update_directory_calls"] += 1
        return identities

    def begin(self, control: dict, *, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        self.metrics = dict(
            http_calls=0, control_bytes=0, http_s=0.0, schema_calls=0, update_directory_calls=0, consume_calls=0
        )
        self.batch = 0
        self.version = control["version"]
        self.staging_budget = control["staging_budget_bytes"]
        self._controls(
            [dict(op="begin", **control, uuid=identity["uuid"]) for identity in self.identities], deadline=deadline
        )
        return dict(success=True)

    def consume(self, round_id: int, metadata: dict, *, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        ids = self._documents("frame", [metadata] * len(self.children), deadline=deadline)
        controls = [
            dict(op="consume", version=self.version, frame=self.batch, metadata_id=identity) for identity in ids
        ]
        # Bound the request before the first real IPC export. No speculative
        # CUDA handle is created and abandoned during control-size planning.
        preview = self._payload([_serialize(c, _CudaWireSize(), _CudaWireSize()) for c in controls])
        if _http_bytes(preview) > MAX_HTTP_BYTES:
            raise ValueError("One GPU bucket IPC control exceeds its HTTP budget")
        replies = self._commands(
            [
                dict(op="export", round=round_id, control=control, count=len(metadata["tensors"]))
                for control in controls
            ],
            deadline=deadline,
        )
        self._request(
            "update_weights_from_tensor", self._payload([item["serialized"] for item in replies]), deadline=deadline
        )
        statuses = self._commands([dict(op="status", round=round_id)] * len(self.children), deadline=deadline)
        self._commands([dict(op="release", round=round_id, consumed=True)] * len(self.children), deadline=deadline)
        self.metrics["consume_calls"] += 1
        self.batch += 1
        names = [entry["name"] for entry in metadata["tensors"]]
        required = sorted(
            {name for status in statuses for name, flag in zip(names, status["full_required"], strict=True) if flag}
        )
        return dict(success=True, full_required=required)

    def finalize(self, update: str, *, timeout: float) -> dict:
        if update != self.version:
            raise ValueError("Final delta version differs from the active update")
        deadline = time.monotonic() + timeout
        empty = dict(bytes=0, entries=[])
        self.receive(self.batch, [empty] * len(self.children), self.staging_budget, timeout=remaining(deadline))
        self.ready(self.batch, timeout=remaining(deadline))
        control = dict(op="finalize", version=update, frames=self.batch)
        replies = self._commands(
            [dict(op="export", round=self.batch, control=control, count=1)] * len(self.children), deadline=deadline
        )
        self._request(
            "update_weights_from_tensor",
            self._payload([reply["serialized"] for reply in replies], flush=True),
            deadline=deadline,
        )
        statuses = self._commands([dict(op="status", round=self.batch)] * len(self.children), deadline=deadline)
        self._commands([dict(op="release", round=self.batch, consumed=True)] * len(self.children), deadline=deadline)
        self._request("update_weight_version", dict(new_version=update, abort_all_requests=False), deadline=deadline)
        return dict(
            success=True,
            force_full_seed=any(reply["full_required"][0] for reply in statuses),
            metrics=dict(self.metrics),
            proxies=self.identities,
        )

    def close(self, *, timeout: float = 10) -> None:
        if not self.children:
            return
        deadline = time.monotonic() + timeout
        self._commands([dict(op="close")] * len(self.children), deadline=deadline)
        for process, connection in self.children:
            connection.close()
            process.wait(timeout=remaining(deadline))
        self.children.clear()


if __name__ == "__main__":
    _child(Connection(int(sys.argv[1])))
