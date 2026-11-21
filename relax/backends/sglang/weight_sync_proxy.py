# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-owned local TP receivers.

CUDA payloads never enter the parent actor.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from multiprocessing.connection import Connection
from typing import Any

from relax.distributed.weight_sync.codec import WIRE_ALIGNMENT, validate_directory
from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup, remaining

from .delta_loader import LOADER as PROXY_MEMORY_LOADER
from .delta_loader import MAX_DOCUMENT_BYTES, MAX_HTTP_BYTES, workspace_bytes
from .delta_shard_layout import ShardFrame, marked_ordinals


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def _http_bytes(payload: dict) -> int:
    # requests' json= body uses these default separators/ASCII escaping.
    return len(json.dumps(payload, allow_nan=False).encode())


def _compressed_directory(data: bytes, tp_size: int) -> str | None:
    # Each TP control contains its own base64 copy, even before adding the
    # outer serializer and IPC handles. Stop incompressible input early.
    cap = (min(len(data) - 1, MAX_HTTP_BYTES // tp_size) // 4) * 3
    compressor = zlib.compressobj(level=1)
    compressed = bytearray()
    for offset in range(0, len(data), MAX_HTTP_BYTES):
        compressed.extend(compressor.compress(data[offset : offset + MAX_HTTP_BYTES]))
        if len(compressed) > cap:
            return None
    compressed.extend(compressor.flush())
    if len(compressed) > cap:
        return None
    return base64.b64encode(compressed).decode("ascii")


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


def _status_buffer(count: int, tail: bytes, device: Any) -> Any:
    import torch

    ack = torch.empty(count + len(tail), dtype=torch.uint8, device=device)
    if tail:
        ack[count:].copy_(torch.frombuffer(bytearray(tail), dtype=torch.uint8))
    return ack


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
        charged = (size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
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
        identity["nccl_net"] = os.getenv("NCCL_NET")
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
                        transport=command["connection"]["transport"] if group is not None else "tcp",
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
                elif op == "locations":
                    keys = range(command["round"], command["stop"])
                    if any(not buffers[k]["complete"] or buffers[k]["exported"] for k in keys):
                        raise ValueError("Ordered consume requires a fenced, unexported prefix")
                    result = dict(
                        success=True,
                        free_bytes=torch.cuda.mem_get_info(device)[0],
                        spans=[(arena.active[k][0], buffers[k]["slab"].numel()) for k in keys],
                    )
                elif op == "export":
                    item = buffers[command["round"]]
                    if not item["complete"]:
                        raise ValueError("Cannot export an incomplete NCCL receive")
                    # Mark before serialization: even an ambiguous send failure
                    # must not cause early producer destruction.
                    slab = item["slab"]
                    if "span" in command:
                        offset, length = command["span"]
                        slab = arena.storage.narrow(0, offset, length)
                        for key in range(command["round"], command["stop"]):
                            buffers[key]["exported"] = True
                    item["exported"] = True
                    ack = _status_buffer(command["count"], command.get("directory_tail", b""), device)
                    # The prefix view also keeps the immutable directory allocation
                    # alive until the worker's original consume/release fence.
                    item["status"] = ack[: command["count"]]
                    if command.get("sentinel"):
                        item["status"].fill_(255)
                    result = dict(success=True, serialized=_serialize(command["control"], slab, ack))
                    del item, slab, ack
                elif op == "status":
                    item = buffers[command["round"]]
                    result = dict(success=True, full_required=item["status"].cpu().tolist())
                    del item
                elif op == "release":
                    for key in range(command["round"], command.get("stop", command["round"] + 1)):
                        item = buffers[key]
                        if item["exported"] and command.get("consumed") is not True:
                            raise ValueError("Exported IPC slab lacks a worker consumption fence")
                        if not item["complete"]:
                            raise ValueError("Cannot release an in-flight NCCL receive")
                        del buffers[key], item
                        arena.release(key)
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
                    # A fenced pre-receive prefix has never exported an IPC
                    # handle. It can be discarded without waiting for the live
                    # generating server to exit. Ambiguous reads/exports keep
                    # the original stronger lifetime protection below.
                    for key in list(buffers):
                        item = buffers[key]
                        if item["complete"] and not item["exported"]:
                            del buffers[key]
                            arena.release(key)
                        del item
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
        self._inline_frame_directory = False
        self._workspace_budget = 0
        self._ordered_capable = self._record_seed = False
        self._seed_directory: dict[str, tuple] = {}
        self._target_requested = self._target_capable = self._target_failed = False
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
        self._inline_frame_directory = connection.get("transport") == "nccl_socket"
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

    def _send_frame_directory(self, data: bytes, identity: str, *, deadline: float) -> None:
        text = data.decode("ascii")
        envelope = {}
        if self._inline_frame_directory:
            compressed = base64.b64encode(zlib.compress(data, level=1))
            if len(compressed) < len(data):
                text = compressed.decode("ascii")
                envelope["decoded_bytes"] = len(data)
        tp_size = len(self.children)
        width = max(1, MAX_HTTP_BYTES // (4 * tp_size))
        while True:
            count = (len(text) + width - 1) // width
            parts = [
                dict(
                    op="document",
                    kind="frame",
                    id=identity,
                    index=i,
                    count=count,
                    data=text[i * width : (i + 1) * width],
                    **envelope,
                )
                for i in range(count)
            ]
            if all(
                _http_bytes(self._payload([_serialize(part) for _ in self.children])) <= MAX_HTTP_BYTES
                for part in parts
            ):
                break
            if width == 1:
                raise ValueError("One proxy control fragment cannot fit the whole-engine HTTP budget")
            width = max(1, width // 2)
        before = self.metrics.get("http_s", 0.0)
        for part in parts:
            self._controls([part] * tp_size, deadline=deadline)
            self.metrics["update_directory_calls"] += 1
        self.metrics["update_directory_http_s"] = self.metrics.get("update_directory_http_s", 0.0) + (
            self.metrics.get("http_s", 0.0) - before
        )

    def _fits_ipc(self, control: dict) -> bool:
        preview = self._payload([_serialize(control, _CudaWireSize(), _CudaWireSize()) for _ in self.children])
        return _http_bytes(preview) <= MAX_HTTP_BYTES

    def begin(self, control: dict, *, timeout: float) -> dict:
        if self._target_failed:
            raise RuntimeError("Target shard publication has an ambiguous outcome")
        deadline = time.monotonic() + timeout
        self.metrics = dict(
            http_calls=0,
            control_bytes=0,
            http_s=0.0,
            schema_calls=0,
            update_directory_calls=0,
            update_directory_http_s=0.0,
            consume_calls=0,
            consume_http_s=0.0,
            consume_ack_s=0.0,
        )
        self.batch = 0
        self.version = control["version"]
        self.staging_budget = control["staging_budget_bytes"]
        self._target_requested = control.get("target_shard_request", False)
        if type(self._target_requested) is not bool or (self._target_requested and len(self.children) != 4):
            raise ValueError("Target shard publication requires four TP owners")
        if control["seed"] or not self._target_requested:
            self._target_capable = self._ordered_capable = False
            self._seed_directory.clear()
        self._record_seed = self._target_requested and control["seed"]
        self._workspace_budget = control.get("receiver_gpu_budget_bytes", 0) - self.staging_budget
        controls = [dict(op="begin", **control, uuid=i["uuid"]) for i in self.identities]
        if self._target_requested:
            for rank, c in enumerate(controls):
                c["target_shard_rank"] = rank
                c["ordered_target_consume"] = True
        self._controls(controls, deadline=deadline)
        return dict(success=True, **({"ordered_target_consume": True} if self._ordered_capable else {}))

    def _shared_controls(self, metadata: dict, deadline: float) -> tuple:
        data = json_bytes(metadata)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError("Proxy document exceeds its bounded worker directory capacity")
        identity = hashlib.sha256(data).hexdigest()
        control = dict(op="consume", version=self.version, frame=self.batch, metadata_id=identity)
        inline = False
        if self._inline_frame_directory:
            compressed = _compressed_directory(data, len(self.children))
            if compressed is not None:
                candidate = dict(control, metadata_zlib=dict(bytes=len(data), data=compressed))
                if self._fits_ipc(candidate):
                    control, inline = candidate, True
        tail = b""
        if not inline and self._inline_frame_directory and not self._target_requested:
            compressed = zlib.compress(data, level=1)
            count = len(metadata["tensors"])
            allocation = ((count + len(compressed) + 511) // 512) * 512
            candidate = dict(control, metadata_ipc=[count, len(compressed), len(data)])
            if len(compressed) <= len(data) and allocation <= self._workspace_budget and self._fits_ipc(candidate):
                # Keep the document fallback when this extra allocation would
                # crowd out the ordinary loader. Direct sparse runs may need
                # one tile table in addition to the current loader group.
                entries = metadata["tensors"]
                ends = metadata.get("loader_ends", [count])
                decode = max(
                    workspace_bytes({"tensors": entries[start:end]})
                    for start, end in zip([0, *ends[:-1]], ends, strict=True)
                )
                tiles = ShardFrame.ordered_workspace([(0, e["count"]) for e in entries if not e["full"]])
                if allocation + decode + tiles <= self._workspace_budget:
                    control, tail = candidate, compressed
        if not inline and not tail:
            if not self._fits_ipc(control):
                raise ValueError("One GPU bucket IPC control exceeds its HTTP budget")
            self._send_frame_directory(data, identity, deadline=deadline)
        return [control] * len(self.children), inline, tail

    def _validate_lanes(self, lanes: list[dict]) -> None:
        if not self._target_requested or not self._target_capable or len(lanes) != 4 or len(self.children) != 4:
            raise ValueError("Partial lanes require a confirmed four-TP capability")
        previous = None
        for rank, lane in enumerate(lanes):
            clean = dict(lane)
            ordinals = marked_ordinals(clean.pop("target_shard", None), rank, len(clean["tensors"]))
            validate_directory(clean, max_bytes=self.staging_budget)
            marked = set(ordinals)
            key = (
                [
                    (e["name"], e["shape"], e["dtype"], e["full"], None if i in marked else e["count"])
                    for i, e in enumerate(clean["tensors"])
                ],
                ordinals,
            )
            if (previous is not None and previous != key) or any(
                clean["tensors"][i]["full"] or clean["tensors"][i]["dtype"] != "bfloat16" for i in ordinals
            ):
                raise ValueError("Target lane logical directories disagree")
            previous = key

    def _lane_controls(self, lanes: list[dict], deadline: float) -> tuple:
        self._validate_lanes(lanes)
        docs, controls, inline = [], [], []
        for lane in lanes:
            data = json_bytes(lane)
            if len(data) > MAX_DOCUMENT_BYTES:
                raise ValueError("Target lane directory exceeds its retained bound")
            ctl = dict(
                op="consume", version=self.version, frame=self.batch, metadata_id=hashlib.sha256(data).hexdigest()
            )
            controls.append(ctl)
            docs.append(data.decode("ascii"))
            compressed = _compressed_directory(data, len(lanes))
            if compressed is not None:
                inline.append(dict(ctl, metadata_zlib=dict(bytes=len(data), data=compressed)))

        def fits(ctl):
            return (
                _http_bytes(self._payload([_serialize(c, _CudaWireSize(), _CudaWireSize()) for c in ctl]))
                <= MAX_HTTP_BYTES
            )

        if len(inline) == len(lanes) and fits(inline):
            return inline, True
        if not fits(controls):
            raise ValueError("Target lane IPC control exceeds whole-engine HTTP budget")
        # All TPs receive the same number of original document operations.
        # Equal partition counts also accommodate unequal lane directory sizes.
        width = MAX_HTTP_BYTES // (4 * len(lanes))
        longest = max(map(len, docs))
        while True:
            count = (longest + width - 1) // width
            if count > 65536:
                raise ValueError("Too many target directory fragments")
            groups = [
                [
                    dict(
                        op="document",
                        kind="frame",
                        id=c["metadata_id"],
                        index=i,
                        count=count,
                        data=d[i * len(d) // count : (i + 1) * len(d) // count],
                    )
                    for c, d in zip(controls, docs, strict=True)
                ]
                for i in range(count)
            ]
            if all(_http_bytes(self._payload([_serialize(c) for c in g])) <= MAX_HTTP_BYTES for g in groups):
                break
            if width == 1:
                raise ValueError("A target directory fragment cannot fit its HTTP budget")
            width = max(1, width // 2)
        for g in groups:
            self._controls(g, deadline=deadline)
            self.metrics["update_directory_calls"] += 1
        return controls, False

    def consume(
        self, round_id: int, metadata: dict | list[dict], *, timeout: float, following: list | None = None
    ) -> dict:
        lanes = isinstance(metadata, (list, tuple))
        if self._target_failed:
            raise RuntimeError("Target shard publication has an ambiguous outcome")
        deadline = time.monotonic() + timeout
        try:
            if following:
                if not lanes or not self._ordered_capable or round_id != self.batch:
                    raise ValueError("Ordered consume requires an acknowledged target prefix")
                return self._ordered(round_id, [metadata, *following], deadline)
            return self._consume(round_id, metadata, deadline, lanes)
        except BaseException:
            if lanes or following:
                self._target_failed = True
            raise

    def _ordered(self, first: int, frames: list, deadline: float) -> dict:
        # The FULL seed installed the same name order in the worker and proxy.
        # Only changing counts and seed ordinals need to cross HTTP.
        compact, workspace = [], []
        for lanes in frames:
            self._validate_lanes(lanes)
            encoded, peak = [], 0
            for lane in lanes:
                entries = lane["tensors"]
                if len(lane["target_shard"]["entries"]) != len(entries):
                    raise ValueError("Only complete target-only frames can share a consume")
                codes = []
                for e in entries:
                    ordinal, shape, dtype = self._seed_directory[e["name"]]
                    if tuple(e["shape"]) != shape or e["dtype"] != dtype:
                        raise ValueError("Target frame disagrees with its seed directory")
                    codes.append((ordinal, e["count"]))
                encoded.append(codes)
                peak = max(peak, ShardFrame.ordered_workspace(codes))
            compact.append(encoded)
            workspace.append(peak)
        locations = self._commands(
            [dict(op="locations", round=first, stop=first + len(frames))] * 4, deadline=deadline
        )
        if len(locations) != 4 or any(len(reply["spans"]) != len(frames) for reply in locations):
            raise ValueError("Missing retained target frame locations")
        for rank, reply in enumerate(locations):
            if any(size != frame[rank]["bytes"] for (_, size), frame in zip(reply["spans"], frames, strict=True)):
                raise ValueError("Target directory disagrees with fenced frame size")

        available = min(self._workspace_budget, *(r["free_bytes"] for r in locations))

        def partition(lo, hi):
            if hi - lo == 1:
                return [(lo, hi, None)]
            count = sum(len(frame[0]) for frame in compact[lo:hi])
            commands = []
            if count + 1024 + sum(workspace[lo:hi]) <= available:
                for rank, reply in enumerate(locations):
                    spans = [(offset, size) for offset, size in reply["spans"][lo:hi] if size]
                    if any(
                        offset != (prev + size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
                        for (prev, size), (offset, _) in zip(spans, spans[1:])
                    ):
                        break
                    start = spans[0][0] if spans else 0
                    length = sum(spans[-1]) - start if spans else 0
                    if length > self.staging_budget:
                        break
                    directory = [frame[rank] for frame in compact[lo:hi]]
                    data = json_bytes(directory)
                    if len(data) > MAX_DOCUMENT_BYTES:
                        break
                    ctl = dict(op="consume", ordered_target=True, frame=first + lo, version=self.version)
                    compressed = _compressed_directory(data, 4)
                    if compressed is None:
                        ctl["metadata"] = directory
                    else:
                        ctl.update(
                            metadata_zlib=dict(bytes=len(data), data=compressed),
                            metadata_id=hashlib.sha256(data).hexdigest(),
                        )
                    commands.append(
                        dict(
                            op="export",
                            round=first + lo,
                            stop=first + hi,
                            span=(start, length),
                            count=count,
                            sentinel=True,
                            control=ctl,
                        )
                    )
            if (
                len(commands) == 4
                and _http_bytes(
                    self._payload([_serialize(c["control"], _CudaWireSize(), _CudaWireSize()) for c in commands])
                )
                <= MAX_HTTP_BYTES
            ):
                return [(lo, hi, commands)]
            mid = (lo + hi) // 2
            return partition(lo, mid) + partition(mid, hi)

        # Plan every real serializer envelope before exporting any IPC handle.
        # A transport/worker error after export is terminal, including a partial ACK.
        for lo, hi, commands in partition(0, len(frames)):
            if commands is None:
                self._consume(first + lo, frames[lo], deadline, True)
                continue
            replies = self._commands(commands, deadline=deadline)
            self._request(
                "update_weights_from_tensor", self._payload([r["serialized"] for r in replies]), deadline=deadline
            )
            statuses = self._commands([dict(op="status", round=first + lo)] * 4, deadline=deadline)
            count = commands[0]["count"]
            if len(statuses) != 4 or any(
                len(s["full_required"]) != count or any(s["full_required"]) for s in statuses
            ):
                raise RuntimeError("Missing ordered target install acknowledgement")
            self._commands(
                [dict(op="release", round=first + lo, stop=first + hi, consumed=True)] * 4, deadline=deadline
            )
            for frame in frames[lo:hi]:
                self._record_target_frame(frame)
            self.metrics["consume_calls"] += 1
            self.metrics["ordered_consume_calls"] = self.metrics.get("ordered_consume_calls", 0) + 1
            self.metrics["ordered_consume_frames"] = self.metrics.get("ordered_consume_frames", 0) + hi - lo
            self.batch += hi - lo
        return dict(success=True, full_required=[], ordered_frames_ack=len(frames))

    def _consume(self, round_id: int, metadata: dict | list[dict], deadline: float, lanes: bool) -> dict:
        if round_id != self.batch:
            raise ValueError("Unexpected logical round")
        tail = b""
        if lanes:
            controls, inline = self._lane_controls(metadata, deadline)
            directory = metadata[0]
        else:
            if "target_shard" in metadata:
                raise ValueError("Partial target mapping requires four ordered lanes")
            controls, inline, tail = self._shared_controls(metadata, deadline)
            directory = metadata
        replies = self._commands(
            [
                dict(
                    op="export",
                    round=round_id,
                    control=c,
                    count=len(directory["tensors"]),
                    **({"sentinel": True} if lanes else {}),
                    **({"directory_tail": tail} if tail else {}),
                )
                for c in controls
            ],
            deadline=deadline,
        )
        before = self.metrics.get("http_s", 0.0)
        self._request(
            "update_weights_from_tensor", self._payload([item["serialized"] for item in replies]), deadline=deadline
        )
        self.metrics["consume_http_s"] = (
            self.metrics.get("consume_http_s", 0.0) + self.metrics.get("http_s", 0.0) - before
        )
        ack_started = time.monotonic()
        statuses = self._commands([dict(op="status", round=round_id)] * len(self.children), deadline=deadline)
        if lanes:
            marked = set(directory["target_shard"]["entries"])
            if len(statuses) != 4 or any(
                len(s["full_required"]) != len(directory["tensors"])
                or any(v not in (0, 1) or (i in marked and v != 0) for i, v in enumerate(s["full_required"]))
                for s in statuses
            ):
                raise RuntimeError("Missing or invalid target shard install acknowledgement")
        self._commands([dict(op="release", round=round_id, consumed=True)] * len(self.children), deadline=deadline)
        self.metrics["consume_ack_s"] = self.metrics.get("consume_ack_s", 0.0) + (time.monotonic() - ack_started)
        if lanes:
            self._record_target_frame(metadata)
        elif self._record_seed:
            for e in directory["tensors"]:
                self._seed_directory[e["name"]] = (len(self._seed_directory), tuple(e["shape"]), e["dtype"])
        self.metrics["consume_calls"] += 1
        if self._inline_frame_directory or lanes:
            metric = (
                "ipc_directory_frames"
                if tail
                else ("inline_directory_frames" if inline else "inline_directory_fallback_frames")
            )
            if tail:
                self.metrics["ipc_directory_bytes"] = self.metrics.get("ipc_directory_bytes", 0) + len(tail)
            self.metrics[metric] = self.metrics.get(metric, 0) + 1
        self.batch += 1
        names = [entry["name"] for entry in directory["tensors"]]
        required = sorted(
            {name for status in statuses for name, flag in zip(names, status["full_required"], strict=True) if flag}
        )
        return dict(success=True, full_required=required)

    def _record_target_frame(self, metadata: list[dict]) -> None:
        stats = self.metrics.setdefault(
            "target_shards",
            [
                dict(tp_rank=r, frames=0, raw_bytes=0, marked_words=0, marked_entries=0, status_acks=0, releases=0)
                for r in range(4)
            ],
        )
        for lane, record in zip(metadata, stats, strict=True):
            ordinals = lane["target_shard"]["entries"]
            record["frames"] += 1
            record["raw_bytes"] += lane["bytes"]
            record["marked_entries"] += len(ordinals)
            record["marked_words"] += sum(lane["tensors"][i]["count"] for i in ordinals)
            record["status_acks"] += 1
            record["releases"] += 1

    def finalize(self, update: str, *, timeout: float) -> dict:
        try:
            return self._finalize(update, timeout=timeout)
        except BaseException:
            if self._target_requested:
                self._target_failed = True
            raise

    def _finalize(self, update: str, *, timeout: float) -> dict:
        if self._target_failed:
            raise RuntimeError("Target shard publication has an ambiguous outcome")
        if update != self.version:
            raise ValueError("Final delta version differs from the active update")
        deadline = time.monotonic() + timeout
        empty = dict(bytes=0, entries=[])
        self.receive(self.batch, [empty] * len(self.children), self.staging_budget, timeout=remaining(deadline))
        self.ready(self.batch, timeout=remaining(deadline))
        control = dict(op="finalize", version=update, frames=self.batch)
        replies = self._commands(
            [
                dict(
                    op="export",
                    round=self.batch,
                    control=control,
                    count=2 if self._target_requested else 1,
                    **({"sentinel": True} if self._target_requested else {}),
                )
            ]
            * len(self.children),
            deadline=deadline,
        )
        self._request(
            "update_weights_from_tensor",
            self._payload([reply["serialized"] for reply in replies], flush=True),
            deadline=deadline,
        )
        statuses = self._commands([dict(op="status", round=self.batch)] * len(self.children), deadline=deadline)
        if self._target_requested:
            if len(statuses) != 4 or any(
                len(r["full_required"]) != 2
                or r["full_required"][0] not in (0, 1)
                or r["full_required"][1] not in (0, 1, 2)
                for r in statuses
            ):
                self._target_failed = True
                raise RuntimeError("Incomplete target shard final acknowledgement")
            capable = all(r["full_required"][0] == 0 and r["full_required"][1] in (1, 2) for r in statuses)
            self._ordered_capable = all(r["full_required"] == [0, 2] for r in statuses)
        self._commands([dict(op="release", round=self.batch, consumed=True)] * len(self.children), deadline=deadline)
        self._request("update_weight_version", dict(new_version=update, abort_all_requests=False), deadline=deadline)
        if self._target_requested:
            self._target_capable = capable
        return dict(
            success=True,
            **({"target_shard_capable": self._target_capable} if self._target_requested else {}),
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
