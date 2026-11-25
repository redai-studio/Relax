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

from relax.distributed.weight_sync.codec import WIRE_ALIGNMENT
from relax.distributed.weight_sync.config import supports_delta_layout
from relax.distributed.weight_sync.gpu_transport import GpuSyncGroup, remaining

from .delta_loader import LOADER as PROXY_MEMORY_LOADER
from .delta_loader import MAX_DOCUMENT_BYTES, MAX_HTTP_BYTES, VERIFY_ACK_BYTES, workspace_bytes
from .delta_pipeline import PIPELINE_LOADER
from .routed_expert_delta import EXPERT_ROUTE_CAPACITY


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
    ack[:count].fill_(255)
    if tail:
        ack[count:].copy_(torch.frombuffer(bytearray(tail), dtype=torch.uint8))
    return ack


def _install_status(item: dict) -> dict:
    result = dict(success=True, full_required=item["status"].cpu().tolist())
    if "expert_routes" in item:
        directory = item["expert_routes"].cpu().numpy().tobytes()
        size = int.from_bytes(directory[:4], "little")
        if not 0 < size <= len(directory) - 4:
            raise ValueError("Missing or oversized scheduler expert-route acknowledgement")
        result["expert_routes"] = json.loads(directory[4 : 4 + size])
    if "verified_bytes" in item:
        result["verified_bytes"] = int.from_bytes(item["verified_bytes"].cpu().numpy().tobytes(), "little")
    return result


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
        identity.update({k: config[k] for k in ("tp_size", "pp_size", "tp_rank", "pp_rank")})
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
                elif op == "export":
                    item = buffers[command["round"]]
                    if not item["complete"]:
                        raise ValueError("Cannot export an incomplete NCCL receive")
                    # Mark before serialization: even an ambiguous send failure
                    # must not cause early producer destruction.
                    slab = item["slab"]
                    item["exported"] = True
                    pipeline = config["pp_size"] > 1
                    tail = command.get("directory_tail", b"")
                    ack = _status_buffer(command["count"], tail + (b"\x00" if pipeline else b""), device)
                    if pipeline:
                        item["completion"] = ack[-1:]
                    # The prefix view also keeps the immutable directory allocation
                    # alive until the worker's original consume/release fence.
                    prefix_count = command["count"]
                    item["status"] = ack[:prefix_count]
                    for key, size in (
                        ("expert_routes", command["control"].get("expert_route_capacity", 0)),
                        ("verified_bytes", command["control"].get("verify_ack_bytes", 0)),
                    ):
                        if size:
                            item[key] = ack[prefix_count : prefix_count + size]
                    result = dict(success=True, serialized=_serialize(command["control"], slab, ack))
                    del item, slab, ack
                elif op == "status":
                    item = buffers[command["round"]]
                    if "completion" in item:
                        while (completed := item["completion"].cpu().item()) == 0:
                            time.sleep(min(0.005, remaining(command["deadline"])))
                        if completed != 1:
                            raise ValueError("Invalid pipeline completion acknowledgement")
                    result = _install_status(item)
                    del item
                elif op == "release":
                    key = command["round"]
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
                    if command.get("abort_group") and group is not None:
                        group.abort()
                        group = None
                    connection.send(dict(success=True, op="close", abort_group=bool(command.get("abort_group"))))
                    break
                else:
                    raise ValueError("Unknown proxy operation")
                connection.send(result)
            except BaseException as exc:
                connection.send(dict(success=False, op=op, error=f"{type(exc).__name__}: {exc}"))
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
        self._publication_failed = False
        self._announce_expert_routes = False
        self._verify = False
        self.tp_size = self.pp_size = 1
        self.children: list[tuple[Any, Connection]] = []
        self._closing: dict[Connection, tuple[bool, bool]] = {}
        self._stopping = False
        self._workers: list[tuple[int, int]] = []
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
        if self._closing or self._stopping:
            raise RuntimeError("Cannot issue commands while retiring GPU proxies")
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
        if not supports_delta_layout(args, len(ranks)):
            raise ValueError("GPU proxy requires one local process per TP/PP coordinate")
        self.tp_size, self.pp_size = args["tp_size"], args["pp_size"]
        if args["base_gpu_id"] < 0 or args["gpu_id_step"] <= 0:
            raise ValueError("GPU proxy requires a nonnegative base device and a positive device step")
        from multiprocessing import Pipe

        import psutil

        root = psutil.Process(self.engine.process.pid)
        workers = [_process_identity(p.pid) for p in [root, *root.children(recursive=True)]]
        if len(workers) <= 1:
            raise ValueError("Cannot verify the existing SGLang worker process tree")
        self._workers = workers
        deadline = time.monotonic() + timeout
        devices = [
            args["base_gpu_id"] + pp * self.tp_size + tp * args["gpu_id_step"]
            for pp in range(self.pp_size)
            for tp in range(self.tp_size)
        ]
        if len(set(devices)) != len(devices):
            raise ValueError("SGLang TP/PP coordinates overlap on a physical GPU")
        for local, rank in enumerate(ranks):
            pp, tp = divmod(local, self.tp_size)
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
                    device=devices[local],
                    tp_size=self.tp_size,
                    pp_size=self.pp_size,
                    tp_rank=tp,
                    pp_rank=pp,
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

    def _payload(self, serialized: list[str], *, flush: bool = False) -> dict:
        loader = PROXY_MEMORY_LOADER
        if self.pp_size > 1:
            if len(serialized) != self.pp_size * self.tp_size:
                raise ValueError("Incomplete SGLang pipeline delivery")
            serialized = [
                _serialize(dict(tp=self.tp_size, rank=rank, stages=serialized[rank :: self.tp_size]))
                for rank in range(self.tp_size)
            ]
            loader = PIPELINE_LOADER
        return dict(serialized_named_tensors=serialized, load_format=loader, flush_cache=flush)

    def _controls(self, controls: list[dict], *, deadline: float, flush: bool = False) -> None:
        self._request(
            "update_weights_from_tensor",
            self._payload([_serialize(c) for c in controls], flush=flush),
            deadline=deadline,
        )

    def _send_frame_directory(self, data: bytes, identity: str, *, deadline: float) -> None:
        ranks = len(self.children)
        self._send_frame_directories([data] * ranks, [identity] * ranks, deadline=deadline)

    def _send_frame_directories(self, documents: list[bytes], identities: list[str], *, deadline: float) -> None:
        texts, envelopes = [], []
        for data in documents:
            envelope = {}
            if self._inline_frame_directory:
                compressed = base64.b64encode(zlib.compress(data, level=1))
                if len(compressed) < len(data):
                    envelope["decoded_bytes"] = len(data)
                    data = compressed
            texts.append(data.decode("ascii"))
            envelopes.append(envelope)
        width = max(1, MAX_HTTP_BYTES // (4 * len(texts)))

        def fragment(index, count, width):
            return [
                dict(
                    op="document",
                    kind="frame",
                    id=key,
                    index=index,
                    count=count,
                    data=text[index * width : (index + 1) * width],
                    **envelope,
                )
                for key, text, envelope in zip(identities, texts, envelopes, strict=True)
            ]

        while True:
            # Shorter lanes pad with empty fragments to keep scheduler calls in lockstep.
            count = (max(map(len, texts)) + width - 1) // width
            if count <= 65536 and all(
                _http_bytes(self._payload([_serialize(c) for c in fragment(i, count, width)])) <= MAX_HTTP_BYTES
                for i in range(count)
            ):
                break
            if width == 1 or count > 65536:
                raise ValueError("Frame directory cannot fit the whole-engine HTTP budget")
            width = max(1, width // 2)
        before = self.metrics.get("http_s", 0.0)
        for i in range(count):
            self._controls(fragment(i, count, width), deadline=deadline)
            self.metrics["update_directory_calls"] += 1
        self.metrics["update_directory_http_s"] = self.metrics.get("update_directory_http_s", 0.0) + (
            self.metrics.get("http_s", 0.0) - before
        )

    def _fits_controls_ipc(self, controls: list[dict]) -> bool:
        preview = self._payload([_serialize(control, _CudaWireSize(), _CudaWireSize()) for control in controls])
        return _http_bytes(preview) <= MAX_HTTP_BYTES

    def _fits_ipc(self, control: dict) -> bool:
        return self._fits_controls_ipc([control] * len(self.children))

    def begin(self, control: dict, *, timeout: float) -> dict:
        if self._publication_failed:
            raise RuntimeError("Receiver publication has an ambiguous outcome")
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
        self._workspace_budget = control.get("receiver_gpu_budget_bytes", 0) - self.staging_budget
        self._announce_expert_routes = bool(control["seed"] and control.get("expert_direct"))
        self._verify = bool(control.get("verify"))
        extra = EXPERT_ROUTE_CAPACITY if self._announce_expert_routes else VERIFY_ACK_BYTES if self._verify else 0
        if extra and extra + 1 + int(self.pp_size > 1) > self._workspace_budget:
            raise ValueError("Expert or verification acknowledgement exceeds receiver workspace")
        controls = [dict(op="begin", **control, uuid=i["uuid"]) for i in self.identities]
        self._publication_failed = True
        self._controls(controls, deadline=deadline)
        self._publication_failed = False
        return dict(success=True)

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
        if not inline and self._inline_frame_directory:
            compressed = zlib.compress(data, level=1)
            count = len(metadata["tensors"])
            status_bytes = count + len(compressed) + int(self.pp_size > 1)
            allocation = ((status_bytes + 511) // 512) * 512
            candidate = dict(control, metadata_ipc=[count, len(compressed), len(data)])
            # Charge the same whole format-2 directory as the scheduler. Use
            # HTTP fragments when the IPC manifest would crowd out decoding.
            if (
                len(compressed) <= len(data)
                and allocation + workspace_bytes(metadata) <= self._workspace_budget
                and self._fits_ipc(candidate)
            ):
                control, tail = candidate, compressed
        if not inline and not tail:
            if not self._fits_ipc(control):
                raise ValueError("One GPU bucket IPC control exceeds its HTTP budget")
            self._send_frame_directory(data, identity, deadline=deadline)
        return [control] * len(self.children), inline, tail

    def _direct_controls(self, lanes: list[dict], deadline: float) -> tuple[list[dict], list[bytes]]:
        encoded = [json_bytes(lane) for lane in lanes]
        identities = [hashlib.sha256(data).hexdigest() for data in encoded]
        if any(len(data) > MAX_DOCUMENT_BYTES for data in encoded):
            raise ValueError("Directed lane exceeds its bounded worker directory capacity")
        base = [dict(op="consume", version=self.version, frame=self.batch, metadata_id=key) for key in identities]
        controls, tails = [], []
        for lane, data, control in zip(lanes, encoded, base, strict=True):
            count = len(lane["tensors"])
            compressed = zlib.compress(data, level=1)
            allocation = ((count + len(compressed) + int(self.pp_size > 1) + 511) // 512) * 512
            if count and len(compressed) <= len(data) and allocation + workspace_bytes(lane) <= self._workspace_budget:
                controls.append(dict(control, metadata_ipc=[count, len(compressed), len(data)]))
                tails.append(compressed)
            else:
                controls.append(
                    dict(control, metadata_zlib=dict(bytes=len(data), data=base64.b64encode(compressed).decode()))
                )
                tails.append(b"")
        if self._fits_controls_ipc(controls):
            return controls, tails
        if not self._fits_controls_ipc(base):
            raise ValueError("Directed IPC handle controls exceed the whole-engine HTTP budget")
        self._send_frame_directories(encoded, identities, deadline=deadline)
        return base, [b""] * len(lanes)

    def consume(self, round_id: int, metadata: dict | list[dict], *, timeout: float) -> dict:
        if self._publication_failed:
            raise RuntimeError("Receiver publication has an ambiguous outcome")
        if round_id != self.batch:
            raise ValueError("Unexpected logical round")
        directed = isinstance(metadata, list)
        lanes = metadata if directed else [metadata] * len(self.children)
        if (
            not lanes
            or len(lanes) != len(self.children)
            or any(
                not isinstance(lane, dict)
                or lane.get("format") != 2
                or lane.get("expert_direct", False) is not directed
                or (directed and lane.get("verify"))
                for lane in lanes
            )
        ):
            raise ValueError("Proxy consume requires the verl wire directory for each local lane")
        deadline = time.monotonic() + timeout
        self._publication_failed = True
        if directed:
            controls, tails = self._direct_controls(lanes, deadline)
            inline = False
        else:
            controls, inline, tail = self._shared_controls(metadata, deadline)
            tails = [tail] * len(lanes)
        entries = [lane["tensors"] for lane in lanes]
        replies = self._commands(
            [
                dict(op="export", round=round_id, control=c, count=len(items), directory_tail=tail)
                for c, items, tail in zip(controls, entries, tails, strict=True)
            ],
            deadline=deadline,
        )
        before = time.monotonic()
        self._request(
            "update_weights_from_tensor", self._payload([item["serialized"] for item in replies]), deadline=deadline
        )
        self.metrics["consume_http_s"] = self.metrics.get("consume_http_s", 0.0) + (time.monotonic() - before)
        ack_started = time.monotonic()
        statuses = self._commands(
            [dict(op="status", round=round_id, deadline=deadline)] * len(self.children), deadline=deadline
        )
        if len(statuses) != len(self.children) or any(
            len(s["full_required"]) != len(items)
            or any(type(v) is not int or v not in (0, 1) for v in s["full_required"])
            for items, s in zip(entries, statuses, strict=True)
        ):
            raise RuntimeError("Missing or invalid delta install acknowledgement")
        self._commands([dict(op="release", round=round_id, consumed=True)] * len(self.children), deadline=deadline)
        self.metrics["consume_ack_s"] = self.metrics.get("consume_ack_s", 0.0) + (time.monotonic() - ack_started)
        self.metrics["consume_calls"] += 1
        if directed:
            self.metrics["expert_direct_frames"] = self.metrics.get("expert_direct_frames", 0) + 1
            self.metrics["expert_direct_lane_bytes"] = self.metrics.get("expert_direct_lane_bytes", 0) + sum(
                lane["bytes"] for lane in lanes
            )
        if self._inline_frame_directory:
            metric = (
                "ipc_directory_frames"
                if any(tails)
                else ("inline_directory_frames" if inline else "inline_directory_fallback_frames")
            )
            if any(tails):
                self.metrics["ipc_directory_bytes"] = self.metrics.get("ipc_directory_bytes", 0) + (
                    sum(map(len, tails)) if directed else len(tails[0])
                )
            self.metrics[metric] = self.metrics.get(metric, 0) + 1
        self.batch += 1
        required = sorted(
            {
                entry["name"]
                for items, status in zip(entries, statuses, strict=True)
                for entry, flag in zip(items, status["full_required"], strict=True)
                if flag
            }
        )
        self._publication_failed = False
        return dict(success=True, full_required=required)

    def finalize(self, update: str, *, timeout: float, final_schema: dict | None = None) -> dict:
        if self._publication_failed:
            raise RuntimeError("Receiver publication has an ambiguous outcome")
        if update != self.version:
            raise ValueError("Final delta version differs from the active update")
        started = time.monotonic()
        deadline = started + timeout
        control = dict(op="finalize", version=update, frames=self.batch)
        count = 1
        if final_schema is not None:
            if (
                len(self.children) != self.tp_size * self.pp_size
                or not isinstance(final_schema, dict)
                or type(final_schema.get("count")) is not int
                or final_schema["count"] <= 0
                or final_schema["count"] + 1 + int(self.pp_size > 1) > self._workspace_budget
                or not isinstance(final_schema.get("digest"), str)
                or len(final_schema["digest"]) != 64
            ):
                raise ValueError("Invalid final capability mask schema or workspace")
            count += final_schema["count"]
            control.update(full_required_count=final_schema["count"], full_required_schema=final_schema["digest"])
        tail = b""
        if self._announce_expert_routes:
            control["expert_route_capacity"] = EXPERT_ROUTE_CAPACITY
            tail = bytes(EXPERT_ROUTE_CAPACITY)
        if self._verify:
            control["verify_ack_bytes"] = VERIFY_ACK_BYTES
            tail = bytes(VERIFY_ACK_BYTES)
        if tail and count + len(tail) + int(self.pp_size > 1) > self._workspace_budget:
            raise ValueError("Final acknowledgement exceeds admitted workspace")
        self._publication_failed = True
        empty = dict(bytes=0, entries=[])
        self.receive(self.batch, [empty] * len(self.children), self.staging_budget, timeout=remaining(deadline))
        self.ready(self.batch, timeout=remaining(deadline))
        replies = self._commands(
            [
                dict(
                    op="export",
                    round=self.batch,
                    control=control,
                    count=count,
                    directory_tail=tail,
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
        ack_started = time.monotonic()
        statuses = self._commands(
            [dict(op="status", round=self.batch, deadline=deadline)] * len(self.children), deadline=deadline
        )
        if (
            not statuses
            or len(statuses) != len(self.children)
            or any(
                len(r["full_required"]) != count
                or any(type(v) is not int or v not in (0, 1) for v in r["full_required"])
                for r in statuses
            )
        ):
            raise RuntimeError("Incomplete delta final acknowledgement")
        routes, verified = [], []
        for index, status in enumerate(statuses if tail else ()):
            identity = self.identities[index]
            owner = {key: identity[key] for key in ("rank", "tp_rank", "pp_rank", "uuid")}
            if self._announce_expert_routes:
                descriptor = status.get("expert_routes")
                if (
                    not isinstance(descriptor, dict)
                    or not isinstance(descriptor.get("routes"), list)
                    or not isinstance(descriptor.get("fallbacks"), dict)
                ):
                    raise RuntimeError("Incomplete scheduler expert routes")
                routes.append(dict(owner, **descriptor))
            if self._verify:
                size = status.get("verified_bytes")
                if type(size) is not int or size <= 0:
                    raise RuntimeError("Incomplete scheduler native FULL verification")
                verified.append(dict(owner, bytes=size))
        if self.pp_size > 1:
            # FIFO at each scheduler makes this control run after its original
            # finalize request (including SGLang's cache flush and TP barrier).
            confirm = dict(op="confirm_flush", version=update)
            replies = self._commands(
                [dict(op="export", round=self.batch, control=confirm, count=0)] * len(self.children),
                deadline=deadline,
            )
            self._request(
                "update_weights_from_tensor", self._payload([r["serialized"] for r in replies]), deadline=deadline
            )
            flushed = self._commands(
                [dict(op="status", round=self.batch, deadline=deadline)] * len(self.children), deadline=deadline
            )
            if len(flushed) != len(self.children) or any(r["full_required"] != [] for r in flushed):
                raise RuntimeError("Incomplete pipeline flush acknowledgement")
        self._commands([dict(op="release", round=self.batch, consumed=True)] * len(self.children), deadline=deadline)
        self.metrics["finalize_ack_s"] = time.monotonic() - ack_started
        self._request("update_weight_version", dict(new_version=update, abort_all_requests=False), deadline=deadline)
        self.metrics["finalize_s"] = time.monotonic() - started
        result = dict(
            success=True,
            force_full_seed=any(reply["full_required"][0] for reply in statuses),
            metrics=dict(self.metrics),
            proxies=self.identities,
        )
        if routes:
            result["expert_routes"] = routes
        if verified:
            result["verified_bytes"] = verified
        if final_schema is not None:
            columns = zip(*(r["full_required"] for r in statuses), strict=True)
            next(columns)  # The leading force-full byte has its own field in the reply.
            result["full_required_mask"] = [int(any(bits)) for bits in columns]
        self._publication_failed = False
        return result

    def close(self, *, timeout: float = 10, after_worker_exit: bool = False, abort_group: bool = False) -> None:
        if not self.children:
            return
        deadline = time.monotonic() + timeout
        # A failed engine shutdown can leave its Ray actor alive. Only an
        # already-started local retirement may be resumed by membership close.
        if after_worker_exit or (self._stopping and abort_group):
            if not self._workers or not _workers_exited(self._workers):
                raise RuntimeError("Cannot retire GPU proxy before its original model workers exit")
            # The source and other engines can still be using the old group.
            # Once this engine's IPC readers are gone, process exit is the
            # local release fence; a unilateral NCCL abort need not complete.
            self._stopping = True
            errors = []
            for process, _ in self.children:
                try:
                    if process.poll() is None:
                        process.kill()
                except OSError as exc:
                    errors.append(exc)
            for process, _ in self.children:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
            for _, connection in self.children:
                connection.close()
            self.children.clear()
            self._closing.clear()
            self._stopping = False
            return
        if self._stopping:
            raise RuntimeError("Local GPU proxy retirement has not completed")
        errors = []
        # Post to all peers before waiting. A child acknowledges before its
        # collective destructor runs, so a join timeout never permits replay.
        for _, connection in self.children:
            if connection not in self._closing:
                self._closing[connection] = (abort_group, False)
                try:
                    connection.send(dict(op="close", abort_group=abort_group))
                except Exception as exc:
                    errors.append(exc)
        for process, connection in self.children:
            try:
                sent_abort, acknowledged = self._closing[connection]
                if not acknowledged:
                    if not connection.poll(max(0, deadline - time.monotonic())):
                        raise TimeoutError("GPU proxy did not acknowledge close")
                    reply = connection.recv()
                    if reply.get("op") != "close":
                        raise RuntimeError(f"Unexpected reply while retiring GPU proxy: {reply}")
                    if reply.get("success") is not True:
                        del self._closing[connection]
                        raise RuntimeError(f"GPU proxy refused close: {reply}")
                    if reply.get("abort_group") is not sent_abort:
                        raise RuntimeError("GPU proxy acknowledged a different retirement mode")
                    self._closing[connection] = (sent_abort, True)
                    connection.close()
                code = process.wait(timeout=max(0, deadline - time.monotonic()))
                if code:
                    raise RuntimeError(f"GPU proxy exited with status {code}")
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        self.children.clear()
        self._closing.clear()


if __name__ == "__main__":
    _child(Connection(int(sys.argv[1])))
