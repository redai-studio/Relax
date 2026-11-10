# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# NIXL calls adapted from radixark/miles at 286634cc; see MILES_LICENSE.
import base64
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from .common import create_agent, layout_of


# An unconfirmed native write may outlive a failed Future or its caller. Keep
# its agent, registrations and source buffers alive until the process exits.
_FAILED_WRITERS: list[Any] = []
_FAILED_TARGETS: list[Any] = []
_FAILED_PACKS: list[tuple] = []
_FAILED_REGISTRATIONS: list[Any] = []


class RegisteredMemory:
    """Own native descriptors and storage before entering registration calls.

    The caller chooses one allocation epoch or a persistent session. It must
    prove original-owner DONE and the receiver fence before releasing memory.
    Partial registration/deregistration failures retain all allocations.
    """

    def __init__(self, agent, parameters: dict) -> None:
        self.agent = agent
        self.parameters = {name: value.detach() for name, value in parameters.items()}
        self.layouts = {name: layout_of(value) for name, value in self.parameters.items()}
        self.registrations = []
        self.failed = self.registered = self.released = False

    def _fail(self) -> None:
        if not self.failed:
            self.failed = True
            _FAILED_REGISTRATIONS.append(self)

    def register(self, *, batched: bool) -> None:
        if self.failed or self.registered or self.released or self.registrations:
            raise RuntimeError("Registration is already active or failed")
        try:
            values = list(self.parameters.values())
            for group in [values] if batched else ([value] for value in values):
                registration = self.agent.get_reg_descs(group)
                self.registrations.append(registration)
                self.agent.register_memory(registration)
            self.registered = True
        except BaseException:
            self._fail()
            raise

    def validate(self, parameters: dict, *, layouts: dict | None = None) -> None:
        actual = layouts if layouts is not None else {name: layout_of(value) for name, value in parameters.items()}
        if self.failed or not self.registered or self.released or actual != self.layouts:
            raise RuntimeError("Registered parameter storage changed; recreate the transfer session")

    def release(self) -> None:
        """Retire only after the caller's transfer completion and CUDA
        fence."""
        if self.failed or not self.registered or self.released:
            raise RuntimeError("Cannot release an inactive or uncertain registration")
        try:
            for registration in self.registrations:
                self.agent.deregister_memory(registration)
        except BaseException:
            self._fail()
            raise
        self.registrations.clear()
        self.parameters.clear()
        self.released = True


class BufferedWriter:
    """Two registered buffers; a failed write never releases source memory."""

    def __init__(self, nbytes: int, timeout: float = 120.0, *, device: torch.device | None = None) -> None:
        self.agent = create_agent()
        self.timeout = timeout
        device = device or torch.device("cpu")
        self.buffers = [
            torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu") for _ in range(2)
        ]
        self.registrations = [self.agent.register_memory(buffer) for buffer in self.buffers]
        self.pending: list[Future | None] = [None, None]
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2p-nixl")
        self.next_index = 0
        self.failed = False
        self.peer = None
        self.generation = None
        self.targets: dict[str, tuple[str, str]] = {}
        self.profile: dict[str, float] = {}
        self._active_range_batch: dict | None = None
        self._retained_failure = False

    def _retain_failure(self) -> None:
        self.failed = True
        if not self._retained_failure:
            self._retained_failure = True
            _FAILED_WRITERS.append(self)

    def _record(self, name: str, value: float) -> None:
        self.profile[name] = self.profile.get(name, 0.0) + value

    def connect_target(self, key: str, manifest: dict) -> None:
        if key in self.targets:
            if self.targets[key][1] != manifest["generation"]:
                raise RuntimeError("remote shard target generation changed")
            return
        peer = self.peer if self.generation == manifest["generation"] else None
        if peer is None:
            peer = self.agent.add_remote_agent(base64.b64decode(manifest["agent_metadata"]))
        self.targets[key] = (peer, manifest["generation"])

    def connect(self, manifest: dict) -> None:
        if self.peer is not None:
            if self.generation != manifest["generation"]:
                raise RuntimeError("P2P target generation changed")
            return
        self.peer = self.agent.add_remote_agent(base64.b64decode(manifest["agent_metadata"]))
        self.generation = manifest["generation"]

    def disconnect_targets(self) -> None:
        """Drop remote registrations only after all submitted writes are DONE.

        Colocated targets must reconnect to fresh metadata after each native
        allocation epoch. The existing strict generation guard stays intact.
        """
        self.finish()
        if self.failed or self._active_range_batch is not None:
            raise RuntimeError("Cannot disconnect an unconfirmed NIXL write")
        peers = {peer for peer, _ in self.targets.values()}
        if self.peer is not None:
            peers.add(self.peer)
        try:
            for peer in sorted(peers):
                self.agent.remove_remote_agent(peer)
        except Exception:
            self._retain_failure()
            raise
        self.targets.clear()
        self.peer = self.generation = None

    def acquire(self) -> tuple[int, torch.Tensor]:
        index = self.next_index
        self.next_index = (index + 1) % 2
        self._wait(index)
        if self.failed:
            raise RuntimeError("Cannot reuse a P2P buffer after an unconfirmed native write")
        return index, self.buffers[index]

    def submit(self, index: int, tensors: dict[str, torch.Tensor], manifest: dict) -> None:
        if self.pending[index] is not None or self.failed:
            raise RuntimeError("P2P buffer still has a reader or an unconfirmed write")
        for name, tensor in tensors.items():
            if tensor.numel() != manifest["params"][name]["nbytes"]:
                raise ValueError(f"Target byte count mismatch: {name}")
        ready = None
        if self.buffers[index].is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.buffers[index].device))
        self.pending[index] = self.pool.submit(self._write, tensors, manifest, ready)

    def _write(self, tensors: dict[str, torch.Tensor], manifest: dict, ready: torch.cuda.Event | None = None) -> None:
        if self.failed:
            raise RuntimeError("P2P writer failed before its queued write started")
        if ready is not None:
            # Native NIXL does not inherit PyTorch stream dependencies.
            torch.cuda.set_device(next(iter(tensors.values())).device)
            started = time.monotonic()
            ready.synchronize()
            self._record("fallback_producer_wait_s", time.monotonic() - started)
            self._record("fallback_producer_wait_calls", 1)
        self._transfer(
            self.agent.get_xfer_descs(list(tensors.values())),
            self.agent.get_xfer_descs(
                [
                    (manifest["params"][name]["address"], tensor.numel(), manifest["gpu_id"])
                    for name, tensor in tensors.items()
                ],
                "VRAM",
            ),
            self.peer,
            sources=tensors,
        )

    def submit_ranges(self, index: int, writes: list[tuple[dict, torch.Tensor]], manifests: dict) -> None:
        """All destinations share one completion fence before this buffer is
        reused."""
        if self.pending[index] is not None or self.failed:
            raise RuntimeError("remote shard buffer unavailable")
        buffer = self.buffers[index]
        if not buffer.is_cuda:
            raise NotImplementedError("Remote shard transfer requires GPU buffers")
        for write, tensor in writes:
            count = write["rows"] * write["row_bytes"]
            manifest = manifests[write["target"]]
            target = manifest["params"][write["param"]]
            end = write["dst_offset"] + (write["rows"] - 1) * write["dst_stride"] + write["row_bytes"]
            if (
                tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
                or tensor.numel() != count
                or tensor.device != buffer.device
                or tensor.data_ptr() < buffer.data_ptr()
                or tensor.data_ptr() + count > buffer.data_ptr() + buffer.numel()
                or write["dst_offset"] < 0
                or end > target["nbytes"]
                or self.targets[write["target"]][1] != manifest["generation"]
            ):
                raise ValueError("Invalid remote shard source buffer or target range")
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(buffer.device))
        self.pending[index] = self.pool.submit(self._write_ranges, writes, manifests, ready)

    def _write_ranges(
        self,
        writes: list[tuple[dict, torch.Tensor]],
        manifests: dict,
        ready: Any,
    ) -> None:
        try:
            if self.failed:
                raise RuntimeError("remote shard writer failed before its queued write started")
            deadline = time.monotonic() + self.timeout
            device = writes[0][1].device
            torch.cuda.set_device(device)
            started = time.monotonic()
            ready.synchronize()
            self._record("ranges_producer_wait_s", time.monotonic() - started)
            self._record("ranges_producer_wait_calls", 1)
            keys = sorted({w["target"] for w, _ in writes})
            parallel = len(keys) > 1
            requests = []
            for key in keys:
                started = time.monotonic()
                sources, targets = [], []
                manifest = manifests[key]
                for write, tensor in writes:
                    if write["target"] != key:
                        continue
                    width, rows, stride = write["row_bytes"], write["rows"], write["dst_stride"]
                    source = tensor.data_ptr()
                    address = manifest["params"][write["param"]]["address"] + write["dst_offset"]
                    for row in range(rows):
                        sources.append((source + row * width, width, device.index))
                        targets.append((address + row * stride, width, manifest["gpu_id"]))
                self._record("python_descriptors_s", time.monotonic() - started)
                self._record("descriptor_pairs", len(sources))
                started = time.monotonic()
                source_descs = self.agent.get_xfer_descs(sources, "VRAM")
                target_descs = self.agent.get_xfer_descs(targets, "VRAM")
                self._record("native_descriptors_s", time.monotonic() - started)
                if parallel:
                    requests.append((source_descs, target_descs, self.targets[key][0]))
                else:
                    self._transfer(source_descs, target_descs, self.targets[key][0], deadline=deadline, sources=writes)
            if parallel:
                self._transfer_many(requests, writes, deadline)
        except Exception:
            self._retain_failure()
            raise

    def _transfer_many(self, requests: list[tuple[Any, Any, str]], sources: Any, deadline: float) -> None:
        """Post every peer before waiting; only a completed batch releases its
        buffer."""
        if self._active_range_batch is not None:
            raise RuntimeError("A previous native transfer batch is still retained")
        records: list[dict] = []
        self._active_range_batch = dict(requests=requests, sources=sources, records=records)
        failure = None
        first_post = None

        def check_deadline() -> None:
            if self.failed:
                raise RuntimeError("P2P writer failed while a native batch was active")
            if time.monotonic() >= deadline:
                raise TimeoutError("NIXL peer batch exceeded the buffer's total deadline")

        def record_done(record: dict) -> None:
            elapsed = time.monotonic() - record["posted"]
            self._record("submit_to_done_s", elapsed)
            self._record(f"parallel_peer_{record['index']}_s", elapsed)

        try:
            for index, (source, target, peer) in enumerate(requests):
                check_deadline()
                started = time.monotonic()
                handle = self.agent.initialize_xfer("WRITE", source, target, peer)
                records.append(dict(index=index, handle=handle, state=None, released=False))
                self._record("initialize_s", time.monotonic() - started)
                self._record("transfers", 1)
            for record in records:
                check_deadline()
                record["posted"] = time.monotonic()
                if first_post is None:
                    first_post = record["posted"]
                record["state"] = self.agent.transfer(record["handle"])
                if record["state"] == "DONE":
                    record_done(record)
                elif record["state"] != "PROC":
                    raise RuntimeError(f"NIXL peer write did not complete: {record['state']}")
                self.profile["max_parallel_inflight"] = max(
                    self.profile.get("max_parallel_inflight", 0), sum(item["state"] == "PROC" for item in records)
                )
            while any(record["state"] == "PROC" for record in records):
                for record in records:
                    if record["state"] != "PROC":
                        continue
                    check_deadline()
                    record["state"] = self.agent.check_xfer_state(record["handle"])
                    if record["state"] == "DONE":
                        record_done(record)
                    elif record["state"] != "PROC":
                        raise RuntimeError(f"NIXL peer write did not complete: {record['state']}")
                if any(record["state"] == "PROC" for record in records):
                    time.sleep(0.0005)
            check_deadline()
            if first_post is not None and len(requests) > 1:
                self._record("parallel_batch_s", time.monotonic() - first_post)
                self._record("parallel_batches", 1)
        except Exception as exc:
            failure = exc
            self._retain_failure()
        # Releasing an active NIXL handle attempts cancellation; it does not
        # provide a completion fence. Retain all unconfirmed handles instead.
        for record in records:
            if record["state"] != "DONE":
                continue
            try:
                self.agent.release_xfer_handle(record["handle"])
                record["released"] = True
            except Exception as exc:
                self._retain_failure()
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        self._active_range_batch = None

    def _transfer(
        self,
        source_descs: Any,
        target_descs: Any,
        peer: str,
        *,
        deadline: float | None = None,
        sources: Any = None,
    ) -> None:
        # Single-peer and sequential writes need the same completion/lifetime
        # contract as parallel peers. Cancellation is not a completion fence.
        self._transfer_many(
            [(source_descs, target_descs, peer)],
            sources,
            deadline if deadline is not None else time.monotonic() + self.timeout,
        )

    def _wait(self, index: int) -> None:
        future = self.pending[index]
        if future is not None:
            try:
                future.result(timeout=self.timeout + 1)
            except Exception:
                # A host timeout does not stop the worker or its native writes.
                # Leave pending intact and poison both buffers, including work
                # already queued on the single transfer thread.
                self._retain_failure()
                raise
            self.pending[index] = None

    def finish(self) -> None:
        for index in range(2):
            self._wait(index)
