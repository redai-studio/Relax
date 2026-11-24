# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Data delivery to engine-owned GPU receivers, inside the original lifecycle.

The caller owns topology, locks, pause/resume, backend snapshots and commit.
This object owns only transport resources and waits for actual receiver fences.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from time import monotonic
from typing import Any

from .codec import WIRE_ALIGNMENT, schema_digest
from .config import RETAINED_DIRECTORY_BYTES, DeltaSyncConfig
from .gpu_transport import GpuSyncGroup
from .tcp_transport import TcpFrameSource


class DeltaTransfer:
    def __init__(self, config: DeltaSyncConfig, device: Any) -> None:
        self.config = config
        self.device = device
        self.engines: tuple = ()
        self.signature: tuple = ()
        self.group: GpuSyncGroup | None = None
        self.tcp: TcpFrameSource | None = None
        self.connection: dict | None = None
        self.ranks: list[list[int]] = []
        self.frame = 0
        self.version: str | None = None
        self.metrics: dict = {}
        self._sender: ThreadPoolExecutor | None = None
        self._pending_frame: Future[set[str]] | None = None
        self._copy_stream: Any = None
        self._host_slabs: list[Any] = [None, None]
        self._host_slot = 0
        self._prepared: tuple[Any, Any, Any] | None = None
        self._active_raw: Any = None
        self._staged: list[tuple[int, dict]] = []
        self._staged_bytes = 0
        self._directory_bytes = sys.getsizeof(self._staged)
        self._control: dict | None = None
        self._final_schema: dict[str, tuple] | None = None
        self._applying = False
        self._consume_inflight = False

    @staticmethod
    def _signature(engines: list, counts: list[int]) -> tuple:
        if not engines or len(engines) != len(counts) or any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("Delta transport requires the original owner's nonempty GPU recipient layout")
        return tuple((engine._actor_id.hex(), count) for engine, count in zip(engines, counts, strict=True))

    def can_reuse(self, engines: list, counts: list[int]) -> bool:
        return self.signature == self._signature(engines, counts)

    def connect(self, engines: list, counts: list[int], *, reconnect: bool = False) -> bool:
        """Return whether receiver identities changed and a FULL seed is
        needed."""
        import ray

        signature = self._signature(engines, counts)
        if signature == self.signature and not reconnect:
            return False
        self._require_drained()
        previous_ids = {engine._actor_id.hex() for engine in self.engines}
        current_ids = {actor_id for actor_id, _ in signature}
        self.close(removed=previous_ids - current_ids)
        self.engines = tuple(engines)
        rank = 1
        self.ranks = []
        for count in counts:
            self.ranks.append(list(range(rank, rank + count)))
            rank += count
        prepared = ray.get(
            [
                engine.delta_weight_sync.remote("prepare", ranks=delivery, timeout=self.config.sync_timeout_s)
                for engine, delivery in zip(self.engines, self.ranks, strict=True)
            ],
            timeout=self.config.sync_timeout_s,
        )
        import torch

        source_uuid = str(torch.cuda.get_device_properties(self.device).uuid).removeprefix("GPU-").lower()
        seen = {source_uuid}
        self.nccl_members = {}
        self.tcp_members = set()
        collective_ranks = []
        routes = []
        for delivery, identities in zip(self.ranks, prepared, strict=True):
            if len(delivery) != len(identities):
                raise ValueError("Proxy identity count disagrees with original recipient topology")
            assigned = []
            for member, identity in zip(delivery, identities, strict=True):
                gpu_uuid = identity["uuid"].removeprefix("GPU-").lower()
                if self.config.transport in ("nccl", "nccl_socket") and gpu_uuid not in seen:
                    collective_rank = len(self.nccl_members) + 1
                    self.nccl_members[str(member)] = collective_rank
                    seen.add(gpu_uuid)
                else:
                    collective_rank = None
                    self.tcp_members.add(str(member))
                assigned.append(collective_rank)
                routes.append(
                    dict(
                        **identity,
                        delivery_id=member,
                        actual_transport="tcp" if collective_rank is None else self.config.transport,
                        collective_rank=collective_rank,
                    )
                )
            collective_ranks.append(assigned)
        if self.config.transport == "nccl_socket" and self.nccl_members:
            # NCCL_NET overrides ncclConfig_t.netName. Reject a conflicting
            # source or proxy before any member blocks in communicator init.
            networks = [os.getenv("NCCL_NET"), *(r["nccl_net"] for r in routes if r["collective_rank"] is not None)]
            if any(net is not None and net.casefold() != "socket" for net in networks):
                self.close()
                raise ValueError("nccl_socket requires NCCL_NET unset or Socket on each collective member")
        host = ray._private.services.get_node_ip_address()
        connection = dict(
            transport=self.config.transport,
            group_name=f"relax-delta-{uuid.uuid4().hex}",
            world_size=1 + len(self.nccl_members),
        )
        if self.nccl_members:
            from relax.utils.http_utils import _wrap_ipv6

            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            connection["init_method"] = f"tcp://{_wrap_ipv6(host)}:{port}"
        if self.tcp_members:
            self.tcp = TcpFrameSource(
                host, timeout_s=self.config.sync_timeout_s, host_budget_bytes=self.config.staging_budget_bytes
            )
        self.connection = connection
        joins = [
            engine.delta_weight_sync.remote(
                "join", connection=connection, ranks=ranks, timeout=self.config.sync_timeout_s
            )
            for engine, ranks in zip(self.engines, collective_ranks, strict=True)
        ]
        if self.nccl_members:
            self.group = GpuSyncGroup(connection, rank=0, device=self.device, timeout_s=self.config.sync_timeout_s)
        joined = ray.get(joins, timeout=self.config.sync_timeout_s)
        self.routes = routes
        self.joined = joined
        self.signature = signature
        return True

    def begin(self, *, version: str, base: str | None, seed: bool, defer_apply: bool = False) -> None:
        self._require_drained()
        if self._staged or self._active_raw is not None or self._prepared is not None:
            raise RuntimeError("Cannot begin with unresolved receiver storage")
        self.frame, self.version = 0, version
        self._applying = False
        self.metrics = dict(
            payload_bytes=0,
            logical_delivery_bytes=0,
            frames=0,
            transfer_apply_s=0.0,
            receive_s=0.0,
            apply_s=0.0,
            prestaged_bytes=0,
            prestaged_frames=0,
            proxy_receive_memory=[],
        )
        self._control = dict(
            version=version,
            base=base,
            seed=seed,
            receiver_gpu_budget_bytes=self.config.receiver_gpu_budget_bytes,
            staging_budget_bytes=self.config.staging_budget_bytes,
        )
        self._final_schema = {} if seed else None
        if not defer_apply:
            self.activate()

    def activate(self) -> None:
        """Called only after the original owner establishes its apply
        window."""
        import ray

        self._require_drained()
        if self._control is None or self._applying:
            raise RuntimeError("A publication may enter its apply window only once")
        ray.get(
            [
                engine.delta_weight_sync.remote("begin", control=self._control, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        self._applying = True

    def _staged_after(self, raw: Any) -> int:
        size = raw.numel()
        return self._staged_bytes + (size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT

    def can_stage(self, metadata: dict, raw: Any) -> bool:
        self._require_drained()
        return self._fits(metadata, self._staged_after(raw), self._retained_directory_size(metadata))

    def _fits(self, metadata: dict, charged: int, retained: int) -> bool:
        return (
            charged <= self.config.staging_budget_bytes
            and self._directory_bytes + retained <= RETAINED_DIRECTORY_BYTES
            and len(json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()) <= RETAINED_DIRECTORY_BYTES
        )

    def _retained_directory_size(self, metadata: dict) -> int:
        """Charge actual Python representations, not their smaller JSON wire.

        Count shared objects once within a frame and conservatively again
        across frames. Four pointer slots per record cover even the first
        append's list overallocation, in addition to the retained tuple/rank.
        This is object allocation accounting, not an RSS/allocator guarantee.
        """
        pending, seen, size = [metadata], set(), 0
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            size += sys.getsizeof(value)
            if isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        return size + sys.getsizeof((self.frame, metadata)) + sys.getsizeof(self.frame) + 4 * struct.calcsize("P")

    def stage(
        self,
        metadata: dict,
        raw: Any,
        *,
        owned_host: bool = False,
        install_worker: ThreadPoolExecutor | None = None,
    ) -> None:
        """Receive immutable bytes and optionally hand receiver completion to
        the install worker."""
        import ray

        if self._consume_inflight:
            raise RuntimeError("Cannot stage while a receiver consume is unresolved")
        charged = self._staged_after(raw)
        retained = self._retained_directory_size(metadata)
        if self._control is None or not self._fits(metadata, charged, retained):
            raise ValueError("Retained delta groups exceed the bounded proxy staging capacity")
        size = raw.numel()
        started = monotonic()
        self._active_raw = raw
        descriptor = None
        if self.tcp is not None:
            # Compatibility transport uses the same wire; only its delivery
            # moves immutable raw bytes through host memory.
            if owned_host:
                # This worker owns the slot through finish() and all ACKs.
                descriptor = self.tcp.publish_slot(memoryview(raw.numpy()).toreadonly(), self.tcp_members)
            else:
                self._host_slabs = [None, None]
                descriptor = self.tcp.publish(raw.cpu().numpy().tobytes(), self.tcp_members)
        batch = dict(bytes=size, entries=[dict(frame=0, offset=0, bytes=size, owner=0)])
        posted = [
            engine.delta_weight_sync.remote(
                "receive",
                round_id=self.frame,
                batches=[batch] * len(ranks),
                budget=self.config.staging_budget_bytes,
                descriptor=descriptor,
                timeout=self.config.sync_timeout_s,
            )
            for engine, ranks in zip(self.engines, self.ranks, strict=True)
        ]
        ray.get(posted, timeout=self.config.sync_timeout_s)
        if self.group is not None:
            # Every NCCL member consumes the same HF-coordinate wire bucket.
            # One broadcast avoids one network copy per TP worker.
            self.group.send(raw, deadline=monotonic() + self.config.sync_timeout_s)
        if self.tcp is not None:
            self.tcp.finish()
        # Source bytes are no longer needed after every transport read fence;
        # SG consumers still hold independently allocated proxy-owned storage.
        self._active_raw = None

        def confirm() -> None:
            memory = self._confirm_receivers()
            elapsed = monotonic() - started
            self.metrics["receive_s"] += elapsed
            self.metrics["transfer_apply_s"] += elapsed
            self.metrics["proxy_receive_memory"] = memory
            self._staged.append((self.frame, metadata))
            self._staged_bytes = charged
            self._directory_bytes += retained
            self.metrics["payload_bytes"] += size
            self.metrics["logical_delivery_bytes"] += size * sum(map(len, self.ranks))
            self.metrics["frames"] += 1
            if not self._applying:
                self.metrics["prestaged_bytes"] += size
                self.metrics["prestaged_frames"] += 1
            self.frame += 1

        if install_worker is None:
            confirm()
        else:
            # Only receiver RPCs remain. The producer can now convert the next
            # bucket, but must drain this single slot before another admission.
            def install() -> set[str]:
                confirm()
                return self.apply()

            self._pending_frame = install_worker.submit(install)

    def _confirm_receivers(self) -> list:
        import ray

        return ray.get(
            [
                engine.delta_weight_sync.remote("ready", round_id=self.frame, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )

    def apply(self) -> set[str]:
        """Consume staged groups in directory order inside the original
        pause."""
        import ray

        if not self._applying:
            raise RuntimeError("Staged payload cannot mutate a generating model")
        if self._consume_inflight:
            raise RuntimeError("Cannot replay an unresolved consume")
        started = monotonic()
        required = set()
        self._consume_inflight = True
        for round_id, metadata in self._staged:
            replies = ray.get(
                [
                    engine.delta_weight_sync.remote(
                        "consume", round_id=round_id, metadata=metadata, timeout=self.config.sync_timeout_s
                    )
                    for engine in self.engines
                ],
                timeout=self.config.sync_timeout_s,
            )
            if len(replies) != len(self.engines) or any(
                not isinstance(reply, dict)
                or reply.get("success") is not True
                or not isinstance(reply.get("full_required"), list)
                for reply in replies
            ):
                raise RuntimeError("Incomplete receiver consume acknowledgement")
            if self._final_schema is not None:
                for entry in metadata["tensors"]:
                    name = entry["name"]
                    if name in self._final_schema:
                        raise ValueError("Duplicate HF input in the ordinary seed schema")
                    self._final_schema[name] = (tuple(entry["shape"]), entry["dtype"])
            required.update(name for reply in replies for name in reply["full_required"])
        self._consume_inflight = False
        self._staged.clear()
        self._staged_bytes = 0
        self._directory_bytes = sys.getsizeof(self._staged)
        elapsed = monotonic() - started
        self.metrics["apply_s"] += elapsed
        self.metrics["transfer_apply_s"] += elapsed
        return required

    def send(self, metadata: dict, raw: Any) -> set[str]:
        """Stage to distinct proxy-owned slices and consume in publication
        order."""
        self.stage(metadata, raw)
        return self.apply()

    def _require_drained(self) -> None:
        if self._pending_frame is not None:
            raise RuntimeError("Drain the pending delta frame before accessing receiver state")

    def drain_frame(self) -> set[str]:
        """Wait for the previous submission, including its requested consume
        ACKs."""
        if self._pending_frame is None:
            return set()
        # Leave a failed future in place. It may represent a partially installed
        # frame and must never permit replay, a new publication or final ACK.
        try:
            required = self._pending_frame.result()
        except BaseException:
            if self._sender is not None:
                self._sender.shutdown(wait=True)
                self._sender = None
            raise
        self._pending_frame = None
        return required

    def prepare_frame(self, raw: Any) -> None:
        """Seal the next TCP frame while the worker delivers the previous
        slot."""
        import torch

        if self.tcp is None or self.group is not None:
            return
        if self._prepared is not None:
            raise RuntimeError("A sealed TCP frame is already waiting for submission")
        if (
            not isinstance(raw, torch.Tensor)
            or raw.dtype != torch.uint8
            or raw.ndim != 1
            or not raw.is_contiguous()
            or raw.numel() > self.config.staging_budget_bytes
        ):
            raise ValueError("TCP submission requires an admitted contiguous byte frame")

        done = None
        if raw.is_cuda and raw.numel():
            slot = self._host_slot
            if self._host_slabs[slot] is None:
                self._host_slabs[slot] = torch.empty(
                    self.config.staging_budget_bytes, dtype=torch.uint8, pin_memory=True
                )
            host = self._host_slabs[slot][: raw.numel()]
            if self._copy_stream is None:
                self._copy_stream = torch.cuda.Stream(device=raw.device)
            self._copy_stream.wait_stream(torch.cuda.current_stream(raw.device))
            with torch.cuda.stream(self._copy_stream):
                host.copy_(raw, non_blocking=True)
                raw.record_stream(self._copy_stream)
                done = self._copy_stream.record_event()
        else:
            host = raw.to(device="cpu", copy=True)

        self._prepared = (raw, host, done)
        self.metrics["sender_pinned_bytes"] = sum(slab.numel() for slab in self._host_slabs if slab is not None)

    def discard_prepared(self) -> None:
        """Fence an unsubmitted outbound copy when its publication aborts."""
        if self._prepared is not None:
            _, _, done = self._prepared
            if done is not None:
                done.synchronize()
            self._prepared = None

    @property
    def pipelined_socket_install(self) -> bool:
        return (
            self.config.transport == "nccl_socket" and self.group is not None and self.tcp is None and self._applying
        )

    def submit_frame(self, metadata: dict, raw: Any, *, applying: bool) -> set[str]:
        """Transfer ownership of a sealed frame until drain_frame().

        Source collectives stay on the caller thread. After a socket broadcast
        has fenced, ordinary receiver readiness and installation can run
        concurrently with conversion of the next bucket.
        """
        self._require_drained()
        if applying != self._applying:
            raise RuntimeError("Delta submission disagrees with the owner's apply window")
        if self.pipelined_socket_install:
            if self._sender is None:
                self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="delta-install")
            self.stage(metadata, raw, install_worker=self._sender)
            return set()
        if self.tcp is None or self.group is not None:
            if applying:
                return self.send(metadata, raw)
            self.stage(metadata, raw)
            return set()
        if self._prepared is None:
            self.prepare_frame(raw)
        source, host, done = self._prepared
        if source is not raw:
            raise RuntimeError("TCP submission differs from its sealed frame")
        self._prepared = None
        self._host_slot = 1 - self._host_slot

        def deliver() -> set[str]:
            if done is not None:
                done.synchronize()
            self.stage(metadata, host, owned_host=True)
            return self.apply() if applying else set()

        if self._sender is None:
            self._sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="delta-tcp-send")
        self._pending_frame = self._sender.submit(deliver)
        return set()

    def finalize(self) -> dict:
        import ray

        self._require_drained()
        if not self._applying or self._staged or self._prepared is not None:
            raise RuntimeError("Cannot finalize before staged payload consumption")
        options = {}
        if self._final_schema is not None:
            options["final_schema"] = dict(count=len(self._final_schema), digest=schema_digest(self._final_schema))
        final_started = monotonic()
        replies = ray.get(
            [
                engine.delta_weight_sync.remote(
                    "finalize", update=self.version, timeout=self.config.sync_timeout_s, **options
                )
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        if (
            len(replies) != len(self.engines)
            or not replies
            or any(
                not isinstance(reply, dict)
                or reply.get("success") is not True
                or type(reply.get("force_full_seed")) is not bool
                for reply in replies
            )
        ):
            raise RuntimeError("Incomplete receiver final acknowledgement")
        full_names = set()
        if self._final_schema is not None:
            for reply in replies:
                bits = reply.pop("full_required_mask", None)
                if (
                    not isinstance(bits, list)
                    or len(bits) != len(self._final_schema)
                    or any(type(v) is not int or v not in (0, 1) for v in bits)
                ):
                    raise RuntimeError("Incomplete receiver final capability mask")
                full_names.update(name for name, bit in zip(self._final_schema, bits, strict=True) if bit)
            self._final_schema = None
        return dict(
            self.metrics,
            finalize_s=monotonic() - final_started,
            final_full_names=sorted(full_names),
            receivers=replies,
            routes=self.routes,
            nccl_group_name=self.connection["group_name"] if self.group is not None else None,
            nccl_world_size=self.connection["world_size"] if self.group is not None else 0,
        )

    def close(self, *, removed: set[str] | None = None) -> None:
        import ray

        # A caller that already observed a failed consume may still retire
        # the connection. A live worker or unresolved read keeps its owner.
        if self._sender is not None:
            self._require_drained()
        if self._active_raw is not None:
            raise RuntimeError("Cannot retire unresolved source transport storage")
        # Even a partial retirement invalidates reuse of the old communicator.
        self.signature = ()
        if self._sender is not None:
            self._sender.shutdown(wait=True)
            self._sender = None
        self.discard_prepared()
        self._host_slabs = [None, None]
        self._copy_stream = None
        closes = [
            engine.delta_weight_sync.remote("close", timeout=self.config.sync_timeout_s, abort_group=bool(removed))
            for engine in self.engines
        ]
        if self.group is not None:
            if removed:
                self.group.abort()
            else:
                self.group.close()
            self.group = None
        if self.tcp is not None:
            self.tcp.close()
            self.tcp = None
        deadline = monotonic() + self.config.sync_timeout_s
        for engine, ref in zip(self.engines, closes, strict=True):
            try:
                ray.get(ref, timeout=max(0, deadline - monotonic()))
            except ray.exceptions.ActorDiedError as exc:
                # Only a permanently dead, replaced actor can no longer own
                # a close RPC. A live old actor must still prove retirement.
                if (
                    isinstance(exc, ray.exceptions.RayTaskError)
                    or not removed
                    or engine._actor_id.hex() not in removed
                ):
                    raise
        self._pending_frame = None
        self.engines, self.signature, self.connection = (), (), None
