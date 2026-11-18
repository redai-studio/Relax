# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Data delivery to engine-owned GPU receivers, inside the original lifecycle.

The caller owns topology, locks, pause/resume, backend snapshots and commit.
This object owns only transport resources and waits for actual receiver fences.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import uuid
from time import monotonic
from typing import Any

from .codec import WIRE_ALIGNMENT
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
        self._active_raw: Any = None
        self._staged: list[tuple[int, dict]] = []
        self._staged_bytes = 0
        self._directory_bytes = sys.getsizeof(self._staged)
        self._control: dict | None = None
        self._applying = False

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
        self.close()
        self.engines = tuple(engines)
        self.signature = signature
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
                if self.config.transport == "nccl" and gpu_uuid not in seen:
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
                        actual_transport="tcp" if collective_rank is None else "nccl",
                        collective_rank=collective_rank,
                    )
                )
            collective_ranks.append(assigned)
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
        return True

    def begin(self, *, version: str, base: str | None, seed: bool, defer_apply: bool = False) -> None:
        if self._staged or self._active_raw is not None:
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
        if not defer_apply:
            self.activate()

    def activate(self) -> None:
        """Called only after the original owner establishes its apply
        window."""
        import ray

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

    def can_stage(self, metadata: dict, raw: Any) -> bool:
        charged = (raw.numel() + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
        directory = len(json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode())
        return (
            self._staged_bytes + charged <= self.config.staging_budget_bytes
            and directory <= RETAINED_DIRECTORY_BYTES
            and self._directory_bytes + self._retained_directory_size(metadata) <= RETAINED_DIRECTORY_BYTES
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

    def stage(self, metadata: dict, raw: Any) -> None:
        """Receive immutable bytes without invoking the live model loader."""
        import ray

        if self._control is None or not self.can_stage(metadata, raw):
            raise ValueError("Retained delta groups exceed the bounded proxy staging capacity")
        started = monotonic()
        self._active_raw = raw
        descriptor = None
        if self.tcp is not None:
            # Compatibility transport uses the same wire; only its delivery
            # moves immutable raw bytes through host memory.
            payload = raw.cpu().numpy().tobytes()
            descriptor = self.tcp.publish(payload, self.tcp_members)
        batch = dict(bytes=raw.numel(), entries=[dict(frame=0, offset=0, bytes=raw.numel(), owner=0)])
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
        memory = ray.get(
            [
                engine.delta_weight_sync.remote("ready", round_id=self.frame, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        elapsed = monotonic() - started
        self.metrics["receive_s"] += elapsed
        self.metrics["transfer_apply_s"] += elapsed
        self.metrics["proxy_receive_memory"] = memory
        self._staged.append((self.frame, metadata))
        self._staged_bytes += (raw.numel() + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
        self._directory_bytes += self._retained_directory_size(metadata)
        self.metrics["payload_bytes"] += raw.numel()
        self.metrics["logical_delivery_bytes"] += raw.numel() * sum(map(len, self.ranks))
        self.metrics["frames"] += 1
        if not self._applying:
            self.metrics["prestaged_bytes"] += raw.numel()
            self.metrics["prestaged_frames"] += 1
        self.frame += 1

    def apply(self) -> set[str]:
        """Consume staged groups in directory order inside the original
        pause."""
        import ray

        if not self._applying:
            raise RuntimeError("Staged payload cannot mutate a generating model")
        started = monotonic()
        required = set()
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
            required.update(name for reply in replies for name in reply["full_required"])
        self._staged.clear()
        self._staged_bytes = 0
        self._directory_bytes = sys.getsizeof(self._staged)
        elapsed = monotonic() - started
        self.metrics["apply_s"] += elapsed
        self.metrics["transfer_apply_s"] += elapsed
        return required

    def send(self, metadata: dict, raw: Any) -> set[str]:
        """Serial delivery retains the same receive and consumption fences."""
        self.stage(metadata, raw)
        return self.apply()

    def finalize(self) -> dict:
        import ray

        if not self._applying or self._staged:
            raise RuntimeError("Cannot finalize before staged payload consumption")
        replies = ray.get(
            [
                engine.delta_weight_sync.remote("finalize", update=self.version, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        return dict(
            self.metrics,
            receivers=replies,
            routes=self.routes,
            nccl_world_size=self.connection["world_size"] if self.group is not None else 0,
        )

    def close(self) -> None:
        import ray

        if self._active_raw is not None:
            raise RuntimeError("Cannot retire unresolved source transport storage")
        closes = [
            engine.delta_weight_sync.remote("close", timeout=self.config.sync_timeout_s) for engine in self.engines
        ]
        if self.group is not None:
            self.group.close()
            self.group = None
        if self.tcp is not None:
            self.tcp.close()
            self.tcp = None
        if closes:
            ray.get(closes, timeout=self.config.sync_timeout_s)
        self.engines, self.signature, self.connection = (), (), None
