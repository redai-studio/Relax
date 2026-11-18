# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Data delivery to engine-owned GPU receivers, inside the original lifecycle.

The caller owns topology, locks, pause/resume, backend snapshots and commit.
This object owns only transport resources and waits for actual receiver fences.
"""

from __future__ import annotations

import socket
import uuid
from time import monotonic
from typing import Any

from .config import DeltaSyncConfig
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

    def connect(self, engines: list, counts: list[int], *, reconnect: bool = False) -> bool:
        """Return whether receiver identities changed and a FULL seed is
        needed."""
        import ray

        if not engines or len(engines) != len(counts) or any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("Delta transport requires the original owner's nonempty GPU recipient layout")
        signature = tuple((engine._actor_id.hex(), count) for engine, count in zip(engines, counts, strict=True))
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

    def begin(self, *, version: str, base: str | None, seed: bool) -> None:
        import ray

        self.frame, self.version = 0, version
        self.metrics = dict(payload_bytes=0, logical_delivery_bytes=0, frames=0, transfer_apply_s=0.0)
        control = dict(
            version=version,
            base=base,
            seed=seed,
            receiver_gpu_budget_bytes=self.config.receiver_gpu_budget_bytes,
            staging_budget_bytes=self.config.staging_budget_bytes,
        )
        ray.get(
            [
                engine.delta_weight_sync.remote("begin", control=control, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )

    def send(self, metadata: dict, raw: Any) -> set[str]:
        """Deliver one complete loader group and return receiver FULL
        constraints."""
        import ray

        if raw.numel() > self.config.staging_budget_bytes:
            raise ValueError("Complete delta loader group exceeds proxy staging capacity")
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
        ray.get(
            [
                engine.delta_weight_sync.remote("ready", round_id=self.frame, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        replies = ray.get(
            [
                engine.delta_weight_sync.remote(
                    "consume", round_id=self.frame, metadata=metadata, timeout=self.config.sync_timeout_s
                )
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        self.metrics["payload_bytes"] += raw.numel()
        self.metrics["logical_delivery_bytes"] += raw.numel() * sum(map(len, self.ranks))
        self.metrics["frames"] += 1
        self.metrics["transfer_apply_s"] += monotonic() - started
        self.frame += 1
        return {name for reply in replies for name in reply["full_required"]}

    def finalize(self) -> dict:
        import ray

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
