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
from collections.abc import Iterator
from itertools import groupby
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
        self._staged: list[tuple[int, dict | list]] = []
        self._staged_bytes = [0] * 4
        self._directory_bytes = sys.getsizeof(self._staged)
        self._control: dict | None = None
        self._applying = False
        self.ordered_target_consume = False
        self._ordered_inflight = False
        self.target_shard_capable = False
        self.target_shard_version = None

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
        self.target_shard_capable = False
        self.target_shard_version = None
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
        if (
            self.config.target_shard_delta
            and tuple(counts) == (4, 4)
            and len(self.nccl_members) == 8
            and not self.tcp_members
        ):
            connection["target_shard_p2p"] = True
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
        self.ordered_target_consume = False
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
        if self.connection and self.connection.get("target_shard_p2p"):
            self._control["target_shard_request"] = True
            self.metrics.update(
                target_shard_frames=0,
                target_shard_lane_bytes=[0] * 4,
                target_shard_entries=0,
                target_shard_changed_words=0,
            )
        if not defer_apply:
            self.activate()

    def activate(self) -> None:
        """Called only after the original owner establishes its apply
        window."""
        import ray

        if self._control is None or self._applying:
            raise RuntimeError("A publication may enter its apply window only once")
        replies = ray.get(
            [
                engine.delta_weight_sync.remote("begin", control=self._control, timeout=self.config.sync_timeout_s)
                for engine in self.engines
            ],
            timeout=self.config.sync_timeout_s,
        )
        self.ordered_target_consume = bool(
            self._control.get("target_shard_request")
            and not self._control["seed"]
            and self.target_shard_capable
            and self.target_shard_version == self._control["base"]
            and len(replies) == len(self.engines) == 2
            and all(
                isinstance(r, dict) and r.get("success") is True and r.get("ordered_target_consume") is True
                for r in replies
            )
        )
        self._applying = True

    def _ordered_frame(self, metadata: dict | list) -> bool:
        if not self.ordered_target_consume or not isinstance(metadata, list) or len(metadata) != 4:
            return False
        # Other frames may invoke the ordinary loader and change live or derived
        # state. Only a consecutive run of entirely direct frames can coalesce.
        return all(
            lane["tensors"] and lane["target_shard"]["entries"] == list(range(len(lane["tensors"])))
            for lane in metadata
        )

    def _staged_after(self, raw: Any) -> list[int]:
        # Ordinary frames reserve the same number of bytes on every GPU. Lane
        # frames have different sizes independently accumulated on each TP.
        sizes = [r.numel() for r in raw] if isinstance(raw, list) else [raw.numel()] * 4
        return [
            used + (size + WIRE_ALIGNMENT - 1) // WIRE_ALIGNMENT * WIRE_ALIGNMENT
            for used, size in zip(self._staged_bytes, sizes, strict=True)
        ]

    def can_stage(self, metadata: dict | list, raw: Any) -> bool:
        return self._fits(metadata, self._staged_after(raw), self._retained_directory_size(metadata))

    def _fits(self, metadata: dict | list, charged: list[int], retained: int) -> bool:
        return (
            max(charged) <= self.config.staging_budget_bytes
            and self._directory_bytes + retained <= RETAINED_DIRECTORY_BYTES
            and len(json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()) <= RETAINED_DIRECTORY_BYTES
        )

    def _retained_directory_size(self, metadata: dict | list) -> int:
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

    def stage(self, metadata: dict | list, raw: Any) -> None:
        """Receive immutable bytes without invoking the live model loader."""
        import ray

        charged = self._staged_after(raw)
        retained = self._retained_directory_size(metadata)
        if self._control is None or not self._fits(metadata, charged, retained):
            raise ValueError("Retained delta groups exceed the bounded proxy staging capacity")
        sharded = isinstance(metadata, list)
        if sharded:
            import torch

            if (
                not self.target_shard_capable
                or self._control["seed"]
                or self.target_shard_version != self._control["base"]
                or self.tcp is not None
                or self.group is None
                or not isinstance(raw, list)
                or len(raw) != len(metadata)
                or len(raw) != 4
            ):
                raise ValueError("Target lanes require the acknowledged TP4 NCCL recipients")
            for tp, (directory, slab) in enumerate(zip(metadata, raw, strict=True)):
                marker = directory["target_shard"]
                if (
                    marker["tp_rank"] != tp
                    or marker["tp_size"] != 4
                    or marker["format"] != 1
                    or slab.device != self.group.device
                    or slab.dtype != torch.uint8
                    or slab.ndim != 1
                    or not slab.is_contiguous()
                    or directory["bytes"] != slab.numel()
                ):
                    raise ValueError("Mismatched source target-lane directory or slab")
            slabs = raw
            peers = [[self.nccl_members[str(r)] for r in ranks] for ranks in self.ranks]
        else:
            slabs = [raw]
        sizes = [slab.numel() for slab in slabs]
        started = monotonic()
        self._active_raw = raw
        descriptor = None
        if self.tcp is not None:
            # Compatibility transport uses the same wire; only its delivery
            # moves immutable raw bytes through host memory.
            payload = raw.cpu().numpy().tobytes()
            descriptor = self.tcp.publish(payload, self.tcp_members)
        batches = [dict(bytes=size, entries=[dict(frame=0, offset=0, bytes=size, owner=0)]) for size in sizes]
        if sharded:
            for batch in batches:
                batch["target_shard"] = True
        posted = [
            engine.delta_weight_sync.remote(
                "receive",
                round_id=self.frame,
                batches=batches if sharded else batches * len(ranks),
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
            if sharded:
                self.group.send_lanes(slabs, peers, deadline=monotonic() + self.config.sync_timeout_s)
            else:
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
        self._staged_bytes = charged
        self._directory_bytes += retained
        self.metrics["payload_bytes"] += sum(sizes)
        self.metrics["logical_delivery_bytes"] += sum(sizes) * (
            len(self.ranks) if sharded else sum(map(len, self.ranks))
        )
        if sharded:
            self.metrics["target_shard_frames"] += 1
            marked = metadata[0]["target_shard"]["entries"]
            self.metrics["target_shard_entries"] += len(marked)
            self.metrics["target_shard_changed_words"] += sum(
                d["tensors"][i]["count"] for d in metadata for i in marked
            )
            for tp, size in enumerate(sizes):
                self.metrics["target_shard_lane_bytes"][tp] += size
        self.metrics["frames"] += 1
        if not self._applying:
            self.metrics["prestaged_bytes"] += sum(sizes)
            self.metrics["prestaged_frames"] += 1
        self.frame += 1

    def apply(self) -> set[str]:
        """Consume staged groups in directory order inside the original
        pause."""
        import ray

        if not self._applying:
            raise RuntimeError("Staged payload cannot mutate a generating model")
        if self._ordered_inflight:
            raise RuntimeError("Cannot replay an unresolved ordered consume")
        started = monotonic()
        required = set()
        for round_id, metadata, following in self._consume_groups():
            options = {"following": following} if following else {}
            if following:
                # A peer may have installed even when Ray reports a timeout.
                # Keep all owners alive and reject replay if any remaining RPC fails.
                self._ordered_inflight = True
            replies = ray.get(
                [
                    engine.delta_weight_sync.remote(
                        "consume", round_id=round_id, metadata=metadata, timeout=self.config.sync_timeout_s, **options
                    )
                    for engine in self.engines
                ],
                timeout=self.config.sync_timeout_s,
            )
            if following and (
                len(replies) != len(self.engines)
                or any(
                    reply.get("success") is not True
                    or type(reply.get("ordered_frames_ack")) is not int
                    or reply["ordered_frames_ack"] != 1 + len(following)
                    for reply in replies
                )
            ):
                raise RuntimeError("Incomplete ordered target-frame acknowledgement")
            required.update(name for reply in replies for name in reply["full_required"])
        self._ordered_inflight = False
        self._staged.clear()
        self._staged_bytes = [0] * 4
        self._directory_bytes = sys.getsizeof(self._staged)
        elapsed = monotonic() - started
        self.metrics["apply_s"] += elapsed
        self.metrics["transfer_apply_s"] += elapsed
        return required

    def _consume_groups(self) -> Iterator[tuple[int, dict | list, list]]:
        for direct, frames in groupby(self._staged, key=lambda frame: self._ordered_frame(frame[1])):
            if direct:
                first, metadata = next(frames)
                yield first, metadata, [item for _, item in frames]
            else:
                for first, metadata in frames:
                    yield first, metadata, []

    def send(self, metadata: dict | list, raw: Any, *, defer_ordered: bool = False) -> set[str]:
        """Stage to distinct proxy-owned slices and consume in publication
        order."""
        ordered = defer_ordered and self._ordered_frame(metadata)
        required = set()
        if defer_ordered and self._staged and (not ordered or not self.can_stage(metadata, raw)):
            required.update(self.apply())
        self.stage(metadata, raw)
        if not ordered:
            required.update(self.apply())
        return required

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
        if self._control and self._control.get("target_shard_request"):
            self.target_shard_capable = len(replies) == len(self.engines) == 2 and all(
                reply.get("success") is True
                and reply.get("force_full_seed") is False
                and reply.get("target_shard_capable") is True
                for reply in replies
            )
            self.target_shard_version = self.version if self.target_shard_capable else None
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
