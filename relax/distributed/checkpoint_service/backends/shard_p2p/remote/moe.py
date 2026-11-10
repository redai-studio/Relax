# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qualified MoE expert owners to native inference shards over NIXL."""

import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist

from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs

from .common import NATIVE_LEASE_PROTOCOL, REMOTE_SHARD_LOADER, named_sources, weight_update_modes
from .moe_qualification import _validate_source_configuration
from .synchronizer import P2PSynchronizer


class MoeP2P(P2PSynchronizer):
    """Deliver current expert shards directly, with native export for other
    weights."""

    def __init__(self, backend: Any, *, reference_catalog: Any) -> None:
        self.reference_catalog = reference_catalog
        self.backend = backend
        self.group = get_gloo_group()
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        args = backend.args
        self._phase(lambda: _validate_source_configuration(backend, self.world_size))
        hosts = [None] * self.world_size
        dist.all_gather_object(
            hosts,
            (
                socket.gethostname(),
                args.tensor_model_parallel_size,
                args.rollout_num_gpus_per_engine,
                args.expert_model_parallel_size,
                args.num_experts,
                args.context_parallel_size,
                args.pipeline_model_parallel_size,
            ),
            group=self.group,
        )
        if len({record[0] for record in hosts}) != 1:
            raise NotImplementedError("This MoE adapter requires all Actor ranks on one node")
        if len({record[1:] for record in hosts}) != 1:
            raise RuntimeError("Actor source or target sizes differ between ranks")
        topology = self._phase(lambda: backend.rollout_topology if self.rank == 0 else None, result_rank=0)
        self.engines = sorted(topology, key=int)
        self.weight_update_sessions = self._phase(lambda: weight_update_modes(topology))
        if len(self.engines) * args.rollout_num_gpus_per_engine != self.world_size:
            raise NotImplementedError("MoE P2P requires one Actor owner per target rank")
        self.endpoints = {
            engine: f"http://{topology[engine]['ip']}:{topology[engine]['port']}" for engine in self.engines
        }

        def validate_session_configuration() -> Path:
            if not Envs.RELAX_WEIGHT_SYNC_SHARED_DIR:
                raise RuntimeError("MoE P2P requires RELAX_WEIGHT_SYNC_SHARED_DIR")
            shared = Path(Envs.RELAX_WEIGHT_SYNC_SHARED_DIR).resolve()
            if not shared.is_dir():
                raise RuntimeError("MoE P2P requires a prevalidated shared metadata directory")
            return shared

        shared = self._phase(validate_session_configuration)
        self.loader = REMOTE_SHARD_LOADER
        self.root = Path(
            self._phase(
                lambda: tempfile.mkdtemp(prefix="moe-p2p-", dir=shared) if self.rank == 0 else None,
                result_rank=0,
            )
        )
        self.key = f"{self.engines[self.rank // args.rollout_num_gpus_per_engine]}-{self.rank % args.rollout_num_gpus_per_engine}"
        self.replica = self.mapping = self.writer = self.specs = self.direct = None
        self.completed_version = self.manifest = None
        self.pool = ThreadPoolExecutor(max_workers=len(self.engines), thread_name_prefix="moe-p2p-control")
        self._snapshot_audit = None
        self._collective_expert_export = self._select_replica_delivery(topology)

        from .plan import fingerprint
        from .replica_overlap import PlanRequests
        from .sender import capture_expert_routing

        self._expert_routing = capture_expert_routing(self)
        self._expert_routing_hash = fingerprint(self._expert_routing)
        # Request ownership/draining is shared by all control operations,
        # independently of warm-plan delivery or replica distribution policy.
        self._plan_requests = PlanRequests()
        if Envs.RELAX_P2P_AUDIT_DIR:
            from .snapshot_audit import NativeSnapshotAudit

            self._snapshot_audit = NativeSnapshotAudit(self, Envs.RELAX_P2P_AUDIT_DIR)

    def _select_replica_delivery(self, topology: dict) -> bool:
        args = self.backend.args
        seeds, remainder = divmod(args.expert_model_parallel_size, args.rollout_num_gpus_per_engine)
        return (
            not remainder
            and seeds >= 2
            and args.num_experts % seeds == 0
            and len(topology) > 1
            and len({info["ip"] for info in topology.values()}) == 1
        )

    def update(self) -> None:
        from .sender import validate_expert_routing

        try:
            self._phase(lambda: validate_expert_routing(self))
            self._phase(self._plan_requests.ready)
            if self._snapshot_audit is not None:
                self._snapshot_audit.prepare_reference()
            super().update()
        except BaseException:
            self._plan_requests.quarantine(self)
            raise

    def _prepare(self, directory: Path, version: int, audit: bool) -> None:
        super()._prepare(directory, version, audit)
        if self._snapshot_audit is not None:
            # Initial P2P prepare has captured its native reference; subsequent
            # versions retain the compact manifest protocol unchanged.
            self._snapshot_audit.control("poison")

    def _validate_manifest(self, manifest: dict, version: int) -> None:
        from .fallback_receipts import RPC_FALLBACK_DELIVERY
        from .receiver import RPC_PLAN_DELIVERY

        engine = self.engines[self.rank // self.backend.args.rollout_num_gpus_per_engine]
        if (
            manifest.get("native_weight_lease_protocol") != NATIVE_LEASE_PROTOCOL
            or manifest.get("weight_update_session") is not self.weight_update_sessions[engine]
        ):
            raise RuntimeError("Receiver did not confirm its registered native weight-update capability")
        fallback = RPC_FALLBACK_DELIVERY
        if manifest.get("fallback_receipt_delivery") != fallback:
            raise RuntimeError("Actor and receiver fallback RPC modes disagree")
        if manifest.get("fallback_receipt_owner") != self.rank:
            raise RuntimeError("Receiver fallback owner differs from its Actor group rank")
        expected = RPC_PLAN_DELIVERY
        if manifest.get("rpc_plan_delivery") != expected:
            raise RuntimeError("Actor and receiver RPC plan modes disagree")
        from .layout import BLOCK_CAPABILITY

        if manifest.get("block_moe_layout") != BLOCK_CAPABILITY:
            raise RuntimeError("Receiver did not validate the expert shard native layout")
        if (
            manifest["version"] != version
            or manifest["tp_rank"] != self.rank % self.backend.args.rollout_num_gpus_per_engine
            or not manifest["hostname"]
            or manifest["pid"] <= 0
            or manifest["server_fields"]["disable_shared_experts_fusion"] is not True
            or manifest["server_fields"]["moe_runner_backend"] != "triton"
        ):
            raise RuntimeError("Invalid remote MoE target identity or runner layout")
        # The initial receipt comes from the addressed engine's loader RPC in
        # this private session directory. Later receipts must retain its exact
        # generation/host/PID/GPU identity, enforced by read_manifest().
        if self.manifest is not None and any(
            manifest[k] != self.manifest[k] for k in ("generation", "hostname", "pid", "gpu_id")
        ):
            raise RuntimeError("Remote MoE target identity changed")

    def _hf_buckets(self, skip_sources: set[str] | None = None) -> Iterator[list[tuple[str, torch.Tensor]]]:
        backend = self.backend
        bucket, nbytes, conversion_error = [], 0, None

        def check_conversion() -> None:
            if conversion_error is not None:
                raise RuntimeError("MoE Bridge conversion failed") from conversion_error

        for experts in (False, True):
            for name, parameter in named_sources(self):
                if (
                    not backend._should_sync_tensor(name, parameter)
                    or (".experts." in name) != experts
                    or (skip_sources and name in skip_sources)
                ):
                    continue
                tensor = backend._megatron.all_gather_param(backend.args, name, parameter)
                values = [(name, tensor)]
                if experts:
                    group = backend._megatron.mpu.get_expert_model_parallel_group()
                    count = dist.get_world_size(group)
                    names = [None] * count
                    dist.all_gather_object(names, name, group=group)
                    tensors = [torch.empty_like(tensor) for _ in range(count)]
                    dist.all_gather(tensors, tensor, group=group)
                    values = list(zip(names, tensors, strict=True))
                if conversion_error is None:
                    try:
                        for global_name, value in values:
                            bucket.extend(backend._bridge_converter.convert(global_name, value))
                    except Exception as error:
                        conversion_error = error
                        bucket.clear()
                nbytes += sum(value.numel() * value.element_size() for _, value in values)
                if nbytes >= backend.args.update_weight_buffer_size:
                    self._phase(check_conversion)
                    yield bucket
                    bucket, nbytes = [], 0
            if nbytes:
                self._phase(check_conversion)
                yield bucket
                bucket, nbytes = [], 0

    def _create_direct(self, manifest: dict) -> Any:
        from .sender import MoeBlockDirectShards

        return MoeBlockDirectShards(self, manifest)
