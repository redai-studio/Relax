# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Adapt the current engine implementation without duplicating its recovery
code.

The legacy actor still serves unmigrated tasks. The managed path instantiates
its implementation locally, in engine-only mode: no second Ray actor, workload
or TransferQueue client owns the engines. This adapter is removed when the
remaining roles no longer need the legacy implementation.
"""

from typing import Any

import ray

from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementPlanner, PlacementRequest
from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot
from relax.utils.http_utils import _wrap_ipv6


def plan_rollout(args: Any, binding: Any, planner: PlacementPlanner) -> None:
    from relax.distributed.ray.rollout import _resolve_sglang_config
    from relax.engine.inference.config import EngineGroupSpec, InferenceModelSpec

    pg, indices, gpu_ids = binding
    view = PlacementGroupView(
        tuple(indices),
        tuple(int(gpu) for gpu in gpu_ids),
        PlacementOwner.CONTROLLER,
        identity=pg,
        node_ids=binding.node_ids,
        bundle_count=pg.bundle_count,
    )
    requests = []
    offset = 0
    for model in _resolve_sglang_config(args).models:
        spec = InferenceModelSpec(
            model.name,
            model_path=model.model_path,
            num_gpus_per_engine=model.num_gpus_per_engine,
            engine_groups=[
                EngineGroupSpec(group.worker_type, group.num_gpus, group.num_gpus_per_engine, dict(group.overrides))
                for group in model.engine_groups
            ],
        ).resolved(args)
        for index, group in enumerate(spec.engine_groups):
            requests.append(
                PlacementRequest(
                    f"rollout/{model.name}/group-{index}",
                    group.worker_type,
                    group.num_gpus,
                    group.num_gpus_per_engine,
                    args.num_gpus_per_node,
                    bundle_offset=offset,
                    active=group.worker_type != "placeholder",
                )
            )
            offset += group.num_gpus
    planner.plan(requests, view)


class RolloutPool:
    def __init__(self, args: Any, binding: Any) -> None:
        from relax.distributed.ray.rollout import RolloutManager, stop_launched_routers

        implementation = RolloutManager.__ray_metadata__.modified_class
        self.backend = implementation.__new__(implementation)
        try:
            implementation.__init__(self.backend, args, binding, inference_only=True)
        except BaseException:
            try:
                if hasattr(self.backend, "_eviction_monitor_stop"):
                    self.backend.dispose()
                elif hasattr(self.backend, "servers"):
                    self.backend._shutdown_all_engines()
            finally:
                try:
                    stop_launched_routers()
                finally:
                    self._dispose_locks()
            raise
        self.synced_actor_ids = {
            engine._actor_id.hex()
            for server in self.backend.servers.values()
            for engine in server.all_engines
            if engine is not None
        }

    def mark_synced(self, actor_ids: set[str]) -> None:
        self.synced_actor_ids.update(actor_ids)

    def observe(self, *, ready: bool, state: LifecycleState) -> tuple[ModelSnapshot, ...]:
        from relax.distributed.ray.rollout import EngineGroupLifecycle, _read_engine_metadata

        snapshots = []
        # Re-read generation-checked metadata on each snapshot/admission. Health
        # monitors can replace slots independently of explicit Manager calls.
        lock = self.backend._engine_lifecycle_lock
        if not lock.acquire(blocking=False):
            return tuple(
                ModelSnapshot(name, state=LifecycleState.STARTING, issues=((name, "topology_busy"),))
                for name in self.backend.servers
            )
        try:
            ready = (
                ready and not self.backend._training_weight_updating and not self.backend._scale_out_weight_updating
            )
            for name, server in self.backend.servers.items():
                replicas = []
                workers = []
                issues = []
                available_types = set()
                for group in list(server.engine_groups):
                    if group.worker_type == "placeholder":
                        continue
                    slots = list(group.all_engines)
                    observations = [
                        _read_engine_metadata(
                            group, slot, engine, group.engine_init_records.get(slot), self.backend._engine_actor_states
                        )
                        for slot, engine in enumerate(slots)
                    ]

                    def confirmed(slot: int) -> bool:
                        engine = slots[slot]
                        record = group.engine_init_records.get(slot)
                        metadata, _, issue = observations[slot]
                        return bool(
                            engine is not None
                            and metadata is not None
                            and issue is None
                            and record is not None
                            and not record.pending_router_registration
                            and (group.is_scaled_out or engine._actor_id.hex() in self.synced_actor_ids)
                            and group.lifecycle_status is EngineGroupLifecycle.ACTIVE
                        )

                    for slot, engine in enumerate(slots):
                        record = group.engine_init_records.get(slot)
                        metadata, actor_state, issue = observations[slot]
                        identity = f"{name}/engine-{group.rank_offset + slot}"
                        pending = record is not None and record.pending_router_registration
                        unsynced = (
                            engine is not None
                            and not group.is_scaled_out
                            and engine._actor_id.hex() not in self.synced_actor_ids
                        )
                        head = slot - slot % group.nodes_per_engine
                        end = head + group.nodes_per_engine
                        eligible = end <= len(slots) and all(confirmed(index) for index in range(head, end))
                        if issue or pending or unsynced:
                            issues.append(
                                (identity, issue or ("weights_pending" if unsynced else "router_registration_pending"))
                            )
                        replica_state = state
                        if actor_state in ("DEAD", "ABSENT"):
                            replica_state = LifecycleState.DEAD
                        elif not eligible:
                            replica_state = LifecycleState.STARTING
                        elif ready:
                            replica_state = LifecycleState.READY
                        replica = ReplicaSnapshot(
                            identity,
                            replica_state,
                            metadata.get("url") if metadata else None,
                        )
                        if slot % group.nodes_per_engine == 0:
                            if group.worker_type in ("prefill", "decode"):
                                workers.append((group.worker_type, replica))
                            else:
                                replicas.append(replica)
                        if slot % group.nodes_per_engine == 0 and eligible and replica_state is LifecycleState.READY:
                            available_types.add(group.worker_type)
                pd = bool(workers)
                serves = ready and ({"prefill", "decode"} <= available_types if pd else "regular" in available_types)
                router = (
                    f"http://{_wrap_ipv6(server.router_ip)}:{server.router_port}"
                    if server.router_ip and server.router_port
                    else None
                )
                snapshots.append(
                    ModelSnapshot(
                        name,
                        tuple(replicas),
                        router,
                        LifecycleState.READY if serves and router else state,
                        admission=bool(serves and router),
                        pd_workers=tuple(workers),
                        issues=tuple(issues),
                    )
                )
            return tuple(snapshots)
        finally:
            lock.release()

    def dispose(self) -> None:
        try:
            self.backend.dispose()
        finally:
            self._dispose_locks()

    def _dispose_locks(self) -> None:
        for name in ("rollout_engine_lock", "_weight_sync_lock"):
            actor = getattr(self.backend, name, None)
            if actor is not None:
                ray.kill(actor)
