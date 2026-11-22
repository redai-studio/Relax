# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static engine implementations hosted inside the public inference owner.

The implementation objects are local, never additional Ray manager actors.
Their placement hooks retain ownership of dedicated teacher PGs; GenRM borrows
its Service PG.
"""

from copy import copy
from typing import Any

import ray

from relax.engine.inference.types import DeploymentMode, LifecycleState, ModelSnapshot, ReplicaSnapshot, Role
from relax.utils.http_utils import _wrap_ipv6


class StaticInferencePool:
    def __init__(self, role: Role, args: Any, binding: Any, options: dict[str, Any]) -> None:
        from relax.engine.inference.config_adapters import static_role_spec

        self.start_sleeping = False
        self.backend: Any = None
        self.spec = static_role_spec(role, args, options)
        self.start_sleeping = self.spec.deployment.mode is DeploymentMode.DEFER or (
            self.spec.deployment.mode is DeploymentMode.SPLIT and bool(getattr(args, "offload_rollout", False))
        )
        if self.start_sleeping:
            args = copy(args)
            if role is Role.GENRM:
                args.genrm_engine_config = dict(args.genrm_engine_config or {}, enable_memory_saver=True)
            else:
                args.teacher_sglang_enable_memory_saver = True
        self.role = role
        self.model_id = self.spec.models[0].model_id
        if role is Role.GENRM:
            ctor_args = (args,)
            ctor_kwargs = {key: options[key] for key in ("bundle_offset", "port_window_index") if key in options}
            ctor_kwargs["dedicated_pg"] = self.spec.deployment.mode is DeploymentMode.DECOUPLED
        else:
            ctor_kwargs = {"num_replicas": options["num_replicas"], "gpus_per_replica": options["gpus_per_replica"]}
            shared = self.spec.deployment.mode is not DeploymentMode.DECOUPLED
            teacher_args = args
            if self.spec.deployment.mode is DeploymentMode.DEFER:
                # The backend indexes the teacher region after rollout. Deferred
                # scoring reuses that region from bundle zero instead.
                teacher_args.rollout_num_gpus = 0
                teacher_args._requires_weights_cpu_backup = True
            ctor_args = (teacher_args,)
            ctor_kwargs.update(pg=binding, shared_pg=shared, bundle_offset=options.get("bundle_offset", 0))
        from relax.distributed.ray.engine_pool import EnginePool

        self.backend = EnginePool.__new__(EnginePool)
        try:
            EnginePool.__init__(self.backend, role.value, *ctor_args, binding, ctor_kwargs)
            if self.start_sleeping:
                # Memory saver enables release; it does not make startup itself
                # memory-free. The launcher must first release the previous owner.
                self.backend.offload()
        except BaseException:
            if hasattr(self.backend, "all_engines"):
                self.dispose()
            raise

    def record_placement(self, planner: Any) -> None:
        from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest

        if self.role is Role.GENRM:
            _, _, offset = self.backend._resolve_placement(0)
            bindings = [(self.backend.pg, False, offset, self.spec.models[0].total_num_gpus)]
        else:
            bindings = [
                (binding, owned, self.backend._resolve_placement(rank)[2], self.backend.gpus_per_replica)
                for rank, (binding, owned) in self.backend._engine_placements.items()
                if rank % self.backend.nodes_per_engine == 0
            ]
        tp = self.spec.models[0].num_gpus_per_engine
        for index, (binding, owned, offset, total) in enumerate(bindings):
            pg, indices, gpu_ids = binding
            view = PlacementGroupView(
                tuple(indices),
                tuple(int(gpu) for gpu in gpu_ids),
                PlacementOwner.MANAGER if owned else PlacementOwner.CONTROLLER,
                identity=pg,
                node_ids=getattr(binding, "node_ids", None),
                local_gpu_ids=getattr(binding, "local_gpu_ids", None),
                bundle_count=getattr(pg, "bundle_count", None),
            )
            planner.plan(
                [
                    PlacementRequest(
                        f"{self.role.value}/{self.model_id}/{index}",
                        "regular",
                        total,
                        tp,
                        self.backend.args.num_gpus_per_node,
                        bundle_offset=offset,
                        phase=self.spec.deployment.phase,
                    )
                ],
                view,
            )

    def observe(self, *, ready: bool, state: LifecycleState) -> tuple[ModelSnapshot, ...]:
        replicas = []
        for rank in range(0, len(self.backend.all_engines), self.backend.nodes_per_engine):
            slots = self.backend.all_engines[rank : rank + self.backend.nodes_per_engine]
            info = self.backend._engine_addr_and_ports.get(rank)
            available = ready and all(engine is not None for engine in slots) and info is not None
            replicas.append(
                ReplicaSnapshot(
                    f"{self.model_id}/replica-{rank}",
                    LifecycleState.READY if available else state,
                    f"http://{_wrap_ipv6(info['host'])}:{info['port']}" if info else None,
                    direct_eligible=False,
                )
            )
        router = getattr(self.backend, "router", None)
        router_ready = ready and router is not None and router.ready
        return (
            ModelSnapshot(
                self.model_id,
                tuple(replicas),
                router_url=router.url if router is not None else None,
                state=state,
                admission=router_ready and any(replica.state is LifecycleState.READY for replica in replicas),
            ),
        )

    def unhealthy_ranks(self) -> list[int]:
        dead = []
        for rank in range(0, len(self.backend.all_engines), self.backend.nodes_per_engine):
            engine = self.backend.all_engines[rank]
            if engine is None:
                dead.append(rank)
                continue
            try:
                if not ray.get(engine.health_generate.remote(), timeout=5):
                    dead.append(rank)
            except Exception:
                dead.append(rank)
        return dead

    def recover(self, dead: list[int]) -> None:
        # Dedicated teachers can now move: clients resolve a fresh Manager
        # snapshot instead of keeping URLs captured at startup.
        if dead:
            # A dead actor cannot withdraw its Router entry. Clear that
            # registry before a replacement can reuse the worker endpoint.
            self.backend._close_router()
        self.backend._retire_engines(dead)
        self.backend._ensure_router_running()
        self.backend.recover()
        if not self.backend.router.ready:
            self.backend.router.activate(self.backend.engines)

    def dispose(self) -> None:
        if self.backend is None:
            return
        self.backend.shutdown()
