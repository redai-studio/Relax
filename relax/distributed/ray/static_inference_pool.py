# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static engine implementations hosted inside the public inference owner.

The implementation objects are local, never additional Ray manager actors.
Their placement hooks retain ownership of dedicated teacher PGs; GenRM borrows
its Service PG.
"""

from typing import Any

import ray

from relax.engine.inference.types import LifecycleState, ModelSnapshot, ReplicaSnapshot, Role
from relax.utils.http_utils import _wrap_ipv6


class StaticInferencePool:
    def __init__(self, role: Role, args: Any, binding: Any, options: dict[str, Any]) -> None:
        from relax.engine.inference.config_adapters import static_role_spec

        self.backend: Any = None
        self.spec = static_role_spec(role, args, options)
        self.role = role
        self.model_id = self.spec.models[0].model_id
        if role is Role.GENRM:
            from relax.distributed.ray.genrm import GenRMManager

            implementation = GenRMManager.__ray_metadata__.modified_class
            ctor_args = (args, binding)
            ctor_kwargs = {key: options[key] for key in ("bundle_offset", "port_window_index") if key in options}
            ctor_kwargs["dedicated_pg"] = True
        else:
            from relax.distributed.ray.teacher_manager import TeacherManager

            implementation = TeacherManager.__ray_metadata__.modified_class
            ctor_args = (args, options["num_replicas"], options["gpus_per_replica"])
            ctor_kwargs = {}
        self.backend = implementation.__new__(implementation)
        try:
            implementation.__init__(self.backend, *ctor_args, **ctor_kwargs)
        except BaseException:
            if hasattr(self.backend, "all_engines"):
                self.dispose()
            raise

    def record_placement(self, planner: Any) -> None:
        from relax.engine.inference.placement import PlacementGroupView, PlacementOwner, PlacementRequest

        if self.role is Role.GENRM:
            bindings = [(self.backend.pg, False, self.backend.bundle_offset, self.spec.models[0].total_num_gpus)]
        else:
            bindings = [
                (binding, owned, 0, self.backend.gpus_per_replica)
                for binding, owned in self.backend._engine_placements.values()
                if owned
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
                    direct_eligible=available,
                )
            )
        return (
            ModelSnapshot(
                self.model_id,
                tuple(replicas),
                state=state,
                admission=any(replica.direct_eligible for replica in replicas),
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
        from relax.distributed.ray.multi_engine_manager import MultiEngineManager

        self.backend._retire_engines(dead)
        # Dedicated teachers can now move: clients resolve a fresh Manager
        # snapshot instead of keeping URLs captured at startup.
        MultiEngineManager.recover(self.backend)

    def dispose(self) -> None:
        if self.backend is None:
            return
        self.backend.shutdown()
        lock = getattr(self.backend, "genrm_engine_lock", None)
        if lock is not None:
            ray.kill(lock)
